import math
import numpy as np
import torch 
from torch import nn 
import scipy

class BoltzmannTransportEquation_HeteroStruct(nn.Module):
    def __init__(self, models, eps_bdry, eps_bulk, traref_params_iv=None, traref_fixed=None, tau_cap_iv=None, tau_coeff_cap=None, bs=1):
        super(BoltzmannTransportEquation_HeteroStruct, self).__init__()
        self.model_cap, self.model_sub = models
        self.cmidx = self.model_cap.nzidx * self.model_sub.nzidx
        self.traref_fixed = traref_fixed
        self.bs = bs

        # self.eps_bdry = eps_bdry
        self.eps_bdry = nn.Parameter(eps_bdry, requires_grad=True)
        self.eps_bulk = nn.Parameter(eps_bulk, requires_grad=True)
        # self.eps_bulk = eps_bulk

        if tau_cap_iv is not None:
            self.tau_cap = nn.Parameter(tau_cap_iv, requires_grad=True) # [ps]
        elif tau_coeff_cap is not None:
            self.tau_cap = None

            self.num_tau_basis = tau_coeff_cap.shape[1]
            self.tau_basis = self.get_basis(self.num_tau_basis)
            self.tau_coeff_cap = nn.Parameter(tau_coeff_cap, requires_grad=True)
        else:
            self.tau_cap = None

        if traref_fixed is None:
            self.traref_params = nn.Parameter(traref_params_iv, requires_grad=True)
        else:
            self.traref_params = traref_params_iv

    def get_basis(self, num_basis):
        omega = self.model_cap.omega[self.model_cap.nzidx.any(dim=1)]
        center_basis = torch.linspace(omega[0], omega[-1], num_basis).to(omega)
        h = center_basis[1] - center_basis[0]
        # basis of shape [num_basis, Nf]
        tau_basis = torch.zeros((num_basis, self.model_cap.Nf)).to(omega)
        assert num_basis >=2

        if num_basis > 2:
            for i in range(0, num_basis):
                if i == 0:
                    tau_basis[i, :len(omega)] += (- 1 / h * omega + center_basis[i+1] / h) \
                        * (omega >= center_basis[i]) * (omega < center_basis[i+1])
                elif i == num_basis-1:
                    tau_basis[i, :len(omega)] += (1 / h * omega - center_basis[i-1] / h) \
                        * (omega > center_basis[i-1]) * (omega <= center_basis[i])
                else:
                    tau_basis[i, :len(omega)] += (1 / h * omega - center_basis[i-1] / h) \
                        * (omega > center_basis[i-1]) * (omega < center_basis[i])
                    tau_basis[i, :len(omega)] += (- 1 / h * omega + center_basis[i+1] / h) \
                        * (omega >= center_basis[i]) * (omega < center_basis[i+1])
        elif num_basis == 2:
            tau_basis[0, :len(omega)] += (- 1 / h * omega + center_basis[1] / h) \
                * (omega >= center_basis[0]) * (omega < center_basis[1])
            tau_basis[1, :len(omega)] += (1 / h * omega - center_basis[0] / h) \
                * (omega > center_basis[0]) * (omega <= center_basis[1])

        return tau_basis 

    def get_tau(self, tau_coeff):
        '''
        tau_coeff of shape: bs_params(b), num_basis(n), Nb(l)
        tau_basis of shape: num_basis(n), Nf(k)
        return tau in [ps]
        '''
        tau = torch.einsum("bnl, nk, kl -> bkl", tau_coeff, self.tau_basis, self.model_cap.nzidx)
        tau[:, self.model_cap.nzidx] = torch.maximum(
            tau[:, self.model_cap.nzidx], 
            5 * self.model_cap.dt * torch.ones(tau[:, self.model_cap.nzidx].shape).to(self.tau_basis)
            )
        return tau
    
    def get_batch_traref(self, traref_params):
        '''
        input:
        traref_params: bs, Nf, Nb
        '''

        cmidx = self.model_cap.nzidx & self.model_sub.nzidx
        traref = torch.zeros((2,2,) + traref_params.shape).to(traref_params)

        # compute traref_params[1,1]/traref_params[0,0] ratio
        ratio = torch.zeros(traref_params.shape).to(traref_params)
        # ratio = T_21 / T_12
        ratio[:, cmidx] = (self.model_cap.dos[cmidx] * self.model_cap.vg[cmidx]) / (self.model_sub.dos[cmidx] * self.model_sub.vg[cmidx] + 1e-12)
        # transmisstion coefficient from sub to cap, detailed balance
        self.idx_tmp_cs = (ratio <= 1.0) * cmidx
        traref[0,0,self.idx_tmp_cs] = traref_params[self.idx_tmp_cs].clone()
        traref[1,1,self.idx_tmp_cs] = traref[0,0,self.idx_tmp_cs] * ratio[self.idx_tmp_cs]

        self.idx_tmp_sc = (ratio > 1.0) * cmidx
        traref[1,1,self.idx_tmp_sc] = traref_params[self.idx_tmp_sc]
        traref[0,0,self.idx_tmp_sc] = traref[1,1,self.idx_tmp_sc] / ratio[self.idx_tmp_sc]
        
        traref = traref.permute(2,0,1,3,4)
        # assert detailed balance
        assert torch.allclose(
            torch.einsum("bk, k -> bk", traref[:,0,0,cmidx], self.model_cap.dos[cmidx] * self.model_cap.vg[cmidx]),
            torch.einsum("bk, k -> bk", traref[:,1,1,cmidx], self.model_sub.dos[cmidx] * self.model_sub.vg[cmidx])
            )
        # reflection coefficient from cap to cap
        traref[:, 0, 1, self.model_cap.nzidx] = 1 - traref[:, 0, 0, self.model_cap.nzidx]
        # reflection coefficient from sub to sub
        traref[:, 1, 0, self.model_sub.nzidx] = 1 - traref[:, 1, 1, self.model_sub.nzidx]
        assert torch.all(traref <= 1.0) and torch.all(traref >= 0.0)

        return traref
    
    def forward(self, t, g, dT_lwall=0, dT_rwall=0):
        '''
        this version of forward allows each batch sample has its own parameters,
        thus we can get statistics of predicted parameters in single run
        g of size: (bs, Nx, Nf, Nb, Nm)
        '''
        if self.traref_fixed is not None:
            # self.traref = self.get_batch_traref(self.traref_params)
            self.traref = self.traref_fixed.to(g)
        else:
            self.traref = self.get_batch_traref(self.traref_params)
        g_cap, g_sub = g[:, :self.model_cap.Nx], g[:, self.model_cap.Nx:]
        dgdt = torch.zeros(g.size()).to(g)

        # update layer temperatures
        self.model_cap.T = self.model_cap.solve_PseudoTemp_interp(g_cap)
        self.model_sub.T = self.model_sub.solve_PseudoTemp_interp(g_sub)

        # interface condition
        cap_rwall_tmp = \
            torch.trapz(
                torch.einsum("bkl, bjklm, m -> bjklm", self.traref[:, 0, 1] * self.model_cap.vg, 
                g_cap[:, None,-1,...,self.model_cap.mu > 0], self.model_cap.mu[self.model_cap.mu > 0]), 
                self.model_cap.mu[self.model_cap.mu>0], dim=-1) - \
            torch.trapz(
                torch.einsum("bkl, bjklm, m -> bjklm", self.traref[:, 1, 1] * self.model_sub.vg, 
                g_sub[:, None,0,...,self.model_sub.mu < 0], self.model_sub.mu[self.model_sub.mu < 0]), 
                self.model_sub.mu[self.model_sub.mu < 0], dim=-1)

        sub_lwall_tmp = - \
            torch.trapz(
                torch.einsum("bkl, bjklm, m -> bjklm", self.traref[:, 1, 0] * self.model_sub.vg, 
                g_sub[:, None,0,...,self.model_sub.mu < 0], self.model_sub.mu[self.model_sub.mu < 0]), 
                self.model_sub.mu[self.model_sub.mu < 0], dim=-1) + \
            torch.trapz(
                torch.einsum("bkl, bjklm, m -> bjklm", self.traref[:, 0, 0] * self.model_cap.vg, 
                g_cap[:, None,-1,...,self.model_cap.mu > 0], self.model_cap.mu[self.model_cap.mu > 0]
                ), self.model_cap.mu[self.model_cap.mu > 0], dim=-1)

        # update cap layer
        # left wall
        cap_lwall = torch.einsum("b, bjklm -> bjklm",
            self.eps_bdry[:, 0], self.model_cap.calc_g_equil(self.model_cap.T_base + dT_lwall)[:, :2])
        cap_lwall[..., self.model_cap.mu > 0] -= \
            1 / torch.trapz(self.model_cap.mu[self.model_cap.mu < 0], self.model_cap.mu[self.model_cap.mu < 0]).abs() * \
            (torch.trapz(
                torch.einsum("b, bjklm, m -> bjklm", (1 - self.eps_bdry[:, 0]), g_cap[:, None,0,...,self.model_cap.mu<0], 
                self.model_cap.mu[self.model_cap.mu<0]), self.model_cap.mu[self.model_cap.mu<0]
            ).repeat_interleave(2, dim=1))[...,None].repeat_interleave(self.model_cap.Nm // 2, dim=-1)
        cap_lwall[:, 1, ..., self.model_cap.mu < 0] = 2 * g_cap[:, 0,...,self.model_cap.mu < 0] - g_cap[:, 1,...,self.model_cap.mu < 0]
        cap_lwall[:, 0, ..., self.model_cap.mu < 0] = 2 * cap_lwall[:, 1,..., self.model_cap.mu < 0] - g_cap[:, 0,...,self.model_cap.mu < 0]

        # right wall
        cap_rwall = torch.zeros((g.size(0), 2, self.model_cap.Nf, self.model_cap.Nb, self.model_cap.Nm)).to(g)
        cap_rwall_tmp = 1 / torch.trapz(self.model_cap.mu[self.model_cap.mu < 0],self.model_cap.mu[self.model_cap.mu < 0]).abs() * \
            torch.einsum("bjkl, kl -> bjkl", cap_rwall_tmp, 1 / (self.model_cap.vg + 1e-12))[...,None].repeat_interleave(self.model_cap.Nm // 2, dim=-1)
        cap_rwall[...,self.model_cap.mu < 0] = cap_rwall_tmp.repeat_interleave(2, dim=1)
        cap_rwall[:, 0,...,self.model_cap.mu > 0] = 2 * g_cap[:, -1,...,self.model_cap.mu > 0] - g_cap[:, -2,...,self.model_cap.mu > 0]
        cap_rwall[:, 1,...,self.model_cap.mu > 0] = 2 * cap_rwall[:, 0,...,self.model_cap.mu > 0] - g_cap[:, -1,...,self.model_cap.mu > 0]
        # construct ghost cells
        self.model_cap.g_gc = self.model_cap.attach_ghostcells(g_cap, [cap_lwall, cap_rwall], [0, -1])
        # obtain cell average rate
        if self.tau_cap is not None:
            self.model_cap.tau = self.tau_cap * 1e-12
            dgdt_cap = self.model_cap.forward(self.model_cap.g_gc)
        elif self.tau_coeff_cap is not None:
            self.model_cap.tau = self.get_tau(self.tau_coeff_cap) * 1e-12
            dgdt_cap = self.model_cap.forward(self.model_cap.g_gc)
        else:
            dgdt_cap = self.model_cap.forward(self.model_cap.g_gc)
        # print("dgdt_cap:")
        # print(dgdt_cap[0,:,0,0,-1])
        # update sub layer
        # left wall
        sub_lwall = torch.zeros((g.size(0), 2, self.model_sub.Nf, self.model_sub.Nb, self.model_sub.Nm)).to(g)
        sub_lwall_tmp = 1 / torch.trapz(self.model_sub.mu[self.model_cap.mu > 0],self.model_sub.mu[self.model_cap.mu > 0]).abs() * \
            torch.einsum("bjkl, kl -> bjkl", sub_lwall_tmp, 1 / (self.model_sub.vg + 1e-12))[...,None].repeat_interleave(self.model_sub.Nm // 2, dim=-1)
        sub_lwall[...,self.model_sub.mu > 0] = sub_lwall_tmp.repeat_interleave(2, dim=1)
        sub_lwall[:, 1,...,self.model_cap.mu < 0] = 2 * g_sub[:, 0, ..., self.model_sub.mu < 0] - g_sub[:, 1, ..., self.model_sub.mu < 0]
        sub_lwall[:, 0,...,self.model_cap.mu < 0] = 2 * sub_lwall[:, 1, ..., self.model_sub.mu < 0] - g_sub[:, 0, ..., self.model_sub.mu < 0]
        
        # right wall
        sub_rwall = torch.einsum("b, bjklm -> bjklm", 
            self.eps_bdry[:, 1], self.model_sub.calc_g_equil(self.model_sub.T_base + dT_rwall)[:, :2])
        sub_rwall[...,self.model_sub.mu < 0] += \
            1 / torch.trapz(self.model_sub.mu[self.model_cap.mu < 0],self.model_sub.mu[self.model_cap.mu < 0]).abs() * \
            (torch.trapz(
                torch.einsum("b, bjklm, m -> bjklm", (1 - self.eps_bdry[:, 1]), g_sub[:, None,-1,...,self.model_sub.mu>0], 
                self.model_sub.mu[self.model_sub.mu>0]
            ), self.model_sub.mu[self.model_sub.mu>0]).repeat_interleave(2, dim=1))[...,None].repeat_interleave(self.model_sub.Nm // 2, dim=-1)
        sub_rwall[:, 0,..., self.model_sub.mu > 0] = 2 * g_sub[:, -1,...,self.model_sub.mu > 0] - g_sub[:, -2,...,self.model_sub.mu > 0]
        sub_rwall[:, 1,..., self.model_sub.mu > 0] = 2 * sub_rwall[:, 0,..., self.model_sub.mu > 0] - g_sub[:, -1,...,self.model_sub.mu > 0]
        # construct ghost cells
        self.model_sub.g_gc = self.model_sub.attach_ghostcells(g_sub, [sub_lwall, sub_rwall], [0, -1])
        # obtain cell average rate
        dgdt_sub = self.model_sub.forward(self.model_sub.g_gc)
        # print("dgdt_sub:")
        # print(dgdt_sub[0,:,0,0,-1])
        
        # bulk energy loss
        dgdt_cap = dgdt_cap - torch.einsum("b, bjklm -> bjklm", self.eps_bulk[:,0], g_cap)
        dgdt_sub = dgdt_sub - torch.einsum("b, bjklm -> bjklm", self.eps_bulk[:,1], g_sub)

        # put together
        dgdt[:, :self.model_cap.Nx] = dgdt_cap
        dgdt[:, self.model_cap.Nx:] = dgdt_sub

        return dgdt