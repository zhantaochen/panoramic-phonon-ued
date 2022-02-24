import numpy as np
from .utils import compute_heatcap
import torch
import torch.nn as nn
from tqdm import tqdm
from scipy import interpolate
import scipy.constants as const
from pymatgen.core.periodic_table import Element

class PhononBTE:
    
    def __init__(self, mater_prop, dt, dx, Lx, Nm, del_T=0.1, T_base=300, flag_approx=False, device='cpu'):
        self.device = device
        self.flag_approx = flag_approx
        self.dt = dt

        # load properties used to calculate debye-waller factor
        self.struct = mater_prop['struct']
        self.pdos_ratio = mater_prop['pdos_ratio']

        # imported units are in rad, s, m, kg
        self.omega = mater_prop['omega'].to(device) 
        self.dos = mater_prop['DOS'].to(device)
        self.vg = mater_prop['vg'].to(device)
        self.tau = mater_prop['tau'].to(device)
        
        self.Nf, self.Nb = self.dos.shape
        
        # convert to ev
        self.hbar = const.hbar / const.value('electron volt') 
        self.k = const.k / const.value('electron volt')
        
        # self.hbar = const.hbar
        # self.k = const.k
        
        self.Lx = Lx
        self.dx = dx
        self.xn = torch.arange(dx / 2, Lx + dx / 2, dx).to(device)
        self.Nx = len(self.xn)
        
        self.dt_max = self.dx / (self.vg.max() * 1e9 / 1e12)
        
        self.Nm = Nm
        self.mu = torch.linspace(-1,1,Nm).to(device)
        
        self.del_T = del_T # resolution for solving lattice temperature
        self.T_base_scalar = T_base
        self.T_base = T_base * torch.ones((1, self.Nx)).to(self.device)
        self.f_base = 1 / (torch.exp(torch.einsum("bj, k", 1 / (self.k * self.T_base), self.hbar * self.omega)) - 1)
        self.nzidx = ((self.dos > 0) * (self.tau > 0) * (self.vg > 0)).to(device)
        
        if self.flag_approx:
            self.Cv = compute_heatcap(self.omega, self.dos, T_base).to(device)
            # self.Cv *=  1e10


    def calc_g_equil(self, T):
        if self.flag_approx:
            g0 = 1 / (4 * np.pi) * torch.einsum("bj, kl -> bjkl", (T - self.T_base), self.Cv)[..., None].repeat_interleave(self.Nm, -1)
        else:
            fT = 1 / (torch.exp(torch.einsum("bj,k -> bjk", 1 / (self.k * T), self.hbar * self.omega)) - 1)
            df = ((fT - self.f_base)[..., None].repeat_interleave(self.Nb, -1))[..., None].repeat_interleave(self.Nm, -1)
            g0 = 1 / (4 * np.pi) * self.hbar * torch.einsum("bjklm,kl,k -> bjklm", df, self.dos, self.omega)
        return g0
        
    def init_distribution(self, T, T_lims=[250, 1000]):
        self.bs = T.shape[0]
        self.T_base = self.T_base_scalar * torch.ones((self.bs, self.Nx)).to(self.device)
        self.f_base = 1 / (torch.exp(torch.einsum("bj, k", 1 / (self.k * self.T_base), self.hbar * self.omega)) - 1)
        self.T = T
        # g is energy density [J / L ** 3] per frequency [rad/s]
        # however we will work in nm and ps for propagation of this distribution
        self.g = self.calc_g_equil(T)
        self.TempList, self.EngList, self.PseudoEngList = self.compute_InterpChart_Full(T_lims[0], T_lims[1])
        self.func_interp = interpolate.interp1d(self.EngList.detach().cpu().numpy(), self.TempList.detach().cpu().numpy())
        self.func_interp_Pseudo = [interpolate.interp1d(self.PseudoEngList.detach().cpu().numpy()[i], self.TempList.detach().cpu().numpy()) for i in range(self.bs)]

    def compute_InterpChart_Full(self, T_min, T_max):
        # omega = self.omega[:,None].repeat_interleave(self.Nb, -1)

        T_list = torch.linspace(T_min, T_max, int((T_max - T_min) // self.del_T)).to(self.g)

        fT = 1 / (torch.exp(torch.einsum("j,k -> jk", 1 / (self.k * T_list), self.hbar * self.omega)) - 1)
        fT0 = 1 / (torch.exp(self.hbar * self.omega / (self.k * self.T_base_scalar)) - 1)
        df = ((fT - fT0)[..., None].repeat_interleave(self.Nb, -1))[..., None].repeat_interleave(self.Nm, -1)
        g_eq = 1 / (4 * np.pi) * self.hbar * torch.einsum("jklm,kl,k -> jklm", df, self.dos, self.omega)

        # coll_integrand_eq = self.hbar * torch.einsum("kl, jklm-> jklm", omega, g_eq)

        coll_integrand_eq = g_eq
        coll_eq = torch.trapz(torch.trapz(coll_integrand_eq, self.mu, dim=-1), self.omega, dim=-2).sum(dim=-1).detach()

        coll_integrand_eq_pseudo = torch.zeros_like(g_eq[None,...].repeat_interleave(self.bs, dim=0))
        if len(self.tau.shape) == 2:
            coll_integrand_eq_pseudo[:,:,self.nzidx,:] = torch.einsum("jkm, bk -> bjkm", g_eq[:,self.nzidx,:], 1 / self.tau[None,self.nzidx].repeat_interleave(self.bs, dim=0))
        elif len(self.tau.shape) == 3:
            coll_integrand_eq_pseudo[:,:,self.nzidx,:] = torch.einsum("jkm, bk -> bjkm", g_eq[:,self.nzidx,:], 1 / self.tau[:,self.nzidx])
        coll_eq_pseudo = torch.trapz(torch.trapz(coll_integrand_eq_pseudo, self.mu, dim=-1), self.omega, dim=-2).sum(dim=-1).detach()

        return T_list.detach(), coll_eq, coll_eq_pseudo

    def update_InterpChart_PseudoTemp(self, T_min, T_max):
        T_list = torch.linspace(T_min, T_max, int((T_max - T_min) // self.del_T)).to(self.g)
        # omega = self.omega[:,None].repeat_interleave(self.Nb, -1)
        fT = 1 / (torch.exp(torch.einsum("j,k -> jk", 1 / (self.k * T_list), self.hbar * self.omega)) - 1)
        fT0 = 1 / (torch.exp(self.hbar * self.omega / (self.k * self.T_base_scalar)) - 1)
        df = ((fT - fT0)[..., None].repeat_interleave(self.Nb, -1))[..., None].repeat_interleave(self.Nm, -1)
        g_eq = 1 / (4 * np.pi) * self.hbar * torch.einsum("jklm,kl,k -> jklm", df, self.dos, self.omega)

        coll_integrand_eq_pseudo = torch.zeros_like(g_eq[None,...].repeat_interleave(self.bs, dim=0))
        if len(self.tau.shape) == 2:
            coll_integrand_eq_pseudo[:,:,self.nzidx,:] = torch.einsum("jkm, bk -> bjkm", g_eq[:,self.nzidx,:], 1 / self.tau[None,self.nzidx].repeat_interleave(self.bs, dim=0))
        elif len(self.tau.shape) == 3:
            coll_integrand_eq_pseudo[:,:,self.nzidx,:] = torch.einsum("jkm, bk -> bjkm", g_eq[:,self.nzidx,:], 1 / self.tau[:,self.nzidx])
        self.PseudoEngList = torch.trapz(torch.trapz(coll_integrand_eq_pseudo, self.mu, dim=-1), self.omega, dim=-2).sum(dim=-1).detach()
        self.func_interp_Pseudo = [interpolate.interp1d(self.PseudoEngList.detach().cpu().numpy()[i], T_list.detach().cpu().numpy()) for i in range(self.bs)]

    def attach_ghostcells(self, g, ghostcells, index):
        g_gc = g.clone()
        for i, _index in enumerate(index):
            if _index == 0:
                g_gc = torch.cat((ghostcells[i], g_gc), dim = 1)
            else:
                g_gc = torch.cat((g_gc, ghostcells[i]), dim = 1)
        return g_gc
        
    def vanLeer(self, r):
        r_abs = torch.abs(r)
        return (r + r_abs) / (1 + r_abs)
    
    def calc_surfvals(self, g_gc):
        fs = torch.zeros((self.bs, self.Nx + 1, self.Nf, self.Nb, self.Nm)).to(self.device)
        dg = g_gc[:, 1:] - g_gc[:, :-1]
        
        vg = torch.einsum("kl,m->klm", self.vg, self.mu) * 1e9 / 1e12 # multiply directional cosine and conver to [nm/ps]
        
        rf = dg[:, :-2, ..., self.mu > 0] / (dg[:, 1:-1, ..., self.mu > 0] + 1e-12)
        fs[..., self.mu > 0] = torch.einsum("bjklm, klm -> bjklm", g_gc[:, 1:-2, ..., self.mu > 0], vg[..., self.mu > 0]) + \
            1 / 2 * torch.einsum("klm, klm, bjklm, bjklm -> bjklm", 
                                 torch.abs(vg[..., self.mu > 0]),
                                 1 - torch.abs(vg[..., self.mu > 0] * self.dt / self.dx), 
                                 self.vanLeer(rf), dg[:, 1:-1, ..., self.mu > 0])
            
        rb = dg[:, 2:, ..., self.mu < 0] / (dg[:, 1:-1, ..., self.mu < 0] + 1e-12)
        fs[..., self.mu < 0] = torch.einsum("bjklm, klm -> bjklm", g_gc[:, 2:-1, ..., self.mu < 0], vg[..., self.mu < 0]) + \
            1 / 2 * torch.einsum("klm, klm, bjklm, bjklm -> bjklm", 
                                 torch.abs(vg[..., self.mu < 0]),
                                 1 - torch.abs(vg[..., self.mu < 0] * self.dt / self.dx),
                                 self.vanLeer(rb), dg[:, 1:-1, ..., self.mu < 0])
        
        return fs
    
    def solve_Temp(self, g):
        g = g.to(self.device)
        if self.flag_approx:
            lhs = torch.zeros((self.bs, self.Nx, self.Nf, self.Nb)).to(self.device)
            rhs = torch.zeros((self.bs, self.Nf, self.Nb)).to(self.device)
            lhs[:, :, self.nzidx] = torch.einsum("bjk, k -> bjk", torch.trapz(g, self.mu, dim=-1)[..., self.nzidx], 1 / self.tau[self.nzidx])
            rhs[:, self.nzidx] = 1 / (4 * np.pi) * self.Cv[self.nzidx] / self.tau[self.nzidx] * 2 # the factor 2 comes from \int_{-1}^{1}\mu d\mu
            LHS = torch.sum(torch.trapz(lhs, self.omega, dim=-2), dim=-1)
            RHS = torch.sum(torch.trapz(rhs, self.omega, dim=-2), dim=-1)
            temperature = LHS / (RHS + 1e-12) + self.T_base
        else:
            omega_tmp = self.omega[:, None].repeat_interleave(self.Nb, -1)
            array_tmp = torch.zeros((self.bs, self.Nx, self.Nf, self.Nb, self.Nm)).to(self.device)
            array_tmp[:, :, self.nzidx, :] = 4 * np.pi / self.hbar * \
                torch.einsum("k, k, bjkm -> bjkm", 
                             1 / omega_tmp[self.nzidx],
                             1 / self.dos[self.nzidx],
                             g[:, :, self.nzidx, :])
            f = torch.trapz(array_tmp, self.mu, dim=-1) / 2 + self.f_base[..., None].repeat_interleave(self.Nb, -1)
            
            # this is an approximated solution ...
            temperature = torch.einsum("k, bjk -> bjk", self.hbar * omega_tmp[self.nzidx],
                                       1 / (self.k * torch.log(1 + 1 / f[:, :, self.nzidx]))).mean(dim=-1)
        
        return temperature

    def solve_Temp_interp(self, g):
        # omega = self.omega[:,None].repeat_interleave(self.Nb, -1)
        # coll_integrand_neq = self.hbar * torch.einsum("kl, bjklm -> bjklm", omega, g)

        coll_integrand_neq = g

        coll_neq = torch.trapz(torch.trapz(coll_integrand_neq, self.mu, dim=-1), self.omega, dim=-2).sum(dim=-1)
        T = torch.from_numpy(self.func_interp(coll_neq.view(-1).detach().cpu().numpy())).to(g)

        return T.view(self.bs, self.Nx)

    def solve_PseudoTemp_interp(self, g):
        # omega = self.omega[:,None].repeat_interleave(self.Nb, -1)
        # coll_integrand_neq = self.hbar * torch.einsum("kl, bjklm -> bjklm", omega, g)

        # coll_integrand_neq = g

        coll_integrand_neq = torch.zeros_like(g)
        if len(self.tau.shape) == 2:
            coll_integrand_neq[:,:,self.nzidx,:] = torch.einsum("bjkm, k -> bjkm", g[:,:,self.nzidx,:], 1 / self.tau[self.nzidx])
        elif len(self.tau.shape) == 3:
            coll_integrand_neq[:,:,self.nzidx,:] = torch.einsum("bjkm, bk -> bjkm", g[:,:,self.nzidx,:], 1 / self.tau[:, self.nzidx])

        coll_neq = torch.trapz(torch.trapz(coll_integrand_neq, self.mu, dim=-1), self.omega, dim=-2).sum(dim=-1)
        # T = torch.from_numpy(self.func_interp(coll_neq.view(-1).detach().cpu().numpy())).to(g)
        T = torch.from_numpy(np.array([self.func_interp_Pseudo[i](coll_neq[i].detach().cpu().numpy()) for i in range(self.bs)])).to(g)

        return T

    def get_tau(self, tau_params):
        '''
    
        Parameters
        ----------
        omega : numpy array
            Frequency in [rad/s].
        tau_params : torch array [A, LA_alpha, LA_beta, TA_alpha, TA_beta]
            invtau_impurity = A * omega ** 4,
            invtau_L = LA_alpha * omega ** 2 * T ** LA_beta,
            invtau_T = TA_alpha * omega ** 2 * T ** TA_beta.
        device : string
            device used by torch.
    
        Returns
        -------
        tau in [s].
    
        '''
        
        tau = torch.zeros((self.Nf, self.Nb)).to(self.device)
        
        invtau_TA = tau_params[0] / self.omega * 1e22 + tau_params[1] * self.omega ** 3 * 1e-29
        # invtau_LA = tau_params[2] / self.omega * 1e22 + tau_params[3] * self.omega ** 3 * 1e-29
        # invtau = tau_params[0] / self.omega * 1e22 + tau_params[1] * self.omega * 1e-3 + tau_params[2] * self.omega ** 3 * 1e-29
        tau[:] = 1 / (invtau_TA + 1)[...,None].repeat_interleave(self.Nb, dim=-1)
        tau = tau * self.nzidx
        tau[self.nzidx] = tau[self.nzidx] + 5 * self.dt * 1e-12
        return tau
    
    def get_batch_tau(self, tau_params):
        '''
    
        Parameters
        ----------
        omega : numpy array
            Frequency in [rad/s].
        tau_params : torch array [A, LA_alpha, LA_beta, TA_alpha, TA_beta]
            invtau_impurity = A * omega ** 4,
            invtau_L = LA_alpha * omega ** 2 * T ** LA_beta,
            invtau_T = TA_alpha * omega ** 2 * T ** TA_beta.
        device : string
            device used by torch.
    
        Returns
        -------
        tau in [s].
    
        '''
        
        tau = torch.zeros((self.bs, self.Nf, self.Nb)).to(self.device)

        invtau_TA = \
            torch.einsum("bl, k -> bkl", tau_params[:, 0, :], 1 / self.omega ** 3 * 1e50) + \
            torch.einsum("bl, k -> bkl", tau_params[:, 1, :], 1 / self.omega ** 2 * 1e36) + \
            torch.einsum("bl, k -> bkl", tau_params[:, 2, :], 1 / self.omega * 1e22) + \
            torch.einsum("bl, k -> bkl", tau_params[:, 3, :], self.omega) + \
            torch.einsum("bl, k -> bkl", tau_params[:, 4, :], self.omega ** 2 * 1e-15) + \
            torch.einsum("bl, k -> bkl", tau_params[:, 5, :], self.omega ** 3 * 1e-31)

        tau[:] = 1 / (invtau_TA)
        tau[:, ~self.nzidx] = 0.0
        tau[:, self.nzidx] = tau[:, self.nzidx] + 10 * 1e-12
        return tau

    def forward(self, g_gc, tau=None):
        
        if tau == None:
            tau = self.tau * 1e12 # convert from [s] to [ps]
        
        fs = self.calc_surfvals(g_gc)
        flux_term = - (fs[:, 1:] - fs[:, :-1]) / self.dx
        
        coll_term = torch.zeros((self.bs, self.Nx, self.Nf, self.Nb, self.Nm)).to(self.device)
        if len(tau.shape) == 2:
            coll_term[:, :, self.nzidx, :] = \
                torch.einsum("bjkm, k -> bjkm", -(g_gc[:, 2:-2,self.nzidx,:] - self.calc_g_equil(self.T)[:, :, self.nzidx, :]), 1 / tau[self.nzidx])
        elif len(tau.shape) == 3:
            coll_term[:, :, self.nzidx, :] = \
                torch.einsum("bjkm, bk -> bjkm", -(g_gc[:, 2:-2,self.nzidx,:] - self.calc_g_equil(self.T)[:, :, self.nzidx,:]), 1 / tau[:, self.nzidx])
        
        dgdt = flux_term + coll_term
        
        return dgdt

    def calc_DebyeWallerFactor_exact(self, g, hkl):
        
        usq = self.calc_msd(g, space_dependent=True)
        
        q = torch.sqrt(torch.sum((hkl * (2*np.pi) / self.latt_const) ** 2, dim=1))
        # exp(-q^2 <u^2> / 3)
        Idecay = torch.exp(- torch.einsum('q, bj -> bqj', q ** 2, usq) / 3)
        Idecay_avg = torch.trapz(Idecay, self.xn, dim=-1) / torch.trapz(torch.ones(self.Nx).to(self.device), self.xn, dim=0)

        return Idecay_avg

    def calc_DebyeWallerFactor(self, temperature, hkl):

        phi_arg = self.debye_temp / temperature
        xi = torch.einsum('i, bj -> bij', torch.linspace(0,1,100).to(self.device), phi_arg)
        integrand = xi / (torch.exp(xi) - 1 + 1e-12)

        phi = torch.trapz(integrand, xi, dim=1) / phi_arg
        B_factor = 11492 * temperature / (Element(self.element).atomic_mass * self.debye_temp ** 2) * phi + \
            2873 / (Element(self.element).atomic_mass * self.debye_temp)

        q = torch.sqrt(torch.sum((hkl * (2*np.pi) / self.latt_const) ** 2, dim=1))
        Idecay = torch.trapz(torch.exp(- 2 * torch.einsum('bj, q -> bjq', B_factor, (q / (4*np.pi)) ** 2)), self.xn, dim=1) / torch.trapz(torch.ones(self.Nx).to(self.device), self.xn, dim=0)

        return Idecay
    
    def calc_msd(self, g, space_dependent=False):            
        assert self.pdos_ratio.shape[1] == self.struct.ntypesp
        usq_spec = []
        num_sites_spec = []
        for i, spec in enumerate(self.struct.symbol_set):
            pdos_ratio = self.pdos_ratio[:,i,None].to(self.device)
            num_sites = (np.array(self.struct.atomic_numbers) == Element(spec).Z).sum()
            num_sites_spec.append(num_sites)

            dos_normed = pdos_ratio * 3 * self.struct.num_sites * self.dos / torch.trapz(self.dos, self.omega, dim=0)
            assert round(torch.trapz(dos_normed, self.omega, dim=0).item()) == (3 * num_sites)

            # g_excite = g_base + g_deviational
            # For g_base, notice here we are considering all solid angle, so the factor 1/(4 * PI) disappears
            # For g_deviational, we first integrate over solid angle -- 2 * pi * trapz(..., mu), 
            #   then normalize the DOS part -- / torch.trapz(..., omega) * 3 * total_num_site * pdos_ratio
            #   the eV constant is multiplied back because the hbar used to calculate g_deviational is previously divided by eV
            g_excite = const.hbar * torch.einsum('bjk, k, kl -> bjkl', self.f_base, self.omega, dos_normed) + \
                    torch.einsum('bjkl, kl -> bjkl',
                        2 * np.pi * torch.trapz(g, self.mu, dim=-1) / torch.trapz(self.dos, self.omega, dim=0) * 3 * self.struct.num_sites,
                        pdos_ratio
                    ) * \
                    const.value('electron volt')
            usq_excite = torch.trapz(torch.einsum('bjkl, k -> bjkl', g_excite, 1 / self.omega ** 2), self.omega, dim=-2).sum(dim=-1)

            # MSD for ground-state fluctuation
            usq_ground = torch.trapz(const.hbar / 2 * torch.einsum('kl,k->kl', dos_normed, 1 / self.omega), self.omega, dim=0).sum(dim=-1)
            
            usq_tmp = 1e20 * (usq_excite + usq_ground) / (num_sites * Element(spec).atomic_mass * const.physical_constants['atomic mass constant'][0])

            usq_spec.append(usq_tmp[...,None])

        usq = torch.einsum('bjs, s -> bj', torch.cat(usq_spec, dim=-1), torch.tensor(num_sites_spec).to(self.device) / self.struct.num_sites)
        usq_avg = torch.trapz(usq, self.xn, dim=-1) / torch.trapz(torch.ones(self.Nx).to(self.device), self.xn, dim=0)

        if space_dependent:
            return usq
        else:
            return usq_avg

    def calc_msd_oldver(self, g, element):
        dos_normed = 3 * self.dos / torch.trapz(self.dos, self.omega, dim=0)
        number = torch.trapz(dos_normed, self.omega, dim=0).sum() / 3
        
        ## if using eV unit
        g_excite = const.hbar * torch.einsum('bjk, k, kl -> bjkl', self.f_base, self.omega, dos_normed) + \
            torch.trapz(g, self.mu, dim=-1) / torch.trapz(self.dos, self.omega, dim=0) * 3 * const.value('electron volt') * 2 * np.pi

        # g_excite = const.hbar * torch.einsum('bjk, k, kl -> bjkl', self.f_base, self.omega, dos) + \
        #     torch.trapz(g, self.mu, dim=-1) / torch.trapz(self.dos, self.omega, dim=0) * 3 * 2 * np.pi

        ## debugging:
        # dos = 9 * 3 * self.omega ** 2 / (self.debye_temp * const.k / const.hbar) ** 3
        # f = 1 / (torch.exp(torch.einsum("j,k", 1 / (const.k * (self.T_base + 633)), const.hbar * self.omega)) - 1)
        # g_excite = const.hbar * torch.einsum('jk,k,kl->jkl', f, self.omega, dos)
        
        usq_excite = torch.trapz(torch.einsum('bjkl, k -> bjkl', g_excite, 1 / self.omega ** 2), self.omega, dim=-2).sum(dim=-1)
        usq_ground = torch.trapz(const.hbar / 2 * torch.einsum('kl,k->kl', dos_normed, 1 / self.omega), self.omega, dim=0).sum(dim=-1)
        # squared displacement in [m]^2
        usq = 1e20 * (usq_excite + usq_ground) / (number * Element(element).atomic_mass * const.physical_constants['atomic mass constant'][0])
        
        ## debugging:
        # print('sqrt(u^2) / (a / sqrt(2)):\n', torch.sqrt(usq) / (self.latt_const / np.sqrt(2)))
        
        usq_avg = torch.trapz(usq, self.xn, dim=-1) / torch.trapz(torch.ones(self.Nx).to(self.device), self.xn, dim=0)

        return usq_avg