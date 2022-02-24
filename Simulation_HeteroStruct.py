#%%
import pickle
import h5py
import numpy as np
import scipy.constants as const
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation
from adjoint_bte.phonon_bte import PhononBTE
from tqdm import tqdm
from pymatgen.core import Structure

import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)

if torch.cuda.is_available():
    device = "cuda:7"
else:
    device = "cpu"
print('torch device:' , device)

flag_exp_setup = True

# Eps & Transmittance
# sample_info = [0, 1, torch.tensor([0.01, 0.43]).to(device)]
# sample_info = [2, 2, torch.tensor([0.01, 0.43]).to(device)]
# sample_info = [3, 3, torch.tensor([0.01, 0.43]).to(device)]

sample_info = {
    "traref_flag": 1,
    "tau_flag": 1,
    "eps_bdry_flag": torch.tensor([0.01, 0.43]).to(device),
    "eps_bulk_flag": torch.tensor([0.001, 0.01]).to(device)
}

# sample_info = {
#     "traref_flag": 2,
#     "tau_flag": 2,
#     "eps_bdry_flag": torch.tensor([0.005, 0.2]).to(device),
#     "eps_bulk_flag": torch.tensor([0.0015, 0.015]).to(device)
# }

# sample_info = {
#     "traref_flag": 3,
#     "tau_flag": 4,
#     "eps_bdry_flag": torch.tensor([0.02, 0.3]).to(device),
#     "eps_bulk_flag": torch.tensor([0.0005, 0.02]).to(device)
# }

flag_TR, flag_Tau = sample_info['traref_flag'], sample_info['tau_flag']
_eps_bdry = sample_info['eps_bdry_flag']
_eps_bulk = sample_info['eps_bulk_flag']

epsbdry_str = "_".join([(str(eps.item())).replace('.', 'd') for eps in np.around(_eps_bdry.cpu().numpy(), 4)])
epsbulk_str = "_".join([(str(eps.item())).replace('.', 'd') for eps in np.around(_eps_bulk.cpu().numpy(), 4)])

OmegaMax, OmegaPoint = 12, 120
phonon_prop_fname = f'data/Au_Si_phonon_OmegaMax{OmegaMax}_OmegaPoint{OmegaPoint}.pkl'
saved_dict_fname = f'data/Au_Si_samples/Simulation_Au_Si_TR{flag_TR}_Tau{flag_Tau}_EpsBdry_{epsbdry_str}_EpsBulk_{epsbulk_str}_OmegaMax{OmegaMax}_OmegaPoint{OmegaPoint}_ExpTP{flag_exp_setup}.pt'
print('saving simulation to: ', saved_dict_fname)

#%%

def compute_heatcap_new(freq_rad, dos, T, struct):
    '''
    Parameters
    ----------
    freq_rad : numpy array
        Frequency in rad/s.
    dos : numpy array
        DESCRIPTION.
    T : numpy array
        Temperature.

    Returns
    -------
    Mode heat capacity.
    f_BE = 1/(exp(hbar*omega/(k*T))-1)
    dfdT = omega*hbar*csch(omega*hbar/(2*k*T))**2/(4*k*T**2)
    '''
    
    hbar = const.hbar
    k = const.k
    dos_normed = 3 * struct.num_sites * dos / torch.trapz(dos, freq_rad, dim=0)

    # y = np.array([float(mpmath.csch(hbar * x / (2 * k * T.cpu().numpy()))) ** 2 for x in freq_rad.cpu().numpy()])
    y = 1 / torch.sinh(hbar * freq_rad / (2 * k * T)) ** 2
    # dfdT = - (1/4)*(1/(k*T))*y
    dfdT = hbar * freq_rad * y / (4 * k * T ** 2)
    Cv = hbar * torch.einsum('kl,k->kl', dos_normed, freq_rad * dfdT) # freq_rad * dos * dfdT
    Cv_per_cubicmeter = 1e30 / struct.volume * Cv # [J/K/m3]

    return Cv_per_cubicmeter

struct_cap = Structure.from_file('data/cif_files/Au_mp-81_conventional_standard.cif')
struct_sub = Structure.from_file('data/cif_files/Si_mp-149_conventional_standard.cif')

with open(phonon_prop_fname, "rb") as f:
    ph_dict = pickle.load(f)

omega_cap = torch.tensor(2*np.pi*ph_dict['Au']["freq"]) * 1e12 # THz -> rad/s
dos_cap, tau_cap, vg_cap = torch.zeros((omega_cap.shape[0], 3)), torch.zeros((omega_cap.shape[0], 3)), torch.zeros((omega_cap.shape[0], 3))
for i, key in enumerate(['TA1', 'TA2', 'LA']):
    dos_cap[:,i] = torch.from_numpy(ph_dict['Au']["DOS"][key]) * 1e15 # [s/m^3]
    tau_cap[:,i] = torch.ones(omega_cap.shape[0]) * 1e-12 # s
    # tau_cap[:,i] = torch.from_numpy(ph_dict[cap_element]["tau"][key]) * 1e-12 # s
    vg_cap[:,i] = torch.from_numpy(ph_dict['Au']["Vg"][key])  # m/s

vg_cap = (vg_cap * dos_cap).sum(dim=1, keepdim=True) / (dos_cap.sum(dim=1, keepdim=True) + 1e-15)
# dos_cap = dos_cap.sum(dim=1, keepdim=True)
dos_cap = (dos_cap * (dos_cap > 1e3)).sum(dim=1, keepdim=True)

# kappa = 1/3 * trapz(Cv * vg^2 * tau, omega)
Cv_Au = compute_heatcap_new(omega_cap.cpu(), dos_cap.cpu(), 300, struct_cap)
tau_avg_Au = 3 * 310 / torch.trapz(Cv_Au * vg_cap.pow(2).cpu(), omega_cap, dim=0)
print("averaged tau for Au: ", tau_avg_Au.item() * 1e12, ' [ps]')
tau_cap = tau_avg_Au * torch.ones_like(dos_cap)
pdos_ratio_cap = torch.ones_like(dos_cap) * (dos_cap > 0)

omega_sub = torch.tensor(2*np.pi*ph_dict["Si"]["freq"]) * 1e12 # THz -> rad/s
dos_sub, tau_sub, vg_sub = torch.zeros((omega_sub.shape[0], 3)), torch.zeros((omega_sub.shape[0], 3)), torch.zeros((omega_sub.shape[0], 3))
for i, key in enumerate(['TA1', 'TA2', 'LA']):
    dos_sub[:,i] = torch.from_numpy(ph_dict["Si"]["DOS"][key]) * 1e15 # [s/m^3]
    tau_sub[:,i] = torch.from_numpy(ph_dict["Si"]["Tau"][key]) * 1e-12 # s
    vg_sub[:,i] = torch.from_numpy(ph_dict["Si"]["Vg"][key])  # m/s
tau_sub = (tau_sub * dos_sub).sum(dim=1, keepdim=True) / (dos_sub.sum(dim=1, keepdim=True) + 1e-15)
vg_sub = (vg_sub * dos_sub).sum(dim=1, keepdim=True) / (dos_sub.sum(dim=1, keepdim=True) + 1e-15)
dos_sub = dos_sub.sum(dim=1, keepdim=True)
pdos_ratio_sub = torch.ones_like(dos_sub) * (dos_sub > 0)

fig, ax = plt.subplots(2,3)
ax[0,0].plot(omega_cap, dos_cap)
ax[0,1].plot(omega_cap, vg_cap)
ax[0,2].plot(omega_cap, tau_cap)
ax[1,0].plot(omega_sub, dos_sub)
ax[1,1].plot(omega_sub, vg_sub)
ax[1,2].plot(omega_sub, tau_sub)
fig.tight_layout()

#%%

cap_prop = {'omega': omega_cap, 'DOS': dos_cap, 'tau': tau_cap, 'vg': vg_cap, 'struct': struct_cap, 'pdos_ratio': pdos_ratio_cap}
sub_prop = {'omega': omega_sub, 'DOS': dos_sub, 'tau': tau_sub, 'vg': vg_sub, 'struct': struct_sub, 'pdos_ratio': pdos_ratio_sub}

"""
Commom properties
"""

Nm = 20
dt = 0.1
print(f"Minimum allowed dx is: {dt * max(vg_cap.max().item(), vg_sub.max().item()) * 1e-3:6.4f}")

T_base = 300
"""
Initialize model for cap
"""
Lx_cap, dx_cap = 5.0, 1.0 # nm
cap_kwargs = {'mater_prop': cap_prop, 'T_base': T_base,
              'dt': dt, 'dx': dx_cap, 'Lx': Lx_cap, 'Nm': Nm, 'device': device}
cap_model = PhononBTE(**cap_kwargs)

"""
Initialize model for sub
"""
Lx_sub, dx_sub = 35.0, 1.0
sub_kwargs = {'mater_prop': sub_prop, 'T_base': T_base,
              'dt': dt, 'dx': dx_sub, 'Lx': Lx_sub, 'Nm': Nm, 'device': device}
sub_model = PhononBTE(**sub_kwargs)

cap_model.init_distribution(cap_model.T_base, T_lims=[250, 2000])
sub_model.init_distribution(sub_model.T_base, T_lims=[250, 2000])

cap_model.msd_base = cap_model.calc_msd(cap_model.g)
sub_model.msd_base = sub_model.calc_msd(sub_model.g)

#%%


exp_data = torch.load(
    f'data/Au_Si_samples/Experiment_HeteroStruct_Au_Si_OmegaMax{OmegaMax}_OmegaPoint{OmegaPoint}.pt', map_location=device)
T0_cap_exp = exp_data['T_hist'][0, None, -1, :cap_model.Nx]
T0_sub_exp = exp_data['T_hist'][0, None, -1, cap_model.Nx:]

T0_cap_add = torch.cat((
    100. + cap_model.T_base,
    300. + cap_model.T_base,
    400. + cap_model.T_base,
    500. + cap_model.T_base,
    600. + cap_model.T_base),
    dim=0)

T0_sub_add = torch.cat((
    200. + sub_model.T_base,
    400. + sub_model.T_base,
    500. + sub_model.T_base,
    600. + sub_model.T_base,
    700. + sub_model.T_base),
    dim=0)

T0_cap = torch.cat((T0_cap_exp, T0_cap_add), dim=0)
T0_sub = torch.cat((T0_sub_exp, T0_sub_add), dim=0)

cap_model.init_distribution(T0_cap, T_lims = [1.0, 2.0 * max(T0_cap.max().item(),T0_sub.max().item())])
sub_model.init_distribution(T0_sub, T_lims = [1.0, 2.0 * max(T0_cap.max().item(),T0_sub.max().item())])

fig = plt.figure()
ax = fig.add_subplot(211)
ax.plot(cap_model.xn.cpu(), cap_model.T[-1].cpu(), '-')
ax.plot(sub_model.xn.cpu() + cap_model.Lx, sub_model.T[-1].cpu(), '-')
ax = fig.add_subplot(212)
ax.plot(cap_model.xn.cpu(), cap_model.g[-1,:,12,-1,0].cpu(), '-')
ax.plot(sub_model.xn.cpu() + cap_model.Lx, sub_model.g[-1,:,12,-1,0].cpu(), '-')
fig.tight_layout()


#%%
fig, ax = plt.subplots(1,1,figsize=(8,6))
ax.plot(cap_model.TempList.cpu(), cap_model.EngList.cpu(), c='0.2')
ax.set_ylabel(r'$\int_{0}^{\infty}\int_{-1}^{1}g^{\mathrm{eq}}(T,\omega,r)\mathrm{d}\mu\mathrm{d}\omega$ (a.u.)',
    fontsize=20, rotation='horizontal')
ax.scatter([300.0], [0.0])
ax.yaxis.set_label_coords(0.15,1.04)
ax.set_xlabel(r'Lattice temperature $T$ (K)', fontsize=20)
fig.tight_layout()
# fig.savefig('figures/figure_SI/LatticeTemp_vs_Eng.pdf', bbox_inches='tight')

fig, ax = plt.subplots(1,1,figsize=(8,6))
ax.plot(cap_model.TempList.cpu(), cap_model.PseudoEngList[0].cpu(), c='0.2')
ax.set_ylabel(r'$\int_{0}^{\infty}\int_{-1}^{1}\frac{g^{\mathrm{eq}}(T_{p},\omega,r)}{\tau(\omega,r)}\mathrm{d}\mu\mathrm{d}\omega$ (a.u.)',
    fontsize=20, rotation='horizontal')
ax.yaxis.set_label_coords(0.15,1.04)
ax.set_xlabel(r'Pseudo-temperature $T_{p}$ (K)', fontsize=20)
fig.tight_layout()
# fig.savefig('figures/figure_SI/PseudoTemp_vs_Eng.pdf', bbox_inches='tight')

#%% set up ground-truth for relaxation time

bs_params = T0_cap.size(0)

def get_tau(mags, pows, model):
    
    invtau = 0.
    for i, (mag, pow) in enumerate(zip(mags, pows)):
        invtau += mag * model.omega.pow(pow)

    tau = 1 / invtau
    tau.clamp_(1e-12, 1e-9)
    tau *= model.nzidx.sum(dim=1)
    return tau

def get_tau_params_by_flag(flag_Tau):

    if flag_Tau == 1:
        mags = np.array([30 * 1e35, 0.1 * 1e-16])
        pows = np.array([-2, 2])
    elif flag_Tau == 2:
        mags = np.array([0.5 * 1e22, 0.00075])
        pows = np.array([-1, 1])
    elif flag_Tau == 3:
        mags = np.array([1 * 1e36, 1 * 1e-30])
        pows = np.array([-2, 3])
    elif flag_Tau == 4:
        mags = np.array([10 * 1e22, 0.00075])
        pows = np.array([-1, 1])

    return mags, pows

if flag_Tau > 0:
    mags, pows = get_tau_params_by_flag(flag_Tau)
    tau_cap = get_tau(mags, pows, cap_model)


kappa_new = 1/3 * torch.trapz(Cv_Au.squeeze() * cap_model.vg.pow(2).cpu().squeeze() * tau_cap.cpu().squeeze(), omega_cap, dim=0).to(device)
print('thermal conductivity: ', kappa_new)

fig, ax = plt.subplots(1,1)
ax.plot(cap_model.omega.cpu(), tau_cap.cpu() * 1e12, '-')

if flag_Tau > 0:
    tau_cap = (tau_cap[None].repeat_interleave(bs_params, dim=0))[...,None].repeat_interleave(cap_model.Nb, dim=0) * (310 / kappa_new.item())
else:
    tau_cap = tau_cap[None].repeat_interleave(bs_params, dim=0) * (310 / kappa_new.item())

ax.plot(cap_model.omega.cpu(), tau_cap[0].cpu() * 1e12, '-')
# ax.set_ylim([0, 1000])
plt.show()

print('current thermal conductivity: ', 1/3 * torch.trapz(Cv_Au.squeeze() * cap_model.vg.pow(2).cpu().squeeze() * tau_cap[0].cpu().squeeze(), omega_cap, dim=0))

cap_model.tau = tau_cap[0].to(device)

# cap_model.init_distribution(T0_cap, T_lims = [280, 2.0 * max(T0_cap.max().item(),T0_sub.max().item())])
# sub_model.init_distribution(T0_sub, T_lims = [280, 2.0 * max(T0_cap.max().item(),T0_sub.max().item())])

#%% set up ground-truth for transmission reflection coefficients

if flag_TR == 0:
    cmidx = cap_model.nzidx & sub_model.nzidx
    # cmidx = ((sub_model.dos * sub_model.vg) / (cap_model.dos * cap_model.vg + 1e-12) < 50) & ((sub_model.dos * sub_model.vg) / (cap_model.dos * cap_model.vg + 1e-12) > 0.02)
    traref = torch.zeros((2, 2, sub_model.Nf, sub_model.Nb)).to(device)
    # transmisstion coefficient from cap to sub
    traref[0,0][cmidx] = 1.0 * (sub_model.dos[cmidx] * sub_model.vg[cmidx]) \
        / ((sub_model.dos[cmidx] * sub_model.vg[cmidx]) + (cap_model.dos[cmidx] * cap_model.vg[cmidx]))
    # transmisstion coefficient from sub to cap, detailed balance
    traref[1,1][cmidx] = 1.0 * (cap_model.dos[cmidx] * cap_model.vg[cmidx]) \
        / ((sub_model.dos[cmidx] * sub_model.vg[cmidx]) + (cap_model.dos[cmidx] * cap_model.vg[cmidx]))
    # assert detailed balance
    assert torch.allclose(traref[0,0][cmidx] * cap_model.dos[cmidx] * cap_model.vg[cmidx], 
                        traref[1,1][cmidx] * sub_model.dos[cmidx] * sub_model.vg[cmidx])
    traref[0,1][cap_model.nzidx] = 1 - traref[0,0][cap_model.nzidx]
    # reflection coefficient from sub to sub
    traref[1,0][sub_model.nzidx] = 1 - traref[1,1][sub_model.nzidx]
    assert torch.all(traref <= 1) and torch.all(traref >= 0)

elif flag_TR == 1:
    cmidx = cap_model.nzidx & sub_model.nzidx
    # cmidx = ((sub_model.dos * sub_model.vg) / (cap_model.dos * cap_model.vg + 1e-12) < 50) & ((sub_model.dos * sub_model.vg) / (cap_model.dos * cap_model.vg + 1e-12) > 0.02)
    traref = torch.zeros((2, 2, sub_model.Nf, sub_model.Nb)).to(device)
    # transmisstion coefficient from cap to sub
    traref[0,0][cmidx] = 0.6 * (sub_model.dos[cmidx] * sub_model.vg[cmidx]) \
        / ((sub_model.dos[cmidx] * sub_model.vg[cmidx]) + (cap_model.dos[cmidx] * cap_model.vg[cmidx]))
    # transmisstion coefficient from sub to cap, detailed balance
    traref[1,1][cmidx] = 0.6 * (cap_model.dos[cmidx] * cap_model.vg[cmidx]) \
        / ((sub_model.dos[cmidx] * sub_model.vg[cmidx]) + (cap_model.dos[cmidx] * cap_model.vg[cmidx]))
    # assert detailed balance
    assert torch.allclose(traref[0,0][cmidx] * cap_model.dos[cmidx] * cap_model.vg[cmidx], 
                        traref[1,1][cmidx] * sub_model.dos[cmidx] * sub_model.vg[cmidx])
    traref[0,1][cap_model.nzidx] = 1 - traref[0,0][cap_model.nzidx]
    # reflection coefficient from sub to sub
    traref[1,0][sub_model.nzidx] = 1 - traref[1,1][sub_model.nzidx]
    assert torch.all(traref <= 1) and torch.all(traref >= 0)

elif flag_TR == 2:
    T_left_, T_right_ = 0.0, 1.0

    cmidx = cap_model.nzidx & sub_model.nzidx
    omega = omega_cap[:,None].repeat_interleave(cap_model.Nb, dim=-1).to(device)
    # cmidx = ((sub_model.dos * sub_model.vg) / (cap_model.dos * cap_model.vg + 1e-12) < 50) & ((sub_model.dos * sub_model.vg) / (cap_model.dos * cap_model.vg + 1e-12) > 0.02)
    traref = torch.zeros((2,2,sub_model.Nf,sub_model.Nb)).to(device)

    ratio = torch.zeros((sub_model.Nf,sub_model.Nb)).to(device)
    # ratio = T_sub2cap / T_cap2sub
    ratio[cmidx] = cap_model.dos[cmidx] * cap_model.vg[cmidx] / (sub_model.dos[cmidx] * sub_model.vg[cmidx] + 1e-12)
    idx_tmp = (ratio < 1.0) * cmidx
    traref[0,0][idx_tmp] = (T_left_ - (T_left_ - T_right_) * (omega - omega[cmidx].min()) / (omega[cmidx].max() - omega[cmidx].min()))[idx_tmp]
    # traref[0,0][idx_tmp] = 1.00 * (omega / omega[cmidx].max())[idx_tmp]
    traref[1,1][idx_tmp] = traref[0,0][idx_tmp] *cap_model.dos[idx_tmp] * cap_model.vg[idx_tmp] / (sub_model.dos[idx_tmp] * sub_model.vg[idx_tmp] + 1e-12)
    
    idx_tmp = (ratio > 1.0) * cmidx
    traref[1,1][idx_tmp] = (T_left_ - (T_left_ - T_right_) * (omega - omega[cmidx].min()) / (omega[cmidx].max() - omega[cmidx].min()))[idx_tmp]
    # traref[1,1][idx_tmp] = 1.00 * (omega / omega[cmidx].max())[idx_tmp]
    traref[0,0][idx_tmp] = traref[1,1][idx_tmp] * sub_model.dos[idx_tmp] * sub_model.vg[idx_tmp] / (cap_model.dos[idx_tmp] * cap_model.vg[idx_tmp] + 1e-12)

    # assert detailed balance
    assert torch.allclose(traref[0,0][cmidx] * cap_model.dos[cmidx] * cap_model.vg[cmidx],
                          traref[1,1][cmidx] * sub_model.dos[cmidx] * sub_model.vg[cmidx])
    traref[0,1][cap_model.nzidx] = 1 - traref[0,0][cap_model.nzidx]
    # reflection coefficient from sub to sub
    traref[1,0][sub_model.nzidx] = 1 - traref[1,1][sub_model.nzidx]
    assert torch.all(traref <= 1) and torch.all(traref >= 0)

elif flag_TR == 3:
    T_left_, T_right_ = 1.0, 0.0

    cmidx = cap_model.nzidx & sub_model.nzidx
    omega = omega_cap[:,None].repeat_interleave(cap_model.Nb, dim=-1).to(device)
    # cmidx = ((sub_model.dos * sub_model.vg) / (cap_model.dos * cap_model.vg + 1e-12) < 50) & ((sub_model.dos * sub_model.vg) / (cap_model.dos * cap_model.vg + 1e-12) > 0.02)
    traref = torch.zeros((2,2,sub_model.Nf,sub_model.Nb)).to(device)

    ratio = torch.zeros((sub_model.Nf,sub_model.Nb)).to(device)
    # ratio = T_sub2cap / T_cap2sub
    ratio[cmidx] = cap_model.dos[cmidx] * cap_model.vg[cmidx] / (sub_model.dos[cmidx] * sub_model.vg[cmidx] + 1e-12)
    idx_tmp = (ratio < 1.0) * cmidx
    traref[0,0][idx_tmp] = (T_left_ - (T_left_ - T_right_) * (omega - omega[cmidx].min()) / (omega[cmidx].max() - omega[cmidx].min()))[idx_tmp]
    # traref[0,0][idx_tmp] = 1.00 * (1 - omega / omega[cmidx].max())[idx_tmp]
    traref[1,1][idx_tmp] = traref[0,0][idx_tmp] * cap_model.dos[idx_tmp] * cap_model.vg[idx_tmp] / (sub_model.dos[idx_tmp] * sub_model.vg[idx_tmp] + 1e-12)
    
    idx_tmp = (ratio > 1.0) * cmidx
    traref[1,1][idx_tmp] = (T_left_ - (T_left_ - T_right_) * (omega - omega[cmidx].min()) / (omega[cmidx].max() - omega[cmidx].min()))[idx_tmp]
    # traref[1,1][idx_tmp] = 1.00 * (1 - omega / omega[cmidx].max())[idx_tmp]
    traref[0,0][idx_tmp] = traref[1,1][idx_tmp] * sub_model.dos[idx_tmp] * sub_model.vg[idx_tmp] / (cap_model.dos[idx_tmp] * cap_model.vg[idx_tmp] + 1e-12)

    # assert detailed balance
    assert torch.allclose(traref[0,0][cmidx] * cap_model.dos[cmidx] * cap_model.vg[cmidx], 
                          traref[1,1][cmidx] * sub_model.dos[cmidx] * sub_model.vg[cmidx])
    traref[0,1][cap_model.nzidx] = 1 - traref[0,0][cap_model.nzidx]
    # reflection coefficient from sub to sub
    traref[1,0][sub_model.nzidx] = 1 - traref[1,1][sub_model.nzidx]
    assert torch.all(traref <= 1) and torch.all(traref >= 0)

fig = plt.figure()
ax = fig.add_subplot(211)
ax.plot(cap_model.omega.cpu(), traref[0,0].cpu(), '-')
ax.plot(sub_model.omega.cpu(), traref[1,1].cpu(), '-')
ax.set_ylim([0,1.1])

#%%

from adjoint_bte.model_heterostructure import BoltzmannTransportEquation_HeteroStruct

bs_params = T0_cap.size(0)
models = (cap_model, sub_model)
traref_params_iv = None

eps_bdry = _eps_bdry[None].repeat_interleave(bs_params, dim=0)
eps_bulk = _eps_bulk[None].repeat_interleave(bs_params, dim=0)

from adjoint_bte.torchdiffeq import odeint

bte_model = BoltzmannTransportEquation_HeteroStruct(
    (cap_model, sub_model), eps_bdry, eps_bulk,
    traref_params_iv=torch.rand((bs_params, cap_model.Nf, cap_model.Nb)).to(device), # placeholder
    traref_fixed=traref[None].repeat_interleave(bs_params, dim=0), tau_cap_iv=tau_cap * 1e12,
    tau_coeff_cap=None, bs=bs_params).to(device)

if flag_exp_setup:
    ts_ = torch.from_numpy(exp_data['t_hist'][:14])
else:
    ts_ = torch.linspace(0, 30, 31)
print(ts_)

cap_model.init_distribution(T0_cap, T_lims = [280, 2.0 * max(T0_cap.max().item(),T0_sub.max().item())])
sub_model.init_distribution(T0_sub, T_lims = [280, 2.0 * max(T0_cap.max().item(),T0_sub.max().item())])
g0 = torch.cat((cap_model.g, sub_model.g), dim=1)


#%%

import time
start_time = time.time()

g_ = odeint(bte_model, g0, ts_, method='euler', options={'step_size': dt, 'return_all_timepoints': False})

T_ = torch.zeros((g_.size(0), g_.size(1), g_.size(2))).to(g_)
msd_ = torch.zeros((g_.size(0), g_.size(1), 2)).to(g_)
for i, g_tmp_ in enumerate(g_):
    T_[i, :, :cap_model.Nx] = cap_model.solve_Temp_interp(g_tmp_[:, :cap_model.Nx])
    T_[i, :, cap_model.Nx:] = sub_model.solve_Temp_interp(g_tmp_[:, cap_model.Nx:])
    msd_[i, :, 0] = cap_model.calc_msd(g_tmp_[:, :cap_model.Nx]) - cap_model.msd_base
    msd_[i, :, 1] = sub_model.calc_msd(g_tmp_[:, cap_model.Nx:]) - sub_model.msd_base

end_time = time.time()
print(f"elapsed time = {time.strftime('%H:%M:%S', time.gmtime(end_time - start_time))}")

#%%

saved_dict = {
    "omega": omega_cap.numpy() * 1e-12 / (2 * np.pi),
    "kw_cap": cap_kwargs, "kw_sub": sub_kwargs,
    "eps_bdry": _eps_bdry,
    "eps_bulk": _eps_bulk,
    "tau": cap_model.tau.detach().cpu().numpy(), 
    "traref": traref.detach().cpu().numpy(),
    "xn": [cap_model.xn.detach().cpu().numpy(), sub_model.xn.detach().cpu().numpy()],
    "t_hist": ts_.detach().cpu().numpy(),
    "T_hist": T_.detach().cpu().numpy(),
    "g_hist": g_.detach().cpu().numpy(),
    "msd_hist": msd_.detach().cpu().numpy()
}
torch.save(saved_dict, saved_dict_fname)

#%%
idx_sample = 2
idx_omega, idx_branch = 20, -1
print(traref[0,0,idx_omega,idx_branch])
t_hist = ts_.detach().cpu().numpy()
g_hist = g_[:, idx_sample, :, idx_omega, idx_branch, [0,-1]].detach().cpu().numpy()
T_hist = T_[:, idx_sample, :].detach().cpu().numpy()
# I_hist = I_[:, idx_sample, :].detach().cpu().numpy()
msd_hist = msd_[:, idx_sample, :].detach().cpu().numpy()
frame_plot = np.linspace(0, len(T_hist)-1, 5).astype('int')

fig = plt.figure(figsize=(9.6,1.6 * len(frame_plot)))
gs = fig.add_gridspec(len(frame_plot),3)
ax = fig.add_subplot(gs[:len(frame_plot),0])
ax.set_xticks([])
ax.set_yticks([])
ax.set_ylabel(r'$T$ (K)', fontsize=15, labelpad=30)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax = fig.add_subplot(gs[:len(frame_plot),1])
ax.set_xticks([])
ax.set_yticks([])
ax.set_ylabel(r'$g$', fontsize=15, labelpad=30)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

for i, N in enumerate(frame_plot):
    ax = fig.add_subplot(gs[i,0])
    ax.plot(cap_model.xn.cpu(), T_hist[N,:cap_model.Nx], 'k')
    ax.plot(sub_model.xn.cpu() + cap_model.Lx, T_hist[N,cap_model.Nx:], 'k')
    ax.set_title(fr"$t={t_hist[N]:5.3f}$ (ps)")
    if i == 0: ax.set_ylim([T_hist[N].min() * 0.9, T_hist[N].max() * 1.1])

    ax = fig.add_subplot(gs[i,1])
    ax.plot(cap_model.xn.cpu(), g_hist[N, :cap_model.Nx,0], 'r')
    ax.plot(cap_model.xn.cpu(), g_hist[N, :cap_model.Nx,1], 'b')
    ax = fig.add_subplot(gs[i,2])
    ax.plot(sub_model.xn.cpu() + cap_model.Lx, g_hist[N, cap_model.Nx:,0], 'r')
    ax.plot(sub_model.xn.cpu() + cap_model.Lx, g_hist[N, cap_model.Nx:,1], 'b')
    ax.set_title(fr"$t={t_hist[N]:5.3f}$ (ps)")
    # ax.set_yscale('log')

# ax.yaxis.tick_right()
ax.yaxis.set_ticks_position('both')
ax.set_xlabel(r'$x$ (nm)', fontsize=15)
fig.tight_layout()

fig, ax = plt.subplots(1,1)
ax.plot(t_hist, msd_hist[:,0], '-o')
ax.plot(t_hist, msd_hist[:,-1], '-o')

#%%

# import meshio
# import meshzoo
# import json
# # from adjoint_bte.model_heterostructure_torchdiffeq import write_mesh

# def write_mesh(g, t, x, omega, fpath=None):
#     points, cells = meshzoo.rectangle_quad(x, omega)
#     print(points)
#     if points.shape[1] == 2:
#         points = np.concatenate([points, np.zeros((len(points),1))], axis=-1)
#     cells = [("quad", cells)]

#     vtk_series = {}
#     vtk_series["file-series-version"] = "1.0"
#     vtk_series["files"] = []
#     for i, _t in enumerate(t):
#         mesh = meshio.Mesh(
#             points, cells,
#             point_data={"g": (g[i]/(g[0] + 1e-6)).reshape(-1, order='F')}
#         )
    
#         if fpath is not None:
#             mesh.write(f"{fpath}/{i}.vtk")
#             vtk_series["files"].append({"name": f"{i}.vtk", "time": _t})

#     with open(f'{fpath}/sol.vtk.series', 'w') as fp:
#         json.dump(vtk_series, fp)

#     return mesh

# x = torch.cat([cap_model.xn, sub_model.xn], dim=0).cpu().numpy()
# t = ts_.numpy()
# g = torch.trapz(g_, cap_model.mu, dim=-1).squeeze()[:,0].cpu().numpy()
# write_mesh(g, t, x, omega.cpu().numpy().squeeze() / 1e12 / (np.pi * 2), fpath='figures/vtk_visual')

