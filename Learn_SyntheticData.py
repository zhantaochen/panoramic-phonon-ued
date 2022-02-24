#%%
import os, time
from datetime import datetime

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from adjoint_bte.utils import loglinspace
from adjoint_bte.phonon_bte import PhononBTE

import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)

if torch.cuda.is_available():
    device = "cuda:7"
else:
    device = "cpu"

flag_trial = False

flag_exp_setup = True
print("flag_exp_setup: ", flag_exp_setup)

flag_onlyFitFilm = False
print("Only fit film signals: ", flag_onlyFitFilm)

# Eps & Transmittance & Tau
# flag_fitEps = True
# flag_fitTraref = True
# flag_fitTau = True

# Eps & Transmittance
flag_fit_EpsBdry = True
flag_fit_EpsBulk = True
flag_fitTraref = True
flag_fitTau = False

# Transmittance & Tau
# flag_fitEps = False
# flag_fitTraref = True
# flag_fitTau = True

# Emps & Tau
# flag_fitEps = True
# flag_fitTraref = False
# flag_fitTau = True

print(
    'Fit EpsBdry: ', flag_fit_EpsBdry, 
    ',    Fit EpsBulk: ', flag_fit_EpsBulk, 
    ',    Fit TR: ', flag_fitTraref, 
    ',    Fit Tau: ', flag_fitTau)

sample_info = {
    "traref_flag": 1,
    "tau_flag": 1,
    "eps_bdry_flag": torch.tensor([0.01, 0.43]).to(device) * flag_fit_EpsBdry,
    "eps_bulk_flag": torch.tensor([0.001, 0.01]).to(device)
}

# sample_info = {
#     "traref_flag": 2,
#     "tau_flag": 2,
#     "eps_bdry_flag": torch.tensor([0.005, 0.2]).to(device) * 0,
#     "eps_bulk_flag": torch.tensor([0.0015, 0.015]).to(device)
# }

# sample_info = {
#     "traref_flag": 3,
#     "tau_flag": 4,
#     "eps_bdry_flag": torch.tensor([0.02, 0.3]).to(device),
#     "eps_bulk_flag": torch.tensor([0.0005, 0.02]).to(device)
# }

noise_level = 0. # noisy_MSD = 1 ((+/-) noise_lvl) * clean_MSD
print('noise level: ', noise_level)

save_grads = True
OmegaMax, OmegaPoint = 12, 120

if flag_exp_setup:
    training_sample_id = [0, ]
else:
    training_sample_id = [0, 1, 2, 3, 4, 5]
print('training samples: ', training_sample_id)


#%%

flag_TR, flag_Tau = sample_info['traref_flag'], sample_info['tau_flag']
eps_bdry = sample_info['eps_bdry_flag']
eps_bulk = sample_info['eps_bulk_flag']

eps_bdry_str = "_".join([(str(eps.item())).replace('.', 'd') for eps in np.around(eps_bdry.cpu().numpy(), 4)])
eps_bulk_str = "_".join([(str(eps.item())).replace('.', 'd') for eps in np.around(eps_bulk.cpu().numpy(), 4)])

simulation_fname = f'data/Au_Si_samples/Simulation_Au_Si_TR{flag_TR}_Tau{flag_Tau}_EpsBdry_{eps_bdry_str}_EpsBulk_{eps_bulk_str}_OmegaMax{OmegaMax}_OmegaPoint{OmegaPoint}_ExpTP{flag_exp_setup}.pt'

time_stamp = datetime.today().strftime('%Y-%m-%d_%H-%M')
path_output = f"outputs/Au_Si_samples/Au_Si_TR{flag_fitTraref}{flag_TR}_Tau{flag_fitTau}{flag_Tau}_EpsBdry{flag_fit_EpsBdry}_{eps_bdry_str}_EpsBulk{flag_fit_EpsBulk}_{eps_bulk_str}_{time_stamp}_Noise_{str(noise_level).replace('.', 'd')}"

path_output += f'_ExpTP{flag_exp_setup}'

print(f'torch device: {device}    ' + f'time stamp: {time_stamp}', flush=True)
print(f'Reading file: {simulation_fname}', flush=True)
print(f'Saving to: {path_output}', flush=True)

#%% setting up hyperparameters

lr_eps_bdry = 0.01
lr_eps_bulk = 0.01

lr_traref = 0.01

lr_taucoeffcap = 2.5

print(f"\nlr_eps_bdry: {lr_eps_bdry:.4f}    ", f"lr_eps_bulk: {lr_eps_bulk:.4f}")
print(f"\nlr_traref: {lr_traref:.4f}    ", f"lr_taucoeffcap: {lr_taucoeffcap:.4f}")

train_data = torch.load(simulation_fname, map_location=device)
print(train_data['t_hist'])
#%%
omega = train_data['omega']
delta_omega = omega[1] - omega[0]

"""
Initialize model for cap
"""
cap_kwargs = train_data['kw_cap']
cap_kwargs['device'] = device
cap_model = PhononBTE(**cap_kwargs)

cap_model.init_distribution(cap_model.T_base)
cap_model.msd_base = cap_model.calc_msd(cap_model.g)

sub_kwargs = train_data['kw_sub']
sub_kwargs['device'] = device
sub_model = PhononBTE(**sub_kwargs)

sub_model.init_distribution(sub_model.T_base)
sub_model.msd_base = sub_model.calc_msd(sub_model.g)

dt = cap_kwargs['dt']
x = torch.cat((cap_model.xn, sub_model.xn + cap_model.Lx), dim=0).cpu().numpy()

#%% 

# read true emissivities
eps_bdry_true = train_data['eps_bdry']
eps_bulk_true = train_data['eps_bulk']
# read true relaxation time
tau_true = torch.from_numpy(train_data['tau'][0]).to(device)
# initial relaxation time params are initialized with various number of basis
# thus not defined here

# read true transmittance and reflectance
traref_true = torch.from_numpy(train_data['traref']).to(device)
# randomly initialize transmission reflection coefficients
cmidx = (cap_model.nzidx & sub_model.nzidx).cpu()

if noise_level > 0:
    msd_hist_true_complete = train_data['msd_hist'][:, training_sample_id, :] * \
        (1 + (np.random.rand(*train_data['msd_hist'][:, training_sample_id, :].shape) - 0.5) * 2 * noise_level)
else:
    msd_hist_true_complete = train_data['msd_hist'][:, training_sample_id, :].copy()

#%%

# loss_func = nn.MSELoss()
loss_func = nn.L1Loss()
print(f'\nloss func: {loss_func}')

# set checkpoint to save models
checkpoint_generator = loglinspace(0.3, 1)
checkpoint = next(checkpoint_generator)

start_time = time.time()

#%%
# from adjoint_bte.AdjointState_AutoDiff import NeuralODE
from adjoint_bte.model_heterostructure import BoltzmannTransportEquation_HeteroStruct
from adjoint_bte.utils_train import train_msd

if not os.path.exists(path_output):
    os.makedirs(path_output)

if flag_trial:
    num_tau_basis = [3, 5, 10]
    t_max = [2, 2, 2]
    max_epoch_init = 2
    max_epoch = [2, 2, 2]
    ratios_lr = [1.0, 1.0, 1.0]
    bs_params = [15, 13, 10]
else:
    num_tau_basis = [3, 5, 10]
    t_max = [60, 60, 60]
    if flag_fitTau:
        max_epoch_init = 1
    else:
        max_epoch_init = 5000
    max_epoch = [399, 300, 300]
    ratios_lr = [1.0, 1.0, 1.0]
    bs_params = [20, 20, 20]

print('\nnum_tau_basis: ', num_tau_basis, '    t_max: ', t_max,
    '\nmax_epoch: ', max_epoch, '    bs_params: ', bs_params)

idx_time = train_data['t_hist'] < t_max[0]
t_hist_true = train_data['t_hist'][idx_time]
T_hist_true = train_data['T_hist'][None, 0, training_sample_id, :]
I_hist_true = None
msd_hist_true = msd_hist_true_complete[idx_time, :, :]

if flag_fit_EpsBdry:
    eps_bdry = torch.rand(bs_params[0], 2).to(device)
else:
    lr_eps_bdry = 0.0
    eps_bdry = eps_bdry_true[None,:].repeat_interleave(bs_params[0], dim=0).to(device)
    print('\nlr_eps_bdry turned to zero.', flush=True)
print('\ninitial eps_bdry:', eps_bdry.mean(dim=0), flush=True)

if flag_fit_EpsBulk:
    eps_bulk = torch.rand(bs_params[0], 2).to(device)
else:
    lr_eps_bulk = 0.0
    eps_bulk = eps_bulk_true[None,:].repeat_interleave(bs_params[0], dim=0).to(device)
    print('\nlr_eps_bulk turned to zero.', flush=True)
print('\ninitial eps_bulk:', eps_bulk.mean(dim=0), flush=True)

if flag_fitTraref:
    traref_params_iv = torch.rand((bs_params[0], cap_model.Nf, cap_model.Nb)) * cmidx
    traref_fixed = None
else:
    traref_params_iv = torch.rand((bs_params[0], cap_model.Nf, cap_model.Nb)) * cmidx
    traref_fixed = traref_true[None].repeat_interleave(bs_params[0], dim=0).to(device)
    lr_traref = 0.0

if flag_fitTau:
    tau_cap_iv = None
    # tau_coeff_cap = 108 * torch.ones((bs_params[0], num_tau_basis[0], cap_model.Nb)).to(device)
    tau_coeff_cap = 216 * torch.rand((bs_params[0], num_tau_basis[0], cap_model.Nb)).to(device)
else:
    tau_cap_iv = 1e12 * tau_true[None,:].repeat_interleave(bs_params[0], dim=0)
    # tau_cap_iv = 108 * torch.ones_like(tau_true[None,:].repeat_interleave(bs_params[0], dim=0)) * cmidx.to(device)
    tau_coeff_cap = None
    lr_taucoeffcap = 0.0
    print('\nlr_taucoeffcap turned to zero.', flush=True)

bte_model = BoltzmannTransportEquation_HeteroStruct(
            (cap_model, sub_model), eps_bdry, eps_bulk, traref_params_iv.to(device),
            traref_fixed=traref_fixed, tau_cap_iv=tau_cap_iv, tau_coeff_cap=tau_coeff_cap, bs=bs_params[0]).to(device)

#%%

if flag_fitTau:
    optimizer = torch.optim.Adam(
        [
            {"params": bte_model.eps_bdry, "lr": lr_eps_bdry},
            {"params": bte_model.eps_bulk, "lr": lr_eps_bulk},
            {"params": bte_model.traref_params, "lr": lr_traref},
            {"params": bte_model.tau_coeff_cap, "lr": 0.0}
        ], lr=1e-2)
else:
    tau_fitting_dict = None
    optimizer = torch.optim.Adam(
        [
            {"params": bte_model.eps_bdry, "lr": lr_eps_bdry},
            {"params": bte_model.eps_bulk, "lr": lr_eps_bulk},
            {"params": bte_model.traref_params, "lr": lr_traref},
            {"params": bte_model.tau_cap, "lr": 0.0}
        ], lr=1e-2)

#%%

traref_params_iv, tau_coeff_iv, eps_bdry_iv, eps_bulk_iv, loss_iv = train_msd(bte_model, dt, bs_params[0], t_hist_true, T_hist_true, msd_hist_true, 
    loss_func, optimizer, max_epoch_init, (traref_true, tau_true, eps_bdry_true, eps_bulk_true),
    {'path_output': path_output, 'start_time': start_time, 'time_stamp': time_stamp, 'num_tau_basis': '0'}, 
    save_grads=save_grads, device=device
    )

#%%

if flag_fitTau:

    for j, _num_tau_basis in enumerate(num_tau_basis):

        if _num_tau_basis > 3:
            loss_func = nn.MSELoss()
            # loss_func = nn.L1Loss()
            print(f'\nloss func: {loss_func}')


        idx_time = train_data['t_hist'] < t_max[j]
        t_hist_true = train_data['t_hist'][idx_time]
        T_hist_true = train_data['T_hist'][idx_time, :, :]
        I_hist_true = None
        msd_hist_true = msd_hist_true_complete[idx_time, :, :]

        if j == 0:
            norm_basis = torch.trapz(bte_model.tau_basis, bte_model.model_cap.omega, dim=-1)
            tau_coords = torch.trapz(torch.einsum("nk, bkl -> bnkl", bte_model.tau_basis, tau_coeff_iv), 
                bte_model.model_cap.omega, dim=-2) / norm_basis[:,None]

            bte_model = BoltzmannTransportEquation_HeteroStruct(
                (cap_model, sub_model), eps_bdry_iv, eps_bulk_iv, traref_params_iv.to(device),
                traref_fixed=traref_fixed, tau_cap_iv=None, 
                tau_coeff_cap=tau_coords, bs=bs_params[j]).to(device)
        else:
            bte_model = BoltzmannTransportEquation_HeteroStruct(
                (cap_model, sub_model), torch.rand(bs_params[j], 2).to(device), torch.rand(bs_params[j], 2).to(device), traref_params_iv.to(device),
                traref_fixed=traref_fixed, tau_cap_iv=None,
                tau_coeff_cap=10 * torch.rand(bs_params[j], _num_tau_basis, 1), bs=bs_params[j]).to(device)

            norm_basis = torch.trapz(bte_model.tau_basis, bte_model.model_cap.omega, dim=-1)
            tau_coords = torch.trapz(torch.einsum("nk, bkl -> bnkl", bte_model.tau_basis, tau_coeff_iv), 
                bte_model.model_cap.omega, dim=-2) / norm_basis[:,None]
            
            if bs_params[j] <= bs_params[j-1]:
                traref_params_iv = traref_params_iv[torch.argsort(loss_iv)][:bs_params[j]]
                tau_coords = tau_coords[torch.argsort(loss_iv)][:bs_params[j]]
                if flag_fit_EpsBdry:
                    eps_bdry_iv = eps_bdry_iv[torch.argsort(loss_iv)][:bs_params[j]]
                else:
                    eps_bdry_iv = eps_bdry_true[None,:].repeat_interleave(bs_params[j], dim=0).to(device)
                
                if flag_fit_EpsBulk:
                    eps_bulk_iv = eps_bulk_iv[torch.argsort(loss_iv)][:bs_params[j]]
                else:
                    eps_bulk_iv = eps_bulk_true[None,:].repeat_interleave(bs_params[j], dim=0).to(device)

            bte_model = BoltzmannTransportEquation_HeteroStruct(
                (cap_model, sub_model), eps_bdry_iv, eps_bulk_iv, traref_params_iv.to(device),
                traref_fixed=traref_fixed, tau_cap_iv=None, 
                tau_coeff_cap=tau_coords, bs=bs_params[j]).to(device)

        optimizer = torch.optim.Adam(
            [
                {"params": bte_model.eps_bdry, "lr": lr_eps_bdry},
                {"params": bte_model.eps_bulk, "lr": lr_eps_bulk},
                {"params": bte_model.traref_params, "lr": lr_traref},
                {"params": bte_model.tau_coeff_cap, "lr": lr_taucoeffcap}
            ], lr=1e-2)
        
        traref_params_iv, tau_coeff_iv, eps_bdry_iv, eps_bulk_iv, loss_iv = train_msd(bte_model, dt, bs_params[j], t_hist_true, T_hist_true, msd_hist_true, 
            loss_func, optimizer, max_epoch[j], (traref_true, tau_true, eps_bdry_true, eps_bulk_true),
            {'path_output': path_output, 'start_time': start_time, 'time_stamp': time_stamp, 'num_tau_basis': _num_tau_basis}, 
            save_grads=save_grads,
            return_best=True, device=device
            )


# %%