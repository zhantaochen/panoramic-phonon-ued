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
    device = "cuda"
else:
    device = "cpu"

flag_trial = False
if flag_trial: device = "cuda:7"

save_grads = True

flag_fit_EpsBdry = False
flag_fit_EpsBulk = True
flag_fitTraref = True
flag_fitTau = False

OmegaMax, OmegaPoint = 12, 120

training_sample_id = [2]
print('training samples: ', training_sample_id)

#%%

simulation_fname = f'data/Au_Si_samples/Experiment_HeteroStruct_Au_Si_OmegaMax{OmegaMax}_OmegaPoint{OmegaPoint}.pt'
time_stamp = datetime.today().strftime('%Y-%m-%d_%H-%M')
path_output = f"outputs/Au_Si_samples/Au_Si_ExperimentData_{time_stamp}"

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

cmidx = (cap_model.nzidx & sub_model.nzidx).cpu()

msd_hist_true_complete = train_data['msd_hist'][:,training_sample_id,:].copy()

#%%

# loss_func = nn.MSELoss()
loss_func = nn.L1Loss()
print(f'\nloss func: {loss_func}')

# set checkpoint to save models
checkpoint_generator = loglinspace(0.3, 1)
checkpoint = next(checkpoint_generator)

start_time = time.time()

#%%
from adjoint_bte.model_heterostructure import BoltzmannTransportEquation_HeteroStruct
from adjoint_bte.utils_train import train_msd

if not os.path.exists(path_output):
    os.makedirs(path_output)

if flag_trial:
    num_tau_basis = [7, 10, 15, 20]
    t_max = [11, 11, 11, 11]
    max_epoch_init = 2
    max_epoch = [2, 2, 2, 2]
    bs_params = [20, 15, 10, 8]
else:
    num_tau_basis = [7, 10, 15]
    t_max = [57, 57, 57]
    # t_max = [7, 7, 7]
    # t_max = [120, 120, 120]
    max_epoch_init = 2000
    max_epoch = [100, 200, 200]
    bs_params = [20, 20, 20]

print('\nnum_tau_basis: ', num_tau_basis, '    t_max: ', t_max,
    '\nmax_epoch: ', max_epoch, '    bs_params: ', bs_params)

idx_time = train_data['t_hist'] < t_max[0]
t_hist_true = train_data['t_hist'][idx_time]
T_hist_true = train_data['T_hist'][None, 0, training_sample_id, :].cpu().numpy()
msd_hist_true = msd_hist_true_complete[idx_time, :, :]
print(t_hist_true)

# torch.manual_seed(0)
eps_bdry = torch.rand(bs_params[0], 2).to(device) * 1
# torch.manual_seed(0)
eps_bulk = torch.rand(bs_params[0], 2).to(device) * 0.1

if not flag_fit_EpsBdry:
    eps_bdry *= 0
    lr_eps_bdry = 0.0
    print('\nlr_eps_bdry turned to zero.', flush=True)
# eps_bdry = torch.tensor([[0.011, 0.431]]).repeat_interleave(bs_params[0], dim=0).to(device)

if not flag_fit_EpsBulk:
    eps_bulk *= 0
    lr_eps_bulk = 0.0
    print('\nlr_eps_bulk turned to zero.', flush=True)

print('\ninitial eps_bdry:\n', eps_bdry, flush=True)
print('\ninitial eps_bulk:', eps_bulk.mean(dim=0), flush=True)


#%% initialize traref with dmm 

# mu_plus_int, mu_minus_int = 1, 1
# def get_traref_dmm():
#     cmidx = cap_model.nzidx & sub_model.nzidx
#     # cmidx = ((sub_model.dos * sub_model.vg) / (cap_model.dos * cap_model.vg + 1e-12) < 50) & ((sub_model.dos * sub_model.vg) / (cap_model.dos * cap_model.vg + 1e-12) > 0.02)
#     traref = torch.zeros((2, 2, sub_model.Nf, sub_model.Nb)).to(device)
#     # transmisstion coefficient from cap to sub
#     traref[0,0][cmidx] = 1.0 * (mu_minus_int * sub_model.dos[cmidx] * sub_model.vg[cmidx]) \
#         / ((mu_minus_int * sub_model.dos[cmidx] * sub_model.vg[cmidx]) + (mu_plus_int * cap_model.dos[cmidx] * cap_model.vg[cmidx]))
#     # transmisstion coefficient from sub to cap, detailed balance
#     traref[1,1][cmidx] = 1.0 * (mu_plus_int * cap_model.dos[cmidx] * cap_model.vg[cmidx]) \
#         / ((mu_minus_int * sub_model.dos[cmidx] * sub_model.vg[cmidx]) + (mu_plus_int * cap_model.dos[cmidx] * cap_model.vg[cmidx]))
#     # assert detailed balance
#     assert torch.allclose(traref[0,0][cmidx] * cap_model.dos[cmidx] * cap_model.vg[cmidx] * mu_plus_int, 
#                         traref[1,1][cmidx] * sub_model.dos[cmidx] * sub_model.vg[cmidx] * mu_minus_int)
#     traref[0,1][cap_model.nzidx] = 1 - traref[0,0][cap_model.nzidx]
#     # reflection coefficient from sub to sub
#     traref[1,0][sub_model.nzidx] = 1 - traref[1,1][sub_model.nzidx]
#     assert torch.all(traref <= 1) and torch.all(traref >= 0)
#     return traref
    
# traref_dmm = get_traref_dmm()
# traref_params_iv = torch.zeros((bs_params[0], cap_model.Nf, cap_model.Nb)).to(device)

# ratio = torch.zeros((sub_model.Nf,sub_model.Nb)).to(device)
# ratio[cmidx] = mu_plus_int * cap_model.dos[cmidx] * cap_model.vg[cmidx] / (mu_minus_int * sub_model.dos[cmidx] * sub_model.vg[cmidx] + 1e-12)

# idx_tmp = (ratio < 1.0) * cmidx.to(device)
# traref_params_iv[:,idx_tmp] = traref_dmm.clone().to(device)[0, 0][idx_tmp]

# idx_tmp = (ratio > 1.0) * cmidx.to(device)
# traref_params_iv[:,idx_tmp] = traref_dmm.clone().to(device)[1, 1][idx_tmp]

# random initialization of transmission coefficients

if flag_fitTraref:
    # torch.manual_seed(0)
    traref_params_iv = torch.rand((bs_params[0], cap_model.Nf, cap_model.Nb)) * cmidx
    traref_fixed = None
else:
    # torch.manual_seed(0)
    traref_params_iv = torch.ones((bs_params[0], cap_model.Nf, cap_model.Nb)) * 0.5 * cmidx
    traref_fixed = None
    lr_traref *= 0.0
    print('\nlr_traref turned to zero.', flush=True)


#%%

from adjoint_bte.model_heterostructure import BoltzmannTransportEquation_HeteroStruct
from adjoint_bte.utils_train import train_msd

tau_cap_iv = 1e12 * train_data['kw_cap']['mater_prop']['tau'][None].repeat_interleave(bs_params[0], dim=0).to(device)
tau_coeff_cap = None
lr_taucoeffcap = 0.0
print('\nlr_taucoeffcap turned to zero.', flush=True)

bte_model = BoltzmannTransportEquation_HeteroStruct(
            (cap_model, sub_model), 
            eps_bdry, eps_bulk, 
            traref_params_iv.to(device),
            traref_fixed=None, 
            tau_cap_iv=tau_cap_iv, 
            tau_coeff_cap=tau_coeff_cap, 
            bs=bs_params[0]).to(device)

optimizer = torch.optim.Adam(
    [
        {"params": bte_model.eps_bdry, "lr": lr_eps_bdry},
        {"params": bte_model.eps_bulk, "lr": lr_eps_bulk},
        {"params": bte_model.traref_params, "lr": lr_traref},
        {"params": bte_model.tau_cap, "lr": lr_taucoeffcap}
    ], lr=0.01)

# optimizer = torch.optim.Adam(
#     [
#         {"params": bte_model.eps_bdry, "lr": lr_eps_bdry},
#         {"params": bte_model.traref_params, "lr": lr_traref},
#         {"params": bte_model.tau_cap, "lr": lr_taucoeffcap}
#     ], lr=0.01)

#%%

traref_params_iv, tau_coeff_iv, eps_bdry_best, eps_bulk_best, loss_iv = train_msd(bte_model, dt, bs_params[0], t_hist_true, T_hist_true, msd_hist_true, 
    loss_func, optimizer, max_epoch_init, (None, None, None, None),
    {'path_output': path_output, 'start_time': start_time, 'time_stamp': time_stamp, 'num_tau_basis': 0}, save_grads=save_grads, 
    device=device
    )
#%%