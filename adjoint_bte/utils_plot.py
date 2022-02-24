import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(
    color=["#003f5c", "#665191", "#d45087", "#ff7c43"]
    )

import numpy as np

#%%

def plot_loss_vs_epoch(loss_epoch, save_fname):
    fig, ax = plt.subplots(1, 1, figsize=(6.4,6.4))
    ax.plot(np.arange(len(loss_epoch)), loss_epoch)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_yscale('log')
    fig.savefig(f'{save_fname}', bbox_inches='tight')
    plt.close()

def plot_tau_params_vs_epoch(tau_params_epoch, tau_params_true, save_fname=None, ylabel=None):
    len_tau_params = len(tau_params_true)
    fig, ax = plt.subplots(len_tau_params, 1, figsize=(6.4,6.4))
    if len_tau_params > 1:
        for i in range(len_tau_params):
            ax[i].plot(np.arange(len(tau_params_epoch)), tau_params_epoch[:,i], '-', color='C0')
            ax[i].hlines(tau_params_true[i], 0, len(tau_params_epoch), ls='--', color='C3')
            if i == len_tau_params-1: ax[i].set_xlabel('Epoch', fontsize=15)
            if ylabel is not None: ax[i].set_ylabel(f'{ylabel[i]}')
    else:
        ax.plot(np.arange(len(tau_params_epoch)), tau_params_epoch, '-', color='C0')
        ax.hlines(tau_params_true, 0, len(tau_params_epoch), ls='--', color='C3')
        ax.set_xlabel('Epoch')
        if ylabel is not None: ax.set_ylabel(f'{ylabel}')
    fig.tight_layout()
    fig.savefig(save_fname, bbox_inches='tight')
    plt.close()

def plot_tau_params_vs_epoch_batch(tau_params_epoch, tau_params_true, save_fname=None, ylabel=None):
    len_tau_params = len(tau_params_true)
    tau_params_avg, tau_params_std = tau_params_epoch.mean(axis=1), tau_params_epoch.std(axis=1)
    fig, ax = plt.subplots(len_tau_params, 1, figsize=(6.4,6.4))
    if len_tau_params > 1:
        for i in range(len_tau_params):
            ax[i].plot(np.arange(len(tau_params_epoch)), tau_params_avg[:,i], '-', color='C0')
            ax[i].fill_between(np.arange(len(tau_params_epoch)), 
                tau_params_avg[:,i] - tau_params_std[:,i], tau_params_avg[:,i] + tau_params_std[:,i],
                alpha=0.25)
            ax[i].hlines(tau_params_true[i], 0, len(tau_params_epoch), ls='--', color='C3')
            if i == len_tau_params-1: ax[i].set_xlabel('Epoch', fontsize=15)
            if ylabel is not None: ax[i].set_ylabel(f'{ylabel[i]}')
    else:
        ax.plot(np.arange(len(tau_params_epoch)), tau_params_avg, '-', color='C0')
        ax.hlines(tau_params_true, 0, len(tau_params_epoch), ls='--', color='C3')
        ax.set_xlabel('Epoch')
        if ylabel is not None: ax.set_ylabel(f'{ylabel}')
    fig.tight_layout()
    if save_fname is not None: fig.savefig(save_fname, bbox_inches='tight')
    plt.close()

def plot_tau_batch(omega, tau, tau_true, ls=['-', '--'], tau_lastep=None, epoch=None, weights=None, save_fname=None):
    num_branch = tau.shape[-1]
    if weights is None:
        tau_mean, tau_std = tau.mean(axis=0), tau.std(axis=0)
    else:
        tau_mean = np.einsum("bkl, b -> kl", tau, weights) / np.sum(weights)
        tau_std = np.sqrt(np.einsum("bkl, b -> kl", (tau - tau_mean) ** 2, weights) / np.sum(weights))
    fig, ax = plt.subplots(num_branch, 1, figsize=(6.4, 6.4))
    if num_branch > 1:
        for i in range(num_branch):
            ax[i].fill_between(
                omega, 
                tau_mean[:, i] - tau_std[:, i], tau_mean[:, i] + tau_std[:, i], 
                color='C0', alpha=0.25)
            ax[i].plot(omega, tau_mean[:, i], ls=ls[0], color='C0', label='Prediction')
            if tau_true is not None: ax[i].plot(omega, tau_true[:,i], ls=ls[1], color='C3', label='Truth')

            if i == num_branch - 1:
                ax[i].set_xlabel('Frequency')
                ax[i].legend(loc='right')
    else:
        ax.fill_between(
            omega, 
            tau_mean[:, 0] - tau_std[:, 0], tau_mean[:, 0] + tau_std[:, 0], 
            alpha=0.25)
        ax.plot(omega, tau_mean[:, 0], ls=ls[0], color='C0', label='Prediction')
        if tau_true is not None: ax.plot(omega, tau_true[:, 0], ls=ls[1], color='C3', label='Truth')
        if tau_lastep is not None: ax.plot(omega, tau_lastep[:, 0], ls=ls[1], color='r', label='Last epoch')
        ax.set_xlabel('Frequency')
        ax.legend(loc='upper right')
        ax.set_title(f'Epoch: {epoch:4d}')
    fig.tight_layout()
    if save_fname is not None: 
        fig.savefig(f'{save_fname}', bbox_inches='tight')
        
    plt.close()

def plot_traref(omega, traref, traref_true, save_fname=None):
    num_branch = traref.shape[-1]
    fig, ax = plt.subplots(num_branch, 1, figsize=(6.4, 6.4))
    for i in range(num_branch):
        ax[i].plot(omega, traref[0, 0, :, i], ls='-', color='C0', label='Prediction')
        ax[i].plot(omega, traref_true[0, 0, :,i], ls='--', color='C3', label='Truth')
        ax[i].plot(omega, traref[1, 1, :, i] + 1, ls='-', color='C0')
        ax[i].plot(omega, traref_true[1, 1, :,i] + 1, ls='--', color='C3')
        if i == num_branch - 1:
            ax[i].set_xlabel('Frequency')
            ax[i].legend(loc='right')
    fig.tight_layout()
    if save_fname is not None: fig.savefig(save_fname, bbox_inches='tight')
    plt.close()

def plot_traref_batch(omega, traref, traref_true, ls=['-', '--'], save_fname=None):
    num_branch = traref.shape[-1]
    traref_mean, traref_std = traref.mean(axis=0), traref.std(axis=0)
    fig, ax = plt.subplots(2, num_branch,  figsize=(num_branch * 6.4, 6.4))
    if num_branch > 1:
        for i in range(num_branch):
            ax[0,i].fill_between(
                omega, 
                traref_mean[0, 0, :, i] - traref_std[0, 0, :, i], traref_mean[0, 0, :, i] + traref_std[0, 0, :, i], 
                color='C0', alpha=0.2)
            ax[0,i].plot(omega, traref_mean[0, 0, :, i], ls=ls[0], color='C0')
            if traref_true is not None: ax[0,i].plot(omega, traref_true[0, 0, :,i], ls=ls[1], color='C3')

            ax[1,i].fill_between(
                omega, 
                traref_mean[1, 1, :, i] - traref_std[1, 1, :, i], traref_mean[1, 1, :, i] + traref_std[1, 1, :, i], 
                color='C0', alpha=0.2)
            ax[1,i].plot(omega, traref_mean[1, 1, :, i], ls=ls[0], color='C0', label='Prediction')
            if traref_true is not None: ax[1,i].plot(omega, traref_true[1, 1, :,i], ls=ls[1], color='C3', label='Truth')
            if i == num_branch - 1:
                ax[1,i].set_xlabel('Frequency')
                ax[1,i].legend(loc='upper right')
    else:
        ax[0].fill_between(
            omega, 
            traref_mean[0, 0, :, 0] - traref_std[0, 0, :, 0], traref_mean[0, 0, :, 0] + traref_std[0, 0, :, 0], 
            color='C0', alpha=0.25)
        ax[0].plot(omega, traref_mean[0, 0, :, 0], ls=ls[0], color='C0')
        if traref_true is not None: ax[0].plot(omega, traref_true[0, 0, :, 0], ls=ls[1], color='C3')

        ax[1].fill_between(
            omega, 
            traref_mean[1, 1, :, 0] - traref_std[1, 1, :, 0], traref_mean[1, 1, :, 0] + traref_std[1, 1, :, 0], 
            color='C0', alpha=0.25)
        ax[1].plot(omega, traref_mean[1, 1, :, 0], ls=ls[0], color='C0', label='Prediction')
        if traref_true is not None: ax[1].plot(omega, traref_true[1, 1, :, 0], ls=ls[1], color='C3', label='Truth')
        ax[1].set_xlabel('Frequency')
        ax[1].legend(loc='upper right')
    fig.tight_layout()
    if save_fname is not None: 
        fig.savefig(f'{save_fname}', bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_epsilons_vs_epoch(epsilons_epoch, epsilons_true, save_fname=None):
    num_epsilons = epsilons_epoch.shape[-1]
    len_epsilons = len(epsilons_epoch)
    fig, ax = plt.subplots(num_epsilons, 1, figsize=(6.4,6.4))
    if num_epsilons > 1:
        for i in range(num_epsilons):
            ax[i].plot(np.arange(len_epsilons), epsilons_epoch[:,i], '-', color='C0')
            if epsilons_true is not None: ax[i].hlines(epsilons_true[i], 0, len_epsilons, ls='--', color='C3')
            if i == num_epsilons-1: ax[i].set_xlabel('Epoch', fontsize=15)
    else:
        ax.plot(np.arange(len_epsilons), epsilons_epoch, '-', color='C0')
        if epsilons_true is not None: ax.hlines(epsilons_true, 0, len_epsilons, ls='--', color='C3')
        ax.set_xlabel('Epoch')
    fig.tight_layout()
    if save_fname is not None: fig.savefig(save_fname, bbox_inches='tight')
    plt.close()

def plot_epsilons_vs_epoch_batch(epsilons_epoch, epsilons_true, save_fname=None):
    num_epsilons = epsilons_epoch.shape[-1]
    len_epsilons = len(epsilons_epoch)
    epsilons_mean, epsilons_std = epsilons_epoch.mean(axis=1), epsilons_epoch.std(axis=1)
    fig, ax = plt.subplots(num_epsilons, 1, figsize=(6.4,6.4))
    if num_epsilons > 1:
        for i in range(num_epsilons):
            ax[i].plot(np.arange(len_epsilons), epsilons_mean[:,i], '-', color='C0')
            ax[i].fill_between(np.arange(len_epsilons), 
            epsilons_mean[:,i] - epsilons_std[:,i], epsilons_mean[:,i] + epsilons_std[:,i],
            alpha=0.25)
            if epsilons_true is not None: ax[i].hlines(epsilons_true[i], 0, len_epsilons, ls='--', color='C3')
            if i == num_epsilons-1: ax[i].set_xlabel('Epoch', fontsize=15)
    else:
        ax.plot(np.arange(len_epsilons), epsilons_epoch, '-', color='C0')
        if epsilons_true is not None: ax.hlines(epsilons_true, 0, len_epsilons, ls='--', color='C3')
        ax.set_xlabel('Epoch')
    fig.tight_layout()
    if save_fname is not None: fig.savefig(save_fname, bbox_inches='tight')
    plt.close()

def plot_mucrits_vs_epoch_batch(mu_crits_epoch, mu_crit_true, mu_lims, save_fname=None):
    mu_lbs, mu_ubs = mu_lims
    num_mucrits = mu_crits_epoch.shape[-1]
    len_mucrits = len(mu_crits_epoch)
    fig, ax = plt.subplots(num_mucrits, 1, figsize=(6.4,6.4))
    if num_mucrits > 1:
        for i in range(num_mucrits):
            ax[i].plot(np.arange(len_mucrits), mu_crits_epoch[:,:,i], '-', color='C0')
            ax[i].hlines(mu_lbs, 0, len_mucrits, ls='-.', color='gray')
            ax[i].hlines(mu_ubs, 0, len_mucrits, ls='-.', color='gray')
            if mu_crit_true is not None: ax[i].hlines(mu_crit_true[i], 0, len_mucrits, ls='--', color='C3')
            if i == num_mucrits-1: ax[i].set_xlabel('Epoch', fontsize=15)
    else:
        ax.plot(np.arange(len_mucrits), mu_crits_epoch[:,:,0], '-', color='C0')
        ax.hlines(mu_lbs, 0, len_mucrits, ls='-.', color='gray')
        ax.hlines(mu_ubs, 0, len_mucrits, ls='-.', color='gray')
        if mu_crit_true is not None: ax.hlines(mu_crit_true, 0, len_mucrits, ls='--', color='C3')
        ax.set_xlabel('Epoch')
    fig.tight_layout()
    if save_fname is not None: fig.savefig(save_fname, bbox_inches='tight')
    plt.close()

def plot_T_comparison(t, x, T_pred, T_true, save_fname=None):
    del_T = np.abs(T_true - T_pred) / T_true
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 6.4))
    c = ax.pcolormesh(x, t, del_T, cmap='GnBu', shading='nearest')
    ax.set_xlabel('x [nm]')
    ax.set_ylabel('t [ps]')
    cbar = fig.colorbar(c, ax=ax)
    cbar.ax.set_ylabel(r'$\frac{|T_{true}-T_{pred}|}{T_{true}}$', labelpad=-40, y=1.15, rotation=0)

    fig.tight_layout()
    if save_fname is not None: fig.savefig(save_fname, bbox_inches='tight')
    plt.close()