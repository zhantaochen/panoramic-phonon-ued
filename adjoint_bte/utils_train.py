import math, sys
import numpy as np
import time

import torch
# from torch.profiler import profile, record_function, ProfilerActivity

from adjoint_bte.utils_plot import plot_loss_vs_epoch, plot_traref_batch, \
    plot_tau_batch, plot_epsilons_vs_epoch_batch

from .utils import compute_heatcap_new


from adjoint_bte.torchdiffeq import odeint_adjoint

def train_msd(
    bte_model, 
    dt, 
    bs_params, 
    t_hist_true, 
    T_hist_true, 
    msd_hist_true,
    loss_func, 
    optimizer, 
    max_epoch, 
    true_params,
    misc_dict, 
    save_grads=False, 
    return_best=False, 
    device='cpu'
):

    Nx_cap = bte_model.model_cap.Nx
    Nx_sub = bte_model.model_sub.Nx
    
    omega = bte_model.model_cap.omega.detach().cpu().numpy()
    cmidx = (bte_model.model_cap.nzidx & bte_model.model_sub.nzidx).cpu()
    idx_samples = np.arange(msd_hist_true.shape[1])
    traref_true, tau_true, eps_bdry_true, eps_bulk_true = true_params

    path_output = misc_dict['path_output']
    start_time = misc_dict['start_time']
    time_stamp = misc_dict['time_stamp']
    num_tau_basis = misc_dict['num_tau_basis']

    loss_epoch, loss_iv_epoch, T_epoch, msd_epoch = [], [], [], []
    eps_bdry_epoch, eps_bulk_epoch, tau_cap_params_epoch, tau_cap_epoch, traref_epoch, grads_epoch = [], [], [], [], [], {}

    for epoch in range(max_epoch):
        N_tpoints = t_hist_true.shape[0]
        idx_sample_ = np.random.choice(idx_samples, bs_params, replace=True)
        T_cap_init = torch.zeros((bs_params, Nx_cap))
        T_sub_init = torch.zeros((bs_params, Nx_sub))
        msd_target = torch.zeros((N_tpoints, bs_params, 2))
        for i, idx_ in enumerate(idx_sample_):
            T_cap_init[i,:] = torch.from_numpy(T_hist_true[0, idx_, :Nx_cap])
            T_sub_init[i,:] = torch.from_numpy(T_hist_true[0, idx_, Nx_cap:])
            msd_target[:,i,:] = torch.from_numpy(msd_hist_true[:, idx_, :])
        #######################################
        # initialization and forward pass
        #######################################
        if hasattr(bte_model, 'tau_coeff_cap'):
            bte_model.model_cap.tau = bte_model.get_tau(bte_model.tau_coeff_cap) * 1e-12
        if bte_model.tau_cap is not None:
            bte_model.model_cap.tau = bte_model.tau_cap * 1e-12
        bte_model.model_cap.init_distribution(T_cap_init.to(device), T_lims = [295, 1.05 * max(T_cap_init.max().item(), T_sub_init.max().item())])
        bte_model.model_sub.init_distribution(T_sub_init.to(device), T_lims = [295, 1.05 * max(T_cap_init.max().item(), T_sub_init.max().item())])

        ts_ = torch.from_numpy(t_hist_true).to(device)
        g0_ = torch.cat((bte_model.model_cap.g, bte_model.model_sub.g), dim=1)
        
        g_ = odeint_adjoint(bte_model, g0_, ts_, method='euler', 
            options={'step_size': dt, 'save_all_timepoints': True, 'requires_grad': False})

        T_ = torch.zeros((g_.size(0), g_.size(1), g_.size(2))).to(g_)
        msd_ = torch.zeros((g_.size(0), g_.size(1), 2)).to(g_)
        for i, g_tmp_ in enumerate(g_):
            T_[i, :, :Nx_cap] = bte_model.model_cap.solve_Temp_interp(g_tmp_[:, :Nx_cap])
            T_[i, :, Nx_cap:] = bte_model.model_sub.solve_Temp_interp(g_tmp_[:, Nx_cap:])
            msd_[i, :, 0] = bte_model.model_cap.calc_msd(g_tmp_[:, :Nx_cap]) - bte_model.model_cap.msd_base
            msd_[i, :, 1] = bte_model.model_sub.calc_msd(g_tmp_[:, Nx_cap:]) - bte_model.model_sub.msd_base
        
        del g_
        # print(msd_)
        # print(msd_target)
        loss = loss_func(msd_, msd_target.to(device))
        loss_iv = (msd_ - msd_target.to(device)).pow(2).mean(dim=(0,-1))

        # print(loss.item())
        #######################################
        # loss on smoothness
        #######################################

        if epoch == 0:
            # initialize the best loss with a large number
            loss_best = 1000

        #######################################
        # save model information to list
        #######################################
        loss_epoch.append(loss.item())
        loss_iv_epoch.append(loss_iv.detach().cpu().numpy())
        T_epoch.append(T_.detach().cpu().numpy())
        msd_epoch.append(msd_.detach().cpu().numpy())
        tau_cap_epoch.append(bte_model.model_cap.tau.detach().cpu().numpy() * 1e12)
        if hasattr(bte_model, 'tau_coeff_cap'):
            tau_cap_params_epoch.append(bte_model.tau_coeff_cap.detach().cpu().numpy())
        eps_bdry_epoch.append(bte_model.eps_bdry.detach().cpu().numpy())
        eps_bulk_epoch.append(bte_model.eps_bulk.detach().cpu().numpy())
        traref_epoch.append(bte_model.traref.detach().cpu().numpy())        

        #######################################
        # back propagate as normal
        #######################################
        
        '''debug record memory usage
        '''
        # mem_params = sum([param.nelement()*param.element_size() for param in bte_model.parameters()])
        # mem_bufs = sum([buf.nelement()*buf.element_size() for buf in bte_model.buffers()])
        # print(f"\nepoch: {epoch}: mem_params: {mem_params}, mem_bufs: {mem_bufs}" + 
        #     f"\n    total size: {sys.getsizeof(bte_model)}")
        '''debug record memory usage
        '''
        
        # bte_model.zero_grad()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #######################################
        # clamp parameters to physical ranges
        #######################################
        
        with torch.no_grad():
            bte_model.eps_bdry[:] = bte_model.eps_bdry.detach().clamp_(0.0, 1.0)
            bte_model.eps_bulk[:] = bte_model.eps_bulk.detach().clamp_(0.0, 1.0)
            bte_model.traref_params[:] = bte_model.traref_params.detach().clamp_(0.0, 1.0) * cmidx.to(device)

            if hasattr(bte_model, 'tau_coeff_cap'):
                bte_model.tau_coeff_cap[:] = bte_model.tau_coeff_cap.detach().clamp_min_(1.0)
            if bte_model.tau_cap is not None:
                bte_model.tau_cap[:] = bte_model.tau_cap.detach().clamp_min_(1.0)

        #######################################
        # print and save training info
        #######################################
        end_time = time.time()
        print(f"\nepoch: {epoch:3d}    ", f"loss: {loss.item():.4e}    ", 
            f"eps_bdry: {np.around(bte_model.eps_bdry.detach().cpu().numpy().mean(axis=0), 4).tolist()}    ",
            f"eps_bulk: {np.around(bte_model.eps_bulk.detach().cpu().numpy().mean(axis=0), 4).tolist()}    ",
            f"elapsed time = {time.strftime('%H:%M:%S', time.gmtime(end_time - start_time))}",
            flush=True)

        #######################################
        # store to local file and make plots
        #######################################
        if save_grads:
            if bte_model.eps_bdry.grad is not None:
                if "eps_bdry" not in grads_epoch.keys():
                    grads_epoch["eps_bdry"] = []
                grads_epoch["eps_bdry"].append(bte_model.eps_bdry.grad.detach().cpu().numpy())

            if bte_model.eps_bulk.grad is not None:
                if "eps_bulk" not in grads_epoch.keys():
                    grads_epoch["eps_bulk"] = []
                grads_epoch["eps_bulk"].append(bte_model.eps_bulk.grad.detach().cpu().numpy())
            
            if bte_model.traref_params.grad is not None:
                if "traref_params" not in grads_epoch.keys():
                    grads_epoch["traref_params"] = []
                grads_epoch["traref_params"].append(bte_model.traref_params.grad[:,bte_model.cmidx].detach().cpu().numpy())

            if hasattr(bte_model, 'tau_coeff_cap'):
                if "tau_coeff_cap" not in grads_epoch.keys():
                    grads_epoch["tau_coeff_cap"] = []
                grads_epoch["tau_coeff_cap"].append(bte_model.tau_coeff_cap.grad.detach().cpu().numpy())

            elif bte_model.tau_cap is not None:
                if "tau_cap" not in grads_epoch.keys():
                    grads_epoch["tau_cap"] = []
                grads_epoch["tau_cap"].append(bte_model.tau_cap.grad[:,bte_model.cmidx].detach().cpu().numpy())

            
            print("Gradients:")
            print("\n".join([f" {key}: {np.abs(grads_epoch[key][-1]).mean()}" for key in grads_epoch.keys()]))
            # print("Gradients")
            # print(grads_epoch["eps_bdry"][-1])

        if epoch % 1 == 0 or epoch == max_epoch - 1:
            
            train_hist = {
                "loss": loss_epoch,
                "loss_iv": loss_iv_epoch,
                "T": T_epoch,
                "msd": msd_epoch,
                "eps_bdry": eps_bdry_epoch,
                "eps_bulk": eps_bulk_epoch,
                "tau_params_cap": tau_cap_params_epoch,
                "tau_cap": tau_cap_epoch,
                "traref": traref_epoch,
                "gradients": grads_epoch
                }
            saved_dict = {
                "state_dict": bte_model.state_dict(),
                "train_hist": train_hist,
                "t_hist_true": t_hist_true,
                "msd_true": msd_hist_true
                }

            torch.save(saved_dict, 
                f"{path_output}/{time_stamp}_trained_model_full_data_{num_tau_basis}.pt")

            plot_loss_vs_epoch(np.array([loss_epoch]).T, 
                save_fname=f'{path_output}/{time_stamp}_loss_epoch_{num_tau_basis}_td.pdf')

            if traref_true is not None: 
                traref_true_plot = traref_true.detach().cpu().numpy()
            else:
                traref_true_plot = None
            plot_traref_batch(omega, bte_model.traref.detach().cpu().numpy(),
                traref_true_plot, ls=['-', '--'], 
                save_fname=f'{path_output}/{time_stamp}_traref_{num_tau_basis}_td.pdf')

            if tau_true is not None: 
                tau_true_plot = tau_true.detach().cpu().numpy() * 1e12
            else:
                tau_true_plot = None

            if epoch > 0:
                plot_tau_batch(omega, bte_model.model_cap.tau.detach().cpu().numpy() * 1e12,
                    tau_true_plot, ls=['-', '--'], tau_lastep=tau_lastep, epoch=epoch,
                    save_fname=f'{path_output}/{time_stamp}_tau_{num_tau_basis}_td.pdf')
            tau_lastep = bte_model.model_cap.tau.detach().cpu().numpy().mean(axis=0) * 1e12
                
            
            if eps_bdry_true is not None: 
                eps_bdry_true_plot = eps_bdry_true.cpu().numpy()
            else:
                eps_bdry_true_plot = None
            plot_epsilons_vs_epoch_batch(np.array(eps_bdry_epoch), eps_bdry_true_plot,
                save_fname=f'{path_output}/{time_stamp}_eps_bdry_{num_tau_basis}_td.pdf')

                
            if eps_bulk_true is not None: 
                eps_bulk_true_plot = eps_bulk_true.cpu().numpy()
            else:
                eps_bulk_true_plot = None
            plot_epsilons_vs_epoch_batch(np.array(eps_bulk_epoch), eps_bulk_true_plot,
                save_fname=f'{path_output}/{time_stamp}_eps_bulk_{num_tau_basis}_td.pdf')
            
        if hasattr(bte_model, 'tau_coeff_cap'):
            tau_returned = bte_model.get_tau(bte_model.tau_coeff_cap)
        else:
            tau_returned = bte_model.tau_cap.detach()

        if loss.item() < loss_best:
            loss_best = loss.item()
            print(f'best loss updated: {loss_best:.3e}', flush=True)
            traref_params_best = bte_model.traref_params.detach()
            tau_returned_best = tau_returned
            eps_bdry_best = bte_model.eps_bdry.detach()
            eps_bulk_best = bte_model.eps_bulk.detach()
            loss_iv_best = loss_iv.detach()

    if return_best:
        return traref_params_best, tau_returned_best, eps_bdry_best, eps_bulk_best, loss_iv_best
    else:
        return bte_model.traref_params.detach(), tau_returned, bte_model.eps_bdry.detach(), bte_model.eps_bulk.detach(), loss_iv.detach()
