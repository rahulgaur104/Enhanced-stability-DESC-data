#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The purpose of this script is to plot the growth rate vs kx, ky and eigenfunction corresponding to the maximum growth rate.
We also want to diagnose various modes to detect and extract only the KBMs and plot them in a shat-dP_ds space.
"""
import os
import pdb
import numpy as np
from netCDF4 import Dataset as ds
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

parent_dir = os.getcwd()


def max_growth_rate(idx0, idx1, idx2, idx3, ky_mode_lim, kx_mode_lim, plot = False):
    # calculates the maximum growth rate and corresponding wavenumber ky, kx
    # Inputs: dofidx, rhoidx and iter, respectively

    #gs2_scan_dirname = f"{parent_dir}/nperiod2_data/rho_{idx0:02d}_alpha_{idx1:02d}/sfac_{idx2:02d}_dP_ds_{idx3:02d}"
    gs2_scan_dirname = f"{parent_dir}/rho_{idx0:02d}_alpha_{idx1:02d}/sfac_{idx2:02d}_dP_ds_{idx3:02d}"
    fname = f"{gs2_scan_dirname}/gs2_input.out.nc"
    #fname = f"{gs2_scan_dirname}/gs2_input_bakdif02.out.nc"
    #try:
    rtg   = ds(fname, 'r')
    ky     = rtg.variables['ky'][:ky_mode_lim].data
    kx     = rtg.variables['kx'][:kx_mode_lim].data
    theta0 = rtg.variables['theta0'][:ky_mode_lim, :kx_mode_lim].data
    
    beta = rtg.variables['beta'][:].data
    
    #pdb.set_trace()   
    idxs   = np.where(np.isnan(rtg.variables['phi2'][:].data) == True)[0]
    
    t      = rtg.variables['t'][:].data 
    
    fnamei = f"{gs2_scan_dirname}/gs2_input.in"
    with open(fnamei, 'r') as f:
        list0 = f.readlines()
    
    # ERROR_PRONE DO NOT CHANGE the number of lines in the gs2_template file
    # This may happen if we aren't cautious
    delt  = eval(list0[98].split('=')[1])
    nstep = eval(list0[99].split('=')[1])
    
    if len(idxs) > 0 and  np.min(idxs) <= 1: # Everything is nan
        #print("first if statement ", idx0, idx1, idx2, "idxs", idxs, rtg.variables['phi2'][:].data)
        gamma_max = np.array([np.nan])
        omega_max = np.array([np.nan])
        omega_max = np.array([np.nan])
        ky_max = np.array([np.nan])
        kx_max = np.array([np.nan])
        print("everything is nan! Run again!")
    elif len(idxs) > 0 and np.min(idxs) > 1: # nan occurs because the eigenfunctions > 1e300
        #print("second if statement ", idx0, idx1, idx2, idxs)
        if np.min(idxs) <=4: # super large growth rate or small nsteps
            idx    = np.min(idxs)
            gammac = np.mean(rtg.variables['omega_average'][0:idx, :ky_mode_lim, :kx_mode_lim, 1].data, axis=0)
            omegac = np.mean(rtg.variables['omega_average'][0:idx, :ky_mode_lim, :kx_mode_lim, 0].data, axis=0)
            phi2   = rtg.variables['phi2_by_mode'][:,:ky_mode_lim, :kx_mode_lim].data
            phi2t  = rtg.variables['phi2'][:].data
            gammaf = 0.5*np.vstack([np.polyfit(t[0:idx], np.log(phi2[0:idx, :, i]), 1)[0] for i in range(len(kx))]).T
        else:
            # Minimum idx at which we encounter a Nan
            idx    = np.min(idxs)
            gammac = np.mean(rtg.variables['omega_average'][idx-4:idx-1, :ky_mode_lim, :kx_mode_lim, 1].data, axis=0)
            omegac = np.mean(rtg.variables['omega_average'][idx-4:idx-1, :ky_mode_lim, :kx_mode_lim, 0].data, axis=0)
            phi2   = rtg.variables['phi2_by_mode'][:,:ky_mode_lim, :kx_mode_lim].data
            phi2t  = rtg.variables['phi2'][:].data
            gammaf = 0.5*np.vstack([np.polyfit(t[idx-4:idx-1], np.log(phi2[idx-4:idx-1, :, i]), 1)[0] for i in range(len(kx))]).T
    
        #gammac[omegac < 0] = 0
        idx_max  = np.where(gammac == np.max(gammac))
        # If idx_max are repeated, i.e, multiple maximas pick the first one.
        # Problematic also, correct it!
        if len(idx_max[0]) > 1 and len(idx_max[1]) > 1 :
            idx_max   = (np.array([idx_max[0][0]]), np.array([idx_max[1][0]]))
        elif len(idx_max[0]) > 1:
            idx_max   = (np.array([idx_max[0][0]]), idx_max[1])
        elif len(idx_max[1]) > 1:
            idx_max   = (idx_max[0], np.array([idx_max[1][0]]))
        else:
            print("idx", idx_max)
    
    
        gamma_max = gammac[idx_max]
        omega_max = omegac[idx_max]
        ky_max    = ky[idx_max[0]]
        kx_max    = theta0[idx_max]
    
    else: # simulation converged successfully
        if (t[-1]-t[0]) < float(nstep*delt): # simulation converged
            gammac = rtg.variables['omega_average'][-1, :ky_mode_lim, :kx_mode_lim, 1].data
            omegac = rtg.variables['omega_average'][-1, :ky_mode_lim, :kx_mode_lim, 0].data
        else:
            gammac  = np.mean(rtg.variables['omega_average'][-4:,:ky_mode_lim, :kx_mode_lim, 1].data, axis=0)
            omegac  = np.mean(rtg.variables['omega_average'][-4:,:ky_mode_lim, :kx_mode_lim, 0].data, axis=0)
            gammac0 = np.mean(rtg.variables['omega_average'][-3:,:ky_mode_lim, :kx_mode_lim, 1].data, axis=0)
            if abs(np.max(gammac) - np.max(gammac0)) > 2E-3:
                phi2   = rtg.variables['phi2_by_mode'][-3:, :ky_mode_lim, :kx_mode_lim].data
                gammaf = 0.5*np.vstack([np.polyfit(t[-3:], np.log(phi2[:, :, i]), 1)[0] for i in range(len(kx))]).T
                gammac = gammaf
    
    
        #gammac[omegac < 0] = 0
        idx_max  = np.where(gammac == np.max(gammac))
    
        if len(idx_max[0]) > 1 and len(idx_max[1]) > 1 :
            idx_max     = (np.array([idx_max[0][0]]), np.array([idx_max[1][0]]))
        elif len(idx_max[0]) > 1:
            idx_max     = (np.array([idx_max[0][0]]), idx_max[1])
        elif len(idx_max[1]) > 1:
            idx_max     = (idx_max[0], np.array([idx_max[1][0]]))
        else:
            print("idx", idx_max)
    
        phi2t  = rtg.variables['phi2'][:].data
    
        gamma_max = gammac[idx_max]
        omega_max = omegac[idx_max]
        ky_max    = ky[idx_max[0]]
        kx_max    = theta0[idx_max]
    
    
    
    
    fname = f"{gs2_scan_dirname}/gs2_input_beta.out.nc"
    #fname = f"{gs2_scan_dirname}/gs2_input.out.nc"
    rtg1  = ds(fname, 'r')
    ky1   = rtg1.variables['ky'][:ky_mode_lim].data
    kx1   = rtg1.variables['kx'][:kx_mode_lim].data
    theta01 = rtg1.variables['theta0'][:ky_mode_lim, :kx_mode_lim].data
    
    beta1 = rtg1.variables['beta'][:].data
    
    #pdb.set_trace()   
    idxs1   = np.where(np.isnan(rtg1.variables['phi2'][:].data) == True)[0]
    
    t1      = rtg1.variables['t'][:].data 
    
    #fnamei1 = f"{gs2_scan_dirname}/gs2_input_beta.in"
    fnamei1 = f"{gs2_scan_dirname}/gs2_input.in"
    with open(fnamei1, 'r') as f1:
        list01 = f1.readlines()
    
    # ERROR_PRONE DO NOT CHANGE the number of lines in the gs2_template file
    # This may happen if we aren't cautious
    delt1  = eval(list01[98].split('=')[1])
    nstep1 = eval(list01[99].split('=')[1])
    
    if len(idxs1) > 0 and  np.min(idxs1) <= 1: # Everything is nan
        print("first if statement ", idx0, idx1, idx2, "idxs", idxs1, rtg1.variables['phi2'][:].data)
        gamma_max1 = np.array([np.nan])
        omega_max1 = np.array([np.nan])
        omega_max1 = np.array([np.nan])
        ky_max1 = np.array([np.nan])
        kx_max1 = np.array([np.nan])
        print("everything is nan! Run again!")
    elif len(idxs1) > 0 and np.min(idxs1) > 1: # nan occurs because the eigenfunctions > 1e300
        print("second if statement ")
        if np.min(idxs1) <=4: # super large growth rate or small nsteps
            idx1    = np.min(idxs1)
            gammac1 = np.mean(rtg1.variables['omega_average'][0:idx, :ky_mode_lim, :kx_mode_lim, 1].data, axis=0)
            omegac1 = np.mean(rtg1.variables['omega_average'][0:idx, :ky_mode_lim, :kx_mode_lim, 0].data, axis=0)
            phi21   = rtg1.variables['phi2_by_mode'][:,:ky_mode_lim, :kx_mode_lim].data
            phi2t1  = rtg1.variables['phi2'][:].data
            #gammaf1 = 0.5*np.vstack([np.polyfit(t1[0:idx], np.log(phi21[0:idx, :, i]), 1)[0] for i in range(len(kx1))]).T
        else:
            # Minimum idx at which we encounter a Nan
            idx1    = np.min(idxs1)
            gammac1 = np.mean(rtg1.variables['omega_average'][idx1-4:idx1-1, :ky_mode_lim, :kx_mode_lim, 1].data, axis=0)
            omegac1 = np.mean(rtg1.variables['omega_average'][idx1-4:idx1-1, :ky_mode_lim, :kx_mode_lim, 0].data, axis=0)
            phi21   = rtg1.variables['phi2_by_mode'][:,:ky_mode_lim, :kx_mode_lim].data
            phi2t1  = rtg1.variables['phi2'][:].data
            #gammaf1 = 0.5*np.vstack([np.polyfit(t[idx1-4:idx1-1], np.log(phi21[idx1-4:idx1-1, :, i]), 1)[0] for i in range(len(kx1))]).T
    
        #gammac1[omegac1 < 0] = 0
        idx_max1  = np.where(gammac1 == np.max(gammac1))
        # If idx_max are repeated, i.e, multiple maximas pick the first one.
        # Problematic also, correct it!
        if len(idx_max1[0]) > 1 and len(idx_max1[1]) > 1 :
            idx_max1   = (np.array([idx_max1[0][0]]), np.array([idx_max1[1][0]]))
        elif len(idx_max1[0]) > 1:
            idx_max1   = (np.array([idx_max1[0][0]]), idx_max1[1])
        elif len(idx_max1[1]) > 1:
            idx_max1   = (idx_max1[0], np.array([idx_max1[1][0]]))
        else:
            print("idx", idx_max1)
    
        rtg1.close()
    
        gamma_max1 = gammac1[idx_max1]
        omega_max1 = omegac1[idx_max1]
        ky_max    = ky1[idx_max1[0]]
        kx_max    = theta01[idx_max1]
    
    else: # simulation converged successfully
        if (t1[-1]-t1[0]) < float(nstep1*delt1): # simulation converged
            gammac1 = rtg1.variables['omega_average'][-1, :ky_mode_lim, :kx_mode_lim, 1].data
            omegac1 = rtg1.variables['omega_average'][-1, :ky_mode_lim, :kx_mode_lim, 0].data
        else:
            gammac1  = np.mean(rtg1.variables['omega_average'][-4:,:ky_mode_lim, :kx_mode_lim, 1].data, axis=0)
            omegac1  = np.mean(rtg1.variables['omega_average'][-4:,:ky_mode_lim, :kx_mode_lim, 0].data, axis=0)
            gammac01 = np.mean(rtg1.variables['omega_average'][-3:,:ky_mode_lim, :kx_mode_lim, 1].data, axis=0)
            #if abs(np.max(gammac) - np.max(gammac1)) > 2E-3:
            #    phi2   = rtg.variables['phi2_by_mode'][-3:, :ky_mode_lim, :kx_mode_lim].data
            #    gammaf1 = 0.5*np.vstack([np.polyfit(t[-3:], np.log(phi2[:, :, i]), 1)[0] for i in range(len(kx))]).T
            #    gammac1 = gammaf1
        
        #gammac1[omegac1 < 0] = 0
        idx_max1  = np.where(gammac1 == np.max(gammac1))
    
        if len(idx_max1[0]) > 1 and len(idx_max1[1]) > 1 :
            idx_max1     = (np.array([idx_max1[0][0]]), np.array([idx_max1[1][0]]))
        elif len(idx_max1[0]) > 1:
            idx_max1     = (np.array([idx_max1[0][0]]), idx_max1[1])
        elif len(idx_max1[1]) > 1:
            idx_max1     = (idx_max1[0], np.array([idx_max1[1][0]]))
        else:
            print("idx", idx_max1)
    
        phi2t1  = rtg1.variables['phi2'][:].data
    
        rtg1.close()
    
        gamma_max1 = gammac1[idx_max1]
        omega_max1 = omegac1[idx_max1]
        ky_max1    = ky1[idx_max1[0]]
        kx_max1    = theta01[idx_max1]
    
    es_heat_flux = rtg.variables['es_heat_flux_by_mode'][-1, :, :ky_mode_lim, :kx_mode_lim].data
    apar_heat_flux = rtg.variables['apar_heat_flux_by_mode'][-1, :, :ky_mode_lim, :kx_mode_lim].data
    bpar_heat_flux = rtg.variables['bpar_heat_flux_by_mode'][-1, :, :ky_mode_lim, :kx_mode_lim].data

    es_part_flux = rtg.variables['es_part_flux_by_mode'][-1, 0, :ky_mode_lim, :kx_mode_lim].data
    apar_part_flux = rtg.variables['apar_part_flux_by_mode'][-1, 0, :ky_mode_lim, :kx_mode_lim].data
    bpar_part_flux = rtg.variables['bpar_part_flux_by_mode'][-1, 0, :ky_mode_lim, :kx_mode_lim].data

    total_heat_flux = es_heat_flux + apar_heat_flux + bpar_heat_flux
    total_part_flux = es_part_flux + apar_part_flux + bpar_part_flux

    D_over_chi_total = total_part_flux/total_heat_flux *  (rtg.variables['tprim'][0].data/rtg.variables['fprim'][0].data)

    total_heat_flux_electrons = total_heat_flux[1]
    total_heat_flux_electrons[total_heat_flux_electrons < 0] = 1e-304
    abs_chii_over_chie = np.abs(total_heat_flux[0]/total_heat_flux[1])

    cutoff = 15
    # If abs_chii_over_chie >> 1, we have ITG.
    # In the mask, True => KBM, False => ITG. Hence, Is ITG = False
    mask_chii_over_chie = 1*[abs_chii_over_chie  < cutoff] + 0*[abs_chii_over_chie > cutoff]

    dgamma_dbeta = (gammac1.T-gammac.T)/(beta1-beta)
    mask_dgamma_dbeta = 0*[dgamma_dbeta < 0] + 1*[dgamma_dbeta >= 0]
    mask_final = mask_dgamma_dbeta * mask_chii_over_chie[0].T

    rtg.close()
    gamma_max = np.max(mask_final[0]*gammac.T)
    #gamma_max = np.max(gammac.T)

    #pdb.set_trace()
    #from matplotlib.colors import BoundaryNorm
    #a = (gammac1.T-gammac.T)/(beta1-beta)
    ## define the colormap
    #cmap = plt.get_cmap('PuOr')
    #
    ## extract all colors from the .jet map
    #cmaplist = [cmap(i) for i in range(cmap.N)]
    ## create the new map
    #cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
    #
    ## define the bins and normalize and forcing 0 to be part of the colorbar!
    #bounds = np.arange(np.min(a),np.max(a), 2.0)
    #idx=np.searchsorted(bounds,0)
    #bounds=np.insert(bounds,idx,0)
    #norm = BoundaryNorm(bounds, cmap.N)
    #
    #
    ## colormap types
    #
    ##plt.figure()
    #plt.contourf(ky1, kx1, a, interpolation='None', norm=norm, cmap=cmap)
    ##plt.imshow(a,interpolation='none',norm=norm,cmap=cmap)
    #plt.colorbar()
    #plt.xlabel(r"$k_y$", fontsize=16) 
    #plt.ylabel(r"$k_x$", fontsize=16) 
    #plt.xticks(fontsize=16)
    #plt.yticks(fontsize=16)
    #plt.title(r"$\gamma$", fontsize=20)
    #plt.tight_layout()
    #plt.savefig("gamma_contours.png", dpi=300)
    
    
    #fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    #
    #def customize_subplot(ax, x, y, z, title, xlabel, ylabel):
    #    cs = ax.contourf(x, y, z, levels=20)
    #    ax.set_title(title, fontsize=16)
    #    ax.set_xlabel(xlabel, fontsize=16)
    #    ax.set_ylabel(ylabel, fontsize=16)
    #    ax.tick_params(axis='both', which='major', labelsize=16)
    #    plt.colorbar(cs, ax=ax)
    #    ax.figure.tight_layout()
    #
    ## Customize each subplot
    #customize_subplot(axs[0, 0], ky, kx, gammac.T, r'$\gamma$', r'$k_x$', r'$k_y$')
    #customize_subplot(axs[0, 1], ky, kx, omegac.T, r'$\omega$', r'$k_x$', r'$k_y$')
    #customize_subplot(axs[1, 0], ky, kx, gammac.T, r'$\gamma$', r'$k_x$', r'$k_y$')
    #customize_subplot(axs[1, 1], ky, kx, omegac.T, r'$\omega$', r'$k_x$', r'$k_y$')
    #
    ## Adjust layout
    #plt.tight_layout()
    
    #plt.contourf(ky, kx, gammac.T, levels=35)
    #plt.xlabel(r"$k_y$", fontsize=16) 
    #plt.ylabel(r"$k_x$", fontsize=16) 
    #plt.xticks(fontsize=16)
    #plt.yticks(fontsize=16)
    #plt.title(r"$\gamma$", fontsize=20)
    #plt.tight_layout()
    #plt.colorbar()
    #plt.savefig("gamma_contours.png", dpi=300)
    
    #plt.figure()
    #plt.contourf(ky, kx, omegac.T, levels=35);
    #plt.xlabel(r"$k_y$", fontsize=16) 
    #plt.ylabel(r"$k_x$", fontsize=16) 
    #plt.xticks(fontsize=16)
    #plt.yticks(fontsize=16)
    #plt.title(r"$\omega$", fontsize=20)
    #plt.tight_layout()
    #plt.colorbar()
    #plt.savefig("omega_contours.png", dpi=300)
    
    
    #plt.figure()
    #plt.contourf(ky1, kx1, gammac1.T, levels=35)
    #plt.xlabel(r"$k_y$", fontsize=16) 
    #plt.ylabel(r"$k_x$", fontsize=16) 
    #plt.xticks(fontsize=16)
    #plt.yticks(fontsize=16)
    #plt.title(r"$\gamma$", fontsize=20)
    #plt.tight_layout()
    #plt.colorbar()
    #plt.savefig("gamma_contours.png", dpi=300)
    
    #plt.figure()
    #plt.contourf(ky1, kx1, omegac1.T, levels=35);
    #plt.xlabel(r"$k_y$", fontsize=16) 
    #plt.ylabel(r"$k_x$", fontsize=16) 
    #plt.xticks(fontsize=16)
    #plt.yticks(fontsize=16)
    #plt.title(r"$\omega$", fontsize=20)
    #plt.tight_layout()
    #plt.colorbar()
    #plt.savefig("omega_contours.png", dpi=300)
    
    #plt.show()
    #except:
    #    print("file not present!")
    #    gamma_max, omega_max, ky_max, kx_max, idx2, idx3 = 0, 0, 0, 0, 0, 0    
    
    return gamma_max, omega_max, ky_max, kx_max, idx2, idx3





####################################################################
###############---------PLOTTING PART---------------################
####################################################################

Ns = int(4)
s_min = 0.2
s_max = 0.95
# normalized flux surface label of surfaces to scan
s_grid = np.linspace(s_min, s_max, Ns)
#s_grid = np.array([0.2, 0.45, 0.7, 0.95]


Nalpha = int(8)
alpha_min = 0
alpha_max = np.pi
# normalized field line label to scan
alpha_grid = np.linspace(alpha_min, alpha_max, Nalpha)

#Number of shat values to scan
N_sfac = int(10)

#Number of pressure gradient values to scan
N_pfac = int(10)

shat_grid = np.linspace(-1, 3, N_sfac)

# number os dP_ds grid points
dP_ds_grid = np.linspace(0.1, -2, N_pfac)

data = np.zeros((Nalpha, N_sfac, N_pfac))

#idx42 = int(2)

for idx42 in np.arange(0,4):
    for j in np.array([0, 1, 2, 3, 4, 5, 6, 7]):
        for k in np.arange(0,N_sfac):
            for l in np.arange(0,N_pfac):
                data[j, k, l], omega, kymax, kxmax, idx2, idx3 = max_growth_rate(idx42, j, k, l, 5, 8)
                #print(data[j,k,l], omega, kymax, kxmax, idx2, idx3)
        #plt.contourf(-1*dP_ds_grid, shat_grid, data[j])
        ##plt.imshow(data[j])
        #plt.show()
    
    fig = plt.figure()
    plt.contourf(-1*dP_ds_grid, shat_grid, np.max(data, axis=0), levels=10, cmap="hot")
    #plt.xlabel(r"$\alpha_{\mathrm{MHD}}$", fontsize=22)
    #plt.ylabel(r"$\hat{s}$", fontsize=22, labelpad=-8)
    #plt.title(r"$$", fontsize=22)
    plt.xlabel(r"pressure gradient", fontsize=32)
    plt.ylabel(r"shear", fontsize=32, labelpad=-8)
    plt.yticks(fontsize=26)
    plt.xticks(fontsize=26)
    cbar = plt.colorbar()
    cbar.set_ticks(cbar.get_ticks()[::int(len(cbar.get_ticks())/6)])
    cbar.ax.tick_params(labelsize=24)
    
    # Get current axis
    ax = plt.gca()
    
    def one_decimal_format(x, pos):
        return f'{x:.1f}'
    
    ax.xaxis.set_major_formatter(FuncFormatter(one_decimal_format))
    
    # Set custom x-ticks if needed
    xmin, xmax = ax.get_xlim()
    step_size = 0.3  # Adjust this value as per your data
    plt.xticks(np.arange(xmin, xmax, step_size), fontsize=20)
    shat0_array  = np.array([-0.03617386, -0.06952115, -0.10695348, -0.16546453])
    dP_ds0_array = np.array([-0.22540299, -0.22540299, -0.22540299, -0.22540299])
    plt.plot(-dP_ds0_array[idx42], shat0_array[idx42], marker='X', ms=16, mew=1, markerfacecolor='lime', markeredgecolor='lime')                               
    plt.tight_layout()
    ##plt.savefig(f"s-alpha-w7x-rho_sp{s_grid*100}.eps")
    ##plt.savefig(f"s-alpha-w7x-rho_sp75.eps")
    #if idx42 == 0:
    #    plt.savefig(f"KBM-s-alpha-w7x-rho_sp15.png", dpi=300)
    #elif idx42 == 1:
    #    plt.savefig(f"KBM-s-alpha-w7x-rho_sp40.png", dpi=300)
    #elif idx42 == 2:
    #    plt.savefig(f"KBM-s-alpha-w7x-rho_sp65.png", dpi=300)
    #else:
    #    plt.savefig(f"KBM-s-alpha-w7x-rho_sp90.png", dpi=300)
    if idx42 == 0:
        plt.savefig(f"KBM-s-alpha-w7x-rho_sp15.pdf", dpi=400)
    elif idx42 == 1:
        plt.savefig(f"KBM-s-alpha-w7x-rho_sp40.pdf", dpi=400)
    elif idx42 == 2:
        plt.savefig(f"KBM-s-alpha-w7x-rho_sp65.pdf", dpi=400)
    else:
        plt.savefig(f"KBM-s-alpha-w7x-rho_sp90.pdf", dpi=400)
#plt.show()





