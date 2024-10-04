#!/usr/bin/env python3
"""
This script performs an s-alpha analysis of tokamak and stellarator equiliria
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
import sys

import multiprocessing as mp
import numpy as np
import pdb

import booz_xform as bxform
from netCDF4 import Dataset as ds
from matplotlib import pyplot as plt

#from gx_geo_vmec import *
from gx_geo_vmec_asym import *
from utils import *

PARENT_DIR_NAME = os.getcwd()
VMEC_FNAME = sys.argv[1]
AXISYM = int(sys.argv[2])
EIKFILE = sys.argv[3]


def check_ball(geo_obj, theta, sfacidx, pfacidx, alphaidx, theta_0):
    """
    Ideal-ballooning Newcomb criterion.

    This function calculates the ideal ballooning growth rate for a general
    3D equilibrium

    Params
    -----------------------------------------------------
    sfac: float
    Hegna-Nakajima prefactor for global magnetic shear
    pfac: float
    Hegna-Nakajima prefactor for pressure gradient
    rhoc: float
    Radius at which you want the growth rate
    alpha: float
    Fieldline at which you need the growth rate
    theta_0: float
    ballooning parameter at which you need the growth rate

    Returns
    ------------------------------------------------------
    The maximum ideal-ballooning growth rate
    """

    bmag = geo_obj.bmag[0, alphaidx, :, 0, 0]
    gradpar = abs(geo_obj.gradpar_theta_b[0, alphaidx, :, 0, 0])
    cvdrift = geo_obj.cvdrift[0, alphaidx, :, sfacidx, pfacidx]
    gbdrift = geo_obj.gbdrift[0, alphaidx, :, sfacidx, pfacidx]
    cvdrift0 = geo_obj.cvdrift0[0, alphaidx, :, sfacidx, 0]
    gds2 = geo_obj.gds2[0, alphaidx, :, sfacidx, pfacidx]
    gds21 = geo_obj.gds21[0, alphaidx, :, sfacidx, pfacidx]
    gds22 = geo_obj.gds22[0, alphaidx, :, sfacidx, 0]

    cvdrift_fth = cvdrift + theta_0 * cvdrift0
    gds2_fth = gds2 + 2 * theta_0 * gds21 + theta_0**2 * gds22

    dPdrho = -1 * np.mean(0.5 * (cvdrift - gbdrift) * bmag**2)

    #isunstable = check_ball_full(dPdrho, theta, bmag, gradpar, cvdrift_fth, gds2_fth)
    #isunstable = check_ball_full2(dPdrho, theta, bmag, gradpar, cvdrift_fth, gds2_fth)
    check_ball_full2(dPdrho, theta, bmag, gradpar, cvdrift_fth, gds2_fth)

    return isunstable


def gamma_ball(geo_obj, theta, sfacidx, pfacidx, alphaidx, theta_0):
    """
    Ideal-ballooning growth rate.

    This function calculates the ideal ballooning growth rate for a general
    3D equilibrium

    Params
    -----------------------------------------------------
    sfac: float
    Hegna-Nakajima prefactor for global magnetic shear
    pfac: float
    Hegna-Nakajima prefactor for pressure gradient
    rhoc: float
    Radius at which you want the growth rate
    alpha: float
    Fieldline at which you need the growth rate
    theta_0: float
    ballooning parameter at which you need the growth rate

    Returns
    ------------------------------------------------------
    The maximum ideal-ballooning growth rate
    """

    bmag = geo_obj.bmag[0, alphaidx, :, 0, 0]
    gradpar = abs(geo_obj.gradpar_theta_b[0, alphaidx, :, 0, 0])
    cvdrift = geo_obj.cvdrift[0, alphaidx, :, sfacidx, pfacidx]
    gbdrift = geo_obj.gbdrift[0, alphaidx, :, sfacidx, pfacidx]
    cvdrift0 = geo_obj.cvdrift0[0, alphaidx, :, sfacidx, 0]
    gds2 = geo_obj.gds2[0, alphaidx, :, sfacidx, pfacidx]
    gds21 = geo_obj.gds21[0, alphaidx, :, sfacidx, pfacidx]
    gds22 = geo_obj.gds22[0, alphaidx, :, sfacidx, 0]

    cvdrift_fth = cvdrift + theta_0 * cvdrift0
    gds2_fth = gds2 + 2 * theta_0 * gds21 + theta_0**2 * gds22

    dPdrho = -1 * np.mean(0.5 * (cvdrift - gbdrift) * bmag**2)

    gamma = gamma_ball_full(dPdrho, theta, bmag, gradpar, cvdrift_fth, gds2_fth)

    return gamma


###################################################################################


nc_obj = ds(VMEC_FNAME, "r")

mpol = nc_obj.variables["mpol"][:].data
ntor = nc_obj.variables["ntor"][:].data

booz_obj = bxform.Booz_xform()
booz_obj.read_wout(VMEC_FNAME)
booz_obj.mboz = int(4 * mpol)
booz_obj.nboz = int(4 * ntor)
booz_obj.run()

# check if your tokamak or stellarator is up-down symmetric
STELLASYM = np.shape(booz_obj.rmns) != (0, 0)

# number of flux surface
LEN_1 = 4
RHO_MIN = 0.15
RHO_MAX = 0.9
RHO_ARR = np.linspace(RHO_MIN, RHO_MAX, LEN_1)
# number of field lines
LEN_2 = 8
ALPHA_MIN = 0.0
ALPHA_MAX = np.pi
ALPHA_ARR = np.linspace(ALPHA_MIN, ALPHA_MAX, LEN_2)

# OP scan parameters
## number of theta_0 values
#LEN_3 = 17
#THETA_0_MIN =-1.0*np.pi
#THETA_0_MAX = 1.0*np.pi
#THETA_0_ARR = np.linspace(THETA_0_MIN, THETA_0_MAX, LEN_3)


# OT scan parameters
# number of theta_0 values
LEN_3 = 11
THETA_0_MIN =-0.75*np.pi
THETA_0_MAX = 0.75*np.pi
THETA_0_ARR = np.linspace(THETA_0_MIN, THETA_0_MAX, LEN_3)

## OP scan parameters
## number of shat points
#LEN_4 = int(32)
#shat_grid = np.linspace(-1, 3, LEN_4)
## number os dP_ds grid points
#LEN_5 = int(16)
#dP_ds_grid = np.linspace(0.1, -0.3, LEN_5)


# OT scan parameters
# number of shat points
LEN_4 = int(24)
shat_grid = np.linspace(-1, 3, LEN_4)
# number os dP_ds grid points
LEN_5 = int(32)
dP_ds_grid = np.linspace(-0.1, 2.0, LEN_5)


# to keep the marginal stability data
ball_scan_arr1 = np.zeros((LEN_1, LEN_2, LEN_3, LEN_4, LEN_5, 2))

# to keep the growth rate data
ball_scan_arr2 = np.zeros((LEN_1, LEN_2, LEN_3, LEN_4, LEN_5))

## OP scan parameters
## We define all the variables fed to the geometry routine
#N_T = 196
#N_POL = 3
#N_THETA = 2 * N_T * N_POL + 1
## This is Boozer theta
#THETA = np.linspace(-N_POL * np.pi, N_POL * np.pi, N_THETA)

## OT scan parameters
# We define all the variables fed to the geometry routine
N_T = 128
N_POL = 4
N_THETA = 2 * N_T * N_POL + 1
# This is Boozer theta
THETA = np.linspace(-N_POL * np.pi, N_POL * np.pi, N_THETA)

# First, we call vmec_fieldlines just to obtain the nominal
# shat and dP_ds values. To do this we call the vmec_fieldlines function
geo_coeffs0 = vmec_fieldlines(
    VMEC_FNAME,
    booz_obj,
    RHO_ARR,
    ALPHA_ARR[0],
    theta1d=np.linspace(-np.pi, np.pi, 11),
    sfac=np.array([1]),
    pfac=np.array([1.0]),
    axisym=AXISYM,
    stellasym=STELLASYM,
)

mu_0 = 4 * np.pi * (1.0e-7)
shat0 = geo_coeffs0.shat
dP_ds0 = mu_0 * geo_coeffs0.d_pressure_d_s

#pdb.set_trace()

# Divide the shat grid and dP_ds_grid with the shat0 and dP_ds0 values
# to obtain the sfac_grid and pfac_grid arrays
sfac_grid_full = shat_grid[None, :] / shat0[:, None]
pfac_grid_full = dP_ds_grid[None, :] / dP_ds0[:, None]

for i in np.arange(1, LEN_1):
    sfac_grid = sfac_grid_full[i]
    pfac_grid = pfac_grid_full[i]
    geo_coeffs = vmec_fieldlines(
        VMEC_FNAME,
        booz_obj,
        RHO_ARR[i],
        ALPHA_ARR,
        theta1d=THETA,
        sfac=sfac_grid,
        pfac=pfac_grid,
        axisym=AXISYM,
        stellasym=STELLASYM,
    )
    legends = []
    #plt.figure(figsize=(8, 6))
    #fig, (ax0, ax1) = plt.subplots(1, 2)
    for j in range(LEN_2):
        for k in range(LEN_3):
            # Setting the number of threads to 1. We don't want multithreading.

            NOP = int(16)

            pool = mp.Pool(processes=NOP)
            results1 = np.array(
                [
                    [
                        pool.apply_async(
                            gamma_ball,
                            args=(geo_coeffs, THETA, l, m, j, THETA_0_ARR[k]),
                        )
                        for l in range(LEN_4)
                    ]
                    for m in range(LEN_5)
                ]
            )
            for m0 in range(LEN_5):
                #ball_scan_arr1[i, j, k, :, m0, :] = np.array(
                #    [results1[m0, l0].get() for l0 in range(LEN_4)]
                #)
                ball_scan_arr2[i, j, k, :, m0] = np.array(
                    [results1[m0, l0].get() for l0 in range(LEN_4)]
                )

            ## results2 = np.array(
            ##    [
            ##        [
            ##            pool.apply_async(gamma_ball, args=(geo_coeffs, THETA, l, m, j, THETA_0_ARR[k]))
            ##            for l in range(LEN_4)
            ##        ]
            ##        for m in range(LEN_5)
            ##    ]
            ## )
            ## for i in range(LEN_5):
            ##    ball_scan_arr2[:, i] = np.array([results2[i, j].get() for j in range(LEN_4)])

            ## Write a code to plot ball_scan_arr1 with respect to sfac_grid and pfac_grid
            ## and save it as a png file
            ## plt.figure()
            ## Plot a contour of ball_scan_arr1 colored white at 0.0 with a line width of 3
            ## plt.contourf(pfac_grid, sfac_grid, ball_scan_arr2, cmap="hot")
            ## plt.colorbar()
            ## Set curve transparency alpha in plt.contour below
            ##pdb.set_trace()
            #cs0 = ax0.contour(
            #    -1 * dP_ds_grid,
            #    shat_grid,
            #    ball_scan_arr1[i, j, k, :, :, 0],
            #    levels=[0.0],
            #    colors=[COLORS[j]],
            #    alpha=(k + 0.5) / (LEN_3 + 0.5),
            #    antialiased=True,
            #    linewidths=1.5,
            #)
 
            ##cs1 = ax1.contour(
            ##    -1 * dP_ds_grid,
            ##    shat_grid,
            ##    ball_scan_arr1[i, j, k, :, :, 1],
            ##    levels=[0.0],
            ##    colors=[COLORS[j]],
            ##    alpha=(k + 0.5) / (LEN_3 + 0.5),
            ##    antialiased=True,
            ##    linewidths=1.5,
            ##)
           
            #cs1 = ax1.contour(
            #    -1 * dP_ds_grid,
            #    shat_grid,
            #    ball_scan_arr1[i, j, k, :, :, 1],
            #    levels = 50,
            #    cmap=COLORMAPS[j],
            #)
            ###pdb.set_trace()
            ##p = cs0.collections[0].get_paths()[0]
            ##v = p.vertices
            ##norm0 = np.linalg.norm(v - np.array([shat0[i], dP_ds0[i]]), axis=1)
            ##idx = np.where(norm0 == np.min(norm0))[0]   
            ##print(f"distnace from marginality {j} {k} = {np.linalg.norm(norm0[idx])}")
            ##    

            ## plt.clabel(cs,levels=[0.0], fmt = legends[-1])
            ## plt.clabel(cs,levels=[0.0])
            ## cs.collections[0].set_label(legends[-1])
    # plt.legend()
    # plt.plot(shat0, -1*dP_ds0, "x", color="limegreen", alpha = (j+1)/(LEN_2+1), mew=5, ms=8)

    ##pdb.set_trace()
    #fig.suptitle('Newcomb metrics')
    #plt.plot(
    #    -1 * dP_ds0[i],
    #    shat0[i],
    #    MARKERS_W_COLORS[i],
    #    alpha=(j + 1) / (LEN_2 + 1),
    #    mew=5,
    #    ms=8,
    #)

    np.save(f"ball_scan_data_{EIKFILE}_surf{i}.npy", np.max(ball_scan_arr2, axis=(1, 2))[i, :, :]) 

    path = f"{PARENT_DIR_NAME}/s-alpha_plots/s-alpha-{EIKFILE}_r{RHO_ARR[i]:.2f}.png"
    #plt.savefig(f"{path}", dpi=600)
    print(f"balllooning s-alpha curve successfully saved at f{path}")
    #plt.close()
    ## plt.show()
