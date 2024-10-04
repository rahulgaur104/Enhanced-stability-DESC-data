#!/usr/bin/env python3

from desc import set_device

import os
import pdb
import sys
import numpy as np
from desc.equilibrium import Equilibrium, EquilibriaFamily
from desc.equilibrium.coords import compute_theta_coords
from desc.objectives import (
    ForceBalance,
    ObjectiveFunction,
    MercierStability,
    BallooningStability,
)
import jax.numpy as jnp
from desc.grid import Grid, LinearGrid
from desc.continuation import solve_continuation_automatic
from matplotlib import pyplot as plt

from matplotlib.ticker import LogLocator, ScalarFormatter

import matplotlib as mpl

mpl.rc("font", family="DejaVu Serif")


comparison = True

# keyword_arr = ["OT", "OH", "OP"]
keyword_arr = ["OH", "OP", "OT"]

for keyword in keyword_arr:
    if keyword == "OP":
        fname_path0 = (
            os.path.dirname(os.getcwd())
            + "/equilibria/OP_nfp3/eq_OP_ball3_033_initial.h5"
        )
        fname_path1 = (
            os.path.dirname(os.getcwd())
            + "/equilibria/OP_nfp3/eq_OP_ball3_033_optimized.h5"
        )
    elif keyword == "OH":
        fname_path0 = (
            os.path.dirname(os.getcwd())
            + "/equilibria/OH_nfp5/eq_OH_ball5_001_initial.h5"
        )
        fname_path1 = (
            os.path.dirname(os.getcwd())
            + "/equilibria/OH_nfp5/eq_OH_ball5_001_optimized.h5"
        )
    else:
        fname_path0 = (
            os.path.dirname(os.getcwd())
            + "/equilibria/OT_nfp1/eq_OT_ball_022_initial.h5"
        )
        fname_path1 = (
            os.path.dirname(os.getcwd())
            + "/equilibria/OT_nfp1/eq_OT_ball_022_optimized.h5"
        )

    if comparison:
        files_list = [fname_path0, fname_path1]
    else:
        files_list = [fname_path0]

    N_equilibria = len(files_list)

    NL = int(100)

    for l in range(N_equilibria):
        eq = Equilibrium.load(files_list[l])
        rho_Merc = np.linspace(0, 1, NL + 1)
        grid = LinearGrid(L=NL, M=34, N=34, NFP=eq.NFP, sym=True)
        data_keys = ["D_Mercier"]
        data = eq.compute(data_keys, grid=grid)
        D_Mercier_array = grid.compress(data["D_Mercier"])
        np.save(f"Mercier_stability/D_Mercier_data_{keyword}_{l}.npy", D_Mercier_array)

    plt.figure(figsize=(6, 5))

    if comparison == True:
        rho_arr = rho_Merc
        D_Mercier_arr0 = np.load(f"Mercier_stability/D_Mercier_data_{keyword}_0.npy")
        D_Mercier_arr1 = np.load(f"Mercier_stability/D_Mercier_data_{keyword}_1.npy")
        plt.plot(rho_arr, D_Mercier_arr0, "-or", ms=2, linewidth=2)
        plt.plot(rho_arr, D_Mercier_arr1, "-og", ms=2, linewidth=2)
        plt.yscale("symlog")
        plt.xlabel(r"$\rho$", fontsize=26)
        plt.ylabel(r"$D_{\mathrm{Merc}}$", fontsize=26)
        plt.xticks(fontsize=20, fontname="DejaVu Serif")
        plt.yticks(fontsize=20, fontname="DejaVu Serif")
        plt.legend(["initial", "optimized"], fontsize=20)

        plt.tight_layout()
        plt.savefig(
            f"Mercier_stability/D_Mercier_comparison_plot_{keyword}.png", dpi=400
        )
        # plt.savefig(f"Mercier_stability/D_Mercier_comparison_plot_{keyword}.pdf", dpi=400)
    else:
        rho_arr = rho_Merc
        gamma_arr0 = np.load("D_Mercier_data_0.npy")
        plt.plot(rho_arr, gamma_arr0, "-or", ms=2, linewidth=2)
        plt.yscale("symlog")
        plt.xlabel(r"$\rho$", fontsize=26)
        plt.ylabel(r"$D_{\mathrm{Merc}}$", fontsize=26)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.tight_layout()
        plt.savefig(f"Mercier_stability/D_Mercier_single_plot_{keyword}.png", dpi=300)
        # plt.savefig(f"Mercier_stability/D_Mercier_single_plot_{keyword}.pdf", dpi=300)

    plt.close()
