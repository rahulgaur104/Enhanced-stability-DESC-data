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
)
import jax.numpy as jnp
from desc.grid import Grid, LinearGrid
from matplotlib import pyplot as plt

from matplotlib.ticker import LogLocator, ScalarFormatter

mu0 = 4 * np.pi * 1e-7

comparison = True

keyword_arr = ["OT", "OH", "OP"]

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

    len0 = int(200)
    radial_grid = LinearGrid(L=len0)
    rho = np.linspace(0, 1, len0 + 1)

    data_keys = ["p", "iota"]

    eq = Equilibrium.load(fname_path0)
    data = eq.compute(data_keys, grid=radial_grid)

    plt.plot(rho, data["p"] / 10**6, "-k", linewidth=3)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.xlabel(r"$\rho$", fontsize=26)
    plt.ylabel(r"$p (\mathrm{MPa})$", fontsize=26)
    plt.tight_layout()
    # plt.savefig(f"input_profiles/{keyword}_pressure_profile.png", dpi=400)
    plt.savefig(f"input_profiles/{keyword}_pressure_profile.pdf", dpi=400)
    plt.close()

    plt.figure()
    plt.plot(rho, data["iota"], "-k", linewidth=3)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.xlabel(r"$\rho$", fontsize=26)
    plt.ylabel(r"$\iota$", fontsize=26)
    plt.tight_layout()
    # plt.savefig(f"input_profiles/{keyword}_iota_profile.png", dpi=400)
    plt.savefig(f"input_profiles/{keyword}_iota_profile.pdf", dpi=400)
    plt.close()
