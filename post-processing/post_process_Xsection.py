#!/usr/bin/env python3
"""
Plot cross sections and magnetic axis position
"""
from desc import set_device

import os
import pdb
import sys
import numpy as np
from desc.equilibrium import Equilibrium, EquilibriaFamily
from desc.equilibrium.coords import compute_theta_coords
import jax.numpy as jnp
from desc.grid import Grid, LinearGrid
from desc.continuation import solve_continuation_automatic
from matplotlib import pyplot as plt

from matplotlib.ticker import LogLocator, ScalarFormatter

from desc.plotting import *

comparison = True

keyword_arr = ["OT", "OH", "OP"]
# keyword_arr = ["OP"]

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
    color_list = ["r", "g"]
    legend_list = ["initial", "optimized"]

    for l in range(N_equilibria):
        plt.figure(figsize=(6, 5))
        eq = Equilibrium.load(files_list[l])
        fig, ax = plot_boundaries([eq], lw=2, color=color_list[l], legend=False)
        # Labeling axes
        ax.set_xlabel(r"$R$", fontsize=30, labelpad=0)
        ax.set_ylabel(r"$Z$", fontsize=30, labelpad=-6)
        ax.tick_params(axis="both", which="major", labelsize=22)  # Larger tick labels
        plt.tight_layout()
        plt.savefig(f"Xsection/{keyword}_section_plot_{legend_list[l]}.png", dpi=400)
        plt.close()
