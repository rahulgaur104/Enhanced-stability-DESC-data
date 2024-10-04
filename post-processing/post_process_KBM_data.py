#!/usr/bin/env python3
"""
This script plots the KBM growth rate vs rho for the various omnigenous equilibria
"""
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
import matplotlib as mpl


keyword_arr = ["OT", "OH", "OP"]

for keyword in keyword_arr:

    growth_rate_initial = np.load(
        os.path.dirname(os.getcwd())
        + f"/analysis/GS2_KBM_analysis/KBM_growth_rate_{keyword}_initial.npy"
    )
    growth_rate_optimized = np.load(
        os.path.dirname(os.getcwd())
        + f"/analysis/GS2_KBM_analysis/KBM_growth_rate_{keyword}_optimized.npy"
    )

    # Square root because the surfaces are chosen in s = normalized toroidal flux
    # but DESC rho = sqrt(s)
    rho_array = np.sqrt(np.linspace(0.05, 0.95, 8))

    plt.figure(figsize=(6, 5))
    plt.plot(rho_array, growth_rate_initial, "-or", ms=2.5, linewidth=2.5)
    plt.plot(rho_array, growth_rate_optimized, "-og", ms=2.5, linewidth=2.5)

    plt.xlabel(r"$\rho$", fontsize=26)
    #plt.ylabel(r"$\gamma_{\mathrm{KBM}}a_{\mathrm{N}}/v_{\mathrm{th,i}}$", fontsize=22)
    plt.ylabel(r"$\gamma a_{\mathrm{N}}/v_{\mathrm{th,i}}$", fontsize=26)
    plt.xticks(np.linspace(0.2, 1, 5), fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(["initial", "optimized"], fontsize=20)
    plt.tight_layout()
    #plt.savefig(f"KBM_stability/{keyword}_KBM_growth_rate_comparison.png", dpi=400)
    plt.savefig(f"KBM_stability/{keyword}_KBM_growth_rate_comparison.pdf", dpi=500)
    plt.show()
