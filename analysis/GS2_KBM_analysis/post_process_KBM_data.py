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

import matplotlib as mpl
mpl.rc('font', family='DejaVu Serif')

keyword_arr = ["OT", "OH", "OP"]
#keyword_arr = ["OP"]

for keyword in keyword_arr:
    growth_rate_initial = np.load(f"KBM_growth_rate_{keyword}_initial.npy")
    growth_rate_optimized = np.load(f"KBM_growth_rate_{keyword}_optimized.npy")
    
    # rho = sqrt(s)
    rho_array = np.sqrt(np.linspace(0.05, 0.95, 8))
    
    
    plt.figure(figsize=(6, 5))
    plt.plot(rho_array, growth_rate_initial, '-or', ms=2.5, linewidth=2.5)
    plt.plot(rho_array, growth_rate_optimized, '-og', ms=2.5, linewidth=2.5)
    
    plt.xlabel(r"$\rho$", fontsize=22)
    plt.ylabel(r"$\gamma_{\mathrm{KBM}}$", fontsize=22)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(["initial", "optimized"], fontsize=18)
    plt.tight_layout()
    plt.savefig(f"{keyword}_KBM_growth_rate_comparison.png", dpi=400)
    plt.show()


