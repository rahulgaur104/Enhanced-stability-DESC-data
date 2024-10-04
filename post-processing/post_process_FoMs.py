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

mpl.rc("font", family="DejaVu Serif")

comparison = True

mu0 = 4 * np.pi * 1e-7

# Manually read and written from the trace files
omni_errors = {
    "OP": np.array(
        [
            [0.8354, 0.8475, 0.8797, 0.9434, 1.013],
            [0.0518, 0.0568, 0.0885, 0.1052, 0.1283],
        ]
    ),
    "OH": np.array(
        [
            [0.0864, 0.1011, 0.1204, 0.1499, 0.2227],
            [0.0433, 0.0519, 0.0691, 0.0979, 0.1655],
        ]
    ),
    "OT": np.array(
        [
            [0.0043, 0.0047, 0.0057, 0.0076, 0.0998, 0.1961],
            [0.0025, 0.0033, 0.0058, 0.0106, 0.0693, 0.0421],
        ]
    ),
}

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
        fname_path2 = (
            os.path.dirname(os.getcwd()) + "/equilibria/OP_nfp3/field_final_new.h5"
        )
        eq0 = Equilibrium.load(f"{fname_path0}")
        eq1 = Equilibrium.load(f"{fname_path1}")
        field = Equilibrium.load(f"{fname_path2}")
    elif keyword == "OH":
        fname_path0 = (
            os.path.dirname(os.getcwd())
            + "/equilibria/OH_nfp5/eq_OH_ball5_001_initial.h5"
        )
        fname_path1 = (
            os.path.dirname(os.getcwd())
            + "/equilibria/OH_nfp5/eq_OH_ball5_001_optimized.h5"
        )
        fname_path2 = (
            os.path.dirname(os.getcwd()) + "/equilibria/OH_nfp5/field_final_new.h5"
        )
        eq0 = Equilibrium.load(f"{fname_path0}")
        eq1 = Equilibrium.load(f"{fname_path1}")
    else:
        fname_path0 = (
            os.path.dirname(os.getcwd())
            + "/equilibria/OT_nfp1/eq_OT_ball_022_initial.h5"
        )
        fname_path1 = (
            os.path.dirname(os.getcwd())
            + "/equilibria/OT_nfp1/eq_OT_ball_022_optimized.h5"
        )
        fname_path2 = (
            os.path.dirname(os.getcwd()) + "/equilibria/OT_nfp1/field_final_new.h5"
        )
        eq0 = Equilibrium.load(f"{fname_path0}")
        eq1 = Equilibrium.load(f"{fname_path1}")
        field = Equilibrium.load(f"{fname_path2}")


    if comparison:
        files_list = [fname_path0, fname_path1]
    else:
        files_list = [fname_path0]

    N_equilibria = len(files_list)

    eq_data_keys = ["a", "Psi", "R0", "<beta>_vol", "current"]

    for l in range(N_equilibria):
        eq = Equilibrium.load(files_list[l])
        eq_data = eq.compute(eq_data_keys)
        # pdb.set_trace()
        for e0 in eq_data_keys:
            if e0 == "a":
                print(e0, 1 / eq_data[e0])
            elif e0 in ["Psi", "current"]:
                print(e0, eq_data[e0][-1])
            else:
                print(e0, eq_data[e0])

        err = np.linalg.norm(omni_errors[f"{keyword}"][l])
        print(f"{keyword} error: {err}")
    print(f"{keyword} complete! \n")
