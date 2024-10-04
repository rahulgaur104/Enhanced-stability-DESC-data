#!/usr/bin/env python3

import pdb
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker

from desc.backend import jax
from desc.equilibrium import Equilibrium
from desc.grid import LinearGrid
from desc.vmec import VMECIO

from scipy.interpolate import griddata
import subprocess as spr

from desc.plotting import *


# keyword_arr = ["OT", "OH", "OP"]
keyword_arr = ["OP"]

for i, keyword in enumerate(keyword_arr):

    if keyword == "OP":
        fname_path0 = (
            os.path.dirname(os.getcwd())
            + "/equilibria/OP_nfp3/eq_OP_ball3_033_initial.h5"
        )
        fname_path1 = (
            os.path.dirname(os.getcwd())
            + "/equilibria/OP_nfp3/eq_OP_ball3_033_optimized.h5"
        )
        eq0 = Equilibrium.load(f"{fname_path0}")
        eq1 = Equilibrium.load(f"{fname_path1}")
    elif keyword == "OH":
        fname_path0 = (
            os.path.dirname(os.getcwd())
            + "/equilibria/OH_nfp5/eq_OH_ball5_001_initial.h5"
        )
        fname_path1 = (
            os.path.dirname(os.getcwd())
            + "/equilibria/OH_nfp5/eq_OH_ball5_001_optimized.h5"
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
        eq0 = Equilibrium.load(f"{fname_path0}")
        eq1 = Equilibrium.load(f"{fname_path1}")

    spr.call([f"cp -r {fname_path0} ./"], shell=True)
    spr.call([f"cp -r {fname_path1} ./"], shell=True)

    eq0 = Equilibrium.load(f"{fname_path0}")
    eq1 = Equilibrium.load(f"{fname_path1}")

    VMECIO.save(eq0, f"wout_{file_name0}.nc", surfs=512)
    jax.clear_caches()
    VMECIO.save(eq1, f"wout_{file_name1}.nc", surfs=512)
