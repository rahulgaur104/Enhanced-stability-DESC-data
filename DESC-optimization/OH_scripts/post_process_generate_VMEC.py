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



file_name1  = "eq_OH_ball5_001_optimized"
fname_path1 = f"{file_name1}.h5"

eq1 = Equilibrium.load(f"{fname_path1}")

VMECIO.save(eq1, f"wout_{file_name1}.nc", surfs=512)



