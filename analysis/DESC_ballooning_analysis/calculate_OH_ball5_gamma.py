#!/usr/bin/env python3
import os
from desc import set_device
set_device("gpu")
import numpy as np
from desc.equilibrium import Equilibrium
from desc.grid import LinearGrid, Grid
from desc.vmec import VMECIO
from desc.backend import jnp
from desc.equilibrium.coords import compute_theta_coords
import pickle
import pdb


#eq = Equilibrium.load("/scratch/gpfs/rg6256/DESC_ball_tests/OH_ball5/eq00_finite_beta_and_iota.h5")
eq = Equilibrium.load("/scratch/gpfs/rg6256/DESC_ball_tests/OH_ball5/eq_OH_ball5_001_optimized.h5")

surfaces = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.55, 0.6, 0.65, 0.67, 0.69, 0.7, 0.75, 0.8, 0.9, 0.95, 0.98, 0.99, 1.0]

M_grid = int(1.5*eq.M)
N_grid = int(1.5*eq.N)

ntor = int(4)
N0 = int(2 * M_grid * N_grid * ntor + 1)
eq_data_keys = ["iota", "a", "psi", "Psi"]

eq_coords = np.zeros((len(surfaces), 3))
eq_coords[:, 0] = np.array([surfaces])

eq_data_grid = Grid(eq_coords)

eq_data = eq.compute(eq_data_keys, grid=eq_data_grid)
iota = eq_data["iota"]

alpha_arr = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.5, 1.6, 1.8, 2.1, 2.3, 2.6, 2.9, 3.0, 3.12])

gamma_max = np.zeros((len(alpha_arr), len(surfaces)))

for i in range(len(alpha_arr)):
    alpha = alpha_arr[i]
    for j in range(len(surfaces)):
        rho = surfaces[j]
        zeta = np.linspace(-jnp.pi*ntor, jnp.pi*ntor, N0)
        sfl_grid = Grid.create_meshgrid([rho, alpha, zeta], coordinates="raz", period=(np.inf, 2*np.pi, np.inf))
        ball_data0 = eq.compute(["ideal ballooning lambda"], grid=sfl_grid)["ideal ballooning lambda"]
        gamma_max[i, j] = np.max(ball_data0)

        print(f"surf number {j} done!")
    print(gamma_max[i])
gamma_max = np.max(gamma_max, axis=0)
print(gamma_max)


#np.savez("OH_ball5/gamma_max_OH_ball5_hres_correct_norm_initial_new_022_wider_zeta1_wider_zeta0", alpha = alpha_arr, rho =np.array(surfaces), lambda_max = gamma_max)
np.savez("OH_ball5/gamma_max_OH_ball5_hres_correct_norm_optimized_new_022_wider_zeta1_wider_zeta0", alpha = alpha_arr, rho =np.array(surfaces), lambda_max = gamma_max)
