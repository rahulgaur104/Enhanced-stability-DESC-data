#!/usr/bin/env python3

from desc import set_device

set_device("gpu")

import numpy as np
from desc.backend import jnp, jax
from desc.geometry import FourierRZToroidalSurface
from desc.equilibrium import EquilibriaFamily, Equilibrium
from desc.equilibrium.coords import compute_theta_coords
from desc.grid import LinearGrid, Grid
from desc.objectives import (
    ForceBalance,
    ObjectiveFunction,
    MercierStability,
    BallooningStability,
    FixBoundaryR,
    FixBoundaryZ,
    FixIota,
    FixPressure,
    FixPsi,
    AspectRatio,
    FixOmniBmax,
    FixOmniShift,
    FixPressure,
    GenericObjective,
    LinearObjectiveFromUser,
    Omnigenity,
    get_fixed_boundary_constraints,
)
from desc.continuation import solve_continuation_automatic
from desc.optimize import Optimizer
from desc.profiles import PowerSeriesProfile, SplineProfile
from desc.magnetic_fields import OmnigenousField
from desc.compute.utils import get_transforms
from desc.vmec_utils import ptolemy_linear_transform
from desc.vmec import VMECIO
from desc.plotting import *
from matplotlib import pyplot as plt
import pickle
import pdb
import sys


equilibrium_path = sys.argv[1]

eq = EquilibriaFamily.load(f"equilibrium_path")

print("Plasma beta is", eq.compute("<beta>_vol")["<beta>_vol"])

constraints = get_fixed_boundary_constraints(eq=eq, profiles=True)
objective = ObjectiveFunction(ForceBalance(eq), deriv_mode="looped")
eq.change_resolution(L=20, M=20, N=20)
eq, _ = eq.solve(
    objective="force",
    constraints=constraints,
    ftol=1e-3,
    xtol=1e-5,
    gtol=1e-5,
    maxiter=70,
    verbose=3,
    copy=True,
)

eq.save(fname + f"_final_hres.h5")

L0 = 200
rho = np.linspace(0, 1, L0 + 1)
grid = LinearGrid(L=L0, M=2 * eq.M, N=2 * eq.N, NFP=eq.NFP, sym=True)
data_keys = ["iota", "D_Mercier"]
data = eq.compute(data_keys, grid=grid)

print("D_Mercier=", grid.compress(data["D_Mercier"]))
print("iota=", grid.compress(data["iota"]))
print("beta=", eq.compute("<beta>_vol")["<beta>_vol"])


# We also calculate how muuch the ballooning stability is ruined!
M_grid = 16
N_grid = 16
ntor = int(4)
N0 = int(2 * M_ball * N_ball * ntor + 1)

eq_data_keys = ["iota", "a", "psi", "Psi"]

eq_coords = np.zeros((len(surfaces_ball), 3))
eq_coords[:, 0] = np.array([surfaces_ball])

eq_data_grid = Grid(eq_coords)
eq_data = eq.compute(eq_data_keys, grid=eq_data_grid)
# Now we compute theta_DESC for given theta_PEST
iota = eq_data["iota"]

alpha = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4])

Nalpha = len(alpha)
Nzeta0 = int(15)

ball_data = np.zeros((len(surfaces_ball), Nalpha, Nzeta0))
gamma_max = np.zeros(
    len(surfaces_ball),
)

for j in range(len(surfaces_ball)):
    rho = surfaces_ball[j] * np.ones((N0 * Nalpha,))
    zeta = np.linspace(-jnp.pi * ntor, jnp.pi * ntor, N0)
    theta_PEST = np.zeros((N0 * Nalpha,))
    zeta_full = np.zeros((N0 * Nalpha,))
    for i in range(Nalpha):
        theta_PEST[N0 * i : N0 * (i + 1)] = alpha[i] + iota[j] * zeta
        zeta_full[N0 * i : N0 * (i + 1)] = zeta

    theta_coords = jnp.array([rho, theta_PEST, zeta_full]).T
    desc_coords = compute_theta_coords(
        eq, theta_coords, L_lmn=eq.L_lmn, tol=1e-10, maxiter=40
    )

    sfl_grid = Grid(desc_coords, sort=False)
    ball_data0 = eq.compute(["ideal_ball_gamma2"], grid=sfl_grid, override_grid=False)[
        "ideal_ball_gamma2"
    ]

    gamma_max[j] = np.max(ball_data0)
    print(f"surf number {j} done!")

print("Final ballooning gamma objective = ", gamma_max)


print("Saving VMEC file!")

VMECIO.save(eq, f"wout_{fname}_DESC.nc", surfs=512)

print("Final equilibrium nestedness = ", eq.is_nested())

fig, ax = plot_section(eq, "|F|", norm_F=True, log=True)
plt.savefig(f"normF.png", dpi=400)
plt.close()

fig, ax = plot_comparison(eqs=[eq])
plt.savefig(f"Xsection.png", dpi=400)
plt.close()

fig, ax = plot_boozer_surface(eq, rho=1)
plt.savefig(f"Boozer_eq.png", dpi=400)
plt.close()

fig, ax = plot_boozer_surface(field, rho=1)
plt.savefig(f"Boozer_field.png", dpi=400)
plt.close()

plt.figure()
theta_grid = np.linspace(0, 2 * np.pi, 300)
zeta_grid = np.linspace(0, 2 * np.pi, 300)
grid = LinearGrid(rho=1.0, theta=theta_grid, zeta=zeta_grid)
fig = plot_3d(eq, name="|B|", grid=grid)
fig.write_html(f"modB_3d.html")
plt.close()
