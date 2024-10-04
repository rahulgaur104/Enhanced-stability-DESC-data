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
    FixCurrent,
    FixPressure,
    FixPsi,
    AspectRatio,
    FixOmniBmax,
    FixOmniMap,
    FixPressure,
    Elongation,
    GenericObjective,
    LinearObjectiveFromUser,
    Omnigenity,
    EffectiveRipple,
    get_fixed_boundary_constraints,
    ToroidalCurrent,
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


#eq = EquilibriaFamily.load(f"eq_final.h5")
#field = OmnigenousField.load(f"field_final.h5")

eq = EquilibriaFamily.load(f"eq06.h5")
field = OmnigenousField.load(f"field06.h5")

fname = "eq"

#eq.change_resolution(L=16, M= 16, N=16, L_grid=30, M_grid=30, N_grid=30)
eq.change_resolution(L=15, M= 15, N=15, L_grid=29, M_grid=29, N_grid=29)
eq = solve_continuation_automatic(eq=eq, objective="force", ftol=1e-3, xtol=1e-5, gtol=1e-5, maxiter=125, verbose=3, pres_step=0.2, bdry_step=0.125)[-1]


num_pitch = 144
ntransit = 10
knots_per_transit = 1024

rho = np.linspace(0, 1, 10)
alpha = np.array([0.])
zeta = np.linspace(0, 2 * np.pi * ntransit, knots_per_transit * ntransit)
grid = eq.get_rtz_grid(
            rho, alpha, zeta, coordinates="raz", period=(np.inf, 2 * np.pi, np.inf)
            )
data = eq.compute("effective ripple", grid=grid, num_pitch=num_pitch)

print("ripple = ", grid.compress(data["effective ripple"]))
fig, ax = plt.subplots()
ax.plot(rho, grid.compress(data["effective ripple"]), marker="o")
plt.savefig("ripple_OP.png", dpi=400)

print("ripple calculation after refinement done!")

eq.save("eq_final.h5")
field.save(f"field_final.h5")


L0 = 200
rho = np.linspace(0, 1, L0+1)
grid = LinearGrid(L=L0, M = eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=True)
data_keys = ["iota", "D_Mercier"]
data = eq.compute(data_keys, grid=grid)

print("D_Mercier=", grid.compress(data["D_Mercier"]));
print("iota=", grid.compress(data["iota"]))
print("beta=", eq.compute("<beta>_vol")["<beta>_vol"])  


surfaces_ball = [0.15, 0.3, 0.4, 0.45, 0.5, 0.6, 0.65, 0.7, 0.8, 0.9, 1.0]

# We also calculate how muuch the ballooning stability is ruined!
M_grid = 15
N_grid = 15
ntor = int(4)
N0 = int(2 * M_grid * N_grid * ntor + 1)

eq_data_keys = ["iota", "a", "psi", "Psi"]

eq_coords = np.zeros((len(surfaces_ball), 3))
eq_coords[:, 0] = np.array([surfaces_ball])

eq_data_grid = Grid(eq_coords)
eq_data = eq.compute(eq_data_keys, grid=eq_data_grid)
# Now we compute theta_DESC for given theta_PEST
iota = eq_data["iota"]

alpha = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.8, 3.1])

Nalpha = len(alpha)
Nzeta0 = int(13)

ball_data = np.zeros((len(surfaces_ball), Nalpha, Nzeta0))
gamma_max = np.zeros(len(surfaces_ball), )

for i in range(3):
    for j in range(len(surfaces_ball)):
        rho = surfaces_ball[j]
        zeta = np.linspace(-jnp.pi*ntor, jnp.pi*ntor, N0)
            
        sfl_grid = Grid.create_meshgrid([rho, alpha, zeta], coordinates="raz", period=(np.inf, 2*np.pi, np.inf))
        ball_data0 = eq.compute(["ideal ball lambda"], grid=sfl_grid)["ideal ball lambda"]
        
        gamma_max[j] = np.max(ball_data0)
        print(f"surf number {j} done!")
    
    print(f"Ballooning gamma objective {i+1} = ", gamma_max)
    print("Shuffling alpha and recalculating the growth rates...\n")

    shift_arr = np.random.default_rng().uniform(-0.1, 0.1, 12)
    alpha = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.8, 3.1])
    alpha[1:] = alpha[1:] + np.reshape(shift_arr, (-1, ))


num_pitch = 144
ntransit = 10
knots_per_transit = 1024

rho = np.linspace(0, 1, 10)
alpha = np.array([0.])
zeta = np.linspace(0, 2 * np.pi * ntransit, knots_per_transit * ntransit)
grid = eq.get_rtz_grid(
            rho, alpha, zeta, coordinates="raz", period=(np.inf, 2 * np.pi, np.inf)
            )
data = eq.compute("effective ripple", grid=grid, num_pitch=num_pitch)

print("ripple = ", grid.compress(data["effective ripple"]))
fig, ax = plt.subplots()
ax.plot(rho, grid.compress(data["effective ripple"]), marker="o")
plt.savefig("ripple_OP.png", dpi=400)


print("Saving VMEC file!")

VMECIO.save(eq, f"wout_{fname}_DESC.nc", surfs=512)

print("Final equilibrium nestedness = ", eq.is_nested())

fig, ax = plot_section(eq, "|F|", norm_F=True, log=True); 
plt.savefig(f"normF.png", dpi=400)  
plt.close()

fig, ax = plot_comparison(eqs=[eq]) 
plt.savefig(f"Xsection.png", dpi=400)  
plt.close()

fig, ax = plot_boozer_surface(eq, rho=1)
plt.savefig(f"Boozer_eq.png", dpi=400)  
plt.close()

fig, ax = plot_boozer_surface(field, rho=1,iota=iota[-1])
plt.savefig(f"Boozer_field.png", dpi=400)  
plt.close()

plt.figure()
theta_grid = np.linspace(0, 2*np.pi, 300)
zeta_grid = np.linspace(0, 2*np.pi, 300) 
grid = LinearGrid(rho = 1.0, theta=theta_grid, zeta=zeta_grid)
fig = plot_3d(eq, name="|B|", grid=grid)
fig.write_html(f"modB_3d.html")
plt.close()
