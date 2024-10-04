#!/usr/bin/env python3

from desc import set_device
set_device("gpu")

import numpy as np
from qsc import Qsc
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
    Elongation,
    AspectRatio,
    FixOmniBmax,
    FixOmniMap,
    FixPressure,
    GenericObjective,
    LinearObjectiveFromUser,
    Omnigenity,
    RotationalTransform,
    QuasisymmetryTwoTerm,
    get_NAE_constraints,
    get_fixed_boundary_constraints,
)
from desc.continuation import solve_continuation_automatic
from desc.optimize import Optimizer
from desc.profiles import PowerSeriesProfile, SplineProfile
from desc.magnetic_fields import OmnigenousField
from desc.compute.utils import get_transforms
from desc.vmec_utils import ptolemy_linear_transform
from desc.plotting import *
from matplotlib import pyplot as plt
from desc.vmec import VMECIO
import pickle

import re
from desc.objectives.normalization import compute_scaling_factors
from desc.compute import get_transforms
from desc.vmec_utils import ptolemy_linear_transform

from scipy.optimize import least_squares
import pdb
import sys

iter0 = int(sys.argv[1])

x0 = np.array([ 0.850079, -0.12873215, -0.07080123])

NFP = int(5)

mirror_ratio = 0.3
radial_shift = -0.00
well_width   = 0.00
well_weight  = 1.0

target_mode = np.array([])
target_amplitude = np.array([])

# setting these values manually
helicity = (1, NFP)
L = int(10)
M = int(10)
N = int(10)
L_shift = int(2)
M_shift = int(1)
N_shift = int(1)
L_well = int(2)
M_well = int(3)

fname = "eq"

surface = FourierRZToroidalSurface(
        R_lmn=[1, 0.1, 0.1],
        Z_lmn=[-0.1, -0.1, 0.00],
        modes_R=[[0, 0], [1, 0], [0, 1]],
        modes_Z=[[-1, 0], [0, -1], [0, -2]],
        NFP=NFP,
        sym=True,
        )

flux = 0.08
pressure_profile = PowerSeriesProfile([8e4, 0, -0., 0,  -8e4])
iota_profile = PowerSeriesProfile([x0[0], 0.0, x0[1], 0.0, x0[2]])

eq = Equilibrium(Psi=flux, NFP=NFP,pressure = pressure_profile, iota=iota_profile, L=L, M=M, N=N, L_grid=int(1.5*L), M_grid=int(1.5*M), N_grid=int(1.5*N), sym=True, surface=surface)
eq = solve_continuation_automatic(
        eq, objective="force", ftol=1e-3, xtol=1e-6, gtol=1e-6, maxiter=100, verbose=3
        )[-1]

eq.save("eq00_low_res_solve.h5")
eq = Equilibrium.load("eq00_low_res_solve.h5")
assert NFP == eq._NFP, "NFP must match!"

field = OmnigenousField(
        L_B=L_well,  # radial resolution of B_lm parameters
        M_B=M_well,  # number of spline knots on each flux surface
        L_x=L_shift,  # radial resolution of x_lmn parameters
        M_x=M_shift,  # eta resolution of x_lmn parameters
        N_x=N_shift,  # alpha resolution of x_lmn parameters
        NFP=NFP,  # number of field periods; should always be equal to Equilibrium.NFP
        helicity=(1, NFP),  # helicity for toroidally closed |B| contours
        B_lm = np.array([[1-mirror_ratio, 1+well_width, 1+mirror_ratio], [radial_shift, radial_shift, radial_shift], [0., 0., 0,]]).flatten()
        )

idx_list = []
x_lmn = np.zeros(field.x_basis.num_modes)
for i in range(np.shape(target_amplitude)[0]):
    idx = np.nonzero((field.x_basis.modes == target_mode[i]).all(axis=1))[0]
    x_lmn[idx] = target_amplitude[i]
    idx_list.append(idx)

field.x_lmn = x_lmn
idx_array = np.array(idx_list).flatten()



def mirrorRatio(params):
    B_lm = params["B_lm"]
    f = jnp.array([B_lm[0] - B_lm[field.M_B],  B_lm[field.M_B - 1] - B_lm[-1], ])
    return f

# dependent variables
M_grid = int(np.ceil(2.5 * M))
N_grid = int(np.ceil(2.5 * N))

NL = 32
grid = LinearGrid(L = NL, M=0, N=0)
rho = np.linspace(0, 1, NL+1)

eq.iota = SplineProfile(x0[0] + x0[1]*rho**2 + x0[2]*rho**4, knots=rho)
eq.current = None
eq, _ = eq.solve(objective="force", maxiter=100, verbose=3)
eq.save(fname + f"{0:02d}_finite_beta_and_iota.h5")

####print("LOADING AN ALREADY SAVED EQ!!!!")
eq = Equilibrium.load(fname + f"{0:02d}_finite_beta_and_iota.h5")

print(eq.compute("<beta>_vol")["<beta>_vol"])  

rho = np.linspace(0, 1, L+1)
grid = LinearGrid(L=L, M = M_grid, N=N_grid, NFP=eq.NFP, sym=True)
data_keys = ["iota", "D_Mercier", "current"]
data = eq.compute(data_keys, grid=grid)

print("current=", grid.compress(data["current"]));
print("D_Mercier=", grid.compress(data["D_Mercier"]));
print("iota", grid.compress(data["iota"]))


eq_weights = [1e3, 1e4, 1e5, 1e6, 1e7]
surfaces_ball = [0.15, 0.3, 0.50, 0.70, 0.85, 0.97] # for (1-x^4)
surfaces_omni = [1.0]
surfaces_QH = [0.2, 0.4, 0.6, 0.8, 1.0]
Mercier_weights = [1.e1]

ntor = int(4)
M_ball = int(10)
N_ball = int(10)
N0 = int(2 * M_ball * N_ball * ntor + 1)

optim_indices = np.array([3], dtype=int)
#optim_indices = np.array([5, 6], dtype=int)
#optim_indices = np.array([5, 6, 7], dtype=int)

# optimize with increasing resolution
for k in optim_indices:

    print("\n---------------------------------------")
    print(f"Optimizing boundary modes M, N <= {k}")
    print("---------------------------------------")
    
    objs_ball = {}
    
    eq_grids_omni = {}
    field_grids_omni = {}
    objs_omni = {}


    objs_QH = {}
    
    eq_ball_weight = 5.0e3
    omni_weight = 1e-4
    weight_QH = 4.5*(22e-0 + 10*k)
    
    for i, rho in enumerate(surfaces_ball):
        shift_arr = np.random.default_rng().uniform(-0.1, 0.1, 12)
        alpha=np.array([[0.0], [0.25], [0.5], [0.75], [1.0], [1.25], [1.5], [1.75], [2.0], [2.25], [2.5], [2.8], [3.1]])
        alpha[1:, :] = alpha[1:, :] + np.reshape(shift_arr, (-1, 1))
 
        if i == 1:
        	objs_ball[rho] = BallooningStability(eq=eq,rho=np.array([rho]), alpha=alpha, zetamax=ntor*jnp.pi, nzeta=N0, weight=eq_ball_weight*2)
        elif i == len(surfaces_ball)-2:
        	objs_ball[rho] = BallooningStability(eq=eq,rho=np.array([rho]), alpha=alpha, zetamax=ntor*jnp.pi, nzeta=N0, weight=eq_ball_weight*2)
        else:
        	objs_ball[rho] = BallooningStability(eq=eq,rho=np.array([rho]), alpha=alpha, zetamax=ntor*jnp.pi, nzeta=N0, weight=eq_ball_weight*2)
        
    for rho in surfaces_omni:
        eq_grids_omni[rho] = LinearGrid(rho=rho, M = int(1*M), N= int(1*N), NFP=eq.NFP)
        field_grids_omni[rho] = LinearGrid(rho=rho, theta=np.linspace(0, 2*np.pi, 6), zeta=np.linspace(0.00, 2*np.pi/NFP, 4), NFP=field.NFP, sym=False)
        objs_omni[rho] = Omnigenity(
            field=field,
            eq=eq,
            eq_grid=eq_grids_omni[rho],
            field_grid=field_grids_omni[rho],
            eta_weight=well_weight,
            weight = omni_weight, 
        )
    
    for rho in surfaces_QH:
        eq_grids_QH = LinearGrid(rho=rho, M = int(2*M), N= int(2*N), NFP=eq.NFP)
        objs_QH[rho] = QuasisymmetryTwoTerm(
            eq=eq,
            helicity=helicity,
            grid=eq_grids_QH,
            weight = weight_QH, 
        )

    Mercier_grid = LinearGrid(M = int(2*M), N = int(2*N), rho=np.array([0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.96, 0.97,0.98, 0.985, 0.99, 1.0]), NFP=eq.NFP, sym=True, axis=False)

    Elongation_grid = LinearGrid(M = 2*int(M), N = 2*int(N), rho=np.array([1.0]), NFP=eq.NFP, sym=True, axis=False)
    
    Curvature_grid = LinearGrid(M = 2*int(M), N = 2*int(N), rho=np.array([1.0]), NFP=eq.NFP, sym=True, axis=False)

    RotationalTransform_grid = LinearGrid(M = 2*int(M), N = 2*int(N), rho=np.linspace(1e-2, 1.0, 111), NFP=eq.NFP, sym=True, axis=True)

    modes_R = np.vstack(
        (
            [0, 0, 0],
            eq.surface.R_basis.modes[
                np.max(np.abs(eq.surface.R_basis.modes), 1) > k, :
            ],
        )
    )
    modes_Z = eq.surface.Z_basis.modes[
        np.max(np.abs(eq.surface.Z_basis.modes), 1) > k, :
    ]
    
    try:
        eq_weight = eq_weights[k-1]
    except:
        eq_weight = eq_weights[-1]
 
    try:
        Mercier_weight = Mercier_weights[k-1]
    except:
        Mercier_weight = Mercier_weights[-1]
   
 
    objective = ObjectiveFunction((MercierStability(eq=eq, grid=Mercier_grid, bounds=(2, np.inf), weight=Mercier_weight),  AspectRatio(eq=eq, bounds=(9.5, 12.5), weight=8e3), Elongation(eq=eq, grid=Elongation_grid, bounds=(0.5, 2.7), weight=1e3), GenericObjective(f="curvature_k2_rho", eq=eq, grid=Curvature_grid, bounds=(-120, 10), weight=2e3),) + tuple(objs_ball.values()) + tuple(objs_omni.values()) + tuple(objs_QH.values()))
    #objective = ObjectiveFunction((MercierStability(eq=eq, grid=Mercier_grid, bounds=(5, np.inf), weight=Mercier_weight),  AspectRatio(eq=eq, bounds=(9.5, 12.5), weight=8e3), Elongation(eq=eq, grid=Elongation_grid, bounds=(0.5, 2.7), weight=1e3), GenericObjective(f="curvature_k2_rho", eq=eq, grid=Curvature_grid, bounds=(-120, 10), weight=2e3),) + tuple(objs_ball.values()) + tuple(objs_QH.values()))

    constraints = (
            ForceBalance(eq=eq, weight=eq_weight),
            FixBoundaryR(eq=eq, modes=modes_R),
            FixBoundaryZ(eq=eq, modes=modes_Z),
            FixPressure(eq=eq),
            FixPsi(eq=eq),
            FixIota(eq=eq),
            FixOmniBmax(field=field, weight=2),
            FixOmniMap(field=field, indices=np.where(field.x_basis.modes[:, 1] == 0)[0]),
    	    # fix the mirror ratio on the magnetic axis
            LinearObjectiveFromUser(mirrorRatio, field, target=[0.3, 1.5]),
            )
    optimizer = Optimizer("proximal-lsq-exact")
    (eq, field), _ = optimizer.optimize(
                (eq, field),
                objective,
                constraints,
                ftol=1e-4,
                xtol=1e-6,
                gtol=1e-6,
                maxiter=72,
                verbose=3,
                options={"initial_trust_ratio":2e-3},
                )


    eq.save(fname + f"{k:02d}.h5")
    field.save(f"field{k:02d}.h5")

    
    L0 = 200
    rho = np.linspace(0, 1, L0+1)
    grid = LinearGrid(L=L0, M = M_grid, N=N_grid, NFP=eq.NFP, sym=True)
    data_keys = ["iota", "D_Mercier", "current"]
    data = eq.compute(data_keys, grid=grid)
    
    print("current=", grid.compress(data["current"]));
    print("D_Mercier=", grid.compress(data["D_Mercier"]));
    test_iota = grid.compress(data["iota"])
    print("iota metrics, max, min, min(abs)", np.max(test_iota), np.min(test_iota), np.min(abs(test_iota)));

    jax.clear_caches()   

if k < 7:
    exit()



