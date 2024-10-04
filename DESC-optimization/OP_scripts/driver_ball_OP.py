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


reload_idx = int(eval(sys.argv[1]))		

NFP = int(3)
aspect_ratio = float(9.5)
torsion = float(0.0)

mirror_ratio = float(0.5)

elongation = float(2.0)
#well_weight = float(2.5)
well_weight = float(1.5)
	
radial_shift = -0.00
well_width = -0.00

# setting these values manually
helicity = (0, NFP)
L = int(11)
M = int(11)
N = int(11)
L_shift = int(3)
M_shift = int(2)
N_shift = int(3)
L_well = int(2)
M_well = int(3)

fname = "eq"

# dependent variables
M_grid = int((2 * M))
N_grid = int((2 * N))
minor_radius = 1 / aspect_ratio
flux = 1.1*np.pi * minor_radius**2  # This assumed B_N = 1.

if reload_idx > 0:
    eq = EquilibriaFamily.load(f"eq{reload_idx:02d}.h5")
    field = OmnigenousField.load(f"field{reload_idx:02d}.h5")
elif reload_idx == -1:
    eq = EquilibriaFamily.load(f"eq_final.h5")
    field = OmnigenousField.load(f"field_final.h5")
else:
    # initial equilibrium
    surface = FourierRZToroidalSurface.from_qp_model(
        major_radius=1,
        aspect_ratio=aspect_ratio,
        mirror_ratio=mirror_ratio,
        elongation=elongation,
        torsion=torsion,
        NFP=NFP,
        positive_iota=True,
        sym=True,
        )
    
    #pressure_profile = PowerSeriesProfile([2.65e4, 0, -0., 0,  -2.65e4])
    #eq = Equilibrium(Psi=flux, NFP=NFP,pressure = pressure_profile,  L=L, M=M, N=N, L_grid=20, M_grid=21, N_grid=21, sym=True, surface=surface)
    #eq = solve_continuation_automatic(
    #        eq, objective="force", ftol=1e-4, xtol=1e-6, gtol=1e-6, maxiter=200, verbose=3
    #        )[-1]

    #eq.save(fname + f"{0:02d}_finite_beta.h5")

    #NL = 200
    #grid = LinearGrid(L = NL, M=0, N=0)
    #rho = np.linspace(0, 1, NL+1)

    #iota_profile = np.polynomial.polynomial.polyfit(rho, eq.compute("iota", grid=grid)["iota"], 5)
    #print("iota_profile = ", iota_profile)

    #### Specified the equilibrium ota
    ###iota_profile = SplineProfile(0.358 + 0.10*rho**2, knots=rho)
    #iota_profile = SplineProfile(0.358 + 0.18*rho**2 - 0.08*rho**4, knots=rho)

    #eq.iota = iota_profile
    #eq.current = None
    #eq = solve_continuation_automatic(
    #        eq, objective="force", ftol=1e-4, xtol=1e-6, gtol=1e-6, maxiter=200, verbose=3
    #        )[-1]
    #eq.save(fname + f"{0:02d}_finite_beta_and_iota.h5")

    #######print("LOADIN AN ALREADY SAVED EQ!!!!")
    eq = Equilibrium.load(fname + f"{0:02d}_finite_beta_and_iota.h5")

if reload_idx == 0:
    start_idx = 1
else:
    start_idx = reload_idx + 1


def mirrorRatio(params):
    B_lm = params["B_lm"]
    f = jnp.array([B_lm[0] - B_lm[field.M_B],  B_lm[field.M_B - 1] - B_lm[-1], ])
    return f

print(eq.compute("<beta>_vol")["<beta>_vol"])  

rho = np.linspace(0, 1, L+1)
grid = LinearGrid(L=L, M = M_grid, N=N_grid, NFP=eq.NFP, sym=True)
data_keys = ["iota", "D_Mercier", "current"]
data = eq.compute(data_keys, grid=grid)

print("current=", grid.compress(data["current"]));
print("D_Mercier=", grid.compress(data["D_Mercier"]));
print("iota", grid.compress(data["iota"]))

## Specified the equilibrium iota
#eq.iota = SplineProfile(grid.compress(data["iota"]), knots=rho)

if reload_idx > 0:
    field = OmnigenousField.load(f"field{reload_idx:02d}.h5")
else:
    field = OmnigenousField(
        L_B=L_well,  # radial resolution of B_lm parameters
        M_B=M_well,  # number of spline knots on each flux surface
        L_x=L_shift,  # radial resolution of x_lmn parameters
        M_x=M_shift,  # eta resolution of x_lmn parameters
        N_x=N_shift,  # alpha resolution of x_lmn parameters
        NFP=eq.NFP,  # number of field periods; should always be equal to Equilibrium.NFP
        helicity=(0, eq.NFP),  # helicity for poloidally closed |B| contours
	#B_lm = np.array([[1-mirror_ratio, 1+well_width, 1+mirror_ratio], [radial_shift, radial_shift, radial_shift], [0., 0., 0,]]).flatten()
	B_lm = np.array([[1-mirror_ratio, 1+well_width, 1+mirror_ratio], [radial_shift, radial_shift, radial_shift], [0., 0., 0,]]).flatten()
        )

eq_weights = [1e3, 1e4, 1e5, 1e6, 1e7]
#surfaces_ball = [0.15, 0.3, 0.45, 0.60, 0.75, 0.98]
#surfaces_ball = [0.45, 0.65, 0.85, 1.0]
surfaces_ball = [0.6, 0.8, 0.98]
#surfaces_omni = [0.2, 0.4, 0.6, 0.8, 0.9, 1.0]
surfaces_omni = [0.4, 0.65]
surfaces_ripple = [0.2, 0.96]
#surfaces_omni = [0.25, 0.45, 0.65, 0.85, 1.0]
#Mercier_weights = [1e3, 1.2e3, 2.5e3, 3.5e3]
#Mercier_weights = [1e4, 3e4, 3e4, 2e5, 4e5, 7e5, 8e5]
Mercier_weights = [2e5, 3e5, 6e5, 6e5]
#Current_weights = [1e-4, 1e-4, 1e-4, 2e-4, 5e-4]

#ntor = int(4)
#M_ball = int(12)
#N_ball = int(11)
ntor = int(4)
M_ball = int(12)
N_ball = int(12)
N0 = int(2 * M_ball * N_ball + 1)

#optim_indices = np.array([3, 6, 9, 11, 11], dtype=int)
#optim_indices = np.array([2, 4, 6, 8, 11], dtype=int)
optim_indices = np.array([2, 4, 6], dtype=int)
#optim_indices = np.array([7], dtype=int)
#optim_indices = np.array([6], dtype=int)

jax.clear_caches()

# optimize with increasing resolution
for k in optim_indices:

    print("\n---------------------------------------")
    print(f"Optimizing boundary modes M, N <= {k}")
    print("---------------------------------------")
    
    objs_ball = {}
    
    eq_grids_omni = {}
    field_grids_omni = {}
    objs_omni = {}
    objs_ripple = {}
    
    eq_ball_weight = 2.5e3 + k*5e2
    omni_weight = 22e-0 + 0.9*k
    
    for i, rho in enumerate(surfaces_ball):
        shift_arr = np.random.default_rng().uniform(-0.1, 0.1, 12)
        alpha=np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.8, 3.1])
        alpha[1:] = alpha[1:] + shift_arr
 
        if i == 0:
        	objs_ball[rho] = BallooningStability(eq=eq,rho=np.array([rho]), alpha=alpha, nturns=ntor,lam0 = 0.0004, w0 =1, w1=8.,  nzetaperturn=N0, weight=eq_ball_weight)
        elif i == len(surfaces_ball)-1:
        	objs_ball[rho] = BallooningStability(eq=eq,rho=np.array([rho]), alpha=alpha, nturns=ntor,lam0 = 0.0004, w0 =1, w1=16.,  nzetaperturn=N0, weight=eq_ball_weight*6)
        else:
        	objs_ball[rho] = BallooningStability(eq=eq,rho=np.array([rho]), alpha=alpha, nturns=ntor,lam0 = 0.0004, w0 =1, w1=16.,  nzetaperturn=N0, weight=eq_ball_weight*6.)
        
    for rho in surfaces_omni:
        eq_grids_omni[rho] = LinearGrid(rho=rho, M = int(28), N= int(28), NFP=eq.NFP, sym=False)
        #eq_grids_omni[rho] = LinearGrid(rho=rho, M = int(30), N= int(30), NFP=eq.NFP, sym=False)
        field_grids_omni[rho] = LinearGrid(rho=rho, theta=np.linspace(0, 2*np.pi, 12), zeta=np.linspace(0, 2*np.pi/NFP, 8), NFP=field.NFP, sym=False)
        objs_omni[rho] = Omnigenity(
            field=field,
            eq=eq,
            eq_grid=eq_grids_omni[rho],
            field_grid=field_grids_omni[rho],
            eta_weight=well_weight,
            weight = omni_weight, 
            #deriv_mode="rev,
        )
 
    for rho in surfaces_ripple:
        objs_ripple[rho] = EffectiveRipple(eq, rho = np.array(rho), alpha = 0, zeta = np.linspace(0, 3*2*np.pi, int(6*38)), num_pitch=34, num_quad = 17, weight=7e5, deriv_mode="fwd")

    Mercier_grid = LinearGrid(M = int(2*M), N = int(2*N), rho=np.array([0.4, 0.6, 0.65, 0.75, 0.80, 0.85, 0.9, 0.95, 1.00]), NFP=eq.NFP, sym=True, axis=False)
    #Mercier_grid = LinearGrid(M = int(2*M), N = int(2*N), rho=np.array([0.6, 0.65, 0.75, 0.80, 0.85, 0.9, 0.95, 1.00]), NFP=eq.NFP, sym=True, axis=False)

    Elongation_grid = LinearGrid(M = 2*int(M), N = 2*int(N), rho=np.array([1.0]), NFP=eq.NFP, sym=True, axis=False)
    
    Curvature_grid = LinearGrid(M = 2*int(M), N = 2*int(N), rho=np.array([1.0]), NFP=eq.NFP, sym=True, axis=False)

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
   

    if k <= 2:
        objective = ObjectiveFunction((MercierStability(eq=eq, grid=Mercier_grid, bounds=(0.25, np.inf), weight=Mercier_weight), AspectRatio(eq=eq, bounds=(8.2, 9.8), weight=8e4), GenericObjective(f="curvature_k2_rho", thing=eq, grid=Curvature_grid, bounds=(-85, 10), weight=2e3),) + tuple(objs_ball.values()) + tuple(objs_omni.values()) + tuple(objs_ripple.values()))
    else:
        objective = ObjectiveFunction((MercierStability(eq=eq, grid=Mercier_grid, bounds=(0.01, np.inf), weight=Mercier_weight), AspectRatio(eq=eq, bounds=(8.2, 9.9), weight=8e4), GenericObjective(f="curvature_k2_rho", thing=eq, grid=Curvature_grid, bounds=(-85, 10), weight=2e3),) + tuple(objs_ball.values()) + tuple(objs_omni.values()) + tuple(objs_ripple.values()))

    constraints = (
            ForceBalance(eq=eq, weight=eq_weight),
            FixBoundaryR(eq=eq, modes=modes_R),
            FixBoundaryZ(eq=eq, modes=modes_Z),
            FixPressure(eq=eq),
            FixIota(eq=eq),
            FixPsi(eq=eq),
            FixOmniBmax(field=field, weight=1),
            FixOmniMap(field=field, indices=np.where(field.x_basis.modes[:, 1] == 0)[0]),
    	    # fix the mirror ratio on the magnetic axis
            LinearObjectiveFromUser(mirrorRatio, field, target=[0.45, 1.3]),
            )

    optimizer = Optimizer("proximal-lsq-exact")
    (eq, field), _ = optimizer.optimize(
                (eq, field),
                objective,
                constraints,
                ftol=1e-4,
                xtol=1e-8,
                gtol=1e-6,
                maxiter=150,
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
    plt.savefig(f"ripple_OP_{k:02d}.png", dpi=400)


    jax.clear_caches()   

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

#alpha = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75])
alpha = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.8, 3.1])

Nalpha = len(alpha)
Nzeta0 = int(13)

ball_data = np.zeros((len(surfaces_ball), Nalpha, Nzeta0))
gamma_max = np.zeros(len(surfaces_ball), )

for i in range(1):
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


print("ballooning analysis before refinement done!")



#eq.change_resolution(L=16, M= 16, N=16, L_grid=30, M_grid=30, N_grid=30)
eq.change_resolution(L=15, M= 15, N=15, L_grid=29, M_grid=29, N_grid=29)
eq = solve_continuation_automatic(eq=eq, objective="force", ftol=1e-3, xtol=1e-5, gtol=1e-5, maxiter=150, verbose=3, pres_step=0.2, bdry_step=0.125)[-1]


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

eq.save(fname + f"_final.h5")
field.save(f"field_final.h5")


L0 = 200
rho = np.linspace(0, 1, L0+1)
grid = LinearGrid(L=L0, M = M_grid, N=N_grid, NFP=eq.NFP, sym=True)
data_keys = ["iota", "D_Mercier"]
data = eq.compute(data_keys, grid=grid)

print("D_Mercier=", grid.compress(data["D_Mercier"]));
print("iota=", grid.compress(data["iota"]))
print("beta=", eq.compute("<beta>_vol")["<beta>_vol"])  


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




