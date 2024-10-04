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
import pdb
import sys


reload_idx = int(eval(sys.argv[1]))

#fname = "NFP1_A+7.5_rc1-0.20_rc2-0.01_rc3-0.01_zs1-0.20_zs2-0.10_zs3-0.01_B2c-0.50_etabar+1.00_I2+1.00_p2-100000.00_mr+0.10_shift-0.05_wd+0.05_wellwt+2.0_tmodez0p1p3_tamp0.52"
#fname = "NFP1_A+10.0_rc1-0.20_rc2-0.01_rc3-0.01_zs1-0.20_zs2-0.10_zs3-0.01_B2c-0.50_etabar+1.00_I2+1.00_p2-100000.00_mr+0.10_shift-0.05_wd+0.05_wellwt+2.0_tmodez0p1m1_tamp0.50"

fname = "NFP1_A+6.0_rc1+0.15_rc2-0.01_rc3-0.01_zs1-0.20_zs2-0.10_zs3-0.01_B2c-0.50_etabar+1.00_I2+1.00_p2-100000.00_mr+0.20_shift-0.02_wd-0.05_wellwt+1.5_tmodez0p1p1_tamp1.05"

args = fname.split("_")

NFP = int(args[0][3:])
aspect_ratio = float(args[1][1:])

rc1 = float(args[2][3:])
rc2 = float(args[3][3:])
rc3 = float(args[4][3:])

zs1 = float(args[5][3:])
zs2 = float(args[6][3:])
zs3 = float(args[7][3:])

B2c = float(args[8][3:])
etabar = float(args[9][6:])
I2 = float(args[10][2:])
p2 = float(args[11][2:])


mirror_ratio = float(args[12][2:])
radial_shift = float(args[13][5:])
well_width = float(args[14][2:])
well_weight = float(args[15][6:])

#print("TMODE AND TAMP ARE BEING MANUALLY SET!")
tmode = args[16][5:]
t0 = []
for i in range(int(len(tmode)/2)):
    if tmode[2*i] == 'z':
        t0.append(0)
    elif tmode[2*i] == 'p':
        t0.append(int(eval(tmode[2*i+1])))
    else: # ch = 'n'
        t0.append(int(-eval(tmode[2*i+1])))

target_mode = np.array([t0])
tamp = args[17][4:]
target_amplitude = np.array([float(tamp)])

# setting these values manually
helicity = (NFP, 0)
L = int(10)
M = int(10)
N = int(10)
L_shift = int(3)
M_shift = int(2)
N_shift = int(2)
L_well = int(2)
M_well = int(3)


if reload_idx > 0:
    eq = EquilibriaFamily.load(f"eq{reload_idx:02d}.h5")
    field = OmnigenousField.load(f"field{reload_idx:02d}.h5")
else:
    # initial NAE solution
    qsc = Qsc(
        nfp=NFP,
        rc=[
            1.00000000e00,
            rc1,
            rc2,
            rc3,
            5.77324841e-04,
            -2.13812436e-05,
            -3.25014052e-07,
            7.33393963e-08,
            1.42375011e-08,
            7.98521016e-10,
        ],
        zs=[
            0.00000000e00,
            zs1,
            zs2,
            zs3,
            1.59769355e-05,
            6.41471931e-06,
            3.47327323e-07,
            -6.49967945e-08,
            -1.39333315e-08,
            -8.47322874e-10,
        ],
        B0=1.0,
        B2c=B2c,
        etabar=etabar,
        I2=I2,
        p2=p2,
        order="r1",
    )


    eq = Equilibrium.from_near_axis(
        qsc,
        r=1 / aspect_ratio,
        L=L,
        M=M,
        N=N,
    )
 

fname = "eq"

# dependent variables
M_grid = int(np.ceil(2.5 * M))
N_grid = int(np.ceil(2.5 * N))

if reload_idx > 0:
    eq = EquilibriaFamily.load(f"eq{reload_idx:02d}.h5")
    field = OmnigenousField.load(f"field{reload_idx:02d}.h5")

    x_lmn = np.zeros(field.x_basis.num_modes)
    # CHANGE THIS LOGIC WHEN LEN(IDX)>1 
    for i in range(np.shape(target_amplitude)[0]):
        idx = np.nonzero((field.x_basis.modes == target_mode[i]).all(axis=1))[0]
    #x_lmn[idx] = target_amplitude[i]
elif reload_idx == -1:
    eq = EquilibriaFamily.load(f"eq_final.h5")
    field = OmnigenousField.load(f"field_final.h5")
else:
    field = OmnigenousField(
            L_B=L_well,  # radial resolution of B_lm parameters
            M_B=M_well,  # number of spline knots on each flux surface
            L_x=L_shift,  # radial resolution of x_lmn parameters
            M_x=M_shift,  # eta resolution of x_lmn parameters
            N_x=N_shift,  # alpha resolution of x_lmn parameters
            NFP=eq.NFP,  # number of field periods; should always be equal to Equilibrium.NFP
            helicity=(eq.NFP, 0),  # helicity for toroidally closed |B| contours
            B_lm = np.array([[1-mirror_ratio, 1+well_width, 1+mirror_ratio], [radial_shift, radial_shift, radial_shift], [0., 0., 0,]]).flatten()
            )
    
    x_lmn = np.zeros(field.x_basis.num_modes)
    for i in range(np.shape(target_amplitude)[0]):
        idx = np.nonzero((field.x_basis.modes == target_mode[i]).all(axis=1))[0]
        x_lmn[idx] = target_amplitude[i]
    
    field.x_lmn = x_lmn

    ##pdb.set_trace()

    #objective = ObjectiveFunction((ForceBalance(eq=eq),))
    #constraints = get_NAE_constraints(eq, qsc, order=1)
    #eq, result = eq.solve(
    #		objective=objective,
    #		constraints=constraints,
    #		optimizer="lsq-exact",
    #		ftol=1e-3,
    #		gtol=1e-6,
    #	 	xtol=1e-6, 
    #		maxiter=200, 
    #		verbose=3, 
    #		copy=True)

    #eq.save(f"{fname}_vacuum.h5")

    #NL = 41
    #grid = LinearGrid(L = NL, M=0, N=0)
    #rho = np.linspace(0, 1, NL+1)


    #eq.pressure = SplineProfile(6.0e4*(1-2*rho**2+rho**4), knots=rho)
    #eq = solve_continuation_automatic(
    #        eq, objective="force", ftol=1e-3, xtol=1e-6, gtol=1e-6, maxiter=100, verbose=3
    #        )[-1]

    #eq.save(fname + f"{0:02d}_finite_beta.h5")

    #NL = 41
    #grid = LinearGrid(L = NL, M=0, N=0)
    #rho = np.linspace(0, 1, NL+1)

    #iota_profile = np.polynomial.polynomial.polyfit(rho, eq.compute("iota", grid=grid)["iota"], 5)

    #print("iota_profile = ", iota_profile)

    ## Specified the equilibrium iota
    #iota_profile = SplineProfile(eq.compute("iota", grid=grid)["iota"], knots=rho)
    #eq.iota = iota_profile
    #eq.current = None
    #eq = solve_continuation_automatic(
    #        eq, objective="force", ftol=1e-3, xtol=1e-6, gtol=1e-6, maxiter=100, verbose=3
    #        )[-1]
    #eq.save(fname + f"{0:02d}_finite_beta_and_iota.h5")

    #print("LOADIN AN ALREADY SAVED EQ!!!!")
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

# Specified the equilibrium iota
eq.iota = SplineProfile(grid.compress(data["iota"]), knots=rho)



eq_weights = [1e3, 1e4, 1e5, 1e6, 1e7]
#surfaces_ball = [0.2, 0.3, 0.5, 0.65, 0.75, 0.83, 0.91]
surfaces_ball = [0.15, 0.3, 0.45, 0.60, 0.75, 0.85]
#surfaces_ball = [0.3, 0.5, 0.65, 0.75, 0.87, 0.97]
#surfaces_omni = [0.2, 0.4, 0.6, 0.8, 1.0]
surfaces_omni = [0.15, 0.3, 0.45, 0.6, 0.75, 0.9]
Mercier_weights = [1.2e3, 2.4e3, 5e3]
#Current_weights = [1e-4, 1e-4, 1e-4, 2e-4, 5e-4]

ntor = int(4)
M_ball = int(12)
N_ball = int(11)
N0 = int(2 * M_ball * N_ball * ntor + 1)

#optim_indices = np.array([1, 2, 3, 4, 5], dtype=int)
optim_indices = np.array([1, 2, 3, 4, 5], dtype=int)
#optim_indices = np.array([9], dtype=int)

# optimize with increasing resolution
for k in optim_indices:

    print("\n---------------------------------------")
    print(f"Optimizing boundary modes M, N <= {k}")
    print("---------------------------------------")
    
    objs_ball = {}
    
    eq_grids_omni = {}
    field_grids_omni = {}
    objs_omni = {}
    
    eq_ball_weight = 5.0e3
    omni_weight = 22e-0 + 3.5*k
    
    for i, rho in enumerate(surfaces_ball):
        shift_arr = np.random.default_rng().uniform(-0.1, 0.1, 13)
        alpha=np.array([[0.0], [0.25], [0.5], [0.75], [1.0], [1.25], [1.5], [1.75], [2.0], [2.25], [2.5], [2.75], [2.9], [3.1]])
        alpha[1:, :] = alpha[1:, :] + np.reshape(shift_arr, (-1, 1))
 
        if i == 1:
        	objs_ball[rho] = BallooningStability(eq=eq,rho=np.array([rho]), alpha=alpha, zetamax=ntor*jnp.pi, nzeta=N0, weight=eq_ball_weight*2)
        elif i == len(surfaces_ball)-2:
        	objs_ball[rho] = BallooningStability(eq=eq,rho=np.array([rho]), alpha=alpha, zetamax=ntor*jnp.pi, nzeta=N0, weight=eq_ball_weight*2)
        else:
        	objs_ball[rho] = BallooningStability(eq=eq,rho=np.array([rho]), alpha=alpha, zetamax=ntor*jnp.pi, nzeta=N0, weight=eq_ball_weight*2)
        
    for rho in surfaces_omni:
        eq_grids_omni[rho] = LinearGrid(rho=rho, M = int(42), N= int(42), NFP=eq.NFP, sym=False)
        #field_grids_omni[rho] = LinearGrid(rho=rho, theta=24, zeta=12, NFP=field.NFP, sym=False)
        field_grids_omni[rho] = LinearGrid(rho=rho, theta=np.linspace(0, 2*np.pi, 24), zeta=np.linspace(0.00, 2*np.pi, 14), NFP=field.NFP, sym=False)
        objs_omni[rho] = Omnigenity(
            field=field,
            eq=eq,
            eq_grid=eq_grids_omni[rho],
            field_grid=field_grids_omni[rho],
            eta_weight=well_weight,
            weight = omni_weight, 
        )
    
    Mercier_grid = LinearGrid(M = int(3*M), N = int(3*N), rho=np.array([0.8, 0.85, 0.9, 0.93, 0.96, 0.97,0.98, 0.985, 0.99, 0.995,  1.0]), NFP=eq.NFP, sym=True, axis=False)

    Elongation_grid = LinearGrid(M = 2*int(M), N = 2*int(N), rho=np.array([1.0]), NFP=eq.NFP, sym=True, axis=False)
    
    Curvature_grid = LinearGrid(M = 3*int(M), N = 3*int(N), rho=np.array([1.0]), NFP=eq.NFP, sym=True, axis=False)

    Iota_grid = LinearGrid(M = 2*int(M), N = 2*int(N), rho=np.array([1.0]), NFP=eq.NFP, sym=True, axis=False)

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
 
   
    objective = ObjectiveFunction((MercierStability(eq=eq, grid=Mercier_grid, bounds=(40, np.inf), weight=Mercier_weight),  AspectRatio(eq=eq, bounds=(6, 7.5), weight=8e3), Elongation(eq=eq, grid=Elongation_grid, bounds=(0.5, 2.7), weight=1e3), GenericObjective(f="curvature_k2_rho", eq=eq, grid=Curvature_grid, bounds=(-110, 10), weight=2e3)) + tuple(objs_ball.values()) + tuple(objs_omni.values()))
    
    constraints = (
            ForceBalance(eq=eq, weight=eq_weight),
            FixBoundaryR(eq=eq, modes=modes_R),
            FixBoundaryZ(eq=eq, modes=modes_Z),
            FixPressure(eq=eq),
            FixIota(eq=eq),
            FixPsi(eq=eq),
            FixOmniBmax(field=field, weight=2),
            FixOmniMap(field=field, indices=np.concatenate((idx, np.where(field.x_basis.modes[:, 1] == 0)[0]))),
    	    # fix the mirror ratio on the magnetic axis
            LinearObjectiveFromUser(mirrorRatio, field, target=[0.7, 1.3]),
            )

    optimizer = Optimizer("proximal-lsq-exact")
    (eq, field), _ = optimizer.optimize(
                (eq, field),
                objective,
                constraints,
                ftol=1e-3,
                xtol=1e-6,
                gtol=1e-6,
                maxiter=50,
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

    print("target_mode =", field.x_lmn[37])

    jax.clear_caches()   
##eq.change_resolution(L=14, M= 14, N=14, L_grid=24, M_grid=24, N_grid=28)
##objective = ObjectiveFunction(ForceBalance(eq), deriv_mode="looped")
##constraints = get_fixed_boundary_constraints(eq=eq, profiles=True)
##eq, _ = eq.solve(objective=objective, constraints=constraints, ftol=1e-3, xtol=1e-5, gtol=1e-5, maxiter=70, verbose=3, copy=True)
##
eq.change_resolution(L=14, M= 14, N=14, L_grid=28, M_grid=28, N_grid=28)
eq = solve_continuation_automatic(eq=eq, objective="force", ftol=1e-3, xtol=1e-5, gtol=1e-5, maxiter=100, verbose=3, pres_step=0.2, bdry_step=0.125)[-1]


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
M_grid = 14
N_grid = 14
ntor = int(3)
N0 = int(2 * M_grid * N_grid * ntor + 1)

eq_data_keys = ["iota", "a", "psi", "Psi"]

eq_coords = np.zeros((len(surfaces_ball), 3))
eq_coords[:, 0] = np.array([surfaces_ball])

eq_data_grid = Grid(eq_coords)
eq_data = eq.compute(eq_data_keys, grid=eq_data_grid)
# Now we compute theta_DESC for given theta_PEST
iota = eq_data["iota"]

#alpha = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75])
alpha = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 2.9, 3.1])

Nalpha = len(alpha)
Nzeta0 = int(13)

ball_data = np.zeros((len(surfaces_ball), Nalpha, Nzeta0))
gamma_max = np.zeros(len(surfaces_ball), )

for i in range(3):
    for j in range(len(surfaces_ball)):
        rho = surfaces_ball[j]*np.ones((N0*Nalpha, ))
        zeta = np.linspace(-jnp.pi*ntor, jnp.pi*ntor, N0)
        theta_PEST = np.zeros((N0*Nalpha,))
        zeta_full = np.zeros((N0*Nalpha, ))
        for k in range(Nalpha):
            theta_PEST[N0*k:N0*(k+1)] = alpha[k] + iota[j] * zeta
            zeta_full[N0*k:N0*(k+1)] = zeta
            
        theta_coords = jnp.array([rho, theta_PEST, zeta_full]).T
        desc_coords = compute_theta_coords(
            eq, theta_coords, L_lmn=eq.L_lmn, tol=1e-10, maxiter=40
        )
    
        sfl_grid = Grid(desc_coords, sort=False)
        ball_data0 = eq.compute(["ideal_ball_gamma2"], grid=sfl_grid, override_grid=False)["ideal_ball_gamma2"]
        
        gamma_max[j] = np.max(ball_data0)
        print(f"surf number {j} done!")
    
    print(f"Ballooning gamma objective {i+1} = ", gamma_max)
    print("Shuffling alpha and recalculating the growth rates...\n")

    alpha = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 2.9, 3.1])
    shift_arr = np.random.default_rng().uniform(-0.1, 0.1, 13)
    alpha=np.array([[0.0], [0.25], [0.5], [0.75], [1.0], [1.25], [1.5], [1.75], [2.0], [2.25], [2.5], [2.75], [2.9], [3.1]])
    alpha[1:, :] = alpha[1:, :] + np.reshape(shift_arr, (-1, 1))


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


