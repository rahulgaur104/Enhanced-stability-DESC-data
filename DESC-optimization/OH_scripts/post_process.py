#!/usr/bin/env python3

import numpy as np
from desc.equilibrium import Equilibrium
from desc.grid import LinearGrid
from matplotlib import pyplot as plt
from desc.magnetic_fields import OmnigenousField

from desc.plotting import *
import pdb

eq0 = Equilibrium.load("eq00_finite_beta_and_iota.h5")
eq = Equilibrium.load("eq_final.h5")

grid = LinearGrid(L=200)
rho = np.linspace(0, 1, 201)
iota = eq.compute("iota", grid=grid)["iota"]

field = OmnigenousField.load("field_final.h5")

#pdb.set_trace()

rho0 = 0.9

#fig, ax = plot_boozer_surface(eq0, rho=rho0)
#plt.show()

fig, ax = plot_boozer_surface(eq, rho=rho0)
plt.show()

fig, ax = plot_boozer_surface(field, rho=rho0, iota=np.interp(rho0, rho, iota))
plt.show()


#fig, ax = plot_section(eq, "|F|", norm_F=True, log=True);
##plt.savefig(f"modF.png", dpi=400)
##plt.close()
#plt.show()
#
#
##fig, ax = plot_section(eq, "|B|", norm_F=True, log=True);
##plt.savefig(f"modB.png", dpi=400)
##plt.close()
#
#
#plt.figure()
#fig, ax = plot_comparison(eqs=[eq0, eq])
#plt.savefig(f"Xsection.png", dpi=400)
#plt.close()
#
#
#theta_grid = np.linspace(0, 2*np.pi, 300)
#zeta_grid = np.linspace(0, 2*np.pi, 300)
#grid = LinearGrid(rho = 1.0, theta=theta_grid, zeta=zeta_grid)
#fig = plot_3d(eq, name="|B|", grid=grid)
#fig.write_html(f"modB.html")
#plt.close()
#
