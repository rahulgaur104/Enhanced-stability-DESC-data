#!/usr/bin/env python3
"""
This script plots the |B| contours on the plasma boundary in Boozer coordinates
"""
import pdb
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker

from desc.equilibrium import Equilibrium
from desc.grid import LinearGrid

from scipy.interpolate import griddata

from desc.plotting import *

keyword_arr = ["OT", "OH", "OP"]
# keyword_arr = ["OT", "OH"]
#keyword_arr = ["OP"]

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

    N = int(200)
    grid = LinearGrid(L=N)
    rho = np.linspace(0, 1, N + 1)

    data_keys = ["iota", "D_Mercier"]

    data0 = eq0.compute(data_keys, grid=grid)
    data1 = eq1.compute(data_keys, grid=grid)

    iota = data0["iota"]

    rho0 = 1.0
    fig, ax, Boozer_data0 = plot_boozer_surface(eq0, rho=rho0, return_data=True)
    plt.close()

    fig, ax, Boozer_data1 = plot_boozer_surface(eq1, rho=rho0, return_data=True)
    plt.close()

    fig, ax, Boozer_data2 = plot_boozer_surface(
        field, rho=rho0, iota=np.interp(rho0, rho, iota), return_data=True
    )
    plt.close()

    np.save("theta_flattened", Boozer_data2["theta_B"])
    np.save("zeta_flattened", Boozer_data2["zeta_B"])
    np.save("B_flattened", Boozer_data2["|B|"])

    # Boozer_data_list = [Boozer_data0, Boozer_data1, Boozer_data2]
    Boozer_data_list = [Boozer_data0, Boozer_data1]

    for i, Boozer_data in enumerate(Boozer_data_list):

        print(i)
        theta_B0 = Boozer_data["theta_B"]
        zeta_B0 = Boozer_data["zeta_B"]
        B0 = Boozer_data["|B|"]

        Theta = theta_B0
        Zeta = zeta_B0

        if i == 2:  # Additional acrobatics to create target field plot
            points_linear = np.array([Theta, Zeta]).T

            # Note that higher-order interpolation or gaussian filtering leads to self-intersection
            # Interpolate the data onto a finer grid using linear interpolation
            theta_fine_linear = np.linspace(Theta.min(), Theta.max(), 150)
            zeta_fine_linear = np.linspace(Zeta.min(), Zeta.max(), 150)
            theta_fine_grid_linear, zeta_fine_grid_linear = np.meshgrid(
                theta_fine_linear, zeta_fine_linear
            )

            B_field = griddata(
                points_linear,
                B0,
                (theta_fine_grid_linear, zeta_fine_grid_linear),
                method="linear",
            )
            Theta, Zeta = theta_fine_grid_linear, zeta_fine_grid_linear

            B0 = B_field.copy()

            trim = 8
            B0_trim = B0[trim:-trim, trim:-trim]
            # Creating the contour plot and manually removing the overlapped nan values
            fig, ax = plt.subplots(figsize=(6, 5))
            if keyword == "OT":
                contour = ax.contour(
                    Zeta,
                    Theta,
                    B0,
                    levels=np.linspace(np.min(B0_trim), np.max(B0_trim), 30)[:-6],
                    cmap="jet",
                )  # Using 'jet' for simple color map
            else:
                # contour = ax.contour(Zeta, Theta, B0, levels=np.linspace(np.min(B0_trim), np.max(B0_trim), 31)[:-2], cmap='jet')  # Using 'jet' for simple color map
                contour = ax.contour(
                    Zeta,
                    Theta,
                    B0,
                    levels=np.linspace(np.min(B0_trim), np.max(B0_trim), 31)[:-1],
                    cmap="jet",
                )  # Using 'jet' for simple color map
                # ax.set_xticks(np.linspace(0, 2, 3))

        else:
            fig, ax = plt.subplots(figsize=(6, 5))
            contour = ax.contour(
                Zeta,
                Theta,
                B0,
                levels=np.linspace(np.min(B0), np.max(B0), 30)[:],
                cmap="jet",
            )

        # Adding a colorbar with larger font size
        cbar = fig.colorbar(contour, ax=ax, orientation="vertical")

        tick_locator = ticker.MaxNLocator(nbins=6)
        cbar.locator = tick_locator

        cbar.ax.tick_params(labelsize=18)  # Change colorbar tick size
        # cbar.set_label('|B| (T)', size=16)  # Colorbar title

        # Labeling axes
        ax.set_xlabel(r"$\zeta_{\mathrm{Boozer}}$", fontsize=26, labelpad=-4)
        ax.set_ylabel(r"$\theta_{\mathrm{Boozer}}$", fontsize=26, labelpad=3)
        ax.tick_params(axis="both", which="major", labelsize=20)  # Larger tick labels

        ## Adding a title and adjusting plot borders for better fit
        # ax.set_title('Contour Plot of |B| in Boozer Coordinates', fontsize=18)
        # plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)
        plt.tight_layout()

        # Increase resolution for publication quality
        if i == 0:
            #plt.savefig(
            #    f"Boozer_contours/Boozer_contour_plot_{keyword}_rho{rho0}_initial.png",
            #    dpi=300,
            #)
            plt.savefig(
                f"Boozer_contours/Boozer_contour_plot_{keyword}_rho{rho0}_initial.pdf",
                dpi=400,
            )
            # plt.show()
            plt.close()
        elif i == 1:
            #plt.savefig(
            #    f"Boozer_contours/Boozer_contour_plot_{keyword}_rho{rho0}_optimized.png",
            #    dpi=300,
            #)
            plt.savefig(
                f"Boozer_contours/Boozer_contour_plot_{keyword}_rho{rho0}_optimized.pdf",
                dpi=400,
            )
            # plt.show()
            plt.close()
        else:
            #plt.savefig(
            #    f"Boozer_contours/Boozer_contour_plot_{keyword}_rho{rho0}_target.png",
            #    dpi=300,
            #)
            plt.savefig(
                f"Boozer_contours/Boozer_contour_plot_{keyword}_rho{rho0}_target.pdf",
                dpi=400,
            )
            # plt.show()
            plt.close()
