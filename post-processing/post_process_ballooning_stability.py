#!/usr/bin/env python3

import os
import pdb
import sys
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.ticker import LogLocator, ScalarFormatter
import matplotlib as mpl

#mpl.rc("font", family="DejaVu Serif")


keyword_arr = ["OH", "OP", "OT"]

for keyword in keyword_arr:
    if keyword == "OP":
        fname_path0 = (
            os.path.dirname(os.getcwd())
            + "/analysis/DESC_ballooning_analysis/OP_ball3_scans/gamma_max_OP_ball3_H100_hres_correct_norm_initial_new_033_wider_zeta1_narrow_zeta0.npz"
        )
        fname_path1 = (
            os.path.dirname(os.getcwd())
            + "/analysis/DESC_ballooning_analysis/OP_ball3_scans/gamma_max_OP_ball3_H100_hres_correct_norm_optimized_new_033_wider_zeta1_narrow_zeta0.npz"
        )
    elif keyword == "OH":
        fname_path0 = (
            os.path.dirname(os.getcwd())
            + "/analysis/DESC_ballooning_analysis/OH_ball5_scans/gamma_max_OH_ball5_hres_correct_norm_initial_new_001_wider_zeta1_wider_zeta0.npz"
        )
        fname_path1 = (
            os.path.dirname(os.getcwd())
            + "/analysis/DESC_ballooning_analysis/OH_ball5_scans/gamma_max_OH_ball5_hres_correct_norm_optimized_new_001_wider_zeta1_wider_zeta0.npz"
        )
    else:
        fname_path0 = (
            os.path.dirname(os.getcwd())
            + "/analysis/DESC_ballooning_analysis/OT_ball_scans/gamma_max_OT_ball_hres_correct_norm_initial_new_022_wider_zeta1_wider_zeta0.npz"
        )
        fname_path1 = (
            os.path.dirname(os.getcwd())
            + "/analysis/DESC_ballooning_analysis/OT_ball_scans/gamma_max_OT_ball_hres_correct_norm_optimized_new_022_wider_zeta1_wider_zeta0.npz"
        )

    files_list = [fname_path0, fname_path1]

    plt.figure(figsize=(6, 5))


    dict0 = np.load(f"{fname_path0}")
    dict1 = np.load(f"{fname_path1}")

    rho_arr =  dict0["rho"]

    gamma_arr0 = dict0["lambda_max"] + 1e-10
    gamma_arr1 = dict1["lambda_max"] + 1e-10


    plt.plot(rho_arr, gamma_arr0, "-or", ms=2, linewidth=2)
    plt.plot(rho_arr, gamma_arr1, "-og", ms=2, linewidth=2)
    plt.xlabel(r"$\rho$", fontsize=26)
    plt.ylabel(r"$\lambda$", fontsize=26)

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(["initial", "optimized"], fontsize=20)

    plt.tight_layout()
    plt.savefig(
        f"ballooning_stability/growth_rate_comparison_plot_{keyword}.pdf", dpi=400
    )
    plt.close()
