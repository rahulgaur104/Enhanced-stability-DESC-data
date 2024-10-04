#!/usr/bin/env python3
from matplotlib import pyplot as plt
import numpy as np

surfaces = np.array([
    0.05,
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.35,
    0.4,
    0.45,
    0.55,
    0.6,
    0.65,
    0.67,
    0.69,
    0.7,
    0.75,
    0.8,
    0.9,
    0.95,
    0.98,
    0.99,
    1.0,
])

y0 = np.load("gamma_max_OT_ball_hres_correct_norm_initial_wider_zeta1.npy")
y1 = np.load("gamma_max_OT_ball_hres_correct_norm_optimized_wider_zeta1.npy")

plt.plot(surfaces, y0, '-or', ms=2, linewidth=2)
plt.plot(surfaces, y1, '-og', ms=2, linewidth=2)

plt.xlabel(r"$\rho$", fontsize=26)
plt.ylabel(r"$\lambda (= \gamma^2)$", fontsize=26, labelpad=-2)

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.legend(["initial", "optimized"], fontsize=20)
plt.tight_layout()

#plt.savefig(f"gamma_comparison_OT_lower_range.png", dpi=400)
plt.savefig(f"gamma_comparison_OT_wider_range.png", dpi=400)


