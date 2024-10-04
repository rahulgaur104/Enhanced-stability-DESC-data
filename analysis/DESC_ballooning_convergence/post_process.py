#!/usr/bin/env python3
from matplotlib import pyplot as plt
import numpy as np

surfaces = np.array(
    [
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
    ]
)

cmap = plt.get_cmap("RdBu")
colors = cmap(np.linspace(0, 1, 5))

y0 = np.load("gamma_max_HELIOTRON_lres_correct_norm.npy")
y1 = np.load("gamma_max_HELIOTRON_mres_correct_norm1.npy")
y2 = np.load("gamma_max_HELIOTRON_nres_correct_norm.npy")
y3 = np.load("gamma_max_HELIOTRON_hres_correct_norm.npy")
y4 = np.load("gamma_max_HELIOTRON_shres_correct_norm.npy")

print(y2)

plt.plot(surfaces, 17 * y0, "-or", ms=2, linewidth=2)
plt.plot(surfaces, 17 * y1, "-og", ms=2, linewidth=2)
plt.plot(surfaces, 17 * y2, "-ob", ms=2, linewidth=2)
plt.plot(surfaces, 17 * y3, "-om", ms=2, linewidth=2)
plt.plot(surfaces, 17 * y3, "-ok", ms=2, linewidth=2)

# plt.plot(surfaces, y0, color=colors[0], ms=2, linewidth=2)
# plt.plot(surfaces, y1, color=colors[1], ms=2, linewidth=2)
# plt.plot(surfaces, y2, color=colors[2], ms=2, linewidth=2)
# plt.plot(surfaces, y3, color=colors[3], ms=2, linewidth=2)
# plt.plot(surfaces, y3, color=colors[4], ms=2, linewidth=2)


plt.xlabel(r"$\rho$", fontsize=20)
plt.ylabel(r"$\lambda (= \gamma^2)$", fontsize=20, labelpad=-1)

plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

plt.legend(
    [
        "L=06, M=04, N=03",
        "L=12, M=08, N=06",
        "L=16, M=12, N=09",
        "L=24, M=16, N=12",
        "L=28, M=20, N=16",
    ],
    fontsize=15,
)
plt.tight_layout()

plt.savefig(f"gamma_resolution_scan_HELIOTRON_correct_norm.png", dpi=400)
# plt.savefig(f"gamma_resolution_scan_HELIOTRON_correct_norm.pdf", dpi=300)
