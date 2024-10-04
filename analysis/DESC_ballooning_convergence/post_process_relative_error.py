#!/usr/bin/env python3


import numpy as np
from matplotlib import pyplot as plt

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

y0 = np.load("gamma_max_HELIOTRON_lres_correct_norm.npy")  # low-res
y1 = np.load("gamma_max_HELIOTRON_mres_correct_norm1.npy")  # medium-res
y2 = np.load("gamma_max_HELIOTRON_nres_correct_norm.npy")  # normal-res
y3 = np.load("gamma_max_HELIOTRON_hres_correct_norm.npy")  # high-res
y4 = np.load("gamma_max_HELIOTRON_shres_correct_norm.npy")  # super-high-res


err0 = np.linalg.norm(y0 - y1)
err1 = np.linalg.norm(y1 - y2)
err2 = np.linalg.norm(y2 - y3)
err3 = np.linalg.norm(y3 - y4)

error_array = np.array([err0, err1, err2, err3])

plt.plot(np.arange(1, 5).astype(int), error_array)

plt.yscale("log")

plt.xlabel(r"$\mathrm{resolution}$", fontsize=20)
plt.ylabel(r"$\mathrm{relative error}$", fontsize=20, labelpad=-2)

plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

plt.tight_layout()

plt.savefig(f"relative_error_HELIOTRON.png", dpi=300)
