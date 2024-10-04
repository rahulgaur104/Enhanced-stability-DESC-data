#!/usr/bin/env python3
"""
Read the max ballooning eigenvalue from a COBRAVMEC output file and compare
it with the DESC ballooning solver.
"""
import numpy as np
import pdb

from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from matplotlib import pyplot as plt

A = np.loadtxt("cobra_grate.HELIOTRON")

ns1 = int(A[0, 2])

nangles = int(np.shape(A)[0] / (ns1 + 1))
B = np.zeros((ns1,))
for i in range(nangles):
    if i == 0:
        B = A[i + 1 : (i + 1) * ns1 + 1, 2]
    else:
        B = np.vstack((B, A[i * ns1 + i + 1 : (i + 1) * ns1 + i + 1, 2]))

gamma1 = np.amax(B, axis=0)

s1 = np.linspace(0, 1, ns1)
s1 = s1 + np.diff(s1)[0]


# RG: gamma_max from my old ideal-ballooning-solver
# The constant factor arises due to an old normalization scheme
# If you rerun this with the current DESC, this factor will be very close to 1.
# Effectively, we are still only scaling the growth rates by a normalization
gamma2 = 17.0 * np.load("gamma_max_HELIOTRON_hres_correct_norm.npy")
s2 = (
    np.array(
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
    ** 2
)

gamma2 = gamma2

# sincel the normalizations are different, we scale the gamma from our
# solver with the ratio of mean positive gamma values from both results
gamma_scaling = np.max(gamma2) / np.max(gamma1)

print("scaling factor = ", gamma_scaling)

plt.plot(np.sqrt(s2), gamma2, "-ob", ms=2, linewidth=2)
plt.plot(np.sqrt(s1), gamma1 * gamma_scaling, "-or", ms=2, linewidth=2)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)


ax = plt.gca()
ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
ax.yaxis.set_major_locator(MaxNLocator(nbins=8))  # Reduce number of ticks

plt.xlabel(r"$\rho$", fontsize=20, labelpad=-0.1)
plt.ylabel(r"$\lambda (=\gamma^2)$", fontsize=20, labelpad=-1)
plt.legend(["DESC", "COBRAVMEC(scaled)"], loc="upper left", framealpha=0.7, fontsize=16)
plt.tight_layout()
plt.savefig("HELIOTRON_unstable_scaling_comparison.pdf", dpi=400)
plt.show()
