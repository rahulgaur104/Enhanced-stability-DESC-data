#!/usr/bin/env python
"""

This script contains a bunch of functions that help calculate the geometric coefficients
and the ideal ballooning growth rate

"""
import pdb
import sys
import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse.linalg import eigs
from scipy.integrate import simps

#####################################################################################################################
########################--------------------BALLOONING SOLVER FUNCTIONS-------------------------#####################
#####################################################################################################################


def gamma_ball_full(
    dPdrho, theta_PEST, B, gradpar, cvdrift, gds2, vguess=None, sigma0=1.2):
    # Inputs  : geometric coefficients(normalized with a_N, and B_N)
    #           on an equispaced theta_PEST grid
    # Outputs : maximum ballooning growth rate gamma
    theta_ball = theta_PEST
    ntheta = len(theta_ball)

    # Note that gds2 is (dpsidrho*|grad alpha|/(a_N*B_N))**2.
    g = np.abs(gradpar) * gds2 / (B)
    c = -1 * dPdrho * cvdrift * 1 / (np.abs(gradpar) * B)
    f = gds2 / B**2 * 1 / (np.abs(gradpar) * B)

    len1 = len(g)

    ##Uniform half theta ball
    theta_ball_u = np.linspace(theta_ball[0], theta_ball[-1], len1)

    g_u = np.interp(theta_ball_u, theta_ball, g)
    c_u = np.interp(theta_ball_u, theta_ball, c)
    f_u = np.interp(theta_ball_u, theta_ball, f)

    # uniform theta_ball on half points with half the size, i.e., only from [0, (2*nperiod-1)*np.pi]
    theta_ball_u_half = (theta_ball_u[:-1] + theta_ball_u[1:]) / 2
    h = np.diff(theta_ball_u_half)[2]
    g_u_half = np.interp(theta_ball_u_half, theta_ball, g)
    g_u1 = g_u[:]
    c_u1 = c_u[:]
    f_u1 = f_u[:]

    len2 = int(len1) - 2
    A = np.zeros((len2, len2))

    A = (
        np.diag(g_u_half[1:-1] / f_u1[2:-1] * 1 / h**2, -1)
        + np.diag(
            -(g_u_half[1:] + g_u_half[:-1]) / f_u1[1:-1] * 1 / h**2
            + c_u1[1:-1] / f_u1[1:-1],
            0,
        )
        + np.diag(g_u_half[1:-1] / f_u1[1:-2] * 1 / h**2, 1)
    )

    # Method without M is approx 3 X faster with Arnoldi iteration
    # Perhaps, we should try dstemr as suggested by Max Ruth. However, I doubt if
    # that will give us a significant speedup
    w, v = eigs(A, 1, sigma=sigma0, v0=vguess, tol=3.0e-7, OPpart="r")
    # w, v  = eigs(A, 1, sigma=1.0, tol=1E-6, OPpart='r')
    # w, v  = eigs(A, 1, sigma=1.0, tol=1E-6, OPpart='r')

    ### Richardson extrapolation
    X = np.zeros((len2 + 2,))
    dX = np.zeros((len2 + 2,))
    # X[1:-1]     = np.reshape(v[:, idx_max].real, (-1,))/np.max(np.abs(v[:, idx_max].real))
    X[1:-1] = np.reshape(v[:, 0].real, (-1,)) / np.max(np.abs(v[:, 0].real))

    X[0] = 0.0
    X[-1] = 0.0

    dX[0] = (-1.5 * X[0] + 2 * X[1] - 0.5 * X[2]) / h
    dX[1] = (X[2] - X[0]) / (2 * h)

    dX[-2] = (X[-1] - X[-3]) / (2 * h)
    dX[-1] = (0.5 * X[-3] - 2 * X[-2] + 1.5 * 0.0) / (h)

    dX[2:-2] = 2 / (3 * h) * (X[3:-1] - X[1:-3]) - (X[4:] - X[0:-4]) / (12 * h)

    Y0 = -g_u1 * dX**2 + c_u1 * X**2
    Y1 = f_u1 * X**2
    # plt.plot(range(len3+2), X, range(len3+2), dX); plt.show()
    gam = simps(Y0) / simps(Y1)

    # return np.sign(gam)*np.sqrt(abs(gam)), X, dX, g_u1, c_u1, f_u1
    # return gam, X, dX, g_u1, c_u1, f_u1
    return gam


def check_ball_full(dPdrho, theta_PEST, B, gradpar, cvdrift, gds2):
    # Inputs  : geometric coefficients(normalized with a_N, and B_N)
    #           on an equispaced theta_PEST grid
    # Outputs : 1 if an equilibrium is unstable, 0 if it's stable

    theta_ball = theta_PEST
    ntheta = len(theta_ball)

    # Note that gds2 is (dpsidrho*|grad alpha|/(a_N*B_N))**2.
    g = np.abs(gradpar) * gds2 / (B)
    c = -1 * dPdrho * cvdrift * 1 / (np.abs(gradpar) * B)
    f = gds2 / B**2 * 1 / (np.abs(gradpar) * B)

    len1 = len(g)

    ##Uniform half theta ball
    theta_ball_u = np.linspace(theta_ball[0], theta_ball[-1], len1)

    g_u = np.interp(theta_ball_u, theta_ball, g)
    c_u = np.interp(theta_ball_u, theta_ball, c)
    f_u = np.interp(theta_ball_u, theta_ball, f)

    delthet = np.diff(theta_ball_u)

    ch = np.zeros((ntheta,))
    gh = np.zeros((ntheta,))
    fh = np.zeros((ntheta,))

    diff = 0.0
    one_m_diff = 1 - diff

    for i in np.arange(1, ntheta):
        ch[i] = 0.5 * (c_u[i] + c_u[i - 1])
        gh[i] = 0.5 * (g_u[i] + g_u[i - 1])
        fh[i] = 0.5 * (f_u[i] + f_u[i - 1])

    cflmax = np.max(np.abs(delthet**2 * ch[1:] / gh[1:]))

    c1 = np.zeros((ntheta,))
    f1 = np.zeros((ntheta,))

    for ig in np.arange(1, ntheta - 1):
        c1[ig] = (
            -delthet[ig] * (one_m_diff * c_u[ig] + 0.5 * diff * ch[ig + 1])
            - delthet[ig - 1] * (one_m_diff * c_u[ig] + 0.5 * diff * ch[ig])
            - delthet[ig - 1] * 0.5 * diff * ch[ig]
        )
        c1[ig] = -delthet[ig] * (one_m_diff * c_u[ig]) - delthet[ig - 1] * (
            one_m_diff * c_u[ig]
        )
        f1[ig] = -delthet[ig] * (one_m_diff * f_u[ig]) - delthet[ig - 1] * (
            one_m_diff * f_u[ig]
        )
        c1[ig] = 0.5 * c1[ig]
        f1[ig] = 0.5 * f1[ig]

    c2 = np.zeros((ntheta,))
    f2 = np.zeros((ntheta,))
    g1 = np.zeros((ntheta,))
    g2 = np.zeros((ntheta,))

    for ig in np.arange(1, ntheta):
        c2[ig] = -0.25 * diff * ch[ig] * delthet[ig - 1]
        f2[ig] = -0.25 * diff * fh[ig] * delthet[ig - 1]
        g1[ig] = gh[ig] / delthet[ig - 1]
        g2[ig] = 1.0 / (
            0.25 * diff * ch[ig] * delthet[ig - 1] + gh[ig] / delthet[ig - 1]
        )

    psi_t = np.zeros((ntheta,))
    psi_t[1] = delthet[0]
    psi_prime = (psi_t[1] / g2[1]) * 0.5

    gamma = 0
    # for ig in np.arange(int((ntheta-1)/2),ntheta-1):
    for ig in np.arange(1, ntheta - 1):
        # pdb.set_trace()
        psi_prime = (
            psi_prime
            + 1 * c1[ig] * psi_t[ig]
            + c2[ig] * psi_t[ig - 1]
            + gamma * (f1[ig] * psi_t[ig] + f2[ig] * psi_t[ig - 1])
        )
        psi_t[ig + 1] = (g1[ig + 1] * psi_t[ig] + psi_prime) * g2[ig + 1]

    # pdb.set_trace()
    if np.isnan(np.sum(psi_t)) or np.isnan(np.abs(psi_prime)):
        print("warning NaN  balls")

    isunstable = 0
    for ig in np.arange(1, ntheta - 1):
        if psi_t[ig] * psi_t[ig + 1] <= 0:
            isunstable = 1
            # print("instability detected... please choose a different equilibrium")
    plt.plot(psi_t)
    plt.show()
    pdb.set_trace()
    return isunstable



def check_ball_full2(dPdrho, theta_PEST, B, gradpar, cvdrift, gds2):
    # Inputs  : geometric coefficients(normalized with a_N, and B_N)
    #           on an equispaced theta_PEST grid
    # Outputs : maximum ballooning growth rate gamma
    theta_ball = theta_PEST
    ntheta = len(theta_ball)

    # Note that gds2 is (dpsidrho*|grad alpha|/(a_N*B_N))**2.
    g = np.abs(gradpar) * gds2 / (B)
    c = -1 * dPdrho * cvdrift * 1 / (np.abs(gradpar) * B)
    f = gds2 / B**2 * 1 / (np.abs(gradpar) * B)

    len1 = len(g)

    ##Uniform half theta ball
    theta_ball_u = np.linspace(theta_ball[0], theta_ball[-1], len1)

    g_u = np.interp(theta_ball_u, theta_ball, g)
    c_u = np.interp(theta_ball_u, theta_ball, c)
    f_u = np.interp(theta_ball_u, theta_ball, f)

    # uniform theta_ball on half points with half the size, i.e., only from [0, (2*nperiod-1)*np.pi]
    theta_ball_u_half = (theta_ball_u[:-1] + theta_ball_u[1:]) / 2
    h = np.diff(theta_ball_u_half)[2]
    g_u_half = np.interp(theta_ball_u_half, theta_ball, g)
    g_u1 = g_u[:]
    c_u1 = c_u[:]
    f_u1 = f_u[:]

    len2 = int(len1) - 2
    A = np.zeros((len2, len2))
    b = np.zeros((len2, ))

    A = (
        np.diag(g_u_half[1:-1] / f_u1[2:-1] * 1 / h**2, -1)
        + np.diag(
            -(g_u_half[1:] + g_u_half[:-1]) / f_u1[1:-1] * 1 / h**2
            + c_u1[1:-1] / f_u1[1:-1],
            0,
        )
        + np.diag(g_u_half[1:-1] / f_u1[1:-2] * 1 / h**2, 1)
    )
    eps = 1e-3
    #b[0] = eps * g_u_half[0] / f_u1[1] * 1 / h**2
    b[0] = eps

    psi_t = np.linalg.inv(A) @ b
    ig_cross = []
    isunstable = 0

    for ig in range(len(psi_t)-1):
        if psi_t[ig] * psi_t[ig + 1] <= 0:
            isunstable = 1
            ig_cross.append(ig)
    
    if isunstable == 1:
        ig_cross_idx = ig_cross[0]
        newcomb_metric = (theta_ball_u[ig_cross_idx+1] - theta_ball_u[0])/(2*theta_ball_u[-1])
    else:
        newcomb_metric = np.tanh(np.cbrt(psi_t[-1])*21)
        #newcomb_metric = abs(psi_t[-1])/theta_ball_u[-1]
        
    plt.plot(psi_t)
    plt.show()
    pdb.set_trace()
    return isunstable, newcomb_metric









