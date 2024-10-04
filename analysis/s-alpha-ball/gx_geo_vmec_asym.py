#!/usr/bin/env python
"""
This a Pythonized geometry module to read tokamak and stellarator equilibria from a VMEC file and caculate the geometric coefficients needed for a GX/GS2 run. Additionally, this module can vary the pressure and iota gradients self-consistently (while respecting MHD force balance) according to the work by Greene and Chance + Hegna and Nakajima and recalculate the geometry coefficients.

Dependencies:
netcdf4, pip install netcdf4
booz_xform, pip install booz_xform

For axisymmetric equilibria
python gx_geo_vmec.py <vmec_filename(with .nc)> 1 <desired output name>

For 3D equilibria
python gx_geo_vmec.py <vmec_filename(with .nc)> 0 <desired output name>

A portion of this script is based on Matt Landreman's vmec_geometry module for the SIMSOPT framework.
For axisymmetric equilibria, make sure that ntor > 1 in the VMEC wout file.
"""

import numpy as np
import sys
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.integrate import cumulative_trapezoid as ctrap
from scipy.integrate import simpson as simps
from netCDF4 import Dataset as ds

import pdb

vmec_fname = sys.argv[1]
axisym = int(eval(sys.argv[2]))
eikfile = sys.argv[3]

mu_0 = 4 * np.pi * (1.0e-7)

################################################################################
########--------------------HELPER FUNCTIONS------------------------############
################################################################################


def nperiod_set(arr, npol, extend=True, brr=None):
    """
    Contract or extend a large array to a smaller one.

    Truncates (or extends) an array to a smaller one. This function gives us the ability to truncate a variable to theta in [-pi, pi].

    Inputs
    ------
    arr: numpy array
    Input array of the dependent variable
    brr: numpy array
    Input array of the independent variable
    extend: boolean
    Whether to extend instead of contract a large array

    Returns
    -------
    A truncated or extended array arr
    """
    if extend is True and npol > 1:
        arr_temp0 = arr - (arr[0] + npol * np.pi)
        arr_temp1 = arr_temp0
        for i in np.arange(1, npol):
            arr_temp1 = np.concatenate((arr_temp1, arr_temp0[1:] + 2 * np.pi * i))

        arr = arr_temp1
    elif brr is None:  # contract the theta array
        eps = 1e-11
        arr_temp0 = arr[arr <= npol * np.pi + eps]
        arr_temp0 = arr_temp0[arr_temp0 >= -npol * np.pi - eps]
        arr = arr_temp0
    else:  # contract the non-theta array using the theta array brr
        eps = 1e-11
        arr_temp0 = arr[brr <= npol * np.pi + eps]
        brr_temp0 = brr[brr <= npol * np.pi + eps]
        arr_temp0 = arr_temp0[brr_temp0 >= -npol * np.pi - eps]
        brr_temp0 = brr_temp0[brr_temp0 >= -npol * np.pi - eps]
        arr = arr_temp0

    return arr



#################################################################################
##############---------------EQUILIBRIUM CALC.------------------#################
#################################################################################


class Struct:
    """
    This class is just a dummy mutable object to which we can add attributes.
    """


def vmec_splines(nc_obj, booz_obj, stellasym=False):
    """
    Initialize radial splines for a VMEC equilibrium.

    Args:
        vmec: a netCDF object

    Returns:
        A structure with the splines as attributes.
    """
    results = Struct()

    rmnc_b = []
    zmns_b = []
    numns_b = []

    d_rmnc_b_d_s = []
    d_zmns_b_d_s = []
    d_numns_b_d_s = []

    ns = nc_obj.variables["ns"][:].data
    s_full_grid = np.linspace(0, 1, ns)
    s_half_grid = s_full_grid[1:] - 0.5 * np.diff(s_full_grid)[0]

    # Boozer quantities are calculated on the half grid by booz_xform
    for jmn in range(int(booz_obj.mnboz)):
        rmnc_b.append(
            InterpolatedUnivariateSpline(s_half_grid, booz_obj.rmnc_b.T[:, jmn])
        )
        zmns_b.append(
            InterpolatedUnivariateSpline(s_half_grid, booz_obj.zmns_b.T[:, jmn])
        )
        numns_b.append(
            InterpolatedUnivariateSpline(s_half_grid, booz_obj.numns_b.T[:, jmn])
        )

        d_rmnc_b_d_s.append(rmnc_b[-1].derivative())
        d_zmns_b_d_s.append(zmns_b[-1].derivative())
        d_numns_b_d_s.append(numns_b[-1].derivative())

    gmnc_b = []
    bmnc_b = []
    d_bmnc_b_d_s = []

    for jmn in range(int(booz_obj.mnboz)):
        gmnc_b.append(
            InterpolatedUnivariateSpline(s_half_grid, booz_obj.gmnc_b.T[:, jmn])
        )
        bmnc_b.append(
            InterpolatedUnivariateSpline(s_half_grid, booz_obj.bmnc_b.T[:, jmn])
        )
        d_bmnc_b_d_s.append(bmnc_b[-1].derivative())


    if stellasym:
        # Creating lists for stellarator asymmetric quantities
        rmns_b = []
        zmnc_b = []
        numnc_b = []

        d_rmns_b_d_s = []
        d_zmnc_b_d_s = []
        d_numnc_b_d_s = []
        gmns_b = []
        bmns_b = []
        d_bmns_b_d_s = []
        # Boozer quantities are calculated on the half grid by booz_xform
        for jmn in range(int(booz_obj.mnboz)):
            rmns_b.append(
                InterpolatedUnivariateSpline(s_half_grid, booz_obj.rmns_b.T[:, jmn])
            )
            zmnc_b.append(
                InterpolatedUnivariateSpline(s_half_grid, booz_obj.zmnc_b.T[:, jmn])
            )
            numnc_b.append(
                InterpolatedUnivariateSpline(s_half_grid, booz_obj.numnc_b.T[:, jmn])
            )

            d_rmns_b_d_s.append(rmns_b[-1].derivative())
            d_zmnc_b_d_s.append(zmnc_b[-1].derivative())
            d_numnc_b_d_s.append(numnc_b[-1].derivative())


        for jmn in range(int(booz_obj.mnboz)):
            gmns_b.append(
                InterpolatedUnivariateSpline(s_half_grid, booz_obj.gmns_b.T[:, jmn])
            )
            bmns_b.append(
                InterpolatedUnivariateSpline(s_half_grid, booz_obj.bmns_b.T[:, jmn])
            )
            d_bmns_b_d_s.append(bmns_b[-1].derivative())



    results.Gfun = InterpolatedUnivariateSpline(s_half_grid, booz_obj.Boozer_G)
    results.Ifun = InterpolatedUnivariateSpline(s_half_grid, booz_obj.Boozer_I)

    # Useful 1d profiles:
    results.pressure = InterpolatedUnivariateSpline(
        s_half_grid, nc_obj.variables["pres"][1:]
    )
    results.d_pressure_d_s = results.pressure.derivative()
    results.psi = InterpolatedUnivariateSpline(
        s_half_grid, nc_obj.variables["phi"][1:] / (2 * np.pi)
    )
    results.d_psi_d_s = results.psi.derivative()
    results.iota = InterpolatedUnivariateSpline(
        s_half_grid, nc_obj.variables["iotas"][1:]
    )
    results.d_iota_d_s = results.iota.derivative()

    # Save other useful quantities:
    results.phiedge = nc_obj.variables["phi"][-1].data
    variables = ["Aminor_p", "nfp", "raxis_cc", "mpol", "ntor"]
    for v in variables:
        results.__setattr__(v, eval("nc_obj.variables['" + v + "'][:].data"))

    variables1 = ["xm_b", "xn_b", "xm_nyq_b", "xn_nyq_b", "mnbooz", "mboz", "nboz"]
    variables2 = ["xm_b", "xn_b", "xm_b", "xn_b", "mnboz", "mboz", "nboz"]
    for k, v in enumerate(variables1):
        results.__setattr__(v, eval("booz_obj." + variables2[k]))

    if stellasym:
    	variables = [
    	    "rmnc_b",
    	    "zmns_b",
    	    "numns_b",
    	    "d_rmnc_b_d_s",
    	    "d_zmns_b_d_s",
    	    "d_numns_b_d_s",
    	    "gmnc_b",
    	    "bmnc_b",
    	    "d_bmnc_b_d_s",
    	    "rmns_b",
    	    "zmnc_b",
    	    "numnc_b",
    	    "d_rmns_b_d_s",
    	    "d_zmnc_b_d_s",
    	    "d_numnc_b_d_s",
    	    "gmns_b",
    	    "bmns_b",
    	    "d_bmns_b_d_s",
    	]
    else:
    	variables = [
    	    "rmnc_b",
    	    "zmns_b",
    	    "numns_b",
    	    "d_rmnc_b_d_s",
    	    "d_zmns_b_d_s",
    	    "d_numns_b_d_s",
    	    "gmnc_b",
    	    "bmnc_b",
    	    "d_bmnc_b_d_s",
    	]
    for v in variables:
        results.__setattr__(v, eval(v))

    return results


#########################################################################################################
#######################------------------GEOMETRY CALCULATION FUN--------------------####################
#########################################################################################################


def vmec_fieldlines(
    vmec_fname,
    booz_obj,
    s,
    alpha,
    theta1d=None,
    phi1d=None,
    phi_center=0,
    sfac=1,
    pfac=1,
    axisym=False,
    stellasym=False,
    res_theta=201,
    res_phi=201,
):
    """
    Geometry routine for GX/GS2.

    Takes in a 1D theta or phi array in boozer coordinates, an array of flux surfaces,
    another array of field line labels and generates the coefficients needed for a local
    stability analysis.
    Additionally, this routinerecalculates the geometric coefficients if the user to wants to vary self-consistently the local pressure gradient and average shear and.
    Inputs:
    ------
    s: List or numpy array
    The normalized toroidal flux psi/psi_boundary
    alpha: list or numpy array
    alpha = theta_b - iota * phi_b is the field line label
    theta1d: numpy array
    Boozer theta
    phi1d: numpy array
    Boozer phi
    sfac: float
    Local variation of the average shear, Geometry is calculated for a total shear of shear*sfac.
    So if want to calculate geometry at 2.5 x nominal shear, sfac = 2.5.
    sfac: float
    Local variation of the pressure gradient. Geometry is calculated for a pressure gradient of dpds*pfac

    Outputs
    -------
    gds22: numpy array
    Flux expansion term
    gds21 numpy array
    Integrated local shear
    gds2: numpy array
    Field line bending
    bmag: numpy array
    normalized magnetic field strength
    gradpar: numpy array
    Parallel gradient b dot grad phi
    gbdrift: numpy array
    Grad-B drift geometry factor
    cvdrift: numpy array
    Curvature drift geometry factor
    cvdrift0: numpy array
    theta_PEST: numpy array
    theta_PEST for the given theta_b array.
    theta_geo: numpy array
    geometric (arctan) theta for the given theta_b array.
    """
    nc_obj = ds(vmec_fname, "r")

    mpol = nc_obj.variables["mpol"][:].data
    ntor = nc_obj.variables["ntor"][:].data

    vs = vmec_splines(nc_obj, booz_obj, stellasym=stellasym)

    # Make sure s is an array:
    try:
        ns = len(s)
    except:
        s = [s]
    s = np.array(s)
    ns = len(s)

    # Make sure alpha is an array
    # For axisymmetric equilibria, all field lines are identical, i.e., your choice of alpha doesn't matter
    try:
        nalpha = len(alpha)
    except:
        alpha = [alpha]
    alpha = np.array(alpha)
    nalpha = len(alpha)

    if (theta1d is not None) and (phi1d is not None):
        raise ValueError("You cannot specify both theta and phi")
    if (theta1d is None) and (phi1d is None):
        raise ValueError("You must specify either theta or phi")
    if theta1d is None:
        nl = len(phi1d)
    else:
        nl = len(theta1d)

    # Now that we have an s grid, evaluate everything on that grid:
    d_pressure_d_s = vs.d_pressure_d_s(s)
    d_psi_d_s = vs.d_psi_d_s(s)
    iota = vs.iota(s)
    d_iota_d_s = vs.d_iota_d_s(s)
    shat = (-2 * s / iota) * d_iota_d_s  # depends on the definitn of rho
    sqrt_s = np.sqrt(s)

    L_reference = vs.Aminor_p

    edge_toroidal_flux_over_2pi = -vs.phiedge / (2 * np.pi)
    toroidal_flux_sign = np.sign(edge_toroidal_flux_over_2pi)
    B_reference = 2 * abs(edge_toroidal_flux_over_2pi) / (L_reference * L_reference)

    xm_b = vs.xm_b
    xn_b = vs.xn_b
    mnmax_b = vs.mnbooz

    G = vs.Gfun(s)
    d_G_d_s = vs.Gfun.derivative()(s)
    I = vs.Ifun(s)
    d_I_d_s = vs.Ifun.derivative()(s)

    rmnc_b = np.zeros((ns, mnmax_b))
    zmns_b = np.zeros((ns, mnmax_b))
    numns_b = np.zeros((ns, mnmax_b))
    d_rmnc_b_d_s = np.zeros((ns, mnmax_b))
    d_zmns_b_d_s = np.zeros((ns, mnmax_b))
    d_numns_b_d_s = np.zeros((ns, mnmax_b))

    gmnc_b = np.zeros((ns, mnmax_b))
    bmnc_b = np.zeros((ns, mnmax_b))
    d_bmnc_b_d_s = np.zeros((ns, mnmax_b))

    delmnc_b = np.zeros((ns, mnmax_b))
    lambmnc_b = np.zeros((ns, mnmax_b))
    betamns_b = np.zeros((ns, mnmax_b))

    for jmn in range(mnmax_b):
        rmnc_b[:, jmn] = vs.rmnc_b[jmn](s)
        zmns_b[:, jmn] = vs.zmns_b[jmn](s)
        numns_b[:, jmn] = vs.numns_b[jmn](s)
        d_rmnc_b_d_s[:, jmn] = vs.d_rmnc_b_d_s[jmn](s)
        d_zmns_b_d_s[:, jmn] = vs.d_zmns_b_d_s[jmn](s)
        d_numns_b_d_s[:, jmn] = vs.d_numns_b_d_s[jmn](s)
        gmnc_b[:, jmn] = vs.gmnc_b[jmn](s)
        bmnc_b[:, jmn] = vs.bmnc_b[jmn](s)
        d_bmnc_b_d_s[:, jmn] = vs.d_bmnc_b_d_s[jmn](s)


    # Import stellarator asymmetric quantities

    rmns_b = np.zeros((ns, mnmax_b))
    zmnc_b = np.zeros((ns, mnmax_b))
    #numns_b = np.zeros((ns, mnmax_b))
    numnc_b = np.zeros((ns, mnmax_b))
    d_rmns_b_d_s = np.zeros((ns, mnmax_b))
    d_zmnc_b_d_s = np.zeros((ns, mnmax_b))
    d_numnc_b_d_s = np.zeros((ns, mnmax_b))

    numnc_b = np.zeros((ns, mnmax_b))
    gmns_b = np.zeros((ns, mnmax_b))
    bmns_b = np.zeros((ns, mnmax_b))
    d_bmns_b_d_s = np.zeros((ns, mnmax_b))

    delmns_b = np.zeros((ns, mnmax_b))
    lambmns_b = np.zeros((ns, mnmax_b))
    betamnc_b = np.zeros((ns, mnmax_b))

    if stellasym:
        for jmn in range(mnmax_b):
            rmns_b[:, jmn] = vs.rmns_b[jmn](s)
            zmnc_b[:, jmn] = vs.zmnc_b[jmn](s)
            numnc_b[:, jmn] = vs.numnc_b[jmn](s)
            d_rmns_b_d_s[:, jmn] = vs.d_rmns_b_d_s[jmn](s)
            d_zmnc_b_d_s[:, jmn] = vs.d_zmnc_b_d_s[jmn](s)
            d_numnc_b_d_s[:, jmn] = vs.d_numnc_b_d_s[jmn](s)
            gmns_b[:, jmn] = vs.gmns_b[jmn](s)
            bmns_b[:, jmn] = vs.bmns_b[jmn](s)
            d_bmns_b_d_s[:, jmn] = vs.d_bmns_b_d_s[jmn](s)


    theta_b = np.zeros((ns, nalpha, nl))
    phi_b = np.zeros((ns, nalpha, nl))

    Vprime = np.zeros((ns, 1))

    if theta1d is None:
        # We are given phi_boozer. Compute theta_boozer
        for js in range(ns):
            phi_b[js, :, :] = phi1d[None, :]
            theta_b[js, :, :] = alpha[:, None] + iota[js] * (phi1d[None, :])
    else:
        # We are given theta_pest. Compute phi:
        for js in range(ns):
            theta_b[js, :, :] = theta1d[None, :]
            phi_b[js, :, :] = (theta1d[None, :] - alpha[:, None]) / iota[js]

    # Now that we know theta_boozer, compute all the geometric quantities
    angle_b = (
        xm_b[:, None, None, None] * (theta_b[None, :, :, :])
        - xn_b[:, None, None, None] * phi_b[None, :, :, :]
    )
    cosangle_b = np.cos(angle_b)
    sinangle_b = np.sin(angle_b)

    R_b = np.einsum("ij,jikl->ikl", rmnc_b, cosangle_b) + np.einsum("ij,jikl->ikl", rmns_b, sinangle_b)
    Z_b = np.einsum("ij,jikl->ikl", zmns_b, sinangle_b) + np.einsum("ij,jikl->ikl", zmnc_b, cosangle_b)

    flipit = 0.0

    if axisym:
        # if R is increasing AND Z is decreasing, we must be moving counter clockwise from
        # the inboard side, otherwise we need to flip the theta coordinate
        if R_b[0][0][0] > R_b[0][0][1] or Z_b[0][0][1] > Z_b[0][0][0]:
            flipit = 1
    else:  # we disable flipit
        flipit = 0

    R_mag_ax = vs.raxis_cc[0]

    #####################################################################################
    #####################------------BOOZER CALCULATIONS--------------###################
    #####################################################################################

    if flipit == 1:
        angle_b = (
            xm_b[:, None, None, None] * (theta_b[None, :, :, :] + np.pi)
            - xn_b[:, None, None, None] * phi_b
        )
    else:
        angle_b = (
            xm_b[:, None, None, None] * theta_b[None, :, :, :]
            - xn_b[:, None, None, None] * phi_b
        )

    cosangle_b = np.cos(angle_b)
    sinangle_b = np.sin(angle_b)
    mcosangle_b = xm_b[:, None, None, None] * cosangle_b
    ncosangle_b = xn_b[:, None, None, None] * cosangle_b
    msinangle_b = xm_b[:, None, None, None] * sinangle_b
    nsinangle_b = xn_b[:, None, None, None] * sinangle_b
    # Order of indices in cosangle_b and sinangle_b: mn_b, s, alpha, l
    # Order of indices in rmnc, bmnc, etc: s, mn_b
    R_b = np.einsum("ij,jikl->ikl", rmnc_b, cosangle_b) + np.einsum("ij,jikl->ikl", rmns_b, sinangle_b)
    d_R_b_d_s = np.einsum("ij,jikl->ikl", d_rmnc_b_d_s, cosangle_b) + np.einsum("ij,jikl->ikl", d_rmns_b_d_s, sinangle_b)
    d_R_b_d_theta_b = -np.einsum("ij,jikl->ikl", rmnc_b, msinangle_b) + np.einsum("ij,jikl->ikl", rmns_b, mcosangle_b)
    d_R_b_d_phi_b = np.einsum("ij,jikl->ikl", rmnc_b, nsinangle_b) - np.einsum("ij,jikl->ikl", rmns_b, ncosangle_b)

    Z_b = np.einsum("ij,jikl->ikl", zmns_b, sinangle_b) + np.einsum("ij,jikl->ikl", zmnc_b, cosangle_b)
    d_Z_b_d_s = np.einsum("ij,jikl->ikl", d_zmns_b_d_s, sinangle_b) + np.einsum("ij,jikl->ikl", d_zmnc_b_d_s, cosangle_b)
    d_Z_b_d_theta_b = np.einsum("ij,jikl->ikl", zmns_b, mcosangle_b) - np.einsum("ij,jikl->ikl", zmnc_b, msinangle_b)
    d_Z_b_d_phi_b = -np.einsum("ij,jikl->ikl", zmns_b, ncosangle_b) + np.einsum("ij,jikl->ikl", zmnc_b, nsinangle_b)

    nu_b = np.einsum("ij,jikl->ikl", numns_b, sinangle_b) + np.einsum("ij,jikl->ikl", numnc_b, cosangle_b)
    d_nu_b_d_s = np.einsum("ij,jikl->ikl", d_numns_b_d_s, sinangle_b) + np.einsum("ij,jikl->ikl", d_numnc_b_d_s, cosangle_b)
    d_nu_b_d_theta_b = np.einsum("ij,jikl->ikl", numns_b, mcosangle_b) - np.einsum("ij,jikl->ikl", numnc_b, msinangle_b)
    d_nu_b_d_phi_b = -np.einsum("ij,jikl->ikl", numns_b, ncosangle_b) + np.einsum("ij,jikl->ikl", numnc_b, nsinangle_b)

    # sqrt_g_booz = (G + iota * I)/B**2
    sqrt_g_booz = np.einsum("ij,jikl->ikl", gmnc_b, cosangle_b) + np.einsum("ij,jikl->ikl", gmns_b, sinangle_b)
    d_sqrt_g_booz_d_theta_b = -np.einsum("ij,jikl->ikl", gmnc_b, msinangle_b) + np.einsum("ij,jikl->ikl", gmns_b, mcosangle_b)
    d_sqrt_g_booz_d_phi_b = np.einsum("ij,jikl->ikl", gmnc_b, nsinangle_b) - np.einsum("ij,jikl->ikl", gmns_b, ncosangle_b)
    modB_b = np.einsum("ij,jikl->ikl", bmnc_b, cosangle_b) + np.einsum("ij,jikl->ikl", bmns_b, sinangle_b)
    d_B_b_d_s = np.einsum("ij,jikl->ikl", d_bmnc_b_d_s, cosangle_b) + np.einsum("ij,jikl->ikl", d_bmns_b_d_s, sinangle_b)

    Vprime = gmnc_b[:, 0] + gmns_b[:, 0]

    delmnc_b[:, 1:] =  gmnc_b[:, 1:] / Vprime[:, None]
    delmns_b[:, 1:] =  gmns_b[:, 1:] / Vprime[:, None]

    betamns_b[:, 1:] = (
        delmnc_b[:, 1:] 
        * 1
        / edge_toroidal_flux_over_2pi
        * mu_0
        * d_pressure_d_s[:, None]
        * Vprime[:, None]
        / (xm_b[1:] * iota[:, None] - xn_b[1:])
    )

    betamnc_b[:, 1:] = (
        delmns_b[:, 1:] 
        * 1
        / edge_toroidal_flux_over_2pi
        * mu_0
        * d_pressure_d_s[:, None]
        * Vprime[:, None]
        / (xm_b[1:] * iota[:, None] - xn_b[1:])
    )

    lambmnc_b[:, 1:] = (
        delmnc_b[:, 1:]
        * (xm_b[1:] * G[:, None] + xn_b[1:] * I[:, None])
        / (
            (xm_b[1:] * iota[:, None] - xn_b[1:])
            * (G[:, None] + iota[:, None] * I[:, None])
        )
    )

    lambmns_b[:, 1:] = (
        delmns_b[:, 1:]
        * (xm_b[1:] * G[:, None] + xn_b[1:] * I[:, None])
        / (
            (xm_b[1:] * iota[:, None] - xn_b[1:])
            * (G[:, None] + iota[:, None] * I[:, None])
        )
    )

    beta_b = np.einsum("ij,jikl->ikl", betamns_b, sinangle_b) + np.einsum("ij,jikl->ikl", betamnc_b, cosangle_b)
    lambda_b = np.einsum("ij,jikl->ikl", lambmnc_b, cosangle_b) + np.einsum("ij,jikl->ikl", lambmns_b, sinangle_b)

    ###################################################################
    # Using R(theta,phi) and Z(theta,phi), compute the Cartesian
    # components of the gradient basis vectors using the dual relations:
    # This calculation is done in Boozer coordinates
    ####################################################################
    phi_cyl = phi_b - nu_b
    sinphi = np.sin(phi_cyl)
    cosphi = np.cos(phi_cyl)
    # X = R * cos(phi):
    d_X_d_theta_b = d_R_b_d_theta_b * cosphi - R_b * sinphi * (-1 * d_nu_b_d_theta_b)
    d_X_d_phi_b = d_R_b_d_phi_b * cosphi - R_b * sinphi * (1 - d_nu_b_d_phi_b)
    d_X_d_s = d_R_b_d_s * cosphi - R_b * sinphi * (-1 * d_nu_b_d_s)
    # Y = R * sin(phi):
    d_Y_d_theta_b = d_R_b_d_theta_b * sinphi + R_b * cosphi * (-1 * d_nu_b_d_theta_b)
    d_Y_d_phi_b = d_R_b_d_phi_b * sinphi + R_b * cosphi * (1 - d_nu_b_d_phi_b)
    d_Y_d_s = d_R_b_d_s * sinphi + R_b * cosphi * (-1 * d_nu_b_d_s)

    # Dual relations
    grad_psi_X = (
        d_Y_d_theta_b * d_Z_b_d_phi_b - d_Z_b_d_theta_b * d_Y_d_phi_b
    ) / sqrt_g_booz
    grad_psi_Y = (
        d_Z_b_d_theta_b * d_X_d_phi_b - d_X_d_theta_b * d_Z_b_d_phi_b
    ) / sqrt_g_booz
    grad_psi_Z = (
        d_X_d_theta_b * d_Y_d_phi_b - d_Y_d_theta_b * d_X_d_phi_b
    ) / sqrt_g_booz

    g_sup_psi_psi = grad_psi_X**2 + grad_psi_Y**2 + grad_psi_Z**2

    # Check varible names
    grad_theta_b_X = (d_Y_d_phi_b * d_Z_b_d_s - d_Z_b_d_phi_b * d_Y_d_s) / (
        sqrt_g_booz * edge_toroidal_flux_over_2pi
    )
    grad_theta_b_Y = (d_Z_b_d_phi_b * d_X_d_s - d_X_d_phi_b * d_Z_b_d_s) / (
        sqrt_g_booz * edge_toroidal_flux_over_2pi
    )
    grad_theta_b_Z = (d_X_d_phi_b * d_Y_d_s - d_Y_d_phi_b * d_X_d_s) / (
        sqrt_g_booz * edge_toroidal_flux_over_2pi
    )

    grad_phi_b_X = (d_Y_d_s * d_Z_b_d_theta_b - d_Z_b_d_s * d_Y_d_theta_b) / (
        sqrt_g_booz * edge_toroidal_flux_over_2pi
    )
    grad_phi_b_Y = (d_Z_b_d_s * d_X_d_theta_b - d_X_d_s * d_Z_b_d_theta_b) / (
        sqrt_g_booz * edge_toroidal_flux_over_2pi
    )
    grad_phi_b_Z = (d_X_d_s * d_Y_d_theta_b - d_Y_d_s * d_X_d_theta_b) / (
        sqrt_g_booz * edge_toroidal_flux_over_2pi
    )

    grad_alpha_X = (
        -phi_b * d_iota_d_s[:, None, None] * grad_psi_X / edge_toroidal_flux_over_2pi
        + grad_theta_b_X
        - iota[:, None, None] * grad_phi_b_X
    )
    grad_alpha_Y = (
        -phi_b * d_iota_d_s[:, None, None] * grad_psi_Y / edge_toroidal_flux_over_2pi
        + grad_theta_b_Y
        - iota[:, None, None] * grad_phi_b_Y
    )
    grad_alpha_Z = (
        -phi_b * d_iota_d_s[:, None, None] * grad_psi_Z / edge_toroidal_flux_over_2pi
        + grad_theta_b_Z
        - iota[:, None, None] * grad_phi_b_Z
    )

    #####################################################################################
    ##############------------LOCAL VARIATION OF A 3D EQUILIBRIUM------------############
    #####################################################################################
    # Calculating the coefficients D1 and D2 needed for Hegna-Nakajima calculation
    # NOTE: 2D functions do not require the alpha dimension. Remove it later.
    # NOTE: This calculation needs to be wrapped in a loop over ns (flux surfaces)
    ## Full flux surface average of various quantities needed to calculate D_HNGC
    ntheta_grid = res_theta
    nphi_grid = res_phi
    theta_b_grid = np.linspace(-np.pi, np.pi, ntheta_grid)
    phi_b_grid = np.linspace(-np.pi, np.pi, nphi_grid)
    th_b_2D, ph_b_2D = np.meshgrid(theta_b_grid, phi_b_grid)

    # grid_structure = (idx_val, ns, nalpha, ntheta_grid, nphi_grid)
    if flipit == 1:
        angle_b_2D = (
            xm_b[:, None, None, None, None] * (th_b_2D[None, None, None, :, :] + np.pi)
            - xn_b[:, None, None, None, None] * ph_b_2D[None, None, None, :, :]
        )
    else:
        angle_b_2D = (
            xm_b[:, None, None, None, None] * (th_b_2D[None, None, None, :, :])
            - xn_b[:, None, None, None, None] * ph_b_2D[None, None, None, :, :]
        )

    cosangle_b_2D = np.cos(angle_b_2D)
    sinangle_b_2D = np.sin(angle_b_2D)

    mcosangle_b_2D = xm_b[:, None, None, None, None] * cosangle_b_2D
    ncosangle_b_2D = xn_b[:, None, None, None, None] * cosangle_b_2D
    msinangle_b_2D = xm_b[:, None, None, None, None] * sinangle_b_2D
    nsinangle_b_2D = xn_b[:, None, None, None, None] * sinangle_b_2D

    lambda_b_2D = np.einsum("ij,jiklm->iklm", lambmnc_b, cosangle_b_2D) + np.einsum("ij,jiklm->iklm", lambmns_b, sinangle_b_2D)

    R_b_2D = np.einsum("ij,jiklm->iklm", rmnc_b, cosangle_b_2D) + np.einsum("ij,jiklm->iklm", rmns_b, sinangle_b_2D)
    d_R_b_d_theta_b_2D = -np.einsum("ij,jiklm->iklm", rmnc_b, msinangle_b_2D) + np.einsum("ij,jiklm->iklm", rmns_b, mcosangle_b_2D)
    d_R_b_d_phi_b_2D = np.einsum("ij,jiklm->iklm", rmnc_b, nsinangle_b_2D) - np.einsum("ij,jiklm->iklm", rmns_b, ncosangle_b_2D)

    d_Z_b_d_theta_b_2D = np.einsum("ij,jiklm->iklm", zmns_b, mcosangle_b_2D) - np.einsum("ij,jiklm->iklm", zmnc_b, msinangle_b_2D)
    d_Z_b_d_phi_b_2D = -np.einsum("ij,jiklm->iklm", zmns_b, ncosangle_b_2D) + np.einsum("ij,jiklm->iklm", zmnc_b, nsinangle_b_2D)

    nu_b_2D = np.einsum("ij,jiklm->iklm", numns_b, sinangle_b_2D) + np.einsum("ij,jiklm->iklm", numnc_b, cosangle_b_2D)
    d_nu_b_d_theta_b_2D = np.einsum("ij,jiklm->iklm", numns_b, mcosangle_b_2D) - np.einsum("ij,jiklm->iklm", numnc_b, msinangle_b_2D)
    d_nu_b_d_phi_b_2D = -np.einsum("ij,jiklm->iklm", numns_b, ncosangle_b_2D) + np.einsum("ij,jiklm->iklm", numnc_b, nsinangle_b_2D)

    sqrt_g_booz_2D = np.einsum("ij,jiklm->iklm", gmnc_b, cosangle_b_2D) + np.einsum("ij,jiklm->iklm", gmns_b, sinangle_b_2D)
    modB_b_2D = np.einsum("ij,jiklm->iklm", bmnc_b, cosangle_b_2D) + np.einsum("ij,jiklm->iklm", bmns_b, sinangle_b_2D)

    #########################################################################
    # We repeat the above exercise to calculate R and Z but use a 2D
    # (theta, phi) grid. This is used to calculate the deformation
    # coefficients that give us the local equilibrium variation
    #########################################################################
    ph_nat_2D = ph_b_2D - nu_b_2D
    sinphi_2D = np.sin(ph_nat_2D)
    cosphi_2D = np.cos(ph_nat_2D)
    # X = R * cos(phi):
    d_X_d_th_b_2D = d_R_b_d_theta_b_2D * cosphi_2D - R_b_2D * sinphi_2D * (
        -1 * d_nu_b_d_theta_b_2D
    )
    d_X_d_phi_2D = d_R_b_d_phi_b_2D * cosphi_2D - R_b_2D * sinphi_2D * (
        1 - d_nu_b_d_phi_b_2D
    )
    # Y = R * sin(phi):
    d_Y_d_th_b_2D = d_R_b_d_theta_b_2D * sinphi_2D + R_b_2D * cosphi_2D * (
        -1 * d_nu_b_d_theta_b_2D
    )
    d_Y_d_phi_2D = d_R_b_d_phi_b_2D * sinphi_2D + R_b_2D * cosphi_2D * (
        1 - d_nu_b_d_phi_b_2D
    )

    grad_psi_X_2D = (
        d_Y_d_th_b_2D * d_Z_b_d_phi_b_2D - d_Z_b_d_theta_b_2D * d_Y_d_phi_2D
    ) / sqrt_g_booz_2D
    grad_psi_Y_2D = (
        d_Z_b_d_theta_b_2D * d_X_d_phi_2D - d_X_d_th_b_2D * d_Z_b_d_phi_b_2D
    ) / sqrt_g_booz_2D
    grad_psi_Z_2D = (
        d_X_d_th_b_2D * d_Y_d_phi_2D - d_Y_d_th_b_2D * d_X_d_phi_2D
    ) / sqrt_g_booz_2D

    g_sup_psi_psi_2D = grad_psi_X_2D**2 + grad_psi_Y_2D**2 + grad_psi_Z_2D**2
    g_sup_psi_psi_2D_inv = 1 / g_sup_psi_psi_2D

    lam_over_g_sup_psi_psi_2D = lambda_b_2D * g_sup_psi_psi_2D_inv

    # Flux surface integrals D1 and D2 are needed to locally vary the gradients of a 3D equilibrium.
    # D1 and D2 are constant for a fixed s, alpha, theta_0, sfac, pfac, phi_b_grid, theta_b_grid

    D1 = np.zeros((ns,))
    D2 = np.zeros((ns,))

    for i in range(ns):
        D1[i] = (
            simps(
                [
                    simps(g_sup_psi_psi_1D_inv, theta_b_grid)
                    for g_sup_psi_psi_1D_inv in g_sup_psi_psi_2D_inv[i][0]
                ],
                phi_b_grid,
            )
            / (2 * np.pi) ** 2
        )

        D2[i] = (
            simps(
                [
                    simps(lam_over_g_sup_psi_psi_1D, theta_b_grid)
                    for lam_over_g_sup_psi_psi_1D in lam_over_g_sup_psi_psi_2D[i][0]
                ],
                phi_b_grid,
            )
            / (2 * np.pi) ** 2
        )


    #x = np.abs(d_G_d_s[:, None, None] + iota[:, None, None] * d_I_d_s[:, None, None] + mu_0 * d_pressure_d_s[:, None, None] * Vprime[:, None, None])
    #pdb.set_trace()

    ## EQUILIBRIUM CHECK: Flux surface averaged MHD force balance.
    #np.testing.assert_allclose(
    #    np.abs(d_G_d_s[:, None, None]
    #    + iota[:, None, None] * d_I_d_s[:, None, None]
    #    + mu_0 * d_pressure_d_s[:, None, None] * Vprime[:, None, None]),
    #    1e-3,
    #    atol=1e-3,
    #)

    # integrated inverse flux expansion term
    intinv_g_sup_psi_psi = ctrap(1 / g_sup_psi_psi, phi_b, initial=0)
    int_lambda_div_g_sup_psi_psi = ctrap(lambda_b / g_sup_psi_psi, phi_b, initial=0)

    # This theta_0 should always be 0
    theta_0 = 0
    spl0 = InterpolatedUnivariateSpline(theta_b[0][0], intinv_g_sup_psi_psi[0][0])
    intinv_g_sup_psi_psi = intinv_g_sup_psi_psi - spl0(theta_0)

    spl1 = InterpolatedUnivariateSpline(
        theta_b[0][0], int_lambda_div_g_sup_psi_psi[0][0]
    )
    int_lambda_div_g_sup_psi_psi = int_lambda_div_g_sup_psi_psi - spl1(theta_0)

    # Additional shear and pressure gradient (in addn. to the nominal vals)
    d_iota_d_s_1 = (
        -(iota[:, None, None, None, None] / (2 * s[:, None, None, None, None]))
        * (sfac[None, None, None, :, None] - 1.0)
        * shat[:, None, None, None, None]
    )
    d_pressure_d_s_1 = (
        mu_0
        * (pfac[None, None, None, None, :] - 1.0)
        * d_pressure_d_s[:, None, None, None, None]
    )

    # The deformation term from Hegna-Nakajima and Green-Chance papers
    # index order = (ns, nalpha, nl, sfac, pfac)
    D_HNGC = (
        1
        / edge_toroidal_flux_over_2pi
        * (
            d_iota_d_s_1
            * (
                intinv_g_sup_psi_psi[:, :, :, None, None]
                / D1[:, None, None, None, None]
                - phi_b[:, :, :, None, None]
            )
            - d_pressure_d_s_1
            * Vprime[:, None, None, None, None]
            * (
                G[:, None, None, None, None]
                + iota[:, None, None, None, None] * I[:, None, None, None, None]
            )
            * (
                int_lambda_div_g_sup_psi_psi[:, :, :, None, None]
                - D2[:, None, None, None, None]
                * intinv_g_sup_psi_psi[:, :, :, None, None]
                / D1[:, None, None, None, None]
            )
        )
    )

    # Now we recalculate some of the geometric coefficients in Boozer coordinates
    # Partially calculated in boozer coordinates
    grad_alpha_dot_grad_psi = (
        grad_alpha_X * grad_psi_X
        + grad_alpha_Y * grad_psi_Y
        + grad_alpha_Z * grad_psi_Z
    )

    # Intergrated local shear L1 is calculated using covariant basis
    # expressions in Hegna and Nakajima
    # We remove the secular part form the integrated local shear grad_alpha_dot_grad_psi_alt
    # NOTE: Potential sign issue here
    L0 = -1 * (
        grad_alpha_dot_grad_psi[:, :, :, None, None]
        / g_sup_psi_psi[:, :, :, None, None]
        + 1
        / edge_toroidal_flux_over_2pi
        * d_iota_d_s[:, None, None, None, None]
        * phi_b[:, :, :, None, None]
    )

    # L1 is the integrated local shear
    L1 = (
        -1 / edge_toroidal_flux_over_2pi * d_iota_d_s_1 * phi_b[:, :, :, None, None]
        + grad_alpha_dot_grad_psi[:, :, :, None, None]
        / g_sup_psi_psi[:, :, :, None, None]
        - D_HNGC
    )

    L2 = d_iota_d_s[:, None, None, None, None] * 1 / edge_toroidal_flux_over_2pi

    # Normal curvature
    # NOTE: Test a case close to a rational surface to check the sign of beta_b
    kappa_n = (
        1
        / modB_b[:, :, :, None, None] ** 2
        * (
            modB_b[:, :, :, None, None] * d_B_b_d_s[:, :, :, None, None]
            + mu_0 * d_pressure_d_s[:, None, None, None, None]
        )
        * 1
        / edge_toroidal_flux_over_2pi
        - beta_b[:, :, :, None, None]
        / (
            2
            * sqrt_g_booz[:, :, :, None, None]
            * (
                G[:, None, None, None, None]
                + iota[:, None, None, None, None] * I[:, None, None, None, None]
            )
        )
        * d_sqrt_g_booz_d_phi_b[:, :, :, None, None]
        + L0
        * (
            G[:, None, None, None, None] * d_sqrt_g_booz_d_theta_b[:, :, :, None, None]
            - I[:, None, None, None, None] * d_sqrt_g_booz_d_phi_b[:, :, :, None, None]
        )
        / (
            2
            * sqrt_g_booz[:, :, :, None, None]
            * (
                G[:, None, None, None, None]
                + iota[:, None, None, None, None] * I[:, None, None, None, None]
            )
        )
    )

    # Geodesic curvature
    kappa_g = (
        G[:, None, None, None, None] * d_sqrt_g_booz_d_theta_b[:, :, :, None, None]
        - I[:, None, None, None, None] * d_sqrt_g_booz_d_phi_b[:, :, :, None, None]
    ) / (
        2
        * sqrt_g_booz[:, :, :, None, None]
        * (
            G[:, None, None, None, None]
            + iota[:, None, None, None, None] * I[:, None, None, None, None]
        )
    )

    B_cross_kappa_dot_grad_alpha_b = (kappa_n + kappa_g * L1) * modB_b[
        :, :, :, None, None
    ] ** 2

    B_cross_kappa_dot_grad_psi_b = kappa_g * modB_b[:, :, :, None, None] ** 2

    grad_alpha_dot_grad_alpha_b = (
        modB_b[:, :, :, None, None] ** 2 / g_sup_psi_psi[:, :, :, None, None]
        + g_sup_psi_psi[:, :, :, None, None] * L1**2
    )
    grad_alpha_dot_grad_psi_b = g_sup_psi_psi[:, :, :, None, None] * L1
    grad_psi_dot_grad_psi_b = (
        g_sup_psi_psi[:, :, :, None, None] * L2
    )  # This is wrong. L2 should be different

    ## Now we calculate the same set of quantities in boozer coordinates after varying the
    ## local gradients.
    bmag = modB_b[:, :, :, None, None] / B_reference
    gradpar_theta_b = (
        -L_reference
        / modB_b[:, :, :, None, None]
        * 1
        / sqrt_g_booz[:, :, :, None, None]
        * iota[:, None, None, None, None]
    )
    gradpar_theta_PEST = (
        -L_reference
        * iota[:, None, None, None, None]
        * 1
        / modB_b[:, :, :, None, None]
        * 1
        / sqrt_g_booz[:, :, :, None, None]
        * (1 - d_nu_b_d_theta_b[:, :, :, None, None])
    )
    gradpar_phi = (
        L_reference / modB_b[:, :, :, None, None] * 1 / sqrt_g_booz[:, :, :, None, None]
    )

    gds2 = (
        grad_alpha_dot_grad_alpha_b
        * L_reference
        * L_reference
        * s[:, None, None, None, None]
    )
    gds21 = (
        grad_alpha_dot_grad_psi_b
        * sfac[None, None, None, :, None]
        * shat[:, None, None, None, None]
        / B_reference
    )
    gds22 = (
        g_sup_psi_psi[:, :, :, None, None]
        * (sfac[None, None, None, :, None] * shat[:, None, None, None, None]) ** 2
        / (
            L_reference
            * L_reference
            * B_reference
            * B_reference
            * s[:, None, None, None, None]
        )
    )

    grho = np.sqrt(
        g_sup_psi_psi[:, :, :, None, None]
        / (
            L_reference
            * L_reference
            * B_reference
            * B_reference
            * s[:, None, None, None, None]
        )
    )

    gbdrift0 = (
        -1.0
        * B_cross_kappa_dot_grad_psi_b
        * 2
        * sfac[None, None, None, :, None]
        * shat[:, None, None, None, None]
        / (modB_b[:, :, :, None, None] ** 2 * sqrt_s[:, None, None, None, None])
        * toroidal_flux_sign
    )
    cvdrift0 = gbdrift0

    cvdrift = (
        -1.0
        * 2
        * B_reference
        * L_reference
        * L_reference
        * sqrt_s[:, None, None, None, None]
        * B_cross_kappa_dot_grad_alpha_b
        / (modB_b[:, :, :, None, None] ** 2)
        * toroidal_flux_sign
    )

    gbdrift = cvdrift + 2 * B_reference * L_reference * L_reference * sqrt_s[
        :, None, None, None, None
    ] * mu_0 * pfac * d_pressure_d_s[:, None, None, None, None] * toroidal_flux_sign / (
        edge_toroidal_flux_over_2pi * modB_b[:, :, :, None, None] ** 2
    )

    cvdrift0 = gbdrift0
    # PEST theta; useful for comparison
    theta_PEST = (
        theta_b[:, :, :, None, None]
        - iota[:, None, None, None, None] * nu_b[:, :, :, None, None]
    )

    # geometric theta; denotes the actual poloidal angle
    theta_geo = np.arctan2(
        Z_b[:, :, :, None, None], R_b[:, :, :, None, None] - R_mag_ax
    )

    # This is half of the total beta_N. Used in GS2 as beta_ref
    beta_N = 4 * np.pi * 1e-7 * vs.pressure(s) / B_reference**2

    int_loc_shr = L0 + L1 + L2
    # Package results into a structure to return:
    results = Struct()
    variables = [
        "iota",
        "d_iota_d_s",
        "d_pressure_d_s",
        "d_psi_d_s",
        "shat",
        "alpha",
        "theta_b",
        "phi_b",
        "theta_PEST",
        "theta_geo",
        "edge_toroidal_flux_over_2pi",
        "R_b",
        "Z_b",
        "beta_N",
        "bmag",
        "gradpar_theta_b",
        "gradpar_theta_PEST",
        "gds2",
        "gds21",
        "gds22",
        "gbdrift",
        "gbdrift0",
        "cvdrift",
        "cvdrift0",
        "grho",
    ]

    for v in variables:
        results.__setattr__(v, eval(v))

    return results


