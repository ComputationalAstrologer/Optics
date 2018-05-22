#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  9 14:55:31 2018
@author: rfrazin

This code is a descendant of Pyr.py.   This version is designed to accurately
reproduce lab measurements, and so allows for different paddings of the two 
FFTs and treats alignments (and possibly other) errors with free parameters.  

params is a dictionary of the basic parameters of the numerical model

"""

import numpy as np

pyrparams = dict()  # dict of nominal optical system paramers
pyrparams['lambda'] = 0.8 # wavelength (microns)
pyrparams['indref'] = 1.45 # pyramid index of refraction
pyrparams['pslope'] = 3. # slope of pyramid faces relative to horizontal (degrees)  
pyrparams['beam_diam'] = 1.e4 # input beam diameter (microns)
pyrparams['d_e2l1'] = 1.e5 # nominal distance from entrance pupil to lens1
pyrparams['f1'] = 1.e6 # focal length of lens #1 (focuses light on pyramid tip)
pyrparams['d_p2l2'] = 1.e5 # distance from pyramid tip to lens 1 (corrected for pyramid glass)
pyrparams['f2'] = 1.e5 # focal length of lens #2 (makes 4 pupil images)
pyrparams['npup'] = 33  # number of pixels in entrace pupil
pyrparams['npad1'] = 2048 # number of points in first FFT
pyrparams['npad2'] = 2048 # number of points in second FFT

# This simulator two dimentions perpendicular to the optical axis and is designed
#   to model the PyWFS in the lab.
class Pyr2D():
    def __init__(self, params=pyrparams):
        self.params = params
        return

# This simulator has only 1 dimension transverse to the optical axis.  The purpose is
#   to test numerical resolution and code algorithms/structures.
class Pyr1D():
    def __init__(self, params=pyrparams):
        return

    # This calculates the field near the focal plane of lens1, which is near the
    #   apex of the pyramid, as a function of both the distance from the entrance
    #   pupil to the lens (den) and the distance from the pupil to the experiment's
    #   focal point (dfo).  This involves 2 Fresnel propagations and quadratic 
    #   factor applied by the lens.
    # Distances are in microns
    def Prop2FirstFocus(self, pfield, den, dfo):
        return(np.nan)

# 1D Fresnel beam prop., analog FT based
#   This evenly samples the spatial frequency over the propagating modes
# g - complex valued field in initial plane
# diam - diameter of beam  NOTE: z, diam, lam must all be in the same units
# z - propagation distance
# lam - wavelength
# dphi_max (degrees) - spatial sampling criterion for chirp function
#    output spatial grid defined by  diam_out and dphi_max
# returns propagated field and output spatial grid
def SlowFresnelProp1D(g, z, diam_in, diam_out, lam = .8, dphi_max=10):
    nx = g.shape[0]
    dx = diam_in/nx
    x = np.linspace(-diam_in/2 + dx/2, diam_in/2 - dx/2, nx) #  input grid
    ds = (dphi_max/180)*lam*z/diam_out  # factors of pi in num and denom cancel
    ns = int(diam_out/ds)
    s = np.linspace(-diam_out/2 + ds/2, diam_out/2 - ds/2, ns)  # output grid
    k = s/(lam*z)  # spatial frequency grid
    Fg = np.zeros(ns).astype('complex')  # propagated version of g
    for m in range(ns):
        for n in range(nx):
            Fg[m] += g[n]*np.exp(1j*x[n]**2/(2*lam*z))*np.exp(-2j*np.pi*k[m]*x[n])*dx
        Fg[m] *= -1j*np.exp(2j*np.pi*z/lam + 1j*s[m]**2/(lam*z))/(lam*z)
    return([Fg, s])

def propTF(u1, L, lam, z):
    M = u1.shape[0]
    N = u1.shape[1]
    assert M == N
    dx = L/M
    fx = np.linspace(-1/(2*dx), 1/(2*dx) - 1/L, M)
    FX, FY = np.meshgrid(fx, fx)
    H = np.exp(-1j*np.pi*lam*z*(FX*FX + FY*FY))
    H = np.fft.fftshift(H)
    U1 = np.fft.fft2(np.fft.fftshift(u1))
    U2 = H*U1
    U2 = np.fft.ifftshift(np.fft.ifft2(U2))
    return(U2)

#rectangle function.  Note to make a square that is length w on a side,
#   use:  X1, Y1 = np.meshgrid(x,y); u = rect1D(X1/2/w)*rect1D(Y1/2/w)
def rect1D(x):
    assert x.ndim == 2
    f = np.zeros(x.shape)
    for m in range(x.shape[0]):
        for n in range(x.shape[1]):
            if abs(x[m,n]) <= 1:
                f[m,n] = 1
    return(f)


#  this zero pad function gives rise to purely real myfft with symm. input
def myzp(f, npad):  # zero-pad function for pupil fields
    if f.ndim == 1:
        nfield = len(f) + 1
        if np.mod(nfield, 2) != 0:
            raise Exception("len(f) must be odd.")
        if npad <= len(f):
            raise Exception("npad must be greater than len(f)")
        if np.mod(npad, 2) == 1:
            raise Exception("npad must be even.")
        g = np.zeros(npad).astype('complex')
        g[int(npad/2) - int(nfield/2) + 1:int(npad/2) + int(nfield/2)] = f
        return(g)
    elif f.ndim == 2:
        if f.shape[0] != f.shape[1]:
            raise Exception("f must be square (if 2D).")
        nfield = f.shape[0] + 1
        if np.mod(nfield, 2) != 0:
            raise Exception("len(f) must be odd.")
        if npad <= f.shape[0]:
            raise Exception("npad must be greater than len(f)")
        if np.mod(npad, 2) == 1:
            raise Exception("npad must be even.")
        g = np.zeros((npad, npad)).astype('complex')
        g[int(npad/2) - int(nfield/2) + 1:int(npad/2) + int(nfield/2),
          int(npad/2) - int(nfield/2) + 1:int(npad/2) + int(nfield/2)] = f
        return(g)
    else:
        raise Exception("Input array must be 1D or 2D.")
        return(np.nan)
