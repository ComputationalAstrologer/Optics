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
from scipy.signal import convolve as conv
from scipy.interpolate import interp1d

pyrparams = dict()  # dict of nominal optical system paramers
pyrparams['wavelength'] = 0.8 # wavelength (microns)
pyrparams['indref'] = 1.45 # pyramid index of refraction
pyrparams['pslope'] = 3. # slope of pyramid faces relative to horizontal (degrees)  
pyrparams['beam_diam'] = 1.e4 # input beam diameter (microns)
pyrparams['d_e2l1'] = 1.e5 # nominal distance from entrance pupil to lens1
pyrparams['f1'] = 1.e6 # focal length of lens #1 (focuses light on pyramid tip)
pyrparams['d_p2l2'] = 1.e5 # distance from pyramid tip to lens 1 (corrected for pyramid glass)
pyrparams['f2'] = 1.e5 # focal length of lens #2 (makes 4 pupil images)
pyrparams['npup'] = 33  # number of pixels in entrace pupil
pyrparams['max_chirp_step_deg'] = 90  #maximum allowed value (degrees) in chirp step for Fresnel prop

# This simulator has only 1 dimension transverse to the optical axis.  The purpose is
#   to test numerical resolution and code algorithms/structures.
class Pyr1D():
    def __init__(self, params=pyrparams):
        return

    # This calculates the field near the focal plane of lens1, which is near the apex of the pyramid
    # pfield - pupil plane field (assume evenly sampled)
    def Prop2FirstFocus(self, pfield, return_dervis=True):
        nx = pfield.shape[0]
        diam0 = self.params['beam_diam']
        dx = diam0/nx
        x = np.linspace(-diam0/2 + dx/2, diam0/2 - dx/2, nx)
        return(np.nan)

    #This resamples the field to cut computation costs.  It judges how accurately the resampled field
    #  reproduces the original after linear interpolation.
    # new_diam - diameter of resampled field.
    def ResampleField(self, g, new_diam):
        return(np.nan)

    def ConvFresnel1D(self, g, d1, d2, z, return_deriv=False):
        lam = self.params['wavelength']
        dPhiTol_deg = self.params['max_chirp_step_deg']
        dx = d1/g.shape[0]
        #first figure out sampling criterion for chirp
        dx_tol = (dPhiTol_deg/180)*lam*z/(d1 + d2)  # factors of pi cancel
        if dx > dx_tol:  # interpolate onto finer grid
            dx = dx_tol
            xo = np.linspace(-d1/2 + dx/2, d1/2 - dx/2, g.shape[0])  # old grid
            nx = int(d1/dx)
            x = np.linspace(-d1/2 + dx/2, d1/2 - dx/2, nx)  # new grid
            dx = x[1] - x[0]
            interp = interp1d(xo, g, 'quadratic', fill_value='extrapolate')
            g = interp(x)
        else:
            nx = g.shape[0]
            x = np.linspace(-d1/2 + dx/2, d1/2 - dx/2, nx)  # grid
        ns = int((d1 + d2)/dx)
        s = np.linspace(-d1/2 - d2/2 + dx/2, d1/2 + d2/2 - dx/2, ns)
        kern = np.exp(1j*np.pi*s*s/(lam*z))  # Fresnel kernel (Goodman 4-16)
        # propagated field is given by h*p
        h = conv(kern, g, mode='same', method='fft')
        #p = -1j*np.exp(2j*np.pi*z/lam)/(lam*z)  # prefactor - causes unwanted oscillations with z
        p = 1/(lam*z)
        if not return_deriv:
            return([p*h,s])
        #dpdz = (1j/(lam*z*z) + 2*np.pi/(lam*lam*z))*np.exp(2j*np.pi*z/lam)  # includes unwanted oscillations
        dpdz = -1/(lam*z*z)
        dkerndz = -1j*np.pi*s*s*kern/(lam*z*z)
        dhdz = conv(dkerndz, g, mode='same', method='fft')
        return([p*h, dpdz*h + p*dhdz, s])


    #Apply thin lens phase transformation.
    # g - electric field impinging on lens
    # x - spatial coordinate
    # ceneter - center of lens relative to zero of x
    # f - lens focal length (same units as x and wavelength)
    def ApplyThinLens(self, g, x, center, f, return_deriv=False):
        lam = self.params['wavelength']
        if g.shape != x.shape:
            raise Exception("Input field and spatial grid must have same dimensions.")
        h = g*np.exp(-1j*np.pi(x - center)*(x - center)/(f*lam))
        if not return_deriv:
            return([h,x])
        dhdc = 2j*np.pi*(x - center)*h/(lam*f)
        return([h, dhdc, x])

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
