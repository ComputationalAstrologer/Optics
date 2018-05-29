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
import matplotlib.pyplot as plt

pyrparams = dict()  # dict of nominal optical system paramers
pyrparams['wavelength'] = 0.8 # wavelength (microns)
pyrparams['indref'] = 1.45 # pyramid index of refraction
pyrparams['pslope'] = 3. # slope of pyramid faces relative to horizontal (degrees)  
pyrparams['beam_diameter'] = 1.e3 # input beam diameter (microns)
pyrparams['D_e_2_l1'] = 20.e3 # nominal distance from entrance pupil to lens1
pyrparams['f1'] = 50.e3 # focal length of lens #1 (focuses light on pyramid tip)
pyrparams['D_l1_2_pyr'] = 50.e3 # distance from pyramid tip to lens 1 (corrected for pyramid glass)
pyrparams['f2'] = 1.e5 # focal length of lens #2 (makes 4 pupil images)
pyrparams['max_chirp_step_deg'] = 60  # maximum allowed value (degrees) in chirp step for Fresnel prop
pyrparams['max_lens_step_deg'] = 20  # maximum step size allowed for lens phase screen

# This simulator has only 1 dimension transverse to the optical axis.  The purpose is
#   to test numerical resolution and code algorithms/structures.
class Pyr1D():
    def __init__(self, params=pyrparams):
        self.params = params
        self.grads = dict()  # dictionary of gradients
        self.field_Start = np.ones(33)  # initial pupil field
        #self.Prop2Apex(self.field_Start)
        self.FocusEntrance(self.field_Start)
        return

    #This runs a test for bringing the entrance pupil to a focus
    def FocusEntrance(self, pfield):
        diam0 = self.params['beam_diameter']
        foc = self.params['f1']
        dx = diam0/pfield.shape[0]
        x = np.linspace(-diam0/2 + dx/2, diam0/2 - dx/2, pfield.shape[0])

        print("initial x.shape= " + str(x.shape))
        lfield, x = self.ApplyThinLens(pfield, x, 0., foc, return_deriv=False)

        print("post lens x.shape= " + str(x.shape))
        
        plt.figure()
        plt.plot(x, np.angle(lfield)*180/np.pi,'bo-')

        lambda_over_D =  self.params['wavelength']*foc/diam0
        diam1 = 5*lambda_over_D
        ffield, x = self.ConvFresnel1D(lfield, x, diam1, foc, return_deriv=False)
        print("post prop x.shape= " + str(x.shape))

        plt.figure()
        plt.plot(x/lambda_over_D, np.real(ffield*np.conj(ffield)),'kx-')

        return

    # This calculates the field near the focal plane of lens1, which is near the apex of the pyramid
    # pfield - pupil plane field (assume evenly sampled)
    def Prop2Apex(self, pfield):
        diam0 = self.params['beam_diameter']
        nxi = pfield.shape[0]
        dxi = diam0/nxi 
        xi = np.linspace(-diam0/2 + dxi/2, diam0/2 - dxi/2, nxi)
        z = self.params['D_e_2_l1']
        [f, df_dparam, x] = self.ConvFresnel1D(pfield, xi,1.5*diam0, z, return_deriv=True)
        self.grads['D_e_2_l1'] = df_dparam
        plt.figure(); plt.plot(x, np.real(f*np.conj(f)))

        foc = self.params['f1']  # g, x, center, f, return_deriv=False)
        for key in self.grads:  # update gradients
            self.grads[key], x = self.ApplyThinLens(self.grads[key], x, 0, foc, return_deriv=False)
        [f, df_dparam, x] = self.ApplyThinLens(f, x, 0, foc, return_deriv=True) # update field
        self.grads['l1_center'] = df_dparam

        lambda_over_D =  self.params['wavelength']*foc/diam0
        diam1 = 10*lambda_over_D
        for key in self.grads:  # propagate gradients
            self.grads[key], x = self.ConvFresnel1D(self.grads[key], x, diam1, foc, return_deriv=False)
        f, df_dparam, x = self.ConvFresnel1D(f, 1.5*diam0, diam1, foc, return_deriv=True) # prop field
        self.grads['D_l1_2_apex'] = df_dparam
        self.field_Apex = f
        plt.figure(); plt.plot(x, np.real(f*np.conj(f)))
        return

    #1D Fresenel prop using convolution in the spatial domain
    # g - vector of complex-valued field in the inital plane
    # x - vector of coordinates in initial plane, corresponding to bin center locaitons
    # d2 - diameter of region to be calculated in output plane
    # z - propagation distance
    # return_derive == True to return deriv. of output field WRT z
    def ConvFresnel1D(self, g, x, d2, z, return_deriv=False):
        if g.shape != x.shape:
            raise Exception("Input field and grid must have same dimensions.")
        lam = self.params['wavelength']
        dPhiTol_deg = self.params['max_chirp_step_deg']
        [dx, diam] = self.GetDxAndDiam(x)
        nx = g.shape[0]
        #first figure out sampling criterion for chirp
        dx_tol = (dPhiTol_deg/180)*lam*z/(diam + d2)  # factors of pi cancel
        if dx > dx_tol:  # interpolate onto finer grid
            dx = dx_tol
            nx = 1 + int(diam/dx)
            dx = diam/nx
            xnew = np.linspace(-diam/2 + dx/2, diam/2 - dx/2, nx)  # new grid
            interp = interp1d(x, g, 'quadratic', fill_value='extrapolate')
            g = interp(xnew)
            x = xnew
        ns = 1 + int((diam + d2)/dx)
        s = np.linspace(-diam/2 - d2/2 + dx/2, diam/2 + d2/2 - dx/2, ns) # spatial grid of output plane
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
        if g.shape != x.shape:
            raise Exception("Input field and spatial grid must have same dimensions.")
        [dx, diam] = self.GetDxAndDiam(x)
        lam = self.params['wavelength']
        max_step = self.params['max_lens_step_deg']*np.pi/180
        dx_tol = max_step*lam*f/(2*np.pi*(diam/2 + np.abs(center)))
        if dx > dx_tol:  # interpolate onto higher resolution grid
            nx = int(diam/dx_tol) + 1
            dx = diam/nx
            xnew = np.linspace(-diam/2 + dx/2, diam/2 + dx/2, nx)
            interp = interp1d(x, g, 'quadratic', fill_value='extrapolate')
            g = interp(xnew)
        else:
            xnew = x
        h = g*np.exp(-1j*np.pi*(xnew - center)*(xnew - center)/(f*lam))
        if not return_deriv:
            return([h, xnew])
        dhdc = 2j*np.pi*(x - center)*h/(lam*f)
        return([h, dhdc, xnew])

    # assuming a regular spaced grid with values at bin centers,
    #  get spacing and diameter from spatial grid x
    def GetDxAndDiam(self, x):  # assumes x values are bin centers
        nx = x.shape[0]
        dx = x[1] - x[0]
        diam = (x[-1] - x[0])*(1 + 1/(nx - 1))
        assert diam > 0
        assert dx > 0
        return([dx, diam])

    #This downsamples the field to cut computation costs.  It judges how accurately the downsampled field
    #  reproduces the original with nearest neighbor interpolation.  The downsampled field is itself
    #  calculated with a more sophisticated interpolation.  Returns downsampled field and new grid.
    # new_diam - diameter of resampled field.
    def DownSampleField(self, g, x, new_diam):
        return(np.nan)



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
