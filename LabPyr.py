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

# dict of nominal optical system paramers
#  units: lengths are in microns, angles are in degrees
pyrparams = dict()
pyrparams['wavelength'] = 0.8 # wavelength (microns)
pyrparams['indref'] = 1.45 # pyramid index of refraction
pyrparams['pslope'] = 4. # slope of pyramid faces relative to horizontal (degrees)  
pyrparams['beam_diameter'] = 1.e3 # input beam diameter (microns)
pyrparams['D_e_2_l1'] = 20.e3 # nominal distance from entrance pupil to lens1
pyrparams['f1'] = 50.e3 # focal length of lens #1 (focuses light on pyramid tip)
pyrparams['D_l1_2_pyr'] = 50.e3 # distance from pyramid tip to lens 1 (corrected for pyramid glass)
pyrparams['apex_diam'] = 8 # diameter of focal spot at pyramid apex in units of lambda_over_D (set by stop)
pyrparams['D_foc_2_l2'] = 50.e3 # distance from focus to lens #2 (includes effective OPL thru prism)
pyrparams['f2'] = 50.e3 # focal length of lens #2 (makes 4 pupil images)
pyrparams['diam_lens2'] = 6.e3 # effective diameter of lens2 (set by a stop)
pyrparams['D_l2_2_detector'] = 10.e3 # distrance from lens2 to detector
pyrparams['detector_width'] = 6.e3 # width of detector
pyrparams['max_chirp_step_deg'] = 30  # maximum allowed value (degrees) in chirp step for Fresnel prop
pyrparams['max_lens_step_deg'] = 15  # maximum step size allowed for lens phase screen
pyrparams['max_pyramid_step_deg'] = 10 # maximum step (degrees) allowed for pyramid phase ramp

# This simulator has only 1 dimension transverse to the optical axis.  The purpose is
#   to test numerical resolution and code algorithms/structures.
class Pyr1D():
    def __init__(self, pupil_field=None, params=pyrparams):
        self.params = params
        self.grads = dict()  # dictionary of gradients
        if pupil_field is None:
            self.field_Start = np.ones(33)  # initial pupil field
        else:
            self.field_Start = pupil_field
        nx = self.field_Start.shape[0]
        diam0 = self.params['beam_diameter']
        dx = diam0/nx
        self.x_Start = np.linspace(-diam0/2 + dx/2, diam0/2 - dx/2, nx)  # initial spatial grid

        #self.Prop1D()
        #self.FocusEntrance1D()
        return

    #This runs a test for bringing the entrance pupil to a focus
    def FocusEntrance1D(self):
        diam0 = self.params['beam_diameter']
        foc = self.params['f1']
        x = self.x_Start

        print("initial x.shape= " + str(x.shape))
        lfield, x = self.ApplyThinLens1D(self.field_Start, x, 0., foc, return_derivs=False)

        print("post lens x.shape= " + str(x.shape))
        
        plt.figure()
        plt.plot(x, np.angle(lfield)*180/np.pi,'bo-')

        lambda_over_D =  self.params['wavelength']*foc/diam0
        diam1 = 5*lambda_over_D
        ffield, x = self.ConvFresnel1D(lfield, x, diam1, foc, return_derivs=False)
        print("post prop x.shape= " + str(x.shape))

        plt.figure()
        plt.plot(x/lambda_over_D, np.real(ffield*np.conj(ffield)),'kx-')
        return

    # This calculates the field near the focal plane of lens1, which is near the apex of the pyramid
    # pfield - pupil plane field (assume evenly sampled)
    def Prop1D(self):

        diam0 = self.params['beam_diameter']
        print("initial x.shape= " + str(self.x_Start.shape))

        # Fresnel prop to lens #1
        z = self.params['D_e_2_l1']
        [f, df_dparam, x] = self.ConvFresnel1D(self.field_Start, self.x_Start,1.2*diam0, z, return_derivs=True)
        self.grads['D_e_2_l1'] = df_dparam

        print("after prop to lens1 x.shape= " + str(x.shape))
        plt.figure(); plt.plot(x, np.real(f*np.conj(f))); 
        plt.title('Intensity impinging on Lens #1'); plt.xlabel("distance ($\mu$m)")

        # apply lens #1
        foc = self.params['f1']
        for key in self.grads:  # update gradients
            self.grads[key], _ = self.ApplyThinLens1D(self.grads[key], x, 0, foc, return_derivs=False)
        f, df_dparam, x = self.ApplyThinLens1D(f, x, 0, foc, return_derivs=True) # update field
        self.grads['l1_center'] = df_dparam

        print("post lens1 x.shape= " + str(x.shape))

        # Fresnel prop to focus
        lambda_over_D =  self.params['wavelength']*foc/diam0  
        diam1 = self.params['apex_diam']*lambda_over_D
        for key in self.grads:  # propagate gradients
            self.grads[key], _ = self.ConvFresnel1D(self.grads[key], x, diam1, foc, return_derivs=False)
        f, df_dparam, x = self.ConvFresnel1D(f, x, diam1, foc, return_derivs=True) # prop field
        self.Focal_field = f*1.0 # ensure deep copy
        self.grads['D_l1_2_apex'] = df_dparam

        print("after prop to focus x.shape= " + str(x.shape))
        plt.figure(); plt.plot(x, np.real(f*np.conj(f)));
        plt.title("Intensity at focus (pyramid apex)"); plt.xlabel("distance ($\mu$m)")

        # Apply phase ramp caused by pyramid
        for key in self.grads:
            self.grads[key], _ = self.ApplyPyramidPhaseRamp1D(self.grads[key], x, 0., return_derivs=False, no_apex=False)
        f, df_dparam, x = self.ApplyPyramidPhaseRamp1D(f, x, 0, return_derivs=True, no_apex=False)
        self.grads['pyramid_center'] = df_dparam

        print("after pyramid x.shape= " + str(x.shape))

        # Fresnel prop to lens #2
        z = self.params['D_foc_2_l2']
        diam2 = self.params['diam_lens2']
        for key in self.grads:  # propagate gradients
            self.grads[key], _ = self.ConvFresnel1D(self.grads[key], x, diam2, z, return_derivs=False)
        f, df_dparam, x = self.ConvFresnel1D(f, x, diam2, z, return_derivs=True)
        self.grads['D_foc_2_l2'] = df_dparam

        plt.figure(); plt.plot(x, np.real(f*np.conj(f)));
        plt.title('Intensity impinging on lens2'); plt.xlabel("distance ($\mu$m)")
        print("after prop to lens2 x.shape= " + str(x.shape))

        #apply lens #2
        foc = self.params['f2']
        for key in self.grads:  # update gradients
            self.grads[key], _ = self.ApplyThinLens1D(self.grads[key], x, 0, foc, return_derivs=False)
        f, df_dparam, x = self.ApplyThinLens1D(f, x, 0, foc, return_derivs=True) # update field
        self.grads['l2_center'] = df_dparam

        print("post lens2 x.shape= " + str(x.shape))

        # Fresnel prop to detector
        z = self.params['D_l2_2_detector']
        diam2 = self.params['detector_width']

        if False:  #  compare full result to maintain_dx results
            # get full result
            fc, xc = self.ConvFresnel1D(f, x, diam2, z, return_derivs=False, maintain_dx=False)
            plt.figure(); plt.plot(xc, np.real(fc*np.conj(fc))); plt.title('Intensity at detector, full res'); plt.xlabel("distance ($\mu$m)")
            print("at detector xc.shape= " + str(xc.shape))
            # get maintain_dx result
            fc, xc = self.ConvFresnel1D(f, x, diam2, z, return_derivs=False, maintain_dx=True)
            plt.figure(); plt.plot(xc, np.real(fc*np.conj(fc))); plt.title('Intensity at detector, maintain_dx'); plt.xlabel("distance ($\mu$m)")
            print("at detector xc.shape= " + str(xc.shape))

        # maintain_dx=True worked well!
        for key in self.grads:  # propagate gradients
            self.grads[key], _ = self.ConvFresnel1D(self.grads[key], x, diam2, z, return_derivs=False, maintain_dx=True)
        f, df_dparam, x = self.ConvFresnel1D(f, x, diam2, z, return_derivs=True, maintain_dx=True)
        self.grads['D_l2_2_detector'] = df_dparam

        print("at detector x.shape= " + str(x.shape))
        plt.figure(); plt.plot(x, np.real(f*np.conj(f))); plt.title('Intensity at detector'); plt.xlabel("distance ($\mu$m)")


        return

    #1D Fresenel prop using convolution in the spatial domain
    # g - vector of complex-valued field in the inital plane
    # x - vector of coordinates in initial plane, corresponding to bin center locaitons
    # diam1 - diameter of region to be calculated in output plane
    # z - propagation distance
    # return_derivs - True to return deriv. of output field WRT z
    # maintain_dx - False allows full sampling of chirp according to self.params['max_chirp_step_deg']
    #             - True forces zeroing the kernel beyond where self.params['max_chirp_step_deg'] 
    #                    exceeded for the dx given in x input array
    #     Thus, True can dramatically shrink the size of the output array
    def ConvFresnel1D(self, g, x, diam1, z, return_derivs=False, maintain_dx=False):
        if g.shape != x.shape:
            raise Exception("Input field and grid must have same dimensions.")
        lam = self.params['wavelength']
        dPhiTol_deg = self.params['max_chirp_step_deg']
        dx, diam = self.GetDxAndDiam(x)
        nx = g.shape[0]
        #first figure out sampling criterion for chirp
        dx_tol = (dPhiTol_deg/180)*lam*z/(diam + diam1)  # factors of pi cancel
        if dx > dx_tol and not maintain_dx:  # interpolate g onto finer grid
            dx = dx_tol
            nx = 1 + int(diam/dx)
            dx = diam/nx
            xnew = np.linspace(-diam/2 + dx/2, diam/2 - dx/2, nx)  # new grid for intial field
            interp = interp1d(x, g, 'quadratic', fill_value='extrapolate')
            g = interp(xnew)
            x = xnew

        ns = int(np.round(diam + diam1)/dx)
        s = np.linspace(-diam/2 - diam1/2 + dx/2, diam/2 + diam1/2 - dx/2, ns) # spatial grid of extended kernel
        nt = int(np.round(diam1/dx))
        t = np.linspace(-diam1/2 + dx/2, diam1/2 - dx/2, nt)  # output spatial grid

        #calculate Fresnel convoltion kernel
        kern = np.exp(1j*np.pi*s*s/(lam*z))  # Fresnel kernel (Goodman 4-16)
        if maintain_dx:  # where does |s| exceed the max step for this dx?
            s_max = lam*z*self.params['max_chirp_step_deg']/(360*dx)
            null_ind = np.where(np.abs(s) > s_max)[0]
            kern[null_ind] = 0
        # propagated field is given by h*p
        h = conv(kern, g, mode='same', method='fft')  # h is on the s spatial grid
        interp = interp1d(s, h, 'quadratic')  # put h on the t spatial grid
        h = interp(t)

        #p = -1j*np.exp(2j*np.pi*z/lam)/(lam*z)  # prefactor - causes unwanted oscillations with z
        p = 1/(lam*z)
        if not return_derivs:
            return([p*h, t])
        #dpdz = (1j/(lam*z*z) + 2*np.pi/(lam*lam*z))*np.exp(2j*np.pi*z/lam)  # includes unwanted oscillations
        dpdz = -1/(lam*z*z)
        dkerndz = -1j*np.pi*s*s*kern/(lam*z*z)
        dhdz = conv(dkerndz, g, mode='same', method='fft')
        interp = interp1d(s, dhdz, 'quadratic')  # do same interpolation for derivative
        dhdz = interp(t)
        return([p*h, dpdz*h + p*dhdz, t])

    #This applies the pyramid's phase ramp, applies re-gridding if phase-step is too big
    # g - input field
    # x - input spatial grid
    # center - location of pyramid center with respect to x (same units as x)
    # return_derivs - duh
    # no_apex - if True, apex is removed and the beam is uniformly tilted
    def ApplyPyramidPhaseRamp1D(self, g, x, center, return_derivs=False, no_apex=False):
        if g.shape != x.shape:
            raise Exception("Input field and spatial grid must have same dimensions.")
        lam = self.params['wavelength']
        pslope = self.params['pslope']*np.pi/180
        pyrparams['indref']
        q = self.params['indref'] - 1  # index of refraction - 1
        ramp = 2*np.pi*q*np.tan(pslope)*np.abs(x - center)/lam  # z = -|x - center|*tan(theta)
        if no_apex:
            ramp = 2*np.pi*q*np.tan(pslope)*(x - center)/lam
        phase_diff = np.abs(ramp[1] - ramp[0])*180/np.pi
        if phase_diff > self.params['max_pyramid_step_deg']:  # if too big, we must resolve the phase ramp better
            dx, diam = self.GetDxAndDiam(x)
            dx *= self.params['max_pyramid_step_deg']/phase_diff
            nx = 1 + int(np.round(diam/dx))
            dx = diam/nx
            xnew = np.linspace(-diam/2 + dx/2, diam/2 - dx/2, nx)
            ramp = 2*np.pi*q*np.tan(pslope)*np.abs(xnew - center)/lam
            if no_apex:
                ramp = 2*np.pi*q*np.tan(pslope)*(xnew - center)/lam
            interp = interp1d(x, g, 'quadratic', fill_value='extrapolate')
            g = interp(xnew)
            x = xnew
        phasor = np.exp(1j*ramp)
        if not return_derivs:
            return([g*phasor, x])
        pos = np.where(x - center >= 0)[0]
        neg = np.where(x - center <  0)[0]
        xpos = x[pos]
        xneg = x[neg]
        aa = - np.ones(xpos.shape)*2*np.pi*q*np.tan(pslope)/lam  # deriv WRT center for positive part
        bb =   np.ones(xneg.shape)*2*np.pi*q*np.tan(pslope)/lam  # negative part
        d_phasor = 1j*phasor*np.hstack([bb,aa])
        return([g*phasor, g*d_phasor, x])

    #Apply thin lens phase transformation.
    # g - electric field impinging on lens
    # x - spatial coordinate
    # ceneter - center of lens relative to zero of x
    # fl - lens focal length (same units as x and wavelength)
    def ApplyThinLens1D(self, g, x, center, fl, return_derivs=False):
        if g.shape != x.shape:
            raise Exception("Input field and spatial grid must have same dimensions.")
        [dx, diam] = self.GetDxAndDiam(x)
        lam = self.params['wavelength']
        max_step = self.params['max_lens_step_deg']*np.pi/180
        dx_tol = max_step*lam*fl/(2*np.pi*(diam/2 + np.abs(center)))
        if dx > dx_tol:  # interpolate onto higher resolution grid
            nx = int(diam/dx_tol) + 1
            dx = diam/nx
            xnew = np.linspace(-diam/2 + dx/2, diam/2 + dx/2, nx)
            interp = interp1d(x, g, 'quadratic', fill_value='extrapolate')
            g = interp(xnew)
        else:
            xnew = x
        h = g*np.exp(-1j*np.pi*(xnew - center)*(xnew - center)/(fl*lam))
        if not return_derivs:
            return([h, xnew])
        dhdc = 2j*np.pi*(xnew - center)*h/(lam*fl)
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