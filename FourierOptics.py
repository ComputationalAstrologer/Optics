#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 12:35:10 2018
   by Richard A. Frazin
   
This is a class of Fourier Optics functions, in which all optical elements are treated as
  thin phase screens.  The Fresnel propagation algorithm, based purely on spatial domain
  convolution, appears to be original.  Note that it takes advantage of FFT acceleration.
   
"""

import numpy as np
from scipy.signal import convolve as conv
from scipy.interpolate import interp1d, interp2d

# It works on a square computation grid, specified by a 1D spatial
#  coordinate vector x.
#params is a dict of stuff you need, including:
# params['wavelength'] - lambda (recommend micron units)
# params['max_chirp_step_deg'] (degrees) controls Fresnel propogation accuracy is controlled by 
# params['max_lens_step_deg'] (degrees) may require increasing resolution for lens phase screen
# params['max_pyramid_phase_step'] may require increasing the resolutin for the pyramid screen
# params['interp_style'] is passed to scipy.interpolate.interp2d (used for up-sampling field)
# params['pyramid_slope_deg'] (degrees) is the slope of the pyramid relative to the horizontal

class FourierOptics():
    def __init__(self, params):
        self.params = params
        return

    #This applies the pyramid's phase ramp, applies re-gridding if phase-step is too big
    # g - input field
    # x - input spatial grid
    # center - location of pyramid center with respect to x (same units as x)
    # return_derivs - duh
    # no_apex - if True, apex is removed and the beam is uniformly tilted
    def ApplyPyramidPhaseRamp1D(self, g, x, center=0, no_apex=False, return_derivs=False):
        if g.shape != x.shape:
            raise Exception("Input field and spatial grid must have same dimensions.")
        if no_apex:
            assert not return_derivs
        lam = self.params['wavelength']
        pslope = self.params['pyramid_slope_deg']*np.pi/180
        q = self.params['indref'] - 1  # index of refraction - 1
        ramp = 2*np.pi*q*np.tan(pslope)*np.abs(x - center)/lam  # z = -|x - center|*tan(theta)
        if no_apex:
            ramp = 2*np.pi*q*np.tan(pslope)*(x - center)/lam
        phase_diff = np.abs(ramp[1] - ramp[0])*180/np.pi
        if phase_diff > self.params['max_pyramid_phase_step']:  # if too big, we must decrase dx
            dx = x[1] - x[0]
            dx_new = dx*self.params['max_pyramid_phase_step']/phase_diff
            [g, xnew] = self.ResampleField1D(g, x, dx_new)
            x = xnew
            ramp = 2*np.pi*q*np.tan(pslope)*np.abs(x - center)/lam
            if no_apex:
                ramp = 2*np.pi*q*np.tan(pslope)*(x - center)/lam

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

    #rotate (degrees) is the offset of the rotation of the pyramid (nominal is 45 deg)
    #   Zero rotation correspons to pyramid edges at 45 degrees to the x-y coords
    #tip (degrees) corresponds to rotating the pyramid about the apex in x-z plane
    #tilt (degress)                                                      y-z
    #   tip and tilt are relative to the pyramid axes
    def ApplyPyramidPhaseRamp2D(self, g, x, center=[0,0], rotate=0, tip=0, tilt=0, return_derivs=False):
        if g.shape[0] != x.shape[0]:
            raise Exception("ApplyPyramidPhaseRamp2D: input field and grid must have same sampling.")
        if g.ndim != 2:
            raise Exception("ApplyPyramidPhaseRamp2D: input field array must be 2D.")
        if g.shape[0] != g.shape[1]:
            raise Exception("ApplyPyramidPhaseRamp2D: input field array must be square.")
        if len(center) != 2:
            raise Exception("ApplyPyramidPhaseRamp2D: center parameter must have len=2.")

        lam = self.params['wavelength']
        indref = self.params['indref']
        tslope = np.tan(self.params['pyramid_slope_deg']*np.pi/180)
        dx, diam = self.GetDxAndDiam(x)
        phase_step = 360*(indref-1)*dx*tslope/lam
        if phase_step > self.params['max_pyramid_phase_step']:
            dxnew = self.params['max_pyramid_phase_step']*lam/(360*tslope)
            [g, x] = self.ResampleField2D(g, x, dxnew, kind=self.params['interp_style'])

        height = self.PyramidHeight2D(x, center, rotate, tip, tilt)
        ramp = np.exp(-2j*np.pi*height*(indref - 1)/lam)
        g *= ramp
        if not return_derivs:
            return([g, x])
        # Unfortunately, I think finite differences is the best option for these derivatives.
        #    Make sure the steps are not to small for the delta x

    def PyramidHeight2D(self, x, center, rotate, tip, tilt):
        #calculate the height of the pyramid face.  Rotate according to (rotate, tip, tilt) params
        tan_slope = np.tan(self.params['pyramid_slope_deg']*np.pi/180.)
        roof = self.params['pyramid_roofsize']/2
        nx = x.shape[0]
        [px, py] = np.meshgrid(x - center[0], x - center[1], indexing='xy')  # note coord shift
        xs = np.zeros(px.shape)
        ys = np.zeros(px.shape)
        zs = np.zeros(px.shape)  # face height (taken to be >= 0)
        rotate = (rotate + 45)*np.pi/180
        for k in range(nx):
            for l in range(nx):
                sx = px[k,l]*np.cos( rotate) + py[k,l]*np.sin(rotate)
                sy = px[k,l]*np.sin(-rotate) + py[k,l]*np.cos(rotate)
                th = np.arctan2(sy, sx)
                xs[k,l] = sx
                ys[k,l] = sy
                if (th >= -np.pi/4) and (th < np.pi/4):  # +x direction
                    if np.abs(sx) >= roof:
                        zs[k,l] = (np.abs(sx) - roof)*tan_slope
                elif (th >= np.pi/4) and (th < 3*np.pi/4):  # +y direction
                    if np.abs(sy) >= roof:
                        zs[k,l] = (np.abs(sy) - roof)*tan_slope
                elif (th >= 3*np.pi/4) or (th < -3*np.pi/4):  # -x direction
                    if np.abs(sx) >= roof:
                        zs[k,l] = (np.abs(sx) - roof)*tan_slope
                else:  # -y direction
                    if np.abs(sy) >= roof:
                        zs[k,l] = (np.abs(sy) - roof)*tan_slope
        # to apply tip, rotate about y-axis.  for tilt, rotate about x-axis
        if tip != 0:
            t = tip*np.pi/180 
            for k in range(nx):
                for l in range(nx):
                    xs[k,l], zs[k,l] = xs[k,l]*np.cos(t) + zs[k,l]*np.sin(t), xs[k,l]*np.sin(-t) + zs[k,l]*np.cos(t)
        if tilt != 0:
            t = tilt*np.pi/180
            for k in range(nx):
                for l in range(nx):
                    ys[k,l], zs[k,l] = ys[k,l]*np.cos(t) + zs[k,l]*np.sin(t), ys[k,l]*np.sin(-t) + zs[k,l]*np.cos(t)
        return(zs)


    #1D Fresenel prop using convolution in the spatial domain
    # g - vector of complex-valued field in the inital plane
    # x - vector of coordinates in initial plane, corresponding to bin center locaitons
    # diam_out - diameter of region to be calculated in output plane
    # z - propagation distance
    # set_dx = [True | False | dx (microns)]
    #   True  - forces full sampling of chirp according to self.params['max_chirp_step_deg']
    #           Note: this can lead to unacceptably large arrays.
    #   False - zeroes the kernel beyond where self.params['max_chirp_step_deg'] 
    #            exceedes the dx given in the x input array.  Note: the output dx will differ
    #            slightly from dx in the x input array.
    #...dx - sets the resolution to dx (units microns).  Note: the output dx will differ 
    #         slightly from this value
    # return_derivs - True to return deriv. of output field WRT z
    def ConvFresnel1D(self, g, x, diam_out, z, set_dx=True, return_derivs=False):
        if g.shape != x.shape:
            raise Exception("Input field and grid must have same dimensions.")
        lam = self.params['wavelength']
        dx, diam = self.GetDxAndDiam(x)
        dPhiTol_deg = self.params['max_chirp_step_deg']
        dx_chirp = (dPhiTol_deg/180)*lam*z/(diam + diam_out)  # sampling criterion for chirp (factors of pi cancel)
        if set_dx == False:
            dx_new = dx
        elif set_dx == True:  # use chirp sampling criterion
            dx_new = dx_chirp
        else:  # take dx_new to be value of set_dx
            if str(type(set_dx)) != "<class 'float'>":
                raise Exception("ConvFresnel1D: set_dx must be a bool or a float.")
            if set_dx <= 0:
                raise Exception("ConvFresnel1D: numerical value of set_dx must be > 0.")
            dx_new = set_dx

        if dx != dx_new:  # interpolate g onto a grid with spacing of (approx) dx_new
            [g, x] = self.ResampleField1D(g, x, dx_new)
            dx = x[1] - x[0]

        # make the kernel grid (s) match x as closely as possible
        ns = int(np.round(diam + diam_out)/dx)  # number of points on extended kernel
        s = np.linspace(-diam/2 - diam_out/2 + dx/2, diam/2 + diam_out/2 - dx/2, ns) # spatial grid of extended kernel
        indices_out = np.where(np.abs(s) < diam_out/2)[0] # get the part of s within the output grid

        #Calculate Fresnel convoltion kernel, (Goodman 4-16)
        #  Note: the factor p = 1/(lam*z) is applied later
        #  Also note: the factor -1j*np.exp(2j*np.pi*z/lam) causes unwanted oscillations with z
        kern = np.exp(1j*np.pi*s*s/(lam*z))  # Fresnel kernel
        if dx > dx_chirp:  # Where does |s| exceed the max step for this dx?
            s_max = lam*z*self.params['max_chirp_step_deg']/(360*dx)
            null_ind = np.where(np.abs(s) > s_max)[0]
            kern[null_ind] = 0
        h = conv(kern, g, mode='same', method='fft')  # h is on the s spatial grid

        p = 1/(lam*z)
        if not return_derivs:
            return([p*h[indices_out], s[indices_out]])
        #dpdz = (-1j/(lam*z*z) + 2*np.pi/(lam*lam*z))*np.exp(2j*np.pi*z/lam)  # includes unwanted oscillations
        dpdz = -1/(lam*z*z)
        dkerndz = -1j*np.pi*s*s*kern/(lam*z*z)
        dhdz = conv(dkerndz, g, mode='same', method='fft')
        s = s[indices_out]
        h = h[indices_out]
        dhdz = dhdz[indices_out]
        return([p*h, dpdz*h + p*dhdz, s])

    #2D Fresenel prop using convolution in the spatial domain
    # g - matrix of complex-valued field in the inital plane
    # x - vector of 1D coordinates in initial plane, corresponding to bin center locaitons
    #        it is assumed that the x and y directions have the same sampling.
    # diam_out - diameter of region to be calculated in output plane
    #    Note that the field is set to 0 outside of disk defined by r = diam_out/2.  This
    #      is because the chirp sampling criteria could be violated outside of this disk.
    # z - propagation distance
    # set_dx = [True | False | dx (microns)]
    #   True  - forces full sampling of chirp according to self.params['max_chirp_step_deg']
    #           Note: this can lead to unacceptably large arrays.
    #   False - zeroes the kernel beyond where self.params['max_chirp_step_deg'] 
    #            exceedes the dx given in the x input array.  Note: the output dx will differ
    #            slightly from dx in the x input array.
    #...dx - sets the resolution to dx (units microns).  Note: the output dx will differ 
    #         slightly from this value
    # return_derivs - True to return deriv. of output field WRT z
    def ConvFresnel2D(self, g, x, diam_out, z, set_dx=True, return_derivs=False):
        if g.shape[0] != x.shape[0]:
            raise Exception("ConvFresnel2D: input field and grid must have same sampling.")
        if g.ndim != 2:
            raise Exception("ConvFresnel2D: input field array must be 2D.")
        if g.shape[0] != g.shape[1]:
            raise Exception("ConvFresnel2D: input field array must be square.")

        lam = self.params['wavelength']
        dx, diam = self.GetDxAndDiam(x)
        dPhiTol_deg = self.params['max_chirp_step_deg']
        dx_chirp = (dPhiTol_deg/180)*lam*z/(diam + diam_out)  # sampling criterion for chirp (factors of pi cancel)
        if set_dx == False:
            dx_new = dx
        elif set_dx == True:  # use chirp sampling criterion
            dx_new = dx_chirp
        else:  # take dx_new to be value of set_dx
            if str(type(set_dx)) != "<class 'float'>":
                raise Exception("ConvFresnel2D: set_dx must be a bool or a float.")
            if set_dx <= 0:
                raise Exception("ConvFresnel2D: numerical value of set_dx must be > 0.")
            dx_new = set_dx

        if dx != dx_new:  # interpolate g onto a grid with spacing of approx dx_new
            [g, x] = self.ResampleField2D(g, x, dx_new, kind=self.params['interp_style'])
            dx = x[1] - x[0]

        # make the kernel grid (s) match x as closely as possible
        ns = int(np.round(diam + diam_out)/dx)  # number of points on extended kernel
        s = np.linspace(-diam/2 - diam_out/2 + dx/2, diam/2 + diam_out/2 - dx/2, ns) # spatial grid of extended kernel
        ind = np.where(np.abs(s) < diam_out/2)[0] # get the part of s within the 1D output grid
        [sx, sy] = np.meshgrid(s, s, indexing='xy')
        i_out = np.where(np.sqrt(sx*sx + sy*sy) > diam_out/2)

        #Calculate Fresnel convoltion kernel, (Goodman 4-16)
        #  Note: the factor p = 1/(lam*z) is applied later
        #  Also note: the factor -1j*np.exp(2j*np.pi*z/lam) causes unwanted oscillations with z
        kern = np.exp(1j*np.pi*(sx*sx + sy*sy)/(lam*z))  # Fresnel kernel
        if dx > dx_chirp:  # Where does |s| exceed the max step for this dx?
            s_max = lam*z*self.params['max_chirp_step_deg']/(360*dx)
            null_ind = np.where(np.sqrt(sx*sx + sy*sy) > s_max)
            kern[null_ind[0], null_ind[1]] = 0
        h = conv(kern, g, mode='same', method='fft')  # h is on the s spatial grid
        h[i_out[0], i_out[1]] = 0.  # zero the field outside the desired region
        h = h[ind[0]:ind[-1] + 1, ind[0]:ind[-1] + 1]
        p = 1/(lam*z)
        if not return_derivs:
            return([p*h, s[ind]])
        #dpdz = (-1j/(lam*z*z) + 2*np.pi/(lam*lam*z))*np.exp(2j*np.pi*z/lam)  # includes unwanted oscillations
        dpdz = -1/(lam*z*z)
        dkerndz = -1j*np.pi*(sx*sx + sy*sy)*kern/(lam*z*z)
        dhdz = conv(dkerndz, g, mode='same', method='fft')
        dhdz[i_out[0], i_out[1]] = 0.
        dhdz = dhdz[ind[0]:ind[-1] + 1, ind[0]:ind[-1] + 1]
        return([p*h, dpdz*h + p*dhdz, s[ind]])

    #Apply thin lens phase transformation.
    # g - electric field impinging on lens
    # x - spatial coordinate
    # center - center of lens relative to zero of x
    # fl - lens focal length (same units as x and wavelength)
    def ApplyThinLens1D(self, g, x, center, fl, return_derivs=False):
        if g.shape != x.shape:
            raise Exception("Input field and spatial grid must have same dimensions.")
        [dx, diam] = self.GetDxAndDiam(x)
        lam = self.params['wavelength']
        max_step = self.params['max_lens_step_deg']*np.pi/180
        dx_tol = max_step*lam*fl/(2*np.pi*(diam/2 + np.abs(center)))
        if dx > dx_tol:  # interpolate onto higher resolution grid
            [g, xnew] = self.ResampleField1D(g, x, dx_tol)
        else:
            xnew = x
        h = g*np.exp(-1j*np.pi*(xnew - center)*(xnew - center)/(fl*lam))
        if not return_derivs:
            return([h, xnew])
        dhdc = 2j*np.pi*(xnew - center)*h/(lam*fl)
        return([h, dhdc, xnew])

    # similar to 1D version, except center must be of length 2
    def ApplyThinLens2D(self, g, x, center, fl, return_derivs=False):
        if g.shape[0] != x.shape[0]:
            raise Exception("ApplyThinLens2D: input field and grid must have same sampling.")
        if g.ndim != 2:
            raise Exception("ApplyThinLens2D: input field array must be 2D.")
        if g.shape[0] != g.shape[1]:
            raise Exception("ApplyThinLens2D: input field array must be square.")
        if len(center) != 2:
            raise Exception("ApplyThinLens2D: center parameter must have len=2.")

        [dx, diam] = self.GetDxAndDiam(x)
        lam = self.params['wavelength']
        max_step = self.params['max_lens_step_deg']*np.pi/180
        dx_tol = max_step*lam*fl/(2*np.pi*(diam/2 + np.sqrt(center[0]**2 + center[1]**2)))
        if dx > dx_tol:  # interpolate onto higher resolution grid
            [g, x] = self.ResampleField2D(g, x, dx_tol, kind=self.params['interp_style'])

        [sx, sy] = np.meshgrid(x, x, indexing='xy')
        sx -= center[0]
        sy -= center[1]
        h = g*np.exp(-1j*np.pi*(sx*sx + sy*sy)/(fl*lam))
        if not return_derivs:
            return([h, x])
        dhdc0 = 2j*np.pi*sx*h/(lam*fl)
        dhdc1 = 2j*np.pi*sy*h/(lam*fl)
        return([h, [dhdc0, dhdc1], x])

    #Resample the field to achieve a target spatial resolution given by dx_new
    #  Returns resampled field and new x
    #g - input field
    #x - initial 1D spatial grid
    #dx_new (microns) - desired spatial resolution
    #kind, fill_value - keywords passed to scipy.interpolate.interp2d
    def ResampleField2D(self, g, x, dx_new, kind='cubic', fill_value=None):
        if g.shape[0] != x.shape[0]:
            raise Exception("ResampleField2D: input field and grid must have same sampling.")
        if g.ndim != 2:
            raise Exception("ResampleField2D: input field array must be 2D.")
        if g.shape[0] != g.shape[1]:
            raise Exception("ResampleField2D: input field array must be square.")

        dx, diam = self.GetDxAndDiam(x)
        nx = int(np.round(diam/dx_new))
        dxnew = diam/nx
        xnew = np.linspace(-diam/2 + dxnew/2, diam/2 - dxnew/2, nx)
        #interp2d doesn't like complex number.  So stupid.
        interp_real = interp2d(x, x, np.real(g), kind=kind, fill_value=fill_value)
        interp_imag = interp2d(x, x, np.imag(g), kind=kind, fill_value=fill_value)
        g = interp_real(xnew, xnew) + 1j*interp_imag(xnew, xnew)
        return([g, xnew])

    #Resample the field to achieve a target spatial resolution given by dx_new
    #  Returns resampled field and new x
    #g - input field
    #x - initial 1D spatial grid
    #dx_new (microns) - desired spatial resolution
    #kind, fill_value - keywords passed to scipy.interpolate.interp1d
    def ResampleField1D(self, g, x, dx_new, kind='quadratic', fill_value='extrapolate'):
        if g.shape != x.shape:
            raise Exception("ResampleField1D: input field and x must have the same shape.")
        dx, diam = self.GetDxAndDiam(x)
        nx = int(np.round(diam/dx_new))
        dxnew = diam/nx
        xnew = np.linspace(-diam/2 + dxnew/2, diam/2 - dxnew/2, nx)
        interp = interp1d(x, g, kind=kind, fill_value=fill_value)
        g = interp(xnew)
        return([g, xnew])

    # assuming a regular spaced grid with values at bin centers,
    #  get spacing and diameter from spatial grid x
    def GetDxAndDiam(self, x):  # assumes x values are bin centers
        nx = x.shape[0]
        dx = x[1] - x[0]
        diam = (x[-1] - x[0])*(1 + 1/(nx - 1))
        assert diam > 0
        assert dx > 0
        return([dx, diam])