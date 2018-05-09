#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
1D Lyot Coronagraph Model
Created on Fri Oct 14 22:25:53 2016
@author: rfrazin

"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as ss
import scipy.optimize as so
import cPickle as pickle

#%%
class spline_fit_Gaussian():
    def __init__(self, height, center, width, spline_density=1):
        assert width > 0 and height > 0
        self.np = 128
        self.std = width  # std dev
        self.center = center
        self.height = height
        self.spline_order = 1  # 1 works best for fitting Gaussians
        self.spline_density = spline_density
        self.ns = int(spline_density/self.std)  # number of splines
        self.x = np.linspace(-.5, .5, self.np)
        self.xs = np.linspace(-.5, .5, self.ns)  # spline centers
        self.g = height*np.exp(- (self.x - center)*(self.x - center)/2/self.std/self.std)
        return
    def spline_eval(self, amp):  # amp is the vector of spline amplitudes
        assert len(amp) == self.ns
        s = np.zeros(self.np)
        dxs = self.xs[1] - self.xs[0]
        for k in range(len(amp)):
            s += amp[k]*ss.bspline((self.x - self.xs[k])/dxs,
                                    self.spline_order)
        return s
    def spline_fit(self):  #return best fit to Gaussian
        B = np.zeros((self.np,self.ns))
        dxs = self.xs[1] - self.xs[0]
        for k in range(self.np):
            for l in range(self.ns):
                B[k,l] = ss.bspline((self.x[k] - self.xs[l])/dxs,
                                    self.spline_order)
        return np.linalg.inv(B.T.dot(B)).dot(B.T).dot(self.g)
    def plot(self):
        plt.figure()
        plt.plot(self.x,self.g, 'k')
        self.fit = self.spline_fit()
        plt.plot(self.x, self.spline_eval(self.fit))
#fit N Gaussians to the 1-D array f, and return coefficient vector
def gaussfit_shape(f, N, std=1.2):
    assert np.ndim(f) == 1
    assert (type(N) is int) and (N > 0)
    B = np.zeros((len(f), N))
    x = np.linspace(0, 1, len(f))
    xs = np.linspace(1./2/N, 1.-1./2/N, N)
    sig = std*(xs[1] - xs[0])
    for l in range(N):
        B[:, l] = np.exp(- (x - xs[l])*(x - xs[l])/sig/sig/2)
    return np.linalg.inv(B.T.dot(B)).dot(B.T).dot(f)
#%%    
def myfft(g):  # for centered arrays
    return np.fft.fftshift(np.fft.fft(np.fft.fftshift(g)))/np.sqrt(len(g))
#periodic sinc. supposed to do the same thing as MATLAB's diric.m,
#   except N = len(g).   g is as 1D array.  N must be a pos int
#   N is the 'width' parameter of the rect fcn of it's DFT.  
def psinc(g, N):
    assert np.ndim(g) == 1
    assert (type(N) is int) and (N > 0) 
    h = np.zeros(len(g))
    for k in range(len(g)):
        h[k] = np.sin(float(N)*g[k]/2)/(N*np.sin(g[k]/2))
        rem = np.mod(g[k], 2.*np.pi)
        if (np.abs(rem) < 1.e-12):
            m = int(np.round(g[k]/2./np.pi))
            h[k] = np.power(-1,m*(N-1))
    return h
#%%
#nfield - number nonzero values in entrance pupil
#npad - size of array that pads nfield
#occulter_size - the size of the occulter in the image plane in units of Lambda/D.
#nact - number of "actuators" on DM (spline points)
#--==> NOTE THAT COMUTATIONALLY THERE ARE nfield-1 ELEMENTS IN THE FIELD!!
#other params are mostly self-explanatory.   The bump* quantities correspond to a Gaussian perturbation in
# the entrance pupil.  The phot_count parameter is the peak number of counts in a pixel for a
# non-coronagraphic image (peak of (self.flat_reference_no_coronagraph)^2).  The converstion factor is
# self.ph_fac.
class lyot_1d():
    def __init__(self, nfield=128, npad=1024, occulter_size=1.99, nact=32,
                 apodizer_width = .2,
                 bump_height=0.05, bump_center = -0.011, bump_width = .071, asupp = .8,
                 dh_start=3., dh_end=9.,
                 phot_count=1.e6):
        assert nfield <= npad/2
        assert ((np.mod(nfield, 2) == 0) and (np.mod(npad, 2) == 0))
        self.nfield = nfield
        self.npad = npad
        self.nact = nact
        self.apw = apodizer_width  # width of amplitude pupil apodizer
        self.bump_height = bump_height  # bump param
        self.bump_center = bump_center  # bump param
        self.bump_width = bump_width    # bump param
        self.aber_support = asupp  # fraction of lyot stop width occupied by aber fcn
        self.dh_start = dh_start  # left limit of dark hole
        self.dh_end = dh_end      # right limit of dark hole
        self.phot_count = phot_count
        self.Lambda = 1.  # wavelength in microns
        self.D = 1.e6  # telescope diameter in microns
        self.bump = None  # pupil plane optical defect set by self.set_bump* functions
        self.fLambda_over_D = float(npad)/(nfield - 1)  # first zero of diffraction pattern (w/o coronagraph), f is effective focal length
        self.f = (float(npad)/(nfield - 1))*self.D/self.Lambda  # focal length in pixel units
        self.Lambda_over_D = float(self.Lambda)/self.D
        self.occulter_size = occulter_size*self.fLambda_over_D  # assumes self.mask as below
        self.mask = self.zp(self.pupil_apodizer())  # pupil plane multiplier
        self.occulter = np.ones(npad).astype('float')
        self.occulter[npad/2 - int(np.ceil(self.occulter_size)):
                      npad/2 + int(np.ceil(self.occulter_size)) + 1] = 0.
        self.flat_at_field_stop = self.prop_to_field_stop(np.ones(npad))
        self.lyot_peak = np.argmax(np.real(self.flat_at_field_stop[npad/2:-1]))
        self.lyot_stop_width = int(np.ceil(0.9*self.lyot_peak))
        self.lyot_mask = np.zeros(npad)
        self.lyot_mask[npad/2 - int(np.ceil(self.lyot_stop_width)): 
                       npad/2 + int(np.ceil(self.lyot_stop_width)) + 1] = 1
        self.flat_at_image_plane = self.prop_to_image_plane(np.ones(npad))  # field, not intensity
        self.flat_reference_no_coronagraph = myfft(myfft(myfft(self.mask)))  # field, not intensity
        tot = self.flat_reference_no_coronagraph*np.conj(self.flat_reference_no_coronagraph)
        tot = np.real(tot)  # intensity of unocculated image
        self.ph_fac = phot_count/np.max(tot)  # conversion factor from intensity to photons
        self.total_photons = self.ph_fac*np.sum(tot) # total number of photos in unocculted image 
        return
    #  this zero pad function gives rise to purely real myfft with symm. input
    def zp(self, f):  # zero-pad function for pupil fields
        assert len(f) == self.nfield - 1
        g = np.zeros(self.npad).astype('complex')
        g[self.npad/2 - self.nfield/2 + 1:self.npad/2 + self.nfield/2] = f
        return g
    def pupil_apodizer(self):
        n = (self.nfield-2)/2
        x = np.linspace(-.5,.5,2*n+1)
        return np.exp(- x*x/(2*self.apw*self.apw))
    # propagate the pupil plane field g to the Lyot field stop the coronagraph
    #  and apply occulting mask
    def prop_to_field_stop(self, g, plotstuff=False):
        assert g.shape[0] == self.npad
        g = g*self.mask
        if plotstuff:
            figpix = 44
            plt.figure(); plt.plot(np.abs(myfft(g))[self.npad/2-figpix:self.npad/2+figpix])
            plt.plot(self.occulter[self.npad/2-figpix:self.npad/2+figpix]*.5*np.max(np.abs(myfft(g))))
        if self.bump is None:
            g = myfft(myfft(g)*self.occulter)
        else:
            assert len(self.bump) == self.nfield - 1
            bump = np.ones(self.npad).astype('complex')
            bump[self.npad/2 - self.nfield/2 + 1:self.npad/2 + self.nfield/2] = self.bump
            g = myfft(myfft(g*bump)*self.occulter)
        return g     
    # propagate the orignal pupil plane field to the image plane, applying both
    #  occulter and Lyot mask.  This routine calls self.prop_to_field_stop
    def prop_to_image_plane(self, g):
        assert g.shape[0] == self.npad
        g = self.prop_to_field_stop(g)
        g = myfft(g*self.lyot_mask)
        return g
    def point_source_prop(self, angle, dm_command=None, multiplier=None):
        assert np.isscalar(angle) and np.isreal(angle)
        x = np.linspace(-self.D/2, self.D/2, self.nfield - 1)
        f = np.exp(1j*(2*np.pi/self.Lambda)*np.sin(-angle*self.Lambda_over_D)*x)
        if multiplier is not None:
            assert len(multiplier) == self.nfield - 1
            f *= multiplier
        if dm_command is None: 
            dm_command = np.zeros(self.nact)
        else:
            assert len(dm_command) == self.nact
        g = self.apply_DM(f, dm_command)
        return self.prop_to_image_plane(g)
    # returns phase advance by DM (ignores incidence angle)
    # command is the vector of DM height commands (nact elements) in units Lambda
    # propagate an off-axis point source through system
    # angle is units of self.Lambda_over_D 
    # multiplier multiplies the pupil plane field
    # sig is used in 'Gaussian' option
    def DM_phase(self, command, fcn = 'Gaussian', sig=1.2):
        assert len(command) == self.nact
        x = np.linspace(-0.50001, 0.5001, self.nfield - 1)  # height evaluation pts
        dx = 1./(self.nact - 1)  # distance between actuators (normalized)
        height = np.zeros(self.nfield - 1)
        for k in range(len(command)):
            center = k*dx - 0.5
            if fcn == 'B-Spline':
                height += command[k]*ss.bspline((x - center)/dx, 3)
            if fcn == 'Gaussian':
                d = 2.*sig*sig*dx*dx
                height += command[k]*np.exp(- (x - center)*(x - center)/d)
        height *= 4*np.pi
        return height
    # apply the DM to an array representing the input field (nfield - 1 elements)
    # dm_command is the DM command (nact elements)
    # dm_step0, dm_step1 are for calculating the pupil field gradient and
    #   Hessian with respect to a given DM command direction
    # if dm_step0 is None, then dm_step1 must be None
    def apply_DM(self, g, dm_command, dm_step0=None, dm_step1=None):
        assert len(g) == self.nfield - 1
        assert len(dm_command) == self.nact and dm_command.ndim == 1
        if dm_step0 is None:
            assert dm_step1 is None
        h = np.zeros(self.npad).astype('complex')
        h[self.npad/2 - self.nfield/2 + 1:self.npad/2 + self.nfield/2] \
              = g*np.exp(1j*self.DM_phase(dm_command))
        if dm_step0 is None:
            return h
        else:
            assert len(dm_step0) == self.nact
            dph = np.zeros(self.npad).astype('complex')
            dph[self.npad/2 - self.nfield/2 + 1:self.npad/2 + self.nfield/2] \
              = self.DM_phase(dm_step0)
            dh = h*1j*dph  # 1st derivative 
            if dm_step1 is None:
                return h, dh
            else:  # 2nd derivative calculation
                assert len(dm_step1) == self.nact
                dph1 = np.zeros(self.npad).astype('complex')
                dph1[self.npad/2 - self.nfield/2 + 1:self.npad/2 + self.nfield/2] \
                  = self.DM_phase(dm_step1)
                ddh = - h*dph*dph1
                return h, dh, ddh
    #create an aberration function in the pupil plane field.  
    #aberration is confined to the lyot_stop_width
    #sets value of self.bump. 
    #this is a zero-base Gaussian aberration with: 
    #    height in units of Lambda
    #    width (std. dev.), center in units of lyot stop width
    #support is the fraction of lyot_stop_width used by function
    def set_bump_Gaussian(self, height=None, center=None, width=None):
        if height is None:
            height = self.bump_height
        if center is None:
            center = self.bump_center
        if width is None:
            width = self.bump_width
        nsupp = int(np.floor(self.aber_support*self.lyot_stop_width))
        x = np.linspace(-.5, .5, 2*nsupp + 1)
        bump = height*np.exp(- (x - center)*(x - center)/(2*width*width))
        b = ((self.nfield - 1) - (2*nsupp + 1))/2
        bump = np.hstack((np.zeros(b),bump,np.zeros(b)))
        self.bump = np.exp(1j*2*np.pi*bump)
        return
    # this sets self.bump
    def set_bump_spline(self, coef, spline_order=1):
        s = self.my_spline_interp(coef, spline_order)
        self.bump = np.exp(1j*2*np.pi*s)
        return 
    # this returns the array of spline interpolated values
    def my_spline_interp(self, coef, spline_order=1):
        assert coef.ndim == 1
        nsupp = int(np.floor(self.aber_support*self.lyot_stop_width))
        x = np.linspace(-.5, .5, 2*nsupp + 1)
        xs = np.linspace(-.5, .5, len(coef))
        s = np.zeros(2*nsupp + 1)
        dxs = xs[1] - xs[0]
        for k in range(len(coef)):
            s += coef[k]*ss.bspline((x - xs[k])/dxs, spline_order)
        b = ((self.nfield - 1) - (2*nsupp + 1))/2
        s = np.hstack((np.zeros(b),s,np.zeros(b)))
        return s
    #this returns tuples of DM configurations and offpoint angles 
    #  to inform the regression
    def config_regress(self, dm_basis='Fourier'):
        dm_mat = None
        if dm_basis == 'Gaussian':
            angles = np.array([-4.])
            std = np.array([1.5, 3]).astype('float')  #widths of Gaussians
            x = np.linspace(0, self.nact - 1, self.nact)
            for s in std:
                c = -1
                while c <= self.nact:
                    c += s
                    f = np.exp(- (x-c)*(x-c)/2/s/s)
                    if dm_mat is None:
                        dm_mat = f
                    else:
                        dm_mat = np.vstack((dm_mat, f))
        elif dm_basis == 'Fourier':
            angles = np.array([-4.])
            ff = (2.*np.pi/self.nact)*np.linspace(0, self.nact/2-1, self.nact/2)
            x = np.linspace(0, self.nact-1, self.nact)
            for f in ff:
                gg = np.exp(-1j*f*x)
                if f < ff[1]:
                    g = np.ones(self.nact)
                else:
                    gr = gaussfit_shape(np.real(gg), self.nact, 1.2)
                    gi = gaussfit_shape(np.imag(gg), self.nact, 1.2)
                    g = np.vstack((gr, gi))
                if dm_mat is None:
                    dm_mat = g 
                else:
                    dm_mat = np.vstack((dm_mat, g))
        elif dm_basis == 'psinc2':
            angles = np.array([-4.0])
            width = self.nact/2.2
            phases = [0., .5*np.pi]
            center = [-.5*width, 0, .5*width]
            x = np.linspace(-np.pi, np.pi, self.nact)
            for ce in center:
                for ph in phases:
                    g = psinc(x, int(np.round(width)))*np.cos(ph + ce*x)
                    gc = gaussfit_shape(g, self.nact, 1.2)
                    if dm_mat is None:
                        dm_mat = gc
                    else:
                        dm_mat = np.vstack((dm_mat,gc))
        elif dm_basis == 'psinc':
            angles = np.array([0])
            phases = [0., np.pi, .5*np.pi, -.5*np.pi]
            #phases = [0., np.pi, .33*np.pi, -.67*np.pi, .17*np.pi, -.83*np.pi]  #make sure these are pairs that differ by pi 
            x = np.linspace(-np.pi, np.pi, self.nact)
            width = np.abs(self.dh_end - self.dh_start)
            center = (self.dh_start + self.dh_end)/2.
            for ph in phases:
                g = psinc(x, int(np.round(width)))*np.cos(ph + center*x)
                gc = gaussfit_shape(g, self.nact, 1.2)
                if dm_mat is None:
                    dm_mat = gc
                else:
                    dm_mat = np.vstack((dm_mat, gc))
        else:
            print "invalid dm_basis string"
            assert False
        return (angles, dm_mat)
    #get the derivative of the image plane field w.r.t. the DM command (Jacobian)
    #    (evaluated at dm_command)
    def jacobian_field(self, dm_command):
        jac = np.zeros((self.nact, self.npad)).astype('complex')
        for k in range(self.nact):
            dm_step = np.zeros(self.nact)
            dm_step[k] = 1.
            field, d_field = self.apply_DM(np.ones(self.nfield - 1), dm_command, dm_step)
            jac[k,:] = self.prop_to_image_plane(d_field)
        return jac
    # similar to jacobian_field, but returns Hessian
    def hessian_field(self, dm_command):
        hes = np.zeros((self.nact, self.nact, self.npad)).astype('complex')
        for k in range(self.nact):
            for l in range(self.nact):
                dm_step0 = np.zeros(self.nact)
                dm_step1 = np.zeros(self.nact)
                dm_step0[k] = 1.
                dm_step1[l] = 1.
                field, d_field, dd_field = self.apply_DM(np.ones(self.nfield - 1), \
                    dm_command, dm_step0, dm_step1)
                hes[k, l, :] = self.prop_to_image_plane(dd_field)
        return hes

#calculate the energy in a dark hole extending between self.dh_start and self.dh_end
#start and end are in units of self.fLambda/D (+ is right of center)
#energy units are in terms of MEAN constrast if contrast_units is set to True
#to include the bump or not, self.bump must be set before call:
#   either use self.bump = None, or self.set_bump_* routines
    def dark_hole_energy(self, dm_command, calc_grad=True, calc_Hessian=False,
                         contrast_units=False):
        assert len(dm_command) == self.nact
        if calc_grad is False:
            assert calc_Hessian is False
        if contrast_units is True:
            assert calc_grad is False
            g = self.flat_reference_no_coronagraph
            normal = np.max(np.real(g*np.conj(g))) # normalize to contrast units
        field = self.prop_to_image_plane(self.apply_DM(np.ones(self.nfield - 1), dm_command))
        p1 = int(np.round(self.dh_start*self.fLambda_over_D)) + self.npad/2
        p2 = int(np.round(self.dh_end*self.fLambda_over_D)) + self.npad/2 + 1
        assert p2 > p1, "problem with dark hole indicies"
        field = field[p1:p2]
        energy = np.real(np.sum(field*np.conj(field)))
        if calc_grad is False:
            if contrast_units is True:
                return energy/normal/(p2-p1)
            else:
                return energy
        else:  # find partial derivatives of energy w.r.t. dm_command
            jac = self.jacobian_field(dm_command)[:, p1:p2]
            grad = np.zeros(self.nact).astype('complex')
            for k in range(self.nact):
                grad[k] = np.sum(jac[k, :]*np.conj(field))
                grad[k] += np.conj(grad[k])
            grad = np.real(grad)
        if calc_Hessian is True:
            fhes = self.hessian_field(dm_command)[:, :, p1:p2]
            hes = np.zeros((self.nact, self.nact)).astype('complex')
            for k in range(self.nact):
                for l in range(self.nact):
                    hes[k, l] = np.sum(fhes[k, l, :]*np.conj(field) +
                                       jac[k]*np.conj(jac[l]))
                    hes[k, l] += np.conj(hes[k, l])
            hes = np.real(hes)
            return energy, grad, hes
        else:
            return energy, grad
    #return Hessian of the dark hole energy, this is just a wrapper
    def dark_hole_hessian(self, dm_command):
        energy, grad, hes = self.dark_hole_energy(dm_command,
                                                  calc_grad=True, calc_Hessian=True)
        return hes
    #find the DM values with the minimum energy, see self.dark_hole_energy
    #dm_command is an initial guess (ok to make random!)
    #t = 'Guassian', 'spline' or 'None'
    def find_min(self, dm_command, t='Gaussian', display=False,
                 spline_coef=None):
        assert len(dm_command) == self.nact
        if t == 'Gaussian':
            self.set_bump_Gaussian()
        elif t == 'spline':
            assert spline_coef is not None
            self.set_bump_spline(spline_coef)
        elif t == 'None':
            self.bump = None
        else:
            print "Unknown bump type"
            assert False
        opts = {'disp':display, 'xtol':1.e-9, 'return_all':True}    
        result = so.minimize(self.dark_hole_energy, dm_command,
                                        method='Newton-CG', options=opts,
                                        jac=True, hess=self.dark_hole_hessian)
        return (result.fun, result.x)

    #call self.find_min a bunch of times to find the best one
    #dm_amp is the smplitude of the random DM amplitude
    def find_best_min(self, spline_coef, dm_amp=.02, ntrials=100):
        assert np.ndim(dm_amp) == 0
        assert spline_coef.ndim == 1
        en_actual = np.zeros(ntrials)
        en_spline = np.zeros(ntrials)
        dmc = np.zeros((self.nact, ntrials))
        for k in range(ntrials):
            dmc[:,k] = dm_amp*np.random.randn(self.nact)
            en, dm = c.find_min(dmc[:,k], t='spline', spline_coef=spline_coef)
            en_spline[k] = en
            dmc[:,k] = dm
            self.set_bump_Gaussian()
            en_actual[k] = self.dark_hole_energy(dmc[:,k], False, False)
        return dmc, en_actual, en_spline
        
    #set plot_range is in units of self.fLambda_over_D, not applied if None
    def plot_solution(self, dm_command, plot_range, angle=0, bump='Gaussian',
                      spline_coef=None,
                      return_handles=False, color=None, label=None):
        assert len(plot_range) == 2
        assert len(dm_command) == self.nact and dm_command.ndim == 1
        if bump == 'Gaussian':
            self.bump = None
            self.set_bump_Gaussian()
        elif bump == 'spline':
            assert spline_coef is not None
            self.set_bump_spline(spline_coef)
        elif bump == 'None':
            self.bump = None
        else:
            assert False, "Unknown bump type."
        g = self.flat_reference_no_coronagraph
        normalization = np.max(np.real(g*np.conj(g)))
        g = self.point_source_prop(angle, dm_command)
        contrast = np.real(g*np.conj(g))/normalization
        pr = np.round(self.fLambda_over_D*np.array(plot_range)).astype('int')
        pixrange = np.arange(pr[1]-pr[0]) + self.npad/2 + int(np.round(pr[0]))
        x = np.linspace(plot_range[0], plot_range[1], len(pixrange))
        if not return_handles:
            plt.plot(x, np.log10(contrast[pixrange]))
            plt.xlabel('image plane coordinate ($\lambda$/D)')
            plt.ylabel('log_{10} contrast')
            return
        else:
            assert color is not None
            assert label is not None
            handle, = plt.plot(x, np.log10(contrast[pixrange]),
                               color=color, label=label, lw=2)
            return handle
            

    #Perform EFC, taking advantage of self.EFC_probe_cost and self.EFC_reg_cost
    # to find the optimal values of the probe amplitude and the regularization
    # parameter respectively.  Thus, these parameters are very highly optimized.
    # Note that a large upper bound on the regularization parameter in thhe
    #  optimization of self.EFC_reg_cost effectively prevents the solution from
    #  worsening due to photon noise.
    # When noise is not included, mirror nonlinearity error is eliminated by 
    #  by a probe size of 10^-5 since there are no aberrations downstream of
    #  the DM.    
    def do_EFC(self, nsteps=10, incl_noise=True, mk_plots=False, reg=.001,
               return_probe_amp=False):
        DHen = np.zeros(nsteps)  # DH energy array
        dmc = np.zeros((self.nact, nsteps))  # DM command array
        probe_amp = np.zeros(nsteps)
        dm = np.zeros(self.nact)
        # setting probe_amp[0] = 0.00258 starts
        #   the probing with probe field at contrast level 1.e-4
        # see self.EFC_regess (return_probe_contrast=True)
        probe_amp[0] = 0.0258
        # get the optimal probe amplitude for the 0th iteration
        dmc[:, 0] = dm
        self.set_bump_Gaussian()
        DHen[0] = self.dark_hole_energy(dm, calc_grad=False, contrast_units=True)
        if incl_noise:
            # this should work now that bug in EFC_regress is fixed
            #res = so.minimize_scalar(self.EFC_probe_cost, args=dm,
            #                         bounds=[1.e-9, .03], method='bounded')
            #probe_amp[0] = res.x
            print "initial probe amplitude: ", probe_amp[0]
            fe, er = self.EFC_regress(dm, probe_amp[0], incl_noise=True,
                                      incl_bump=True, make_plots=mk_plots)
        else:
            probe_amp[0] = 1.e-5
            fe = self.EFC_regress(dm, probe_amp[0], incl_noise=False,
                                  incl_bump=True)
        ddm = self.EFC_step(fe, dm, max_step=None, svd_bound=None, reg_const=reg)
        dm += ddm
        for k in 1 + np.array(range(nsteps - 1)):
            dmc[:, k] = dm
            self.set_bump_Gaussian()
            DHen[k] = self.dark_hole_energy(dm, calc_grad=False, contrast_units=True)
            if incl_noise:
                probe_amp[k] = probe_amp[0]*np.sqrt(np.sqrt(DHen[k]/DHen[0]))
                fe, er = self.EFC_regress(dm, probe_amp[k], incl_noise=True,
                                          incl_bump=True, make_plots=mk_plots)
            else: 
                probe_amp[k] = probe_amp[0]
                fe = self.EFC_regress(dm, probe_amp[k], incl_noise=False,
                                      incl_bump=True)
            ddm= self.EFC_step(fe, dm, max_step=None, svd_bound=None, reg_const=reg)
            dm += ddm
        if return_probe_amp:
            return DHen, dmc, probe_amp
        else: 
            return DHen, dmc
    #This optimal size of the EFC probe is a balance between the SNR benefits
    # of a large probe step and nonlinearity inherent in a large probe step.
    # Here, we have the benefit of knowing the true field, which  can be used
    # to define the cost of a given probe size, as well as known shot noise 
    # statistics returned by EFC_regress.
    #amp is the probe amplitude
    #dm is the current state of the DM
    def EFC_probe_cost(self, amp, dm):
        assert np.ndim(amp) == 0
        assert np.ndim(dm) == 1 and len(dm) == self.nact
        self.set_bump_Gaussian()
        ft = self.point_source_prop(0,dm)  # true value of field
        n0 = np.round(self.fLambda_over_D*self.dh_start).astype('int')  # dh begin pixel
        n0 += self.npad/2
        n1 = np.round(self.fLambda_over_D*self.dh_end).astype('int')  # dh end pixel
        n1 += self.npad/2
        et = ft[n0:n1+1]
        #get error due to nonlinearity of DM deformation, so incl_bump = False
        e1 = self.EFC_regress(dm, amp, incl_noise=False, incl_bump=False)[n0:n1+1]
        er1 = np.real((et - e1)*np.conj(et - e1))
        e2, er2 = self.EFC_regress(dm, amp, incl_noise=True, incl_bump=False)
        er2 = er2[n0:n1+1,:]
        er = er1 + np.sum(er2, axis=1)
        return np.sum(er)
    #do the regression step of EFC, to estimate the field
    #dmc is the base DM command, about which probe steps are applied
    #amp (scalar) is the amplitude of the probe steps
    #incl_noise adds Gaussian shot noise -> returns error variances due to shot noise
    #incl_bump=True includes self.bump when calculating "measured" intensities. 
    #   useful for evaluating error due to DM nonlinearity (in resulting field)
    #returns only field estimate if incl_noise is False.  also returns error if True
    def EFC_regress(self, dmc, amp, incl_noise=False, incl_bump=True,
                    make_plots=False, return_probe_contrast=False):
        assert np.ndim(dmc) == 1 and len(dmc) == self.nact
        assert np.ndim(amp) == 0
        if make_plots:
            assert incl_noise
        if incl_bump:
            self.set_bump_Gaussian()  # to get "real" intensity measurements
        else:
            self.bump = None  # turn off bump
        e_est = np.zeros(self.npad).astype('complex')
        probe = amp*self.config_regress(dm_basis='psinc')[1]  # [4 or 6,nact] matrix
        if return_probe_contrast:  # bump not included
            self.bump = None
            g = self.flat_reference_no_coronagraph
            normal = np.max(np.real(g*np.conj(g))) # normalize to contrast units
            e0 = self.point_source_prop(0, dm_command=(dmc + probe[0,:]))
            e1 = self.point_source_prop(0, dm_command=(dmc + probe[1,:]))
            e2 = self.point_source_prop(0, dm_command=(dmc + probe[2,:]))
            e3 = self.point_source_prop(0, dm_command=(dmc + probe[3,:]))            
            dI0 = np.real((e0 - e1)*np.conj(e0 - e1))/normal
            dI2 = np.real((e2 - e3)*np.conj(e2 - e3))/normal
        else:  # bump may be included
            e0 = self.point_source_prop(0, dm_command=(dmc + probe[0,:]))
            e1 = self.point_source_prop(0, dm_command=(dmc + probe[1,:]))
            e2 = self.point_source_prop(0, dm_command=(dmc + probe[2,:]))
            e3 = self.point_source_prop(0, dm_command=(dmc + probe[3,:]))            
        if not incl_noise:
            i0 = np.real(e0*np.conj(e0)) - np.real(e1*np.conj(e1))
            i2 = np.real(e2*np.conj(e2)) - np.real(e3*np.conj(e3)) 
        else:
            err = np.zeros((self.npad,2))  # real,imag error variance
            u0 = np.real(e0*np.conj(e0))*self.ph_fac  # convert to counts
            du = np.sqrt(u0)*np.random.randn(self.npad)  # photon noise
            u0n = (u0 + du)/self.ph_fac # convert back 
            u1 = np.real(e1*np.conj(e1))*self.ph_fac  # convert to counts
            du = np.sqrt(u1)*np.random.randn(self.npad)  # photon noise
            u1n = (u1 + du)/self.ph_fac # convert back
            u2 = np.real(e2*np.conj(e2))*self.ph_fac  # convert to counts
            du = np.sqrt(u2)*np.random.randn(self.npad)  # photon noise
            u2n = (u2 + du)/self.ph_fac # convert back 
            u3 = np.real(e3*np.conj(e3))*self.ph_fac  # convert to counts
            du = np.sqrt(u3)*np.random.randn(self.npad)  # photon noise
            u3n = (u3 + du)/self.ph_fac # convert back            
            i0 = u0n - u1n
            i2 = u2n - u3n
        n0 = np.round(self.fLambda_over_D*self.dh_start).astype('int')  # dh begin pixel
        n0 += self.npad/2
        n1 = np.round(self.fLambda_over_D*self.dh_end).astype('int')  # dh end pixel
        n1 += self.npad/2
        if n1 < n0:
            n0, n1 = n1, n0  # cool way to swap in Python!
        if return_probe_contrast:
            return .5*np.mean(dI0[n0:n1+1] + dI2[n0:n1+1])
        if make_plots:
            xx = np.linspace(self.dh_start, self.dh_end, 1+n1-n0)
            crap0 = u0[n0:n1+1] - u1[n0:n1+1]
            ecrap0 = np.sqrt(u0[n0:n1+1] + u1[n0:n1+1])
            crap2 = u2[n0:n1+1] - u3[n0:n1+1]
            ecrap2 = np.sqrt(u2[n0:n1+1] + u3[n0:n1+1])
            f, ax = plt.subplots(2, 1, sharex=True, sharey=False)
            ax[0].errorbar(xx, crap0, ecrap0, lw=2)
            ax[1].set_xlabel('pixel position ($\lambda$/D)', fontsize='large')
            ax[0].set_ylabel('intensity diff. (counts)', fontsize='large')
            ax[0].set_title('probes 1 and 2')
            ax[1].errorbar(xx, crap2, ecrap2, lw=2)
            ax[1].set_ylabel('intensity diff. (counts)', fontsize='large')
            ax[1].set_title('probes 3 and 4')            
        self.bump = None  # use approx system to estimate DM influence matrix
        DMin = self.jacobian_field(dmc).T   #[self.npad, self.nact] DMin fluence matrix 
        for k in np.linspace(n0,n1,1+n1-n0).astype('int'):
            y = np.array([[i0[k]],[i2[k]]])  #,[i4[k]]])  #  this assumes pairs of opposite probes
            A = 4*np.array([[np.real(DMin[k,:]).dot(probe[0,:]),
                             np.imag(DMin[k,:]).dot(probe[0,:])],
                            [np.real(DMin[k,:]).dot(probe[2,:]), 
                             np.imag(DMin[k,:]).dot(probe[2,:])]])  #,
            if incl_noise:
                Cov = np.array([[u0[k]+u1[k], 0],  # error covariance
                                [0, u2[k]+u3[k]]])/self.ph_fac/self.ph_fac
                iCov = np.linalg.inv(Cov)
            else:
                iCov = np.eye(len(y))
            Q = np.linalg.inv(A.T.dot(iCov.dot(A)))
            x = Q.dot(A.T.dot(iCov.dot(y)))
            e_est[k] = x[0][0] + 1j*x[1][0]
            if incl_noise:
                err[k,0] = Q[0,0]
                err[k,1] = Q[1,1]
        if make_plots:
            ind = np.linspace(n0,n1,1+n1-n0).astype('int')
            f, ax = plt.subplots(2, 1, sharex=True, sharey=False)
            ax[0].errorbar(xx, np.real(e_est[ind]), np.sqrt(err[ind,0]), lw=2)
            ax[1].errorbar(xx, np.imag(e_est[ind]), np.sqrt(err[ind,1]), lw=2)
            ax[1].set_xlabel('pixel position ($\lambda$/D)', fontsize='large')
            ax[0].set_ylabel('real(field value)', fontsize='large')
            ax[1].set_ylabel('imag(field value)', fontsize='large')
            ax[0].set_title('dark hole field estimate', fontsize='large')
        if incl_noise:
            return e_est, err
        else: 
            return e_est
    #Take a DM step to implement EFC with Tikhonov regularization
    #fe is estimated value of the field from self.EFC_regress
    #dm is the current state of the DM
    #3 methods to pick regularization parameter:
    #  set max_step , svd_bound and reg_const are None
    #  set svd_bound (.1 is good), max_step and reg_conts are None
    #  set reg_const is const regularization param (.001 is good)
    #      svd_bound, max_step must be None
    def EFC_step(self, fe, dm, max_step=None, svd_bound=None, reg_const=.001):
        assert np.ndim(fe) == 1 and len(fe) == self.npad
        assert np.ndim(dm) == 1 and len(dm) == self.nact
        i0 = np.round(self.fLambda_over_D*self.dh_start).astype('int')  # dh begin pixel
        i0 += self.npad/2
        i1 = np.round(self.fLambda_over_D*self.dh_end).astype('int')  # dh end pixel
        i1 += self.npad/2    
        if i1 < i0:
            i0, i1 = i1, i0  # cool way to swap in Python!
        ee = fe[i0:i1+1]
        self.bump = None
        H = self.jacobian_field(dm)[:,i0:i1+1].T  #DM influence matrix
        HTH = np.real(np.conj(H.T).dot(H))
        ev_max = np.real(np.linalg.eig(HTH)[0][0])
        if max_step is not None:
            assert svd_bound is None
            assert reg_const is None
            assert max_step < 1. and max_step > 0.
            r = 1.e-12*ev_max  # regularization parameter
            step_size = 2.*max_step
            while step_size > max_step:
                M = np.linalg.inv(HTH + r*np.eye(self.nact))    
                step = - M.dot(np.real(np.conj(H.T).dot(ee)))
                step_size = np.max(np.abs(step))
                r = 1.5*r
        elif svd_bound is not None:  #use knowledge of true system
            assert reg_const is None
            assert max_step is None
            self.junk = (dm, H, HTH, ee, ev_max)
            reg_res = so.minimize_scalar(self.EFC_reg_cost, bounds=[0, svd_bound],
                                         method='bounded')
            r = reg_res.x
            print 'reg_param/ev_max = ', r
            M = np.linalg.inv(HTH + ev_max*r*np.eye(self.nact))
            step = - M.dot(np.real(np.conj(H.T).dot(ee)))
            self.junk = None
        elif reg_const is not None:
            assert svd_bound is None
            assert max_step is None
            assert reg_const > 0.
            M = np.linalg.inv(HTH + ev_max*reg_const*np.eye(self.nact))
            step = - M.dot(np.real(np.conj(H.T).dot(ee)))
        else:
            assert False, "parameters chosen incorrectly."
        return step
    #This does the probe estimation of the field in the DH for EFC.
    #First it determines the optimal probe amplitude
    #cost function to find regularization parameter for the EFC step.
    #This method uses knowledge of the true system, so it cannot be
    #  realized in practice.
    #r is the regularization parameter
    #HTH matrix - see self.EFC_step
    def EFC_reg_cost(self, r):
        dm = self.junk[0]
        H = self.junk[1]
        HTH = self.junk[2]
        ee = self.junk[3]
        ev_max = self.junk[4]
        M = np.linalg.inv(HTH + r*ev_max*np.eye(self.nact))
        step = - M.dot(np.real(np.conj(H.T).dot(ee)))
        self.set_bump_Gaussian()
        return self.dark_hole_energy(dm + step, calc_grad=False)
    #regression on the spline coefficients without noise
    #n_images is the number of images used in the regression, each assoicated with a
    #   random DM deformation
    #n_spline is the number of spline coefficients to be estimated
    #self.dm_amp is the std. deviation of the DM amplitudes (units of self.Lambda)
    #offpoint (in units of self.Lambda_over_D)
    #FOV (in units of self.fLambda_over_D) is the field-of-view used in the regression
    #height, center width are passsed to self.point_source_prop
    #reg is a regularization parameteterv
    def spline_regress(self, dm_basis='Fourier', dm_amp = 0.2, FOV = 15,
                       spline_order=1, spline_density = 3, reg = 1.e-14,
                       incl_noise=True, make_plots=False, return_error_only=False):
        npix = int(2*self.fLambda_over_D*FOV)  # no. of pixels taken per image
        assert npix < self.npad - 2
        self.dm_amp = dm_amp
        self.FOV = FOV
        self.spline_order = spline_order
        # create a Gaussian bump and fit a b-spline
        sfg = spline_fit_Gaussian(self.bump_height, self.bump_center,
                                  self.bump_width, spline_density)
        coef_opt = sfg.spline_fit()
        n_spline = len(coef_opt)
        self.offpoint, self.dm_trials = self.config_regress(dm_basis)
        n_exposures = len(self.offpoint)*np.shape(self.dm_trials)[0]
        print "This sequence has ", n_exposures, "exposures."
        n0 = np.shape(self.dm_trials)[0]
        n_images = len(self.offpoint)*n0
        D = np.zeros((n_images*npix, n_spline))  # system matrix
        y = np.zeros((n_images*npix, 1))  # measured image
        y0 = np.zeros((n_images*npix, 1))  # aberration free image
        self.pixrange = np.arange(npix).astype('int') + self.npad/2 - npix/2
        for n in range(len(self.offpoint)):
            angle = self.offpoint[n]
            for k in range(np.shape(self.dm_trials)[0]):
                i0 = n*n0*npix + k*npix
                i1 = i0 + npix
                dm = self.dm_amp*self.dm_trials[k, :]
                self.bump = None  # get rid of bump
                g = self.point_source_prop(angle, dm_command=dm)
                y0[i0:i1, 0] = np.real(g*np.conj(g))[self.pixrange]
                self.set_bump_Gaussian()  # self.set_bump_spline(coef_opt); 
                g = self.point_source_prop(angle, dm_command=dm)
                y[i0:i1, 0] = np.real(g*np.conj(g))[self.pixrange]
                self.bump = None
                for l in range(n_spline):
                    co = np.zeros(n_spline)
                    co[l] = 1
                    spl = self.my_spline_interp(co, spline_order)
                    a =  1j*2*np.pi*self.point_source_prop(angle,
                                            dm_command=dm, multiplier=spl)
                    b = np.conj(self.point_source_prop(angle, dm_command=dm))
                    D[i0:i1, l] = 2*np.real(a*b)[self.pixrange]
        if incl_noise:  # add shot noise
            iCov = self.ph_fac/y.T[0]  # diag of inverse shot noise covariance
            dy = ((np.sqrt(self.ph_fac*y.T)*np.random.randn(len(y))).T)/self.ph_fac
            yn = y + dy
        else:
            yn = y
            iCov = np.ones(len(y))
        # force 0th coef to 0, makes matrix numerically invertible
        D = np.vstack((D, np.zeros(n_spline)))
        D[-1, 0] = np.amax(D, axis=(0, 1))
        yn = np.vstack((yn, 0))
        y0 = np.vstack((y0, 0))
        y = np.vstack((y, 0))
        iCov = np.hstack((iCov, np.max(iCov)))
        #
        DTD = D.T.dot(np.diag(iCov)).dot(D)
        eigvals = np.linalg.eigvalsh(DTD)
        regmat = reg*np.max(eigvals)*np.eye(n_spline)
        DTD += regmat
        iDTD = np.linalg.inv(DTD)
        coef_err = np.sqrt(np.diag(iDTD))
        coef_est = iDTD.dot(D.T).dot(np.diag(iCov)).dot(yn - y0)
        coef_est = np.reshape(coef_est, (len(coef_est),))
        if return_error_only:
            return coef_err
        if make_plots:
            plt.figure()
            plt.plot((y-y0)*self.ph_fac, 'g-')
            plt.plot((yn-y0)*self.ph_fac, 'c-')
            plt.plot(D.dot(coef_opt)*self.ph_fac, 'r--')
            plt.plot(D.dot(coef_est)*self.ph_fac, 'b:')
            plt.title('y-y0 (counts)')
            plt.figure()
            plt.plot(np.log10(np.linalg.eigvalsh(D.T.dot(D))), 'go')
            plt.title('log10 eigenvalues')
            plt.figure()
            ax = np.linspace(-self.aber_support*.5, self.aber_support*.5, len(coef_opt))
            # if incl_noise:
            #    plt.errorbar(range(len(coef_est)), coef_est, coef_err)
            # plt.plot(range(len(coef_est)), coef_est, 'ks-')
            plt.plot(ax, coef_opt, 'ro:')
            # get nonlinear solution (important!)
        self.regmat = regmat
        result = so.minimize(self.cost_nonlin, coef_est, method='Powell',
                             jac=False, args=(yn, D[-1, 0], iCov),
                             options={'xtol': 1.e-6, 'ftol': 1.e-6})
        if make_plots:
            if incl_noise:
                plt.errorbar(ax, result.x, coef_err)
            else:
                plt.plot(ax, result.x, 'kx')
            plt.title('spline coefficients')
            plt.ylabel('spline value (units of $\lambda$)')
            plt.xlabel('pupil plane position (units of D)')
        return result.x

# Calculate least-square cost of a given spline coefficient vector
# To be called by a member function that does linear regression first.
# coef - vector of coefficients to be optimized
# y - noisy (or not) measurements (col vector)
# maxD is the value used to pin the 0th coefficient to 0
#   (note the y and iCov need to already account for this extra component).
# iCov - diagonal elements of inverse covariance matrix, set to unity otherwise
# some member variables are set by self.spline_regress
# gradient and hessian functions have been removed due to bugs
    def cost_nonlin(self, coef, y, maxD, iCov=None,
                    calc_grad=False, calc_hess=False):
        assert calc_grad is False and calc_grad is False
        assert coef.ndim == 1
        assert y.ndim == 2 and y.shape[1] == 1  # true for column vector
        assert maxD.ndim == 0
        if iCov is not None:
            assert iCov.ndim == 1 and len(iCov) == len(y)
        else:
            iCov = np.ones(len(y))
        npix = int(2*self.fLambda_over_D*self.FOV)
        n0 = np.shape(self.dm_trials)[0]
        n_images = len(self.offpoint)*n0
        z = np.zeros((n_images*npix, 1))
        for n in range(len(self.offpoint)):
            angle = self.offpoint[n]
            for k in range(np.shape(self.dm_trials)[0]):
                i0 = n*n0*npix + k*npix
                i1 = i0 + npix
                dm = self.dm_amp*self.dm_trials[k, :]
                self.set_bump_spline(coef, self.spline_order)
                g = self.point_source_prop(angle, dm_command=dm)
                z[i0:i1, 0] = np.real(g*np.conj(g))[self.pixrange]
        # add final equation (makes 0th value of coef = 0)
        z = np.vstack((z,maxD*coef[0]))        
        cost = (y-z).T.dot(np.diag(iCov).dot(y-z))[0][0]
        cost += coef.T.dot(self.regmat.dot(coef))
        return cost

#if __name__ == "__main__":

if False:  # make pickle
    ntrials = 9
    spl_dens = 2.5
    pickle_name = 'experiment_3.pickle'
    c = lyot_1d(bump_height=.1, phot_count=3.e4*40./31.)  # 30 EFC iterations
    en_3 = np.zeros(ntrials)  # holds mean contrast
    dm_3 = np.zeros((c.nact, ntrials))
    coef_est_3 = None
    for k in range(ntrials):
        print k
        coef_est = c.spline_regress(spline_density=spl_dens)
        if coef_est_3 is None:
            coef_est_3 = np.zeros((len(coef_est), ntrials))
        coef_est_3[:, k] = coef_est
    for k in range(ntrials):
        print k
        en, dm = c.find_min(np.zeros(c.nact), t='spline',
                            spline_coef=coef_est_3[:, k])
        dm_3[:, k] = dm
        c.set_bump_Gaussian()
        en_3[k] = c.dark_hole_energy(dm, calc_grad=False, contrast_units=True)
    info_3 = {'nphot': c.phot_count, 'height': c.bump_height, 'spline_density': spl_dens}
    fp = open(pickle_name, 'w')
    pickle.dump((coef_est_3, en_3, dm_3, info_3), fp)
    fp.close()
   
if False: # make pickle
    ntrials = 9
    spl_dens = 2.5
    pickle_name = 'experiment_1.pickle'
    c = lyot_1d(bump_height=.1, phot_count=1.e7*80./31.)  # 20 EFC iterations
    en_0 = np.zeros(ntrials)  # holds mean contrast
    dm_0 = np.zeros((c.nact, ntrials))
    coef_est_0 = None
    for k in range(ntrials):
        print k
        coef_est = c.spline_regress(spline_density=spl_dens)
        if coef_est_0 is None:
            coef_est_0 = np.zeros((len(coef_est), ntrials))
        coef_est_0[:, k] = coef_est
    for k in range(ntrials):
        print k
        en, dm = c.find_min(np.zeros(c.nact), t='spline',
                            spline_coef=coef_est_0[:, k])
        dm_0[:, k] = dm
        c.set_bump_Gaussian()
        en_0[k] = c.dark_hole_energy(dm, calc_grad=False, contrast_units=True)
    info = {'nphot': c.phot_count, 'height': c.bump_height, 'spline_density': spl_dens}
    fp = open(pickle_name, 'w')
    pickle.dump((coef_est_0, en_0, dm_0, info), fp)
    fp.close()

if False:  #almost make a pickle
    pickle_name = 'experiment_1.pickle'
    c1 = lyot_1d(bump_height=.1, phot_count=1.e7*80./31.)
    fp = open(pickle_name, 'r')
    (coef_est_1, en_1, dm_1, info) = pickle.load(fp)
    fp.close()
    print info
    print c1.phot_count
    ntrials = len(en_1)
    en_1o = np.zeros(en_1.shape)
    dm_1o = np.zeros(dm_1.shape)
    for k in range(ntrials):
        print k
        dmc, en_tru, en_spl = c1.find_best_min(coef_est_1[:, k], dm_amp=.03, ntrials=8)
        best_ind = np.argmin(en_spl)
        dm_1o[:, k] = dmc[:, best_ind]
        c1.set_bump_Gaussian()
        en_1o[k] = c1.dark_hole_energy(dm_1o[:,k], calc_grad=False, contrast_units=True)

if False:    # load pickle #3
    pickle_name = 'experiment_3.pickle'
    c3 = lyot_1d(bump_height=.1, phot_count=3.e4*40./31.)
    fp = open(pickle_name, 'r')
    (coef_est_3, en_3, dm_3, info_3) = pickle.load(fp)
    fp.close()
    print info_3
    print c3.phot_count
        
make_figs = False
save_figs = False

if make_figs:
    reg = .001  # this works best
    c6 = lyot_1d(bump_height=.1, phot_count=1.e6)
    DHen6, dmc6 = c6.do_EFC(100, True, False, reg=reg)
    c9 = lyot_1d(bump_height=.1, phot_count=1.e9)
    DHen9, dmc9 = c9.do_EFC(100, True, False, reg=reg)

    pickle_name = 'experiment_2.pickle'
    c2 = lyot_1d(bump_height=.1, phot_count=1.e4*40./31.)
    fp = open(pickle_name, 'r')
    (coef_est_2, en_2, dm_2, info_2) = pickle.load(fp)
    fp.close()
    print info_2
    print c2.phot_count

    pickle_name = 'experiment_1.pickle'
    c1 = lyot_1d(bump_height=.1, phot_count=1.e7*80./31.)
    fp = open(pickle_name, 'r')
    (coef_est_1, en_1, dm_1, info_1) = pickle.load(fp)
    fp.close()
    print info_1
    print c1.phot_count

    # spline fit stuff
    spl_dens = 2.5
    sfg = spline_fit_Gaussian(c6.bump_height, c6.bump_center,
                              c6.bump_width, spl_dens)
    coef_opt = sfg.spline_fit()
    n_spline = len(coef_opt)
    aber_opt = sfg.spline_eval(coef_opt)
    aber_fit = sfg.spline_eval(coef_est_2[:,0])
    plt.figure()
    h0, = plt.plot(sfg.x, 360*sfg.g, 'r-', lw=1, label="true aberration")
    h1, = plt.plot(sfg.x, 360*aber_opt, 'k:', lw=3, label="spline fit, no noise")
    h2, = plt.plot(sfg.x, 360*aber_fit,'b-', lw=2, label="spline fit, 4.7e6 photons")
    plt.ylabel("phase delay (deg)", fontsize='large')
    plt.xlabel("pupil plane position (D)", fontsize='large')
    plt.title("entrance pupil phase aberration", fontsize='large')
    plt.axis([-.5,.5,-2,55])
    plt.legend(handles=[h0, h1, h2],loc=1)
    if save_figs:
        plt.savefig('figs/entrance_aber_fit.eps')
    
    coef_err = c2.spline_regress(spline_density=2.5, return_error_only=True)
    plt.errorbar(range(n_spline),coef_est_2[:,0],coef_err,lw=2)
    h2, = plt.plot(range(n_spline),coef_est_2[:,0], 'b-', lw=2, label='4.7e6 photons')
    h1, = plt.plot(range(n_spline),coef_opt, 'k:', lw=3, label='no noise')
    plt.title('fitted spline coefficients', fontsize='large')
    plt.xlabel('index number', fontsize='large')
    plt.ylabel('coefficient value',fontsize='large')
    plt.legend(handles=[h1,h2], loc=0)
    if save_figs:
        plt.savefig('figs/spline_fit.eps')

    
    # make mean contrast comparisons
    ratio = c6.total_photons/c6.phot_count
    rich6_total_photons = ratio*info_2['nphot']*31
    rich9_total_photons = ratio*info_1['nphot']*31
    print rich6_total_photons/1.e6, rich6_total_photons/1.e6
    m6 = np.mean(en_2)
    e6 = np.array([m6 - np.std(en_2), m6 + np.std(en_2)])
    m9 = np.mean(en_1)
    e9 = np.array([m9 - np.std(en_1), m9 + np.std(en_1)])
    plt.figure()
    erich6, = plt.plot([0,0],np.log10(e6),'k', lw=2.5)
    erich9, = plt.plot([0,0],np.log10(e9),'b', lw=2.5)        
    efc6, = plt.plot(np.log10(DHen6), 'r:x', label='EFC: 4.69e7 photons/iteration')
    efc9, = plt.plot(np.log10(DHen9), 'c:*', label='EFC: 4.69e10 photons/iteration')
    rich6, = plt.plot([0], np.log10(m6),'ks',
                     label='Lights On: 4.7e6 total photons')
    rich9, = plt.plot([0],np.log10(m9),'bo',
                     label='Lights On: 9.4e9 total photons')
    plt.axis([-3,100,-10,-2])
    plt.ylabel('log$_{10}$ mean contrast', fontsize='large')
    plt.xlabel('iteration #', fontsize='large')
    plt.title('mean hole contrast vs. iteration #', fontsize='large')
    plt.legend(handles=[efc6, efc9, rich6, rich9])
    if save_figs:
        plt.savefig('figs/ContrastVsIteration.eps')
    
    
    dm2_best = dm_2[:, np.argmin(en_2)]
    dm1_best = dm_1[:, np.argmin(en_1)]
    plotrange = [-1,15]
    axlims = [plotrange[0], plotrange[1], -12, 2.5]
    plt.figure()
    h8 = c6.plot_solution(np.zeros(c6.nact), plotrange, return_handles=True,
                          bump='None',color='b',label='raw, no aberration')
    h0 = c6.plot_solution(np.zeros(c6.nact), plotrange, return_handles=True,
                          color='m',label='raw, with aberration')
    h6 = c6.plot_solution(dmc6[:,10], plotrange, return_handles=True,
                          color='r',label='EFC: 4.7e8 total photons')
    h2 = c6.plot_solution(dm2_best, plotrange, return_handles=True,
                          color='k',label='Lights On: 4.7e6 total photons')
    plt.legend(handles=[h8, h0, h6, h2], loc=0)
    plt.axis(axlims)
    plt.plot([c6.dh_start, c6.dh_start],[axlims[2],axlims[3]],'k-', lw=4)
    plt.plot([c6.dh_end, c6.dh_end],[axlims[2],axlims[3]],'k-', lw=4)
    plt.xlabel('image plane position ($\lambda$/D)', fontsize='large')
    plt.ylabel('log$_{10}$ conrast', fontsize='large')
    plt.title('stellar image', fontsize='large')
    if save_figs:
        plt.savefig('figs/StellarImage-6.eps')

    plt.figure()
    h8 = c6.plot_solution(np.zeros(c6.nact), plotrange, return_handles=True,
                          bump='None',color='b',label='raw, no aberration')
    h0 = c6.plot_solution(np.zeros(c6.nact), plotrange, return_handles=True,
                          color='m',label='raw, with aberration')
    h9 = c6.plot_solution(dmc9[:,20], plotrange, return_handles=True,
                          color='r',label='EFC: 9.4e11 total photons')
    h1 = c6.plot_solution(dm1_best, plotrange, return_handles=True,
                          color='k',label='Lights On: 9.4e9 total photons')
    plt.legend(handles=[h8, h0, h9, h1], loc=0)
    plt.axis(axlims)
    plt.plot([c6.dh_start, c6.dh_start],[axlims[2],axlims[3]],'k-', lw=4)
    plt.plot([c6.dh_end, c6.dh_end],[axlims[2],axlims[3]],'k-', lw=4)
    plt.xlabel('image plane position ($\lambda$/D)', fontsize='large')
    plt.ylabel('log$_{10}$ conrast', fontsize='large')
    plt.title('stellar image', fontsize='large')
    if save_figs:
        plt.savefig('figs/StellarImage-9.eps')

        
    # EFC procedure stuff
    c6.bump = None
    amp = 0.01
    probe = amp*c6.config_regress(dm_basis='psinc')[1]
    field0 = np.real(c6.point_source_prop(0, np.zeros(c6.nact)))
    field = np.zeros((len(field0), 4)).astype('complex')
    for k in range(4):
        field[:,k] = c6.point_source_prop(0, amp*probe[k,:])
    plot_range = [-20,20]
    pr = np.round(c6.fLambda_over_D*np.array(plot_range)).astype('int')
    pixrange = np.arange(pr[1]-pr[0]) + c6.npad/2 + int(np.round(pr[0]))
    x = np.linspace(plot_range[0], plot_range[1], len(pixrange))
    field0 = field0[pixrange]
    field = field[pixrange, :]

    plt.figure()
    px = np.linspace(-.5*c6.npad/c6.nfield, .5*c6.npad/c6.nfield, c6.npad)
    h1, = plt.plot(px, np.angle(c6.apply_DM(np.ones(c6.nfield-1),
                   probe[0, :]))*180/np.pi, 'g', label='probe 1')
    h2, = plt.plot(px, np.angle(c6.apply_DM(np.ones(c6.nfield-1),
                   probe[1, :]))*180/np.pi, 'r', label='probe 2')
    h3, = plt.plot(px, np.angle(c6.apply_DM(np.ones(c6.nfield-1),
                   probe[2, :]))*180/np.pi, 'c', label='probe 3')
    h4, = plt.plot(px, np.angle(c6.apply_DM(np.ones(c6.nfield-1),
                   probe[3, :]))*180/np.pi, 'm', label='probe 4')
    plt.axis([-.5, .5, -9, 9])
    plt.title('probe shapes', fontsize='large')
    plt.legend(handles=[h1, h2, h3, h4], loc=0)
    plt.xlabel('pupil plane position (D)')
    plt.ylabel('phase delay (deg)')
    if save_figs:
        plt.savefig('figs/probe_shapes.jpg')

    f, ax = plt.subplots(2, 2, sharey=False, sharex=True)
    h0, = ax[0,0].plot(x, field0, 'k', label='no probe')
    ax[0,0].set_title('real(DM flat)')
    h0, = ax[0,1].plot(x, np.zeros(len(x)), 'k', label='no probe')
    ax[0,1].set_title('imag(DM flat)')        
    h1, = ax[1,0].plot(x, np.real(field[:, 0])-field0, 'g', label='probe 1')
    h2, = ax[1,0].plot(x, np.real(field[:, 1])-field0, 'r', label='probe 2')
    h3, = ax[1,0].plot(x, np.real(field[:, 2])-field0, 'c', label='probe 3')
    h4, = ax[1,0].plot(x, np.real(field[:, 3])-field0, 'm', label='probe 4')
    ax[1,0].set_title('real(probe - flat)')
    #ax[1,0].legend(handles=[h0, h1, h2, h3, h4], loc=0)
    h0, = ax[1,1].plot(x, np.zeros(len(x)), 'k', label='no probe')
    h1, = ax[1,1].plot(x, np.imag(field[:, 0]), 'g', label='probe 1')
    h2, = ax[1,1].plot(x, np.imag(field[:, 1]), 'r', label='probe 2')
    h3, = ax[1,1].plot(x, np.imag(field[:, 2]), 'c', label='probe 3')
    h4, = ax[1,1].plot(x, np.imag(field[:, 3]), 'm', label='probe 4')
    ax[1,1].set_title('imag(probe - flat)')
    #ax[1,1].legend(handles=[h0, h1, h2, h3, h4], loc=0)
    ax[1,0].set_xlabel('image plane position ($\lambda$/D)', fontsize='large')
    ax[1,1].set_xlabel('image plane position ($\lambda$/D)', fontsize='large')
    ax[0,0].set_ylabel('field value', fontsize='large')
    ax[1,0].set_ylabel('field value', fontsize='large')
    if save_figs:
        plt.savefig('figs/probe_fields.jpg')

    DH_en6, dmc6 = c6.do_EFC(nsteps=5, mk_plots=True)
    if save_figs:
        if False:  # watch the figure numbers!
            for k in range(5):
                fignum = 2*k + 2
                plt.figure(fignum)
                fname = 'figs/EFC_field_est_iter' + str(k) + '.jpg'
                plt.savefig(fname)
                fignum = 2*k + 1
                plt.figure(fignum)
                fname = 'figs/probe_intensity_diff_iter' + str(k) + '.jpg'
                plt.savefig(fname)