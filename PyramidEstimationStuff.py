#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 10:23:51 2021

@author: rfrazin
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.signal import decimate as dc
import PupilMap as PM
import LabPyr as LP
import FourierOptics


szp = 51  # linear size of pupil plane
prad = 15  # radius of entrance pupil within pupil plane

parms = dict()
parms['beam_diameter'] = 1.e3 # input beam diameter (microns)
parms['wavelength'] = 0.6328 # wavelength (microns)
parms['pyramid_slope_deg'] = 0.4 # slope of pyramid faces relative to horizontal (degrees)
parms['n_starting_points'] = szp  # number of resolution elements in initital beam diameter
parms['f1'] = 40.e3 # focal length of lens #1 (focuses light on pyramid tip)
parms['D_e_to_l1'] = 2*parms['f1'] # nominal distance from entrance pupil to lens1 (microns)
parms['lens1_fill_diameter'] = parms['beam_diameter']*4  #  computational beam width at lens#1.  This matters!
parms['D_l1_to_pyr'] = parms['f1'] # nominal distance from lens1 to pyramid tip
parms['D_l1_to_detector'] = 2*parms['f1']  # distance from lens to detector in 4f system
parms['apex_diam'] = 40 # diameter of focal spot at pyramid apex in units of lambda_over_D (set by stop)
parms['beam_diameter_at_pyramid'] = parms['apex_diam']*parms['f1']*parms['wavelength']/parms['beam_diameter']
parms['detector_width'] = None # width of detector
parms['max_chirp_step_deg'] = 30  # maximum allowed value (degrees) in chirp step for Fresnel prop
parms['max_lens_step_deg'] = 20 # maximum step size allowed for lens phase screen
parms['max_plane_step_deg'] = 20  # (degrees) maximum phase step allowed for planar phase screen
parms['max_pyramid_phase_step'] = 20  # maximum step (degrees) allowed for pyramid phase ramp
parms['max_fractional_change'] = 1.e-5  # maximum change tolerated in certain finite differences
parms['interp_style'] = 'linear'  # type of 2d interpolator for field resampling

WOM = LP.WorkingOpticalModels(params=parms)


#This gives the Jacobian of the detector intensity for several representations
#  of the phase and amplitude.  See my 2018 JOSA-A paper on the PyWFS.
#sm - the system matrix (complex valued).
#  shape is (n detector pixels, n of wavefront parameters)
#lpt - the linearization point should be None or a vector containing both
#  phases and amplitudes having length 2*sm.shape[1]
#kd - at which detector pixel is the Jacobian to be evaluated?
#rtype - how is the wavefront represented?
#  choices are 'PhasePixel' (phase only), 'PolarPixel' (amp and phase),
#  'AmpPixel' (ampltiude only)
def PixelPartialDeriv(sm, kd, lpt=None, rtype='PhasePixel'):
    assert rtype in ['PhasePixel', 'PolarPixel', 'AmpPixel']
    if lpt is None:
        lpt = np.hstack((np.zeros((sm.shape[1],)), np.ones((sm.shape[1],))))
    else: assert lpt.shape ==  (2*sm.shape[1],)
    lptp = lpt[:sm.shape[1]]
    lpta = lpt[sm.shape[1]:]
    u = lpta*np.exp(1j*lptp)  # pupil field evaluated at ltp
    v = sm[kd,:].dot(u)  # detector field at kd, evaluated at ltp
    if rtype == 'PhasePixel':
        g = 1j*np.exp(1j*lptp)*lpta*sm[kd,:]*v.conj()
    elif rtype == 'AmpPixel':
        g = np.exp(1j*lptp)*sm[kd,:]*v.conj()
    elif rtype == 'PolarPixel':
        q = np.exp(1j*lptp)*sm[kd,:]*v.conj()
        g1 = 1j*q*lpta
        g = np.hstack((g1,q))
    else: raise Exception("PixelPartialDerivative: bad rtype")
    return( 2*np.real(g) )

#This produces the rather large Jacobian matrix, G, of the detector intensity.
#Note that least-squares regression can be performed without explicitly 
#  forming G, by treating on column at a time.
#See PixelPartialDeriv for arguments
def FullJacobian(sm, lpt=None, rtype='PhasePixel'):
    assert rtype in ['PhasePixel', 'PolarPixel', 'AmpPixel']
    if rtype == 'PhasePixel' or rtype == 'AmpPixel':
        G = np.zeros(sm.shape)
    elif rtype == 'PolarPixel':
        G = np.zeros((sm.shape[0], 2*sm.shape[1]))
    else: raise Exception("FullJacobian: bad rtype")
    for k in range(sm.shape[0]):
        G[k,:] = PixelPartialDeriv(sm, k, lpt, rtype)
    return(G)



#create initial pupil plane
pmap, ipmap = PM.PupilMap(N=szp, pixrad=prad, return_inv=True)
nppix = len(pmap)

MakeSysmatPyr = False
if MakeSysmatPyr:
    #create the system matrix for the WFS
    nx = 240
    sysmat = np.zeros((nx*nx, nppix)).astype('complex')
    xu = np.linspace(- parms['beam_diameter']/2, parms['beam_diameter']/2, szp)
    for k in range(nppix):
        if np.mod(k, 50) == 0: print(k, " of 709")
        #set up field
        u = np.zeros((nppix,)).astype('complex')
        u[k] = 1.0 
        uu =  PM.EmbedPupilVec(u, pmap, szp)

        result = WOM.PropF4ReflectiveNSidedPyramid(g=uu, x=xu, SlopeDeviations=None, FaceCenterAngleDeviations=[30,30,30], pyr_dist_error=0.,
                                          N=3, NominalSlope=None, return_derivs=False, print_stuff=False)

        df = result['field']
        #make it smaller.  more filtering destroys the phase information
        dfr = dc(df, 8, ftype='fir', axis=0, zero_phase=True)
        dfr = dc(dfr, 8, ftype='fir', axis=1, zero_phase=True)
        dfr = dfr[23:23+240,280-240:280]
        sysmat[:,k] = dfr.reshape(nx**2,)
    print('sysmat is done.')

LoadSysmat = False
if LoadSysmat:
    fn = 'Pickles/PyWFSsysmat4sides709_175x175.pickle'
    #fn = 'Pickles/PyWFSsysmat3sides709_175x175.pickle'
    fid = open(fn, 'rb')
    sysmat = pickle.load(fid); fid.close()
    nr = sysmat.shape[0];  nc = sysmat.shape[1]; nx = np.sqrt(nr).astype('int')

EstimateStuff = False
if EstimateStuff:
    strehl = 0.5; psig = np.sqrt( - np.log(strehl))
    asig = 0.1
    amp = 1. + asig*np.random.randn(nc)
    ph = psig*np.random.randn(nc)
    u = amp*np.exp(1j*ph)
    v = sysmat.dot(u)
    vv = np.real(v*v.conj())
    I = vv.reshape(nx,nx)

#this should be the same as the reflective pyramid model above, but without the 
#pyramid.  It's an effort to understand the source of the information loss.
def PropF4(u_start, x_start, params=parms, diagnostics=False):
        FO = FourierOptics.FourierOptics(params)
        obd = params['D_e_to_l1'] # nominal distance from entrance pupil to lens1 (microns)
        imd = params['D_l1_to_detector']
        diam0 = params['lens1_fill_diameter']

        # propagate field to lens
        z = obd
        field2d, x2d = FO.ConvFresnel2D(u_start, x_start, diam0, z, set_dx=True, return_derivs=False)

        if diagnostics:
            plt.figure()
            dx = x2d[1] - x2d[0]
            br2d = np.real(field2d*field2d.conj()); br2d /= np.max(br2d)
            plt.imshow(br2d, extent=[x2d[0], x2d[-1], x2d[0], x2d[-1]])
            plt.colorbar();
            plt.title('intensity at lens1, z = ' + str(z/1.e4) +  ' cm, $\Delta x$ = ' + str(dx))
            plt.xlabel('x (microns)')
            plt.ylabel('y (microns)')

        #apply phase screen due to lens
        focal_length = params['f1']
        lens_center = [0, 0]
        field2d, x2d = FO.ApplyThinLens2D(field2d, x2d, lens_center, focal_length, return_derivs=False)

        #propagate to image plane
        z = imd
        diam1 = params['beam_diameter']
        field2d, x2d = FO.ConvFresnel2D(field2d, x2d, diam1, z, set_dx=True, return_derivs=False)
        field2d /= 3.31e-5

        if diagnostics:
            zd = obd + imd
            br2d = np.real(field2d*field2d.conj()); br2d /= np.max(br2d)
            plt.figure()
            dx = x2d[1] - x2d[0]
            plt.imshow(br2d, extent=[x2d[0], x2d[-1], x2d[0], x2d[-1]])
            plt.colorbar();
            plt.title('intensity at detector, z = ' + str(zd/1.e4) +  ' cm, $\Delta x$ = ' + str(dx))
            plt.xlabel('x (microns)')
            plt.ylabel('y (microns)')

        return({'x': x2d, 'field': field2d})


MakeSysmatF4 = False
if MakeSysmatF4:
    u = np.zeros((nppix,)).astype('complex')
    k=0; u[k] = 1.0 
    uu =  PM.EmbedPupilVec(u, pmap, szp)
    xu = np.linspace(- parms['beam_diameter']/2, parms['beam_diameter']/2, szp)
    result = PropF4(uu, xu, params=parms, diagnostics=False)
    nv = result['x'].shape[0]
    sysmat = np.zeros((nv*nv, nppix)).astype('complex')
    for k in range(nppix):
        if np.mod(k, 50) == 0: print(k, " of 709")
        #set up field
        u = np.zeros((nppix,)).astype('complex')
        u[k] = 1.0 
        uu =  PM.EmbedPupilVec(u, pmap, szp)
        result = PropF4(uu, xu, params=parms, diagnostics=False)
        sysmat[:,k] = result['field'].reshape(nv*nv,)
    print('sysmat is done.')