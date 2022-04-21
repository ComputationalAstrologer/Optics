#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 24 16:45:03 2021
@author: Richard Frazin

These codes are utilities for calculating Jacobians for optical systems once a 
 system matrix ('sm' below) has been calculated 

"""



import numpy as np


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
