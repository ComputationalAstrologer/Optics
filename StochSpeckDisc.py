#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 11:09:53 2021

@author: rfrazin
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle, os
import PupilMap as PM

loc = "/w/Wavefronts/OnSky_FF_8mag_oversize_splineDM_03RN_03r0_25L0_035Gain_25AOstart"

szp = 50  # linear size of pupil plane
prad = 25  # radius of entrance pupil within pupil plane
pmap, ipmap = PM.PupilMap(N=szp, pixrad=prad, return_inv=True)
nppix = len(pmap)  # number of pupil pixels
lam = 1.  # wavelength in microns
contrast = 1.e-4  
planetDist = 4.  #distance from center in untis of lambda/D
planetAngl = 217.*np.pi/180  # angle of planet in radians


def MyGaussian(length,fwhm):  #for wavelet analysis with scipy.signal.cwt
     x = np.arange(length)
     c = (length-1)/2
     std = fwhm/2.355
     f = np.exp( - ((x-c)**2)/2/std**2)
     f /= np.sum(f)
     return(f)  

#load AO residual wavefronts
LOAD = True
nfiles2load = 6
load_measured_wavefronts = False
if LOAD:
    filecount = -1
    for fn in os.listdir(loc):
        filecount += 1
        if filecount > nfiles2load: 
            break
        fnp = open(os.path.join(loc,fn),'rb')
        d = pickle.load(fnp); fnp.close()
        if filecount == 0:
            wft = d['AOres'][ :, 2:-1]
            assert wft.shape[0] == nppix
            if load_measured_wavefronts:
                wfm = d['WFmeas'][:, 2:-1]
        else:
            wft = np.hstack((wft, d['AOres'][ :, 2:-1]))
            if load_measured_wavefronts:
                wfm = np.hstack((wfm, d['WFmeas'][ :, 2:-1]))
    if load_measured_wavefronts:
        wfm = wfm.T
    wft = wft.T


circ = np.real(PM.EmbedPupilVec(np.ones((1976,)),pmap,szp))

#make pupil field for planet w/o AO residual, phasor for starlight at a planet location
up = np.sqrt(contrast)*np.ones((szp,szp)).astype('complex')
sp = np.ones((szp,szp)).astype('complex')
s = np.linspace(-np.pi, np.pi, szp)
(xx, yy) = np.meshgrid(s,s,indexing='xy'); del s
alphax = planetDist*np.cos(planetAngl)
alphay = planetDist*np.sin(planetAngl)
for ky in range(szp):
    for kx in range(szp):
        ph = alphax*xx[ky,kx] + alphay*yy[ky,kx]
        up[ky,kx] *= np.exp(1j*ph)
        sp[ky,kx] *= np.exp(-1j*ph)

#make pupil field for star w/o AO residual


#make time series at the planet's central pixel
nt = wft.shape[0]  # number of time steps
bp = contrast*np.abs(np.sum(np.exp(1j*wft), axis=1))**2  # planetary brightness at center
st = np.exp(- np.var(wft, axis=1))  # Strehl raio according to the Ruze formula
bs = np.zeros((nt,))  # star brightness at planet center
fs = np.zeros((nt,)).astype('complex')  # electric field of star at planet pixel
for tt in range(nt):
    sfield = PM.EmbedPupilVec(np.exp(1j*wft[tt,:]), pmap, szp)
    sfield -= np.ones(sfield.shape)  #  this includes a coronagraph effect
    fs[tt] = np.sum(sfield*sp)
    bs[tt] = np.real(fs[tt]*np.conj(fs[tt]))

