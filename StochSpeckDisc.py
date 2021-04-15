#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 11:09:53 2021

@author: rfrazin
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle, os
from scipy.optimize import minimize as MIZE
import PupilMap as PM
from IntensityDistributions import Frazinian, ModifiedRician, ModifiedRicianPlusConstant

loc = "/w/Wavefronts/OnSky_FF_8mag_oversize_splineDM_03RN_03r0_25L0_035Gain_25AOstart"

szp = 50  # linear size of pupil plane
prad = 25  # radius of entrance pupil within pupil plane
pmap, ipmap = PM.PupilMap(N=szp, pixrad=prad, return_inv=True)
nppix = len(pmap)  # number of pupil pixels
lam = 1.  # wavelength in microns
contrast = 1.e-4  
planetDist = 2.4  #distance from center in untis of lambda/D
planetAngl = 267.*np.pi/180  # angle of planet in radians



#load AO residual wavefronts
LOAD = True
nfiles2load = 3
load_measured_wavefronts = True
correction_factor = np.sqrt(2.11)  # measured WF spatial variance is too big
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
                wfm = d['WFmeas'][:, 2:-1]/correction_factor
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


nt = wft.shape[0]  # number of time steps
bp = contrast*np.abs(np.sum(np.exp(1j*wft), axis=1))**2  # planetary brightness at center
st = np.exp(- np.var(wft, axis=1))  # Strehl raio according to the Ruze formula
bs = np.zeros((nt,))  # star brightness at planet center
fs = np.zeros((nt,)).astype('complex')  # electric field of star at planet pixel
if load_measured_wavefronts:  # 'p' means 'predicted from measured WF'
    stp = np.exp(- np.var(wfm, axis=1))
    bsp = np.zeros((nt,))
    fsp = np.zeros((nt,)).astype('complex')
for tt in range(nt):
    sfield = PM.EmbedPupilVec(np.exp(1j*wft[tt,:]), pmap, szp)
    sfield -= np.ones(sfield.shape)  #  this includes a coronagraph effect
    fs[tt] = np.sum(sfield*sp)
    bs[tt] = np.real(fs[tt]*np.conj(fs[tt]))
    if load_measured_wavefronts:
        sfield = PM.EmbedPupilVec(np.exp(1j*wfm[tt,:]), pmap, szp)
        sfield -= np.ones(sfield.shape)  #  this includes a coronagraph effect
        fsp[tt] = np.sum(sfield*sp)
        bsp[tt] = np.real(fsp[tt]*np.conj(fsp[tt]))

#make a histogram
drop_frac = 0.05
nbins = 50
bsh = 1.0*bs  # deep copy
bsh.sort()
bsh = bsh[: int((1 - drop_frac)*len(bsh))]
edges = np.linspace(0, bsh[-1], nbins+1)
centers = np.linspace(edges[0], edges[-2], nbins) + .5*(edges[1] - edges[0])
bsh = np.histogram(bsh, bins=edges, density=True)[0]
tstring = 'd = ' + str(planetDist) + ', th = ' +  str(planetAngl*180/np.pi)
scale = np.median(bsh)
plt.figure(); plt.plot(centers, bsh,'bo'); plt.title(tstring)
print( np.abs(np.mean(fs)), np.std(np.real(fs)), np.std(np.imag(fs)))


#this calculates the misfit of the fitted function to the observed histogram.
#  It also returns the gradient.  
#v - vector of parameters fed to func
#y - vector of histogram frequencies
#centers - centers of histogram bins (these are intensity values)
#func - the function used to model the histogram.  Must take the 'return_derivs' argument
#scale - scaling of the chi-squared metric 
#ignore_cc - forces correlation coef to be zero, due to optimization difficulties (only valid for Frazinian)
def ChiSqHist(v, y, centers, func, scale, ignore_cc=False):
    assert y.shape == centers.shape
    if func == Frazinian: 
        assert ( (len(v) == 4) or (len(v) == 3) )
    elif func == ModifiedRician: assert len(v) == 2
    elif func == ModifiedRicianPlusConstant: assert len(v) ==3
    else: raise Exception("Error: 'func' is not implemented.") 
    if func == Frazinian:
        if ignore_cc:
            assert len(v) == 3
            Q = Frazinian(centers, v[0], v[1], v[2], 0., ignore_cc=True, return_derivs=True)
        else:
            assert len(v) == 4
            Q = Frazinian(centers, v[0], v[1], v[2], v[3], ignore_cc=False, return_derivs=True)
    elif func == ModifiedRician:
        Q = ModifiedRician(centers, v[0], v[1], return_derivs=True)
    elif func == ModifiedRicianPlusConstant:
        Q = ModifiedRicianPlusConstant(centers, v[0], v[1], v[2], return_derivs=True)

    ym = Q[0]  # modeled histogram values
    ch = 0.5*np.sum( (ym - y)**2 )/(scale**2)
    g = np.zeros((len(v),))  # gradient values
    for k in range(len(v)):
        g[k] = np.sum( (ym - y)*Q[k+1] )/(scale**2)

    return((ch, g))


v = np.array([50., 90.]) 
out2 = MIZE(ChiSqHist, v, args=(bsh, centers, ModifiedRician, scale), method='CG', jac=True, bounds=None)
v = out2['x']
print("v = ", v)
fit = ModifiedRician(centers, v[0], v[1], return_derivs=False)
plt.plot(centers, fit, 'kp:');
v = np.array([1.5*v[0],v[1]/2,2*v[1], 0.1])
out1 = MIZE(ChiSqHist, v, args=(bsh, centers, Frazinian, scale), method='CG', jac=True, bounds=None)
v = out1['x']
print("v = ", v)
fit = Frazinian(centers, v[0], v[1], v[2], v[3], ignore_cc=False, return_derivs=False)
plt.plot(centers, fit, 'rx:');



def MyGaussian(length,fwhm):  #for wavelet analysis with scipy.signal.cwt
     x = np.arange(length)
     c = (length-1)/2
     std = fwhm/2.355
     f = np.exp( - ((x-c)**2)/2/std**2)
     f /= np.sum(f)
     return(f)  
