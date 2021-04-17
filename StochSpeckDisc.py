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
from IntensityDistributions import ChiSqHist, Frazinian, ModifiedRician, ModifiedRicianPlusConstant

wfloc = "/w/Wavefronts/OnSky_FF_8mag_oversize_splineDM_03RN_03r0_25L0_035Gain_25AOstart"
CMfile = "./Lyot4489x1976Dmatrix.npy"  # system matrix for coronagraph
lam = 1.1  # wavelength in microns

contrast = 1.e-3
planetDist = 0.4  #distance from center in untis of lambda/D
planetAngl = 10.*np.pi/180  # angle of planet in radians

szp = 50  # linear size of pupil plane
szf = 67  # linear size of focal plane
prad = 25  # radius of entrance pupil within pupil plane
pmap, ipmap = PM.PupilMap(N=szp, pixrad=prad, return_inv=True)
nppix = len(pmap)  # number of pupil pixels
circ = np.real(PM.EmbedPupilVec(np.ones((nppix,)),pmap,szp))

#locations for random sample of stellar field
nstloc = 31
stloc = []; strad = []; stang = []; stf = []
for k in range(nstloc):
    stf.append([])  # each element will be list of complex field values
    strad.append(8*np.random.rand()*( (szf/2)/13 ))  # 13 is the extent of the image along the axes for "./Lyot4489x1976Dmatrix.npy"
    stang.append(2*np.pi*np.random.rand())
    stloc.append(( int(szf/2 + strad[k]*np.cos(stang[k])) ,  # y,x pixel values
                   int(szf/2 + strad[k]*np.sin(stang[k])) ))

#load AO residual wavefronts and coronagraph model
LOAD = True
nfiles2load = 3
load_measured_wavefronts = True
correction_factor = np.sqrt(2.11)  # measured WF spatial variance is too big
if LOAD:
    D = np.load(CMfile)  # coronagraph system matrix
    assert D.shape == (szf*szf, nppix)
    filecount = -1
    for fn in os.listdir(wfloc):
        filecount += 1
        if filecount > nfiles2load: 
            break
        fnp = open(os.path.join(wfloc,fn),'rb')
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

nt = wft.shape[0]
for kt in range(nt):
    fs = D.dot(wft[kt, :])
    for ks in range(nstloc):
        fld = fs[stloc[k][1], stloc[k][0]]
        stf[k].append( fld )


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
print( np.abs(np.mean(fs)), np.std(np.real(fs)), np.std(np.imag(fs)),
      np.corrcoef(np.real(fs), np.imag(fs))[0,1])





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




