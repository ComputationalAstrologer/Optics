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

contrast = 1.e-3  #planet contrast

szp = 50  # linear size of pupil plane
szf = 67  # linear size of focal plane
prad = 25  # radius of entrance pupil within pupil plane
pmap, ipmap = PM.PupilMap(N=szp, pixrad=prad, return_inv=True)
nppix = len(pmap)  # number of pupil pixels
circ = np.real(PM.EmbedPupilVec(np.ones((nppix,)),pmap,szp))  # this is just a circle embedded in a square array

#load AO residual wavefronts and coronagraph model
LOAD = True
nfiles2load = 1
load_measured_wavefronts = False
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


#locations for random sample of stellar field
nloc = 5
loc = []; angls = []; dpix=[]; 
for k in range(nloc):  # this code block uses PM.FindPerfectAngle to adjust the input angles
    r = (np.random.rand()*7.5 + 0.5)*2*np.pi
    t = np.random.rand()*2.*np.pi
    (aa, dpixx, success) = PM.FindPerfectAngle([r*np.sin(t), r*np.cos(t)], D, szp, ipmap)
    if success:
        angls.append((aa[0], aa[1]))
        dpix.append( np.ravel_multi_index(dpixx, (szf, szf), order='C'))  #detector pixel corresponding to the angle (aa[0], aa[1])
nloc = len(dpix)

nt = wft.shape[0]  #number of wavefronts
sf = np.zeros((nloc, nt)).astype('complex')  # stellar field time-series at each loc
sb = np.zeros((nloc, nt))  # stellar brightness time-series at each loc
pb = np.zeros((nloc, nt))  # planetary brightnesstime-series at each loc
s = np.linspace(-.5, .5, szp)  # scaled 1D pupil coordinate
(xx, yy) = np.meshgrid(s,s,indexing='xy'); del s  # 2D pupil coords.  xx increases to the right.  xx[k,:] is increasing.  xx[:,k] is constant
pphasor = np.zeros((nppix, nloc)).astype('complex')  # pupil phasor corresponding a point source at loc[k]
phas = np.zeros((szp, szp)).astype('complex')
for k in range(nloc):
    for ky in range(szp):
        for kx in range(szp):
            phas[ky,kx] = np.exp( 1j*(angls[k][0]*yy[ky,kx] + angls[k][1]*xx[ky,kx]  ) )  # this ordering is correct for 'xy' indexing of np.meshgrid
    pphasor[:,k] = PM.ExtractPupilVec(phas, ipmap)  #planet phasor

for kt in range(nt):
    for k in range(nloc):
        sf[k, kt] = ( D.dot(wft[kt, :]) )[dpix[k]]  # stellar field
        sb[k, kt] = np.real( sf[k, kt]*np.conj(sf[k, kt]) )  # stellar intensity
        pb[k, kt] =   contrast*np.abs(( D.dot(wft[kt,:]*pphasor[:,k]) )[dpix[k]])**2  #planet intensity

#make histograms
drop_frac = 0.05
nbins = 50
sbh = []  # this will contain histograms of the stellar intensity
edges = []  # bin edges
cntrs = []  # bin centers
for kl in range(nloc):
    sb[k,:].sort()  # sort intensities in ascending order
    q = sb[k,: int((1- drop_frac)*nt)]
    edges.append(np.linspace(0, q[-1], nbins+1))
    cntrs.append(np.linspace(edges[-1][0], edges[-1][-2], nbins) + .5*(edges[-1][1] - edges[-1][0]))
    sbh.append(np.histogram(q, bins=edges, density=True)[0])
    rr = np.sqrt(angls[k][0]**2 + angls[k][1]**2)/2/np.pi
    th = np.arctan2(angls[k][0], angls[k][1])*180/np.pi
    tstring = 'd = ' + str( rr ) + ', th = ' +  str(th)
    #fit histogram
    meanI = np.sum(cen[kl]*sth[kl])*(cen[kl][1] - cen[kl][0])
    v = np.array([.2*np.sqrt(meanI), np.sqrt(meanI)]) 
    out = MIZE(ChiSqHist, v, args=(sth[kl], cen[kl], ModifiedRician), method='CG', jac=True, bounds=None)
    v = out['x']
    fit = ModifiedRician(cen[kl], v[0], v[1], return_derivs=False)
    plt.figure(); plt.plot(cen[kl], sth[kl],'bo-', cen[kl], fit, 'rx:'); plt.title(tstring);
#    v = np.array([0.1*np.min([v[0],v[1]]) , v[0], v[1]])
#    out = MIZE(ChiSqHist, v, args=(sth[kl], cen[kl], ModifiedRicianPlusConstant), method='CG', jac=True, bounds=None)
#    v = out['x']
#    fit = ModifiedRicianPlusConstant(cen[kl], v[0], v[1], v[2], return_derivs=False)
#    plt.plot(cen[kl], fit, 'mp:');






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






v = out2['x']
print("v = ", v)

plt.plot(centers, fit, 'kp:');
v = np.array([1.5*v[0],v[1]/2,2*v[1], 0.1])
out1 = MIZE(ChiSqHist, v, args=(bsh, centers, Frazinian, scale), method='CG', jac=True, bounds=None)
v = out1['x']
print("v = ", v)
fit = Frazinian(centers, v[0], v[1], v[2], v[3], ignore_cc=False, return_derivs=False)
plt.plot(centers, fit, 'rx:');




