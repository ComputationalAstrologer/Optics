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
circ = np.real(PM.EmbedPupilVec(np.ones((nppix,)),pmap,szp))

#locations for random sample of stellar field
nstloc = 11
stloc = []; strad = []; stang = []; stf = []; sth = []; cen =[]
for k in range(nstloc):
    cen.append([])  # histogram bin centers
    stf.append([])  # each element will be list of complex field values
    sth.append([])  # each element will be a histogram of intensity values
    strad.append(8*np.random.rand())  # radius in lam/D units
    rrr = strad[-1]*((szf/2)/13 )  # 13 is the extent of the image along the axes for "./Lyot4489x1976Dmatrix.npy"
    stang.append(np.random.rand()*360)
    the = stang[-1]*np.pi/180
    stloc.append(( int(szf/2 + rrr*np.cos(the)) ,  # y,x pixel values
                   int(szf/2 + rrr*np.sin(the)) ))

#load AO residual wavefronts and coronagraph model
LOAD = True
nfiles2load = 3
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

nt = wft.shape[0]
for kt in range(nt):
    fs = (D.dot(wft[kt, :])).reshape(szf,szf)
    for ks in range(nstloc):
        fld = fs[stloc[ks][1], stloc[ks][0]]
        stf[ks].append( fld )

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



#make histograms
drop_frac = 0.05
nbins = 50
for kl in range(nstloc):
    sti = []
    for ki in range(len(stf[kl])):
        sti.append( np.real(stf[kl][ki]*np.conj(stf[kl][ki])) )
    sti = np.array(sti)
    sti.sort()  # put intensities in ascending order
    sti = sti[: int((1 - drop_frac)*len(sti))]  # drop largest values
    edges = np.linspace(0, sti[-1], nbins+1)
    cen[kl] = np.linspace(edges[0], edges[-2], nbins) + .5*(edges[1] - edges[0])
    sth[kl] = np.histogram(sti, bins=edges, density=True)[0]
    tstring = 'd = ' + str(strad[kl]) + ', th = ' +  str(stang[kl])
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




