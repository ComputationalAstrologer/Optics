#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 16:22:41 2017

"""

import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
from os import path
import time
import Pyr as pyr

#%%

strehl = 0.4
pcount = 1.e7
reg = 1.e-7
ntrials = 1

sigma = pyr.StrehlToSigma(strehl)

print("initialzing...")
t0 = time.time()
p = pyr.Pyr()
t0 = (time.time() - t0)/60.
print("done initializing in ", str(t0), "minutes.")

c0 = np.zeros(p.nc)

#%%  see how much of the light falls in the pupil images
ntrials = 30
strehl = np.linspace(.08, .999, 16)
sigma = pyr.StrehlToSigma(strehl)
frac = np.zeros((len(strehl), ntrials))
for k in range(len(strehl)):
    for tr in range(ntrials):
        ph = sigma[k]*np.random.randn(p.nc)
        ph -= np.mean(ph)
        If, dIf = p.Intensity(ph)
        Ip, dIp = p.PupilImageIntensity(ph, g=None, slopes=False, normalized=True)
        frac[k, tr] = np.sum(Ip)/np.sum(If)
    mean_frac = np.mean(frac, axis=1)
    std_frac = np.std(frac, axis=1)

#%%
plt.figure(41)
plt.plot(strehl,1 - mean_frac, 'k-', lw=1)
plt.errorbar(strehl,1 - mean_frac, yerr=std_frac, fmt="none", elinewidth=3)
plt.xlabel('Strehl ratio', FontSize='large')
plt.ylabel('fraction lost',FontSize='large')
plt.title('Light Outside of Pupil Images', FontSize='large')
plt.xticks(.1*np.arange(10) + .1)

if True:
    dir = '/Users/rfrazin/docs/Papers/JOSAA/pyramid/figs/'
    fn = dir + 'light_lost.eps'
    plt.savefig(fn)



#%% eigen analysis stuff

If, dIf = p.Intensity(c0)
Ip, dIp = p.PupilImageIntensity(c0, g=None, slopes=False, normalized=True)
Is, dIs, Istot = p.PupilImageIntensity(c0, g=None, slopes=True, normalized=True)
Ir, dIr, Irtot = p.PupilImageIntensity(c0, g=None, slopes=True, normalized=False)

ur, sr, vr = np.linalg.svd(dIr, full_matrices=False, compute_uv=True)
uf, sf, vf = np.linalg.svd(dIf, False, True)
up, sp, vp = np.linalg.svd(dIp, False, True)
us, ss, vs = np.linalg.svd(dIs, False, True)
sr /= np.max(sr)
sp /= np.max(sp)
ss /= np.max(ss)
sf /= np.max(sf)

pIr = np.linalg.pinv(dIr, rcond=1.e-6)
pIf = np.linalg.pinv(dIf, rcond=1.e-6)
pIs = np.linalg.pinv(dIs, rcond=1.e-6)
pIp = np.linalg.pinv(dIp, rcond=1.e-6)

#%%

picklename = 'slopes_vs_whole.p'
if not path.isfile(picklename):
    print 'file: ', picklename, 'not found.  doing calculations.'

    ntrials = 60
    strehl = np.hstack((np.linspace(.1,.7,7),np.linspace(.8, .999, 10)))
    sigma = pyr.StrehlToSigma(strehl)
    
    mean_s = np.zeros(len(strehl))
    mean_r = np.zeros(len(strehl))
    mean_f = np.zeros(len(strehl))
    mean_p = np.zeros(len(strehl))
    std_p = np.zeros(len(strehl))
    std_r = np.zeros(len(strehl))
    std_s = np.zeros(len(strehl))
    std_f = np.zeros(len(strehl))
    score_s = np.zeros((len(strehl), ntrials))
    score_f = np.zeros((len(strehl), ntrials))
    score_r = np.zeros((len(strehl), ntrials))
    score_p = np.zeros((len(strehl), ntrials))
    
    
    for k in range(len(strehl)):
        for tr in range(ntrials):    
            ph = sigma[k]*np.random.randn(p.nc)
            ph -= np.mean(ph)
            
            ys, _1, _2 = p.PupilImageIntensity(ph, g=None, slopes=True, normalized=True)
            yr, _1, _2 = p.PupilImageIntensity(ph, g=None, slopes=True, normalized=False)
            yp, _1 = p.PupilImageIntensity(ph, g=None, slopes=False, normalized=True)
            yf, _1 = p.Intensity(ph)
    
            xs = np.dot(pIs, ys - Is)
            xp = np.dot(pIp, yp - Ip)
            xr = np.dot(pIr, yr - Ir)
            xf = np.dot(pIf, yf - If)
    
            score_s[k, tr] = np.std(ph - xs)
            score_r[k, tr] = np.std(ph - xr)
            score_f[k, tr] = np.std(ph - xf)
            score_p[k, tr] = np.std(ph - xp)
    
    std_r = np.std(score_r, axis=1)*180/np.pi
    std_f = np.std(score_f, axis=1)*180/np.pi
    std_p = np.std(score_p, axis=1)*180/np.pi
    std_s = np.std(score_s, axis=1)*180/np.pi
    
    mean_r = np.mean(score_r, axis=1)*180/np.pi
    mean_f = np.mean(score_f, axis=1)*180/np.pi
    mean_p = np.mean(score_p, axis=1)*180/np.pi
    mean_s = np.mean(score_s, axis=1)*180/np.pi

    stuff = {}
    stuff['ntrials'] = ntrials
    stuff['strehl'] = strehl
    stuff['sigma'] = sigma
    stuff['mean_r'] = mean_r
    stuff['mean_s'] = mean_s
    stuff['mean_f'] = mean_f
    stuff['mean_p'] = mean_p
    stuff['std_r'] = std_r
    stuff['std_s'] = std_s
    stuff['std_f'] = std_f
    stuff['std_p'] = std_p

    fp = open(picklename, 'w')
    pickle.dump(stuff, fp)
    fp.close()

else:
    fp = open(picklename, 'r')
    stuff = pickle.load(fp)
    fp.close()
    
    ntrials = stuff['ntrials']
    strehl = stuff['strehl']
    sigma = stuff['sigma']
    mean_r = stuff['mean_r']
    mean_s = stuff['mean_s']
    mean_f = stuff['mean_f']
    mean_p = stuff['mean_p']
    std_r = stuff['std_r']
    std_s = stuff['std_s']
    std_f = stuff['std_f']
    std_p = stuff['std_p']




#%%
lsigma = np.log10(sigma*180/np.pi)
lmean_s = np.log10(mean_s)
lmean_f = np.log10(mean_f)
lmean_r = np.log10(mean_r)
lmean_p = np.log10(mean_p)
lstd_s = np.log10(mean_s + std_s) - np.log10(mean_s)
lstd_f = np.log10(mean_f + std_f) - np.log10(mean_f)
lstd_r = np.log10(mean_r + std_r) - np.log10(mean_r)
lstd_p = np.log10(mean_p + std_p) - np.log10(mean_p)


plt.figure(33)
plt.clf()
h0, = plt.plot(strehl, sigma*180/np.pi, 'k-', lw=4, label='no gain')
hs, = plt.plot(strehl, mean_s, ':', color='r', lw=3, label='NormalizedSlope')
plt.errorbar(strehl, mean_s, yerr=std_s, fmt="none", elinewidth=2, ecolor='r')
hp, = plt.plot(strehl, mean_p, 'm-', lw=2, label='FourImages')
plt.errorbar(strehl, mean_p, yerr=std_p, fmt="none", elinewidth=2, ecolor='m')
hr, = plt.plot(strehl, mean_r, 'b-.', lw=2, label='UnnormalizedSlope')
plt.errorbar(strehl, mean_r, yerr=std_r, fmt="none", elinewidth=2, ecolor='b')
hf, = plt.plot(strehl, mean_f, '--', color='orange', lw=3, label='AllPixels') 
plt.errorbar(strehl, mean_f, yerr=std_f, fmt="none", elinewidth=2, ecolor='orange')
plt.legend(handles=[h0, hr,hs,hp,hf], loc=1)
plt.xlabel('input Strehl ratio', FontSize='large')
plt.ylabel('RMS error (deg)', FontSize='large')
plt.title('Psuedo-inverse solutions, no noise',FontSize='large')
plt.xlim((0.84,.99))
plt.xticks(0.85 + .02*np.arange(8))
plt.ylim((-2,35.))

if True:
    dir = '/Users/rfrazin/docs/Papers/JOSAA/pyramid/figs/'
    fn = dir + 'pinv_solutions-HiStrehl.eps'
    plt.savefig(fn)

plt.figure(34)
plt.clf()
h0, = plt.plot(strehl, sigma*180/np.pi, 'k-', lw=4, label='no gain')
hs, = plt.plot(strehl, mean_s, ':', color='r', lw=3, label='NormalizedSlope')
plt.errorbar(strehl, mean_s, yerr=std_s, fmt="none", elinewidth=3, ecolor='r')
hp, = plt.plot(strehl, mean_p, 'm-', lw=2, label='FourImages')
plt.errorbar(strehl, mean_p, yerr=std_p, fmt="none", elinewidth=4, ecolor='m')
hr, = plt.plot(strehl, mean_r, 'b-.', lw=2, label='UnnormalizedSlope')
plt.errorbar(strehl, mean_r, yerr=std_r, fmt="none", elinewidth=2, ecolor='b')
hf, = plt.plot(strehl, mean_f, '--', color='orange', lw=3, label='AllPixels') 
plt.errorbar(strehl, mean_f, yerr=std_f, fmt="none", elinewidth=2, ecolor='orange')
plt.legend(handles=[h0, hr,hs,hp,hf], loc=1)
plt.xlabel('input Strehl ratio', FontSize='large')
plt.ylabel('RMS error (deg)', FontSize='large')
plt.title('Psuedo-inverse solutions, no noise',FontSize='large')
plt.xlim((.09,.851))
plt.ylim((0,140.))

if False:
    dir = '/Users/rfrazin/docs/Papers/JOSAA/pyramid/figs/'
    fn = dir + 'pinv_solutions-LoStrehl.eps'
    plt.savefig(fn)




#%%
plt.figure(10)
a = np.linspace(0, p.nc-2, p.nc-1).astype('int')
hf, = plt.plot(a, sf[0:-1], '--', color='orange', lw=3, label='AllPixels')
hp, = plt.plot(a, sp[0:-1], 'm-', lw=2,label='FourImages')
plt.plot([796],[0], 'ko', markeredgecolor='k', markerfacecolor='w',markersize=8,markeredgewidth=2)
hs, = plt.plot(a, ss[0:-1], 'r:', lw=3, label='NormalizedSlope')
plt.plot([796],[0], 'o', markeredgecolor='r', markerfacecolor='r',markersize=5,markeredgewidth=2)
hr, = plt.plot(a, sr[0:-1], 'b-.', lw=2, label='UnnormalizedSlope')
plt.plot([796],[0], '.', markeredgecolor='b', markerfacecolor='b',markersize=2,markeredgewidth=2)
plt.xlabel('index', FontSize='large')
plt.ylabel('normalized singular value', FontSize='large')
plt.title('Intensity Jacobian Singular Values', FontSize='large')
plt.ylim((-.01,1.))
plt.yticks(([0,.05, .1, .15, .2, .3, .4, .6, .8, 1.]))
plt.legend(handles=[hr,hs,hp,hf], loc=1)

if True:
    dir = '/Users/rfrazin/docs/Papers/JOSAA/pyramid/figs/'
    fn = dir + 'singular_values.eps'
    plt.savefig(fn)
