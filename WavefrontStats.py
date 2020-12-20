#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 11:48:00 2020

@author: rfrazin
"""

import numpy as np
import os, pickle
from AndersonDarlingTest import ADtest

loc = './Wavefronts'

M = 1976

LOAD = True
if LOAD:
    filecount = -1
    for fn in os.listdir(loc):
        filecount += 1
        fnp = open(os.path.join(loc,fn),'rb')
        d = pickle.load(fnp); fnp.close()
        if filecount == 0:
            wft = d['AOres'][ :, 2:-1]
            assert wft.shape[0] == M
            wfm = d['WFmeas'][:, 2:-1]
        else:
            wft = np.hstack((wft, d['AOres'][ :, 2:-1]))
            wfm = np.hstack((wfm, d['AOres'][ :, 2:-1]))

del wfm
wft = wft.T
TIME_BIN = True
if TIME_BIN:
    bs = 50  # bin size
    nbins = wft.shape[0]//bs
    wfb = np.zeros((nbins, bs, M))
    mu = np.zeros((nbins, M))
    var = np.zeros((nbins, M))  # these are the variances of each pixel, not a covariance matrix
    for k in range(nbins):
        wfb[k,:,:] = wft[k*bs:(1+k)*bs, :]
        mu[k,:] = np.mean(wfb[k,:,:], axis=0)
        var[k,:] = np.var(wfb[k,:,:], axis=0, ddof=1)
    MUCORR = False
    if MUCORR:
        smc = len(np.correlate(mu[:,0], mu[:,0],'full'))
        mucorr = np.zeros((smc, M))
        for k in range(M):
            mucorr[:,k] = np.correlate(mu[:,k], mu[:,k],'full')
    ADTEST = True
    if ADTEST:
        npix = 3
        #pix = np.random.randint(0,M,npix)
        pix = [1379, 1105,  659]
        adscore = np.zeros((nbins,npix))
        for kb in range(nbins):
            if np.mod(kb,50) == 0:
                print('bin ', kb, ' is done.')
            for k in range(npix):
                adscore[kb, k] = ADtest(wfb[kb,:, pix[k]])



TOTAL_STATS = False
if TOTAL_STATS:
    mu_true = np.zeros((M,))
    mu_meas = np.zeros((M,))
    mu_diff = np.zeros((M,))
    cov_true = np.zeros((M,M))
    cov_meas = np.zeros((M,M))
    cov_diff = np.zeros((M,M))
    
    filecount = 0
    wfcount = 0
    for fn in os.listdir(loc):
        fnp = open(os.path.join(loc, fn), 'rb')
        d = pickle.load(fnp); fnp.close()
        wf_true = d['AOres'][:, 2:-1]  # true wavefronts. the first two and last one are trash
        wf_meas = d['WFmeas'][:, 2:-1]  # measured wavefronts
        wf_diff = wf_true - wf_meas
        nwf = wf_true.shape[1]  # number of measured wavefront
        assert wf_meas.shape == wf_true.shape
        assert wf_meas.shape[0] == M
        filecount += 1
        for k in range(wf_meas.shape[1]):
            mu_true += wf_true[:,k]
            mu_meas += wf_meas[:,k]
            mu_diff += wf_diff[:,k]
            cov_true += np.outer(wf_true[:,k], wf_true[:,k])
            cov_meas += np.outer(wf_meas[:,k], wf_meas[:,k])
            cov_diff += np.outer(wf_diff[:,k], wf_diff[:,k])
            wfcount += 1
        if filecount == 5: break  # each file takes 2 minutes!
    cov_true /= wfcount
    cov_meas /= wfcount
    cov_diff /= wfcount
    mu_diff /= wfcount
    mu_true /= wfcount
    mu_meas /= wfcount
    cov_true -= np.outer(mu_true, mu_true)
    cov_meas -= np.outer(mu_meas, mu_meas)


