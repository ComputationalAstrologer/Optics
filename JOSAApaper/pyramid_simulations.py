#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 15:35:48 2017

@author: frazin
"""

import matplotlib.pyplot as plt
import numpy as np
import cPickle as pickle
import time
import Pyr as pyr

do_calculations = False
make_plots = True


k_st = None
ntrials = 12
strehl = np.array([.8, .5, .4, .3, .2, .1])
pcount = np.array([1.e7, 1.e5])
sigma = pyr.StrehlToSigma(strehl)
regparam = np.array([.001, .01, .05, .2, .4])
nstrehl = len(strehl)
ncount = len(pcount)
nreg = len(regparam)

if make_plots:
    sig_mean = np.zeros(nstrehl)

    nl5_mean = np.zeros((nstrehl, nreg))
    nl5_st = np.zeros((nstrehl, nreg))
    nl7_mean = np.zeros((nstrehl, nreg))
    nl7_st = np.zeros((nstrehl, nreg))
    ls5_mean = np.zeros((nstrehl, nreg))
    ls5_st = np.zeros((nstrehl, nreg))
    ls7_mean = np.zeros((nstrehl, nreg))
    ls7_st = np.zeros((nstrehl, nreg))

    ls5_er = np.zeros(nstrehl)
    ls7_er = np.zeros(nstrehl)
    nl5_er = np.zeros(nstrehl)
    nl7_er = np.zeros(nstrehl)

    for ks in range(nstrehl):
        fname = 'pickles/pyrsim_stats' + str(strehl[ks]) + '.p'
        fp = open(fname, 'r')
        phase_std, nl_std, ls_std = pickle.load(fp)
        fp.close()
        sig_mean[ks] = np.mean(phase_std)

        for kr in range(nreg):
            ls5_mean[ks, kr] = np.mean(ls_std[:, kr, 1])
            ls5b = np.argmin(ls5_mean, axis=1)  # best regularization
            ls5_best = np.min(ls5_mean, axis=1)
            ls5_st[ks, kr] = np.std(ls_std[:, kr, 1])

            ls7_mean[ks, kr] = np.mean(ls_std[:, kr, 0])
            ls7b = np.argmin(ls7_mean, axis=1)
            ls7_best = np.min(ls7_mean, axis=1)
            ls7_st[ks, kr] = np.std(ls_std[:, kr, 0])

            nl5_mean[ks, kr] = np.mean(nl_std[:, kr, 1])
            nl5b = np.argmin(nl5_mean, axis=1)
            nl5_best = np.min(nl5_mean, axis=1)
            nl5_st[ks, kr] = np.std(nl_std[:, kr, 1])

            nl7_mean[ks, kr] = np.mean(nl_std[:, kr, 0])
            nl7b = np.argmin(nl7_mean, axis=1)
            nl7_best = np.min(nl7_mean, axis=1)
            nl7_st[ks, kr] = np.std(nl_std[:, kr, 0])

        for ks in range(nstrehl):
            ls5_er[ks] = ls5_st[ks, ls5b[ks]]
            ls7_er[ks] = ls7_st[ks, ls7b[ks]]
            nl5_er[ks] = nl5_st[ks, nl5b[ks]]
            nl7_er[ks] = nl7_st[ks, nl7b[ks]]

    figdir = '/Users/frazin/docs/Papers/JATIS/Pyramid1/figs/'

    plt.figure(10);
    h0, = plt.plot(pyr.SigmaToStrehl(sig_mean), sig_mean, 'k', lw=3, label='no gain');
    h1, = plt.plot(pyr.SigmaToStrehl(sig_mean), ls5_best, 'rD-', ms=4, lw=2, label='linear');
    plt.errorbar(pyr.SigmaToStrehl(sig_mean), ls5_best, ls5_er, fmt='none', ecolor='r', elinewidth=2);
    h2, = plt.plot(pyr.SigmaToStrehl(sig_mean), nl5_best, 'bo-', ms=4, label='nonlinear');
    plt.errorbar(pyr.SigmaToStrehl(sig_mean), nl5_best, nl5_er, fmt='none', ecolor='b', elinewidth=2);
    plt.xlabel('input Strehl ratio', fontsize='large')
    plt.ylabel('phase estimate error std', fontsize='large')
    plt.legend(handles=[h0, h1, h2],fontsize='large')
    plt.title('$10^5$ photons', fontsize='large')
    plt.axis('tight');
    plt.axis([.006, .84, 0., 1.49]);
    fname = figdir + 'error_std5.eps'
    plt.savefig(fname)

    plt.figure(20);
    h0, = plt.plot(pyr.SigmaToStrehl(sig_mean), sig_mean, 'k', lw=3, label='no gain');
    h1, = plt.plot(pyr.SigmaToStrehl(sig_mean), ls7_best, 'rD-', ms=4, lw=2, label='linear');
    plt.errorbar(pyr.SigmaToStrehl(sig_mean), ls7_best, ls7_er, fmt='none', ecolor='r', elinewidth=2);
    h2, = plt.plot(pyr.SigmaToStrehl(sig_mean), nl7_best, 'bo-', ms=4, label='nonlinear');
    plt.errorbar(pyr.SigmaToStrehl(sig_mean), nl7_best, nl7_er, fmt='none', ecolor='b', elinewidth=2);
    plt.xlabel('input Strehl ratio', fontsize='large')
    plt.ylabel('phase estimate error std', fontsize='large')
    plt.legend(handles=[h0, h1, h2],fontsize='large')
    plt.title('$10^7$ photons', fontsize='large')
    plt.axis('tight');
    plt.axis([.006, .84, 0., 1.49])
    fname = figdir + 'error_std7.eps'
    plt.savefig(fname)

    plt.figure(13);
    ls_erbar = -2*ls5_best*pyr.SigmaToStrehl(ls5_best)*ls5_er
    nl_erbar = -2*nl5_best*pyr.SigmaToStrehl(nl5_best)*nl5_er
    h0, = plt.plot(pyr.SigmaToStrehl(sig_mean), pyr.SigmaToStrehl(sig_mean), 'k', lw=3, label='no gain');
    h1, = plt.plot(pyr.SigmaToStrehl(sig_mean), pyr.SigmaToStrehl(ls5_best), 'rD-', ms=4, lw=2, label='linear');
    plt.errorbar(pyr.SigmaToStrehl(sig_mean), pyr.SigmaToStrehl(ls5_best), ls_erbar, fmt='none', ecolor='r', elinewidth=2);
    h2, = plt.plot(pyr.SigmaToStrehl(sig_mean), pyr.SigmaToStrehl(nl5_best), 'bo-', ms=4, label='nonlinear');
    plt.errorbar(pyr.SigmaToStrehl(sig_mean), pyr.SigmaToStrehl(nl5_best), nl_erbar, fmt='none', ecolor='b', elinewidth=2);
    plt.xlabel('input Strehl ratio', fontsize='large')
    plt.ylabel('Strehl of phase estimate error', fontsize='large')
    plt.legend(handles=[h0, h1, h2],fontsize='large')
    plt.title('$10^5$ photons', fontsize='large')
    plt.axis('tight');
    plt.axis([.006, .84, 0., 1]);
    fname = figdir + 'error_strehl5.eps'
    plt.savefig(fname)

    plt.figure(23);
    ls_erbar = -2*ls7_best*pyr.SigmaToStrehl(ls7_best)*ls7_er
    nl_erbar = -2*nl7_best*pyr.SigmaToStrehl(nl7_best)*nl7_er
    h0, = plt.plot(pyr.SigmaToStrehl(sig_mean), pyr.SigmaToStrehl(sig_mean), 'k', lw=3, label='no gain');
    h1, = plt.plot(pyr.SigmaToStrehl(sig_mean), pyr.SigmaToStrehl(ls7_best), 'rD-', ms=4, lw=2, label='linear');
    plt.errorbar(pyr.SigmaToStrehl(sig_mean), pyr.SigmaToStrehl(ls7_best), ls_erbar, fmt='none', ecolor='r', elinewidth=2);
    h2, = plt.plot(pyr.SigmaToStrehl(sig_mean), pyr.SigmaToStrehl(nl7_best), 'bo-', ms=4, label='nonlinear');
    plt.errorbar(pyr.SigmaToStrehl(sig_mean), pyr.SigmaToStrehl(nl7_best), nl_erbar, fmt='none', ecolor='b', elinewidth=2);
    plt.xlabel('input Strehl ratio', fontsize='large')
    plt.ylabel('Strehl of phase estimate error', fontsize='large')
    plt.legend(handles=[h0, h1, h2],fontsize='large')
    plt.title('$10^7$ photons', fontsize='large')
    plt.axis('tight');
    plt.axis([.006, .84, 0., 1]);
    fname = figdir + 'error_strehl7.eps'
    plt.savefig(fname)

    # force stop of execution
if not do_calculations:
    raise Exception("not doing calculations.")

p = pyr.Pyr(NDim=2, basis='PhasePixel', RegMat='ID')
MaxIt = 10  # for PhasePixel each Hessian eval is about 5 min

phase_std = np.zeros(ntrials)
ls_std = np.zeros((ntrials, nreg, ncount))
nl_std = np.zeros((ntrials, nreg, ncount))

t0 = time.time()
print 'Strehl is ' + str(strehl[k_st])

cp0 = np.zeros(p.nc)
I0, dI0 = p.Intensity(cp0)
Itot = np.sum(I0)
Int0 = I0.reshape((p.npix, p.npix))

for tr in range(ntrials):
    phase = np.random.randn(p.nc)
    phase -= np.mean(phase)
    ph = sigma[k_st]*phase
    phase_std[tr] = np.std(ph)

    for kp in range(len(pcount)):

        I, dI = p.Intensity(ph)
        y = I*(pcount[kp]/Itot)
        noise_sig = np.sqrt(y)
        y += noise_sig*np.random.randn(len(y))
        y = y*(Itot/pcount[kp])
        noise_sig = noise_sig*(Itot/pcount[kp])
        wt = np.divide(1, noise_sig*noise_sig)
        wt = wt/np.mean(wt)
        wt = None  # weighting messes up the numerics

        for kr in range(1):  # use this solution to make the others faster
            cls0, cov_ls = p.LeastSqSol(y, wt, cp0, RegParam=regparam[kr], ZeroPhaseMean=True)
            cnl0, stats = p.FindMinCost(cls0, y, wt, RegParam=regparam[kr], AmpCon=False, MaxIt=10, LsqStart=False)
            ls_std[tr, kr, kp] = np.std(cls0 - ph)
            nl_std[tr, kr, kp] = np.std(cnl0 - ph)
        for kr in np.arange(nreg-1) + 1:
            cls0, cov_ls = p.LeastSqSol(y, wt, cp0, RegParam=regparam[kr], ZeroPhaseMean=True)
            cnl0, stats = p.FindMinCost(cnl0, y, wt, RegParam=regparam[kr], AmpCon=False, MaxIt=2, LsqStart=False)
            ls_std[tr, kr, kp] = np.std(cls0 - ph)
            nl_std[tr, kr, kp] = np.std(cnl0 - ph)

print (time.time() - t0)/3600.

fname = 'pyrsim_stats' + str(strehl[k_st]) + '.p'
filep = open(fname, 'w')
pickle.dump((phase_std, nl_std, ls_std), filep)
filep.close()




if False:

    phase_std0 = [.464, .834, .906, 1.137, 1.242]  # std of phase
    ls7_std0 = [.1125, .482, .690, .928, 1.06]
    nl7_std0 = [.0097, .219, .488, .796, .861]
    ls6_std0 = [.1152, .483, .691, .930, 1.06]
    nl6_std0 = [.0285, .219, .486, .789, .870]
    ls5_std0 = [.1498, .498, .692, .938, 1.07]
    nl5_std0 = [.0982, .277, .417, .722, .954]

    phase_std1 = [.903, 1.09, 1.27]
    ls7_std1   = [.645, .830,  1.09]
    nl7_std1   = [.317, .432,  .877]
    ls6_std1   = [.645, .830,  1.09]
    nl6_std1   = [.361, .442, .892]
    ls5_std1   = [.645, .837, 1.09]
    nl5_std1   = [.378, .498, .912]


