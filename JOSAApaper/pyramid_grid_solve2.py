#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 15:47:09 2017

@author: frazin
"""
import numpy as np
import cPickle as pickle
import time
import Pyr as pyr

#this does unregularized solutions only

def grid_solve_pool(example_index):

    print "initialzing..."
    t0 = time.time()
    p = pyr.Pyr()
    t0 = (time.time() - t0)/60.
    print "done initializing in ", str(t0), "minutes."

    filename_prefix = 'pickles/nonlin_strehl9p4_result'

    #strehl = [.005, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    #photon_counts = [1.e4, 1.e7]
    #dnoise = [0., 4.]

    strehl = [0.9]
    photon_counts = [1.e4]
    dnoise = [0.]

    regp = 1.e-9
    ntrials = 24
    nl_maxit = 300

    c0 = np.zeros(p.nc)
    I0, dI0 = p.Intensity(c0)
    Itot = np.sum(I0)

    exm = []  # list of experiments
    for pc in photon_counts:
        for dn in dnoise:
            for st in strehl:
                grid = dict()
                grid['has_data'] = False  # will be changed when grid is filled
                grid['sigma'] = pyr.StrehlToSigma(st)
                grid['strehl'] = st
                grid['pcount'] = pc
                grid['det_noise'] = dn
                grid['reg_param'] = regp
                grid['ntrials'] = ntrials
                grid['nl_maxit'] = nl_maxit
                grid['lin_score'] = 0*np.zeros(ntrials)
                grid['lin_cost'] = 0*np.zeros(ntrials)
                grid['nl_score'] = np.zeros(ntrials)
                grid['nl_cost'] = np.zeros(ntrials)
                grid['nl_result'] = []  # list of result dict
                exm.append(grid)

    k = example_index  # do kth grid point
    print 'gridpoint: started ', k, ' of ', len(exm) - 1
    t0 = time.time()
    exm[k]['has_data'] = True
    sigma = exm[k]['sigma']
    pcount = exm[k]['pcount']
    d_noise = exm[k]['det_noise']
    for tr in range(exm[k]['ntrials']):
        ph = sigma*np.random.randn(p.nc)
        ph -= np.mean(ph)
        I, dI = p.Intensity(ph)
        y = I*(pcount/Itot)
        noise_sig = np.sqrt(y) + d_noise
        y += noise_sig*np.random.randn(len(y))
        y *= (Itot/pcount)
        noise_sig *= (Itot/pcount)
        wt = None  # for ML set to 1/noise_sig^2 --> precision issues
        reg = exm[k]['reg_param']
        ls_x, var_x = p.LeastSqSol(y, wt, c0, RegParam=reg, ZeroPhaseMean=True)
        ls_cost, dls_cost = p.Cost(ls_x, y, wt, RegParam=reg)
        exm[k]['lin_score'][tr] = np.std(ls_x - ph)
        exm[k]['lin_cost'][tr] = ls_cost
        nl_x, res = p.FindMinCost(ls_x, y, wt=None, RegParam=reg, method='BFGS',
                                  MaxIt=exm[k]['nl_maxit'], LsqStart=False, AmpCon=False)
        exm[k]['nl_score'][tr] = np.std(nl_x - ph)
        exm[k]['nl_cost'][tr] = res.fun
        res.x = None  # avoid storing large items
        res.hess_inv = None
        res.jac = None
        res.allvecs = None
        exm[k]['nl_result'].append(res)
    exm[k]['mean_lin_score'] = np.mean(exm[k]['lin_score'])
    exm[k]['std_lin_score'] = np.std(exm[k]['lin_score'])
    exm[k]['mean_nl_score'] = np.mean(exm[k]['nl_score'])
    exm[k]['std_nl_score'] = np.std(exm[k]['nl_score'])

    fname = filename_prefix + str(example_index) + '.p'
    with open(fname, 'w') as fp:
        pickle.dump(exm[k], fp)
    fp.close()

    t0 = (time.time() - t0)/60.
    print 'gridpoint: finished ', k, ' of ', len(exm) - 1, " in ", t0, " minutes."

    return  # end of function

if __name__ == "__main__":
    grid_solve_pool(0)
