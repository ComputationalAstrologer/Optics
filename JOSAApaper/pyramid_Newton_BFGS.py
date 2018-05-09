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

# The idea of this code is to see how many BFGS iterations are needed
#   to do at least as well as N Newton iterations, where N=1,2,...

print "initialzing..."
t0 = time.time()
p = pyr.Pyr()
t0 = (time.time() - t0)/60.
print "done initializing in ", str(t0), "minutes."

strehl = 0.8
sigma = pyr.StrehlToSigma(strehl)
pcount = 1.e9
dnoise = 1.
reg = 1.e-9
bfgs_maxit = 300
newton_maxit = 5

c0 = np.zeros(p.nc)
I0, dI0 = p.Intensity(c0)
Itot = np.sum(I0)

ph = sigma*np.random.randn(p.nc)
ph -= np.mean(ph)
I, dI = p.Intensity(ph)
y = I*(pcount/Itot)
noise_sig = np.sqrt(y) + dnoise
y += noise_sig*np.random.randn(len(y))
y *= (Itot/pcount)
noise_sig *= (Itot/pcount)
wt = None  # for ML set to 1/noise_sig^2 --> precision issues
ls_x, var_x = p.LeastSqSol(y, wt, c0, RegParam=reg, ZeroPhaseMean=True)
ls_cost, dls_cost = p.Cost(ls_x, y, wt, RegParam=reg)
ls_score = np.std(ls_x - ph)
newton_x = 1.*ls_x  # deep copy

print "Least square cost: ", str(ls_cost), ". Least square score: ", np.std(ph - ls_x), "."

bnres = []  # contains results
for k in range(newton_maxit):
    ex = dict()
    ex['reg_param'] = reg
    ex['ls_score'] = ls_score
    ex['ls_cost'] = ls_cost
    ex['ls_x'] = ls_x
    t0 = time.time()
    newton_x, res = p.FindMinCost(newton_x, y, wt=None, RegParam=reg, method='Newton-CG', MaxIt=1, LsqStart=False, AmpCon=False)
    print "Newton iteration 0, cost: ", res.fun, ". score: ", np.std(ph-newton_x), ". time: ", (t0 -  time.time())/3600., " hours."
    ex['newton_itcount'] = k+1
    ex['newton_cost'] = res.fun
    ex['newton_score'] = np.std(newton_x - ph)
    ex['newton_result'] = res
    ex['bfgs_cost'] = ex['newton_cost'] + 1.
    ex['bfgs_itcount'] = 1
    while ex['bfgs_cost'] > ex['newton_cost']:
        bfgs_x, res = p.FindMinCost(ls_x, y, wt=None, RegParam=reg, method='BFGS',
                                    MaxIt=ex['bfgs_itcount'], LsqStart=False, AmpCon=False)
        ex['bfgs_cost'] = res.fun
        ex['bfgs_result'] = res
        ex['bfgs_score'] = np.std(bfgs_x - ph)
        if ex['bfgs_itcount'] > bfgs_maxit:
            break
        ex['bfgs_itcount'] += 1
    bnres.append(ex)
    with open('newton_bfgs.p', 'w') as fp:
        pickle.dump([ph, y, bnres], fp)

