#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created oNcnt Tue Feb 13 18:08:39 2024
@author: rfrazin

This explores the statistical properties of nonlinear probing for EFC
The objective is to estimate the speckle field and incoherent intensity
  using known probe fields under the assumption of Poisson noise.
All fields are in units of \sqrt(photons/exposure).  Intensities are in
  units of photons/exposure
At this time, this estimates the log of the incoherent intensity 

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

#This calculates the -1*(log likelihood) under a poisson distribution
#   Ncntote that the Ncnt! term is not included
#Ncnt - a list (or array) of count values (measurements)
#I - a list (or array) of intensity values (units are counts, but need not be integer valued.  I is the response variable
#    note len(I) must equal len(Ncnt)
#Ig  - optional list (not array!) of intensity gradients. If provided the gradient will be output
#Igg - optional list (not array!) of intensity hessians.  If provided, the hessian will be output.
#      requires gradient to be provided.
def NegLLPoisson(Ncnt, I, Ig=None, Igg=None):
    M = len(Ncnt)  # M is the number of "measurements"
    if len(I) != M:
        print("Input counts (M) and the number of intensity values do not match.")
        assert False
    if Ig is not None:
        assert len(Ig) == M        
    if Igg is not None:
        assert Ig is not None
        assert len(Igg) == M
    negll = 0.  #negative log-likelihood value
    negllg = 0.
    negllgg = 0.
    for m in range(M): #fist calculate LL, LLg, LLgg and then multiply by -1 when done
        s = Ncnt[m]*np.log(I[m]) - I[m]
        negll -= s
        if Ig is not None:
            dlnPds = Ncnt[m]/I[m] - 1.
            sg = dlnPds*Ig[m]
            negllg -= sg
            if Igg is not None:        
                d2lnPds2 = -1.*Ncnt[m]/(I[m]**2)
                sgg = d2lnPds2*np.outer(Ig[m],Ig[m]) + dlnPds*Igg[m] 
                negllgg -= sgg
    if Igg is not None:
        return (negll, negllg, negllgg)
    if Ig is not None:
        return (negll, negllg)
    return negll

#This produces the intensity in units of photons at a single pixel
# x is the state vector
#   x[0] - quantity that specifies incoherent intensit -- see kwarg "IncModel"
#   x[1] - speckle field (real part)
#   x[2] - speckle field (imag part)
# probes - a list, tuple, or array of complex numbers corresponding to probe fields
# IncModel - if 'log' -  the input parameter is the log  of the incoherent intensity (removes positivity requirement)
#          - if 'sqrt' - the input parameter is the sqrt of the incoherent intensity.  There is no positivity requirement
def Intensity(x, probes, return_grad=False, return_hess=False, IncModel='sqrt'):
    assert IncModel == "sqrt" or IncModel == "log"
    out = []
    if return_grad: out_grad = []
    if return_hess:  # it's a constant diag!
        out_hess = []
        s_gg = 2.*np.eye(3)
        if IncModel == 'sqrt': pass
        elif IncModel == 'log': s_gg[0,0] = np.exp(x[0]); 

    sp = x[1] + 1j*x[2]  # speckle field
    for l in range(len(probes)):
        tc = probes[l] + sp   # total coherent field
        if IncModel == 'log':
            iinc = np.exp(x[0])
            iinc_g = np.exp(x[0])
        elif IncModel == 'sqrt':
            iinc = x[0]**2
            iinc_g = 2*x[0]
        
        s = iinc + np.real(tc*np.conj(tc))
        out.append(s)
        if return_grad:
            s_g = np.zeros((3,))
            s_g[0] = iinc_g
            s_g[1] = 2*x[1] + 2*np.real(probes[l])
            s_g[2] = 2*x[2] + 2*np.imag(probes[l])
            out_grad.append(s_g)
        if return_hess:
            out_hess.append(s_gg)
    if not return_grad and not return_hess:
        return out
    if return_grad and not return_hess:
        return (out, out_grad)
    if return_hess and not return_grad:
        return (out, out_hess)
    if return_grad and return_hess:
        return (out, out_grad, out_hess)



#IncMocel - see comments for the same kwarg in the Intensity function
def MonteCarloRun(Ntrials=500, IncModel='sqrt', Estimator='NonLin'):
    assert IncModel == 'sqrt' or IncModel == 'log'
    assert Estimator == 'NonLin' or Estimator == 'Pairwise'
    sc = 1. # scale factor for all intensities
    z  = sc*20 # incoherent intensity - note this an intensity not a field!
    f  = np.sqrt(sc*30)*np.exp(1j*np.pi/6)   # field units 
    p1 = np.sqrt(sc*150)*np.exp(1j*np.pi*7/8)  # field units
    p2 = p1*np.exp(-1j*np.pi/2)  # field units
    probes = [0., p1, p2]
    if Estimator == 'Pairwise':
        sr2 = np.sqrt(2)
        z /= 2.; f /= sr2; p1 /= sr2; p2 /= sr2
        probes = [0, 0, p1, -p1, p2, -p2]
        
    if IncModel == 'log':
       x_true = np.array([np.log(z) , np.real(f), np.imag(f)])
    elif IncModel == 'sqrt':
       x_true = np.array([np.sqrt(z), np.real(f), np.imag(f)])
    Itrue = Intensity(x_true,probes,False,False)
   
    def PairwiseEstimator(Imeas):
        assert len(Imeas) == 6
        xhat = np.zeros((3,))
        I0 =  Imeas[0] + Imeas[1]
        I1 = (Imeas[2] - Imeas[3])/4
        I2 = (Imeas[4] - Imeas[5])/4
        mat = np.zeros((2,2))
        mat[0,0] = np.real(probes[2]); mat[0,1] = np.imag(probes[2])
        mat[1,0] = np.real(probes[4]); mat[1,1] = np.imag(probes[4])
        q = np.linalg.pinv(mat).dot(np.array([I1,I2]))
        xhat[1] = q[0]; xhat[2] = q[1]
        if IncModel == 'sqrt':
            xhat[0] = np.sqrt(np.abs( I0 - 2*(xhat[1]**2 + xhat[2]**2) ))
        elif IncModel == 'log':
            xhat[0] = np.log(np.abs( I0 - 2*(xhat[1]**2 + xhat[2]**2) ))
        return(xhat)
        
    #This is funciton given to the minimizer
    def WrapperNegllPoisson(x, Ncnt):
        I, Ig = Intensity(x, [0.,p1,p2],True,False)
        nll, nllg = NegLLPoisson(Ncnt, I, Ig, None)
        return (nll, nllg)
    def WrapperNegllPoissonHess(x, Ncnt):
        I, Ig, Igg = Intensity(x, [0.,p1,p2],True,True)
        nll, nllg, nllgg = NegLLPoisson(Ncnt, I, Ig, Igg)
        return nllgg

    #This performs local optimization with a series of starting points for |f| and its phase
    #Imeas is one series of probe measurements.
    # assumes 0th probe is 0, which provides un upper limit on |f|
    def GridSearchOptimize(Imeas):  
        fun = WrapperNegllPoisson
        funH = WrapperNegllPoissonHess
        magmax = np.sqrt(Imeas[0] + 2*np.sqrt(Imeas[0]))
        mag = magmax*np.logspace(-4,0,5,base=2)
        phase  = np.linspace(0, 2*np.pi*(9-1)/9, 9)
        funval = np.zeros((len(mag),len(phase)))
        xvals = np.zeros((len(mag), len(phase), 3))
        ops = {'disp':False, 'maxiter': 50}
        for km in range(len(mag)):
            for kp in range(len(phase)):
                mg = mag[km]
                ph = phase[kp]
                x = np.zeros((3,))
                Iinc = Imeas[0] - mg**2
                if Iinc <= 0.1: Iinc = 0.1
                x[0] = np.log(Iinc)
                #    x[0] = 0.
                f = mg*np.exp(1j*ph)
                x[1] = np.real(f)
                x[2] = np.imag(f)
                result = minimize(fun,x,args=(Imeas),method='Newton-CG',jac=True, hess=funH, options=ops)
                funval[km,kp] = result['fun']
                xvals[km,kp,:] = result['x']
        bestindex = np.unravel_index(np.argmin(funval),funval.shape)
        return(xvals[bestindex[0],bestindex[1],:])
            
    Imeas = np.zeros((Ntrials,len(probes)))
    xhat = np.zeros((Ntrials, 3))
    for k in range(Ntrials):
        for p in range(len(probes)):
            Imeas[k,p] = np.random.poisson(Itrue[p],1)[0]
        if Estimator == 'NonLin':
            xhat[k,:] = GridSearchOptimize(Imeas[k,:])
        elif Estimator == 'Pairwise':
            assert False  # under construction
        if IncModel == 'sqrt':  # ensure a positive estimate 
            xhat[k,0] = np.abs(xhat[k,0])  
    Ihat = Intensity(xhat, probes,False,False)
    return (xhat, x_true, Itrue, Ihat)
    
    
        
        


