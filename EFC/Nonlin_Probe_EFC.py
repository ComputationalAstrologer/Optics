#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 18:08:39 2024
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
#   Note that the N! term is not included
#N - a list (or array) of count values (measurements)
#I - a list (or array) of intensity values (units are counts, but need not be integer valued.  I is the response variable
#    note len(I) must equal len(N)
#Ig  - optional list (not array!) of intensity gradients. If provided the gradient will be output
#Igg - optional list (not array!) of intensity hessians.  If provided, the hessian will be output.
#      requires gradient to be provided.
def NegLLPoisson(N, I, Ig=None, Igg=None):
    M = len(N)  # M is the number of "measurements"
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
        s = N[m]*np.log(I[m]) - I[m]
        negll -= s
        if Ig is not None:
            dlnPds = N[m]/I[m] - 1.
            sg = dlnPds*Ig[m]
            negllg -= sg
            if Igg is not None:        
                d2lnPds2 = -1.*N[m]/(I[m]**2)
                sgg = d2lnPds2*np.outer(Ig[m],Ig[m]) + dlnPds*Igg[m] 
                negllgg -= sgg
    if Igg is not None:
        return (negll, negllg, negllgg)
    if Ig is not None:
        return (negll, negllg)
    return negll

#This produces the intensity in units of photons at a single pixel
# x is the state vector
#   x[0] - log(incoherent intensity)
#   x[1] - speckle field (real part)
#   x[2] - speckle field (imag part)
# probes - a list, tuple, or array of complex numbers corresponding to probe fields
def intensity(x, probes, return_grad=False, return_hess=False):
    out = []
    if return_grad: out_grad = []
    if return_hess:  # it's a constant diag!
        out_hess = []
        s_gg = 2.*np.eye(3)
        s_gg[0,0] = np.exp(x[0]); 

    sp = x[1] + 1j*x[2]  # speckle field
    for l in range(len(probes)):
        tc = probes[l] + sp   # total coherent field
        s = np.exp(x[0]) + np.real(tc*np.conj(tc))
        out.append(s)
        if return_grad:
            s_g = np.zeros((3,))
            s_g[0] = np.exp(x[0])
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


def MonteCarloRun(Ntrials=1):
    sc = 1. # scale factor for all intensities
    z  = sc*20 # incoherent intensity
    f  = np.sqrt(sc*30)*np.exp(1j*np.pi/6)
    p1 = np.sqrt(sc*150)*np.exp(1j*np.pi*7/8)
    p2 = p1*np.exp(-1j*np.pi/2)
    probes = [0., p1, p2]
    
    Itrue = intensity([z, np.real(f), np.imag(f)],probes,False,False)
    Imeas = np.zeros((Ntrials,len(probes)))
    for k in range(Ntrials):
        for p in range(len(probes))
            Imeas[k,p] = np.random.poisson(Itrue[p],1)[0]
        
    
    return(None)
    

    
        
        


