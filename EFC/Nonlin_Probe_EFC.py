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

sc = 1. # scale factor for all fields
z  = sc*20 # incoherent intensity
f  = np.sqrt(sc*30)*np.exp(1j*np.pi/6)
p1 = np.sqrt(sc*150)*np.exp(1j*np.pi*7/8)
p2 = p1*np.exp(-1j*np.pi/2)

        

#This produces the intensity in units of photons at a single pixel
# x is the input vector
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

    sp = x[1] + 1j*x[2]
    for l in range(len(probes)):
        tc = probes[l] + sp   # total coherent field
        s = np.exp(x[0]) + 2*np.real(tc*np.conj(tc))
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

    
        
        


