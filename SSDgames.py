#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 13:21:32 2021
@author: Richard Frazin

This does various experiments with SSD
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize as MIZE
from IntensityDistributions import ModifiedRicianPlusConstant, Frazinian

#This an unweighted least-squares function for the Modified Rician
#v - a vector of length 2 containing the 'a' and 'sig' parameters of the MR
#y - the frequency values of the intensity histogram
#x - the intensity values corresponding to the frequency
def ChiSqModifiedRicianPlusConstant(v, y, x):
    assert len(v) == 3
    assert y.shape == x.shape
    (ym, dc, da, ds) = ModifiedRicianPlusConstant(x, v[0], v[1], v[2], return_derivs=True)
    ch = 0.5*np.sum( (ym - y)**2 )  # chi-squared value
    d0 = np.sum( (ym - y)*dc )
    d1 = np.sum( (ym - y)*da )
    d2 = np.sum( (ym - y)*ds )
    g = np.array([d0,d1,d2])
    return((ch, g))

def ChiSqFrazinian(v, y, x):
    assert len(v) == 4
    assert y.shape == x.shape
    (ym, da, dr, di, dc) = Frazinian(x, v[0], v[1], v[2], v[3], return_derivs=True)
    ch = 0.5*np.sum( (ym - y)**2 )
    d0 = np.sum( (ym - y)*da )
    d1 = np.sum( (ym - y)*dr )
    d2 = np.sum( (ym - y)*di )
    d3 = np.sum( (ym - y)*dc )
    # add boundary penalty for correlation coefficient
    g = np.array([d0, d1, d2, d3])
    return((ch, g))


if False:
    #generate intensities from ModifiedRicianPlusConstant
    ci = 1.
    ai = 2.
    si = 1.
    xi = (ai*ai + si*si)*np.linspace(0.01, 4., 50)
    pi = ModifiedRicianPlusConstant(xi, ci, ai, si, return_derivs=False)
    plt.figure(); plt.plot(xi, pi, 'ko-')
    
    
    guess = np.array([ai,si,si,0.2])
    out = MIZE(ChiSqFrazinian, guess, args=(pi,xi), method='CG', jac=True, bounds=None)
    v = out['x']
    print("v = ", v)
    fit = Frazinian(xi, v[0], v[1], v[2], v[3], return_derivs=False)
    plt.plot(xi, fit, 'rx:')


#note that Frazinian(xi, 3., 0.5, 2.5, 0.) gives a very similar distribution
#  ModifiedRicianPlusConstant(xi, 0.45, 3.45, 0.795), thus providing an example in which
#  non-circular normal statistics can effectively mimic a planet.
#Another example is Frazinian(xi, 5., 0.5, 2.5, 0.) and 
#   ModifiedRicianPlusConstant(3.9, 5.1,, 1.) or ModifiedRicianPlusConstant(.2,5.4,.76)

if True:

    #generate intensities from a non-circular distribution
    ca = 5. # constant phasor amplitude
    cr = 0.5  # std of real part of random phasor
    ci = 2.5  # std of imag part of random phasor
    cc = 0.  # correlation coef of real and imag parts
    xi = ((ca + cr)**2 + ci*ci)*np.linspace(.01,2.,50)  #intensity bins
    pi = Frazinian(xi, ca, cr, ci, cc)
    plt.figure(); plt.plot(xi,pi,'ko-');
    
    #note that this optimization problem has many local minima.  Probably best to
    #  implement with many random starting points.
    guess = np.array([3.9, 3.5, 1.])
    out = MIZE(ChiSqModifiedRicianPlusConstant, guess, args=(pi,xi),method='CG', jac=True, bounds=None)
    v = out['x']; print("v = ", v);
    fit = ModifiedRicianPlusConstant(xi, v[0], v[1], v[2], return_derivs=False)
    plt.plot(xi,fit,'rx:');
