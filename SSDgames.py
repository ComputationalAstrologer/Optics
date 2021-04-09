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
from IntensityDistributions import ModifiedRician, Frazinian

#generate intensities from a non-circular distribution

ca = 3. # constant phasor amplitude
cr = 1.5  # std of real part of random phasor
ci = 2.5  # std of imag part of random phasor
cc = .9  # correlation coef of real and imag parts
xi = ((ca + cr)**2 + ci*ci)*np.linspace(.01,2,50)  #intensity bins
pi = Frazinian(xi, ca, cr, ci, cc)

plt.figure(); plt.plot(xi,pi,'ko-');


#This an unweighted least-squares function for the Modified Rician
#v - a vector of length 2 containing the 'a' and 'sig' parameters of the MR
#y - the frequency values of the intensity histogram
#x - the intensity values corresponding to the frequency
def ChiSqModifiedRician(v, y, x):
    assert len(v) == 2
    assert y.shape == x.shape
    (ym, da, ds) = ModifiedRician(x, v[0], v[1], return_derivs=True)
    ch = 0.5*np.sum( (ym - y)**2 )  # chi-squared value
    d0 = np.sum( (ym -y)*da )
    d1 = np.sum( (ym -y)*ds )
    g = np.array([d0,d1])
    return((ch, g))

guess = np.array([3.5, 1.5])
out = MIZE(ChiSqModifiedRician, guess, args=(pi,xi),method='CG',
           jac=True, bounds=None)

