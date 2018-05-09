#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 16:41:33 2017

@author: frazin
"""

import matplotlib.pyplot as plt
import numpy as np
import Pyr as pyr

#average solutions over random input phase

p = pyr.Pyr(NDim=2, basis='PhasePixel', RegMat='ID')

ntrials = 100
pixind = [34, 34]

meanabshes = np.zeros((p.nc, p.nc))

for k in range(ntrials):
    ph = 2*np.pi*np.random.rand(p.nc)
    h = p.IntensityHessianAtPixel(pixind, ph)
    meanabshes += np.abs(h)/ntrials

#histogram stuff
vals = meanabshes.reshape(p.nc*p.nc,)
mv = np.max(vals)
# define bin edges

bins = [1.01*mv, .1*mv, .01*mv, .001*mv, 1.e-4*mv, 1.e-5*mv, 1.e-6*mv, 0.]
bins.reverse()
hist = np.histogram(vals, bins=bins, density=False)
