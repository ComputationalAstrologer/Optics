#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 15:47:09 2017

@author: frazin
"""
import numpy as np
import matplotlib.pyplot as plt
import Pyr as pr

py = pr.Pyr()

strehl = 0.3
sigma = pr.StrehlToSigma(strehl)

ph = sigma*np.random.randn(py.nc)
It = py.Intensity(ph)[0]
It = It.reshape(125, 125)
I_noint = py.IntensityBlock3Faces(ph, show_indiv=True)
I_noint = I_noint.reshape(125, 125)

plt.figure()
plt.imshow(np.sqrt(It))
plt.colorbar()
plt.title("$\sqrt{\mathrm{True \; PyWFS \; image}}$", FontSize='large')
plt.xlabel("pixel index", FontSize='large')
plt.ylabel("pixel index", FontSize='large')


diffratio = np.divide(np.abs(It-I_noint), It)
plt.figure()
plt.imshow(np.log10(diffratio))
plt.colorbar()
plt.title("log relative difference image", FontSize='large')
plt.xlabel("pixel index", FontSize='large')
plt.ylabel("pixel index", FontSize='large')
