#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 10:49:32 2020
@author: rfrazin

This program shows how standard spline interpolation can be used to simulate
a deformable mirror (DM).  This should be a pretty good DM simulation, since
DMs do behave kind of like splines.

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import LSQBivariateSpline as SPL


#create an image, xim is the image coordinates
nim = 50
im = np.random.rand(nim, nim)
xim = np.linspace(-.5, .5, nim)
xx, yy = np.meshgrid(xim,xim)
xx = xx.reshape((nim*nim,))
yy = yy.reshape((nim*nim,))

#create a set of coordinates for the spline knots.  The spline knots are like
#  the actuators.  Their locations are specified by xsp
nsp  = 25
xsp = np.linspace(-.4999, .4999,nsp) 

myspl = SPL(xx, yy, im.reshape((nim*nim,)), xsp, xsp, kx=3, ky=3)

#smoothed image
sim = myspl(xim, xim)
#Alex says this output needs to be rotated 90 deg (+/- ?)...