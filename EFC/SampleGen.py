# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 11:50:18 2024
@author: Richard Frazin
"""

import numpy as np
import matplotlib.pyplot as plt
import EFC


#pick region of interest
pl = EFC.MakePixList([115,145,138,168],(256,256))
A = EFC.EFC(HolePixels=pl, SpeckleFactor=0.)
Czero = np.zeros((1089,))
f0x = A.Field(Czero,'X','Hole','phase',False,0.)  # get nominal field in region
f0y = A.Field(Czero,'Y','Hole','phase',False,0.)


nsamp = 1024
inputs = np.zeros((nsamp,2,len(Czero)))
labels = np.zeros(inputs.shape)
ampp = 0.2 # amplitude std
phap = 0.4 # phase std
for ks in range(nsamp):
    coef = (1. + ampp*np.random.randn(len(Czero)))*np.exp(1j*phap*np.random.randn(len(Czero)))
    f1x = A.Shx@coef - f0x
    f1y = A.Shy@coef - f0y
    f1y = f1y.reshape((31,31))
    inputs[ks,0,:] = np.real(f1x)
    inputs[ks,1,:] = np.imag(f1x)
    labels[ks,0,:] = np.real(f1y)
    labels[ks,1,:] = np.imag(f1y)

   