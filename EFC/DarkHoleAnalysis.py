#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 18:09:24 2024
@author: rfrazin

This provides analysis based on EFC class in EFC.py. 
It's kind of a grabbag of random crap

"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
import EFC
fig = plt.figure
MakePixList = EFC.MakePixList
EFC = EFC.EFC  # need this to load the pickle

#get EFC instance w/ dark hole command
stuff = pickle.load(open('minus10HoleWithSpeckles33x33.pickle','rb'))
Z = stuff['EFC class'];
Cdh = stuff['DH command']  # Dark Hole command (phase values not DM height)
A = EFC(MakePixList([152,167,152,167], (256,256)), Z.SpeckleFactor)

#nominal cross field
Ey_nom = A.Shy@np.exp(1j*Cdh)
mdEy_nom = np.median(np.abs(Ey_nom))
Nmat, V = A.MakeNmat(Cdh, A.HolePixels)
NN = np.concatenate( (np.real(Nmat), np.imag(Nmat)) , axis=0)
target = np.zeros((NN.shape[0], 2))
target[:NN.shape[0]/2,0]  = mdEy_nom
target[-NN.shape[0]: ,1]  = mdEy_nom
pNN = np.linalg.pinv(NN)



    

                         