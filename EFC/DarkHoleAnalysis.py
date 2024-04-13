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
crap = pickle.load(open('minus9HoleWithSpeckles21x21.pickle','rb'))
G = crap['G']
#dark hole command
CGbest = crap['CGbest']; del(crap)



assert False

#make the M matrix for a 14x14 spot inside the dark hole
smpixlist = MakePixList([150,163,150,163],(256,256))
#the rows of V, e.g. V[3,:] form the singular basis of the domain of M 
U, svals, V = np.linalg.svd(G.MakeMmat(CGbest,smpixlist))

#plot singular values
sv = np.zeros((441,))
sv[:392] = svals
fig(); plt.plot(np.log10(1.e-15 + sv),'ko-'); plt.title('Singular Values')
plt.xlabel('index'); plt.ylabel('log10(Singular Values)');


Iddx = lambda v : np.mean(G.PolIntensity(CGbest + v, XorY='X', region='Full', DM_mode='phase', return_grad=False)[smpixlist])
Iddy = lambda v : np.mean(G.PolIntensity(CGbest + v, XorY='Y', region='Full', DM_mode='phase', return_grad=False)[smpixlist])
alpha = np.linspace(0,12,91)
ivals = np.arange(370,441)
Ix = np.zeros((len(ivals),len(alpha)))
Iy = 0.*Ix
for ki in range(len(ivals)):
    print('ki =', ki)
    for k in range(len(alpha)):
        Ix[ki, k] = Iddx(V[ivals[ki],:]*alpha[k])
        Iy[ki, k] = Iddy(V[ivals[ki],:]*alpha[k])

fig(); plt.plot(alpha,np.log10(Ix.T)); plt.title('X');
fig(); plt.plot(alpha,np.log10(Iy.T)); plt.title('Y');



#this is slow
#for ki in range(441):
#    for k in range(len(alpha)):
#        Ix[ki,k] = Iddx( alpha[k]*V[ki,:] )
#        Iy[ki,k] = Iddy( alpha[k]*V[ki,:] )


#Ibg = np.zeros(alpha.shape)
#Ism = np.zeros(alpha.shape)
#Icr = np.zeros(alpha.shape)
#for k in range(len(alpha)):
#    Ibg[k] = Iddx( alpha[k]*V[ 2,:])
#    Ism[k] = Iddx( alpha[k]*V[-2,:])
#    Icr[k] = Iddy( alpha[k]*V[-2,:])

#fig(); plt.plot(alpha,np.log10(Ibg),'k:',alpha,np.log10(Icr),'bx-' ,alpha,np.log10(Ism),'r-.'); 
#plt.title("DarkDark Intensity as a fcn of $\\alpha$"); plt.xlabel('$\\alpha$'); plt.ylabel('Mean Intensity');



