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
#make sure you are in Optics/EFC
Vn = A.GetNull(Cdh, pixlist=None, sv_thresh=1.e-3)  # get null space
abest_im = np.load("abest_im.npy"); abest_re = np.load("abest_re.npy");  #modulation commands
assert (Vn.shape[1] == len(abest_im)) and (Vn.shape[1] == len(abest_re))
abEx0 = np.abs(A.Shx@np.exp(1j* Cdh               ) + A.sphx*A.SpeckleFactor)
abEx1 = np.abs(A.Shx@np.exp(1j*(Cdh + Vn@abest_re)) + A.sphx*A.SpeckleFactor)
abEx2 = np.abs(A.Shx@np.exp(1j*(Cdh + Vn@abest_im)) + A.sphx*A.SpeckleFactor)
Ey0 = A.Shy@np.exp(1j* Cdh               ); 
Ey1 = A.Shy@np.exp(1j*(Cdh + Vn@abest_re)); 
Ey2 = A.Shy@np.exp(1j*(Cdh + Vn@abest_im)); 
Ey0_re = np.real(Ey0); Ey0_im = np.imag(Ey0)
Ey1_re = np.real(Ey1); Ey1_im = np.imag(Ey1)
Ey2_re = np.real(Ey2); Ey2_im = np.imag(Ey2)

fig(); plt.plot(abEx0,'ko',abEx1,'ro',abEx2,'co'); 
plt.title('|Ex| for nominal and probed states');
fig(); plt.plot(Ey0_re,'ko',Ey0_im,'k*',Ey1_re,'ro',Ey1_im,'rx',Ey2_re,'co',Ey2_im,'cx');
plt.title("|Re(Ey)| and |Im(Ey)| in nominal and probed states");






Ey_nom = A.Shy@np.exp(1j*Cdh)  # nominal cross field
mdEy_nom = np.median(np.abs(Ey_nom))
target = np.zeros((2*len(Ey_nom),2))
target[ len(Ey_nom):,0] = mdEy_nom*np.ones((len(Ey_nom),)) - np.real(A.Shy)@Cdh
target[-len(Ey_nom):,1] = mdEy_nom*np.ones((len(Ey_nom),)) - np.imag(A.Shy)@Cdh



V = A.GetNull(Cdh, A.HolePixels)  # the cols of V are the basis vectors
pSV = np.linalg.pinv(np.concatenate((np.real(A.Shy@V),np.imag(A.Shy@V)),axis=0))
cv = pSV@target  # null space coefficient vectors corresponding to target fields
Cd = V@cv  # DM command vectors corresponding to targets



Ixnom = A.PolIntensity(Cdh,          'X','Hole','phase',False,None)
Ix0 = A.PolIntensity(Cdh + Cd[:,0]/10,'X','Hole','phase',False,None)
Ix1 = A.PolIntensity(Cdh + Cd[:,1]/10,'X','Hole','phase',False,None)
fig(); plt.plot(Ixnom,'ko',Ix0,'rx',Ix1,'g*');

t = np.random.randn(1089);  t /= np.sqrt( (t*t).sum())
E0r = lambda c: np.real(A.Shx@np.exp(1j*(Cdh + c*V[:,0])))
E0i = lambda c: np.imag(A.Shx@np.exp(1j*(Cdh + c*V[:,0])))
E1r = lambda c: np.real(A.Shx@np.exp(1j*(Cdh + c*V[:,90])))
E1i = lambda c: np.imag(A.Shx@np.exp(1j*(Cdh + c*V[:,90])))
Etr = lambda c: np.real(A.Shx@np.exp(1j*(Cdh + c*t)))
Eti = lambda c: np.imag(A.Shx@np.exp(1j*(Cdh + c*t)))

F0r = lambda c: np.real(A.Shy@np.exp(1j*(Cdh + c*V[:,0])))
F0i = lambda c: np.imag(A.Shy@np.exp(1j*(Cdh + c*V[:,0])))
F1r = lambda c: np.real(A.Shy@np.exp(1j*(Cdh + c*V[:,90])))
F1i = lambda c: np.imag(A.Shy@np.exp(1j*(Cdh + c*V[:,90])))



fig();plt.plot(E0i(0),'ko',E0i(1),'kx',E1i(1),'rx');
 








