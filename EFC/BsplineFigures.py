#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 26 15:42:26 2025
@author: Richard Frazin

This makes figures illustrating the use cubic B-spline basis functions
"""

import numpy as np
import matplotlib.pyplot as plt
from os import path as ospath  #needed for isfile(), join(), etc.
from sys import path as syspath

machine = "homeLinux"
#machine = "officeWindows"
if machine == "homeLinux":
    MySplineToolsLocation = "/home/rfrazin/Py/Optics"
    PropMatLoc = "/home/rfrazin/Py/EFCSimData/"
elif machine == 'officeWindows':
    MySplineToolsLocation = "F:/Python/Optics"
    PropMatLoc = "F:/MyOpticalSetups/EFC Papers/DataArrays"
syspath.insert(0, MySplineToolsLocation)
import Bspline3 as BS  # this module is in MySplineToolsLocation

figdir = "./PaperFigures/"


#%%
#interpolate a simple real-valued function and show contributions
#set up B-spline with 1 knot on each end for padding to handle edge effects
x = np.linspace(-np.pi, np.pi*129/131,131)  # 1D coord
nkn = 15  # number of knots
delta = (x.max()-x.min())/(nkn-2)
S = BS.UnivariateCubicSpline(x, nkn, Xmin=(x.min()-delta/2), Delta=delta)

f = 4. + np.cos(5*x + np.pi/4)  # fcn to interpolate
co = S.GetSplCoefs(f)
fhat = S.SplInterp(co)  # interpolated function

splcen = [];
for k in range(len(co)): splcen.append(S.splcen[k])

spline = lambda x, center : BS.Bspline3_1D_vec(x - center, S.Delta)

plt.figure(figsize=(8 ,4.8));  # default is (6.4, 4.8) in inches!
plt.title('Real-Valued Interpolation Example',fontsize=18)  # default is 16 in pts
plt.plot(x,f,'k-',linewidth=2,label='orig. fun');
plt.plot(x,fhat,'r-',linewidth=2,label='interp. fun');
plt.plot(splcen,co,'bH',markersize=6,label='coef. value');
plt.plot(splcen,np.zeros((len(splcen),)),'kx',markersize=9,markeredgewidth=4,label='knot loc.');
for k in range(len(co)):
   if k == len(co)//2:
      plt.plot(x, co[k]*spline(x, splcen[k]),"b--",linewidth=1,label='indiv.');
   else:
      plt.plot(x, co[k]*spline(x, splcen[k]),"b--",linewidth=1);
plt.xlim(splcen[0]-1.4, splcen[-1] +0.2);
plt.legend(loc='upper left', bbox_to_anchor=(0,1));
plt.xlabel("'x' spatial coord.")
plt.ylabel('function value')
#plt.savefig(figdir + 'Interp1D_realfcn.jpg')


#%%
#  interpolate a DM phase sheet in 1D
x = np.linspace(-0.5, 0.5, 165);
Delta21=1./21; Delta33 = 1./31  # setting delta33 to 1/31 is for padding
#B15 = BS.BivariateCubicSpline(xx.flatten(),yy.flatten(),15,Xmin=-0.5+Delta15,Delta=Delta15)
#B11 = BS.BivariateCubicSpline(xx.flatten(),yy.flatten(),11,Xmin=-0.5+Delta11,Delta=Delta11)
#B15 = BS.BivariateCubicSpline(xx.flatten(),yy.flatten(),15,Xmin=-0.5+Delta15,Delta=Delta15)
B21 = BS.UnivariateCubicSpline(x,21,Xmin=-0.5+Delta21,Delta=Delta21)
B33 = BS.UnivariateCubicSpline(x,33,Xmin=-0.5-Delta33,Delta=Delta33)
#21 random phases between -pi and pi, normal random deviates.  std=1.0 rad
phrnd = np.array([-1.6040482 ,  0.7249395 ,  0.97271276, -1.95491176,  0.99068236, 0.89580684, -1.63097499,  0.0581289 ,  0.16723109,  0.73545627, 0.38529965, -0.75767653, -0.47477813, -0.01023433, -0.89190528, 1.34632011,  1.25190763, -0.90265665,  0.49165466, -0.36632656,1.38786673])
phrnd -= np.mean(phrnd)

#%%
#example: random DM command

co21 = np.pi*phrnd
height = B21.SplInterp(co21)
phasor21 = np.exp(1j*height)
co33 = B33.GetSplCoefs(phasor21);
phasor33 = B33.SplInterp(co33)

height_ratios = [1,2.5]
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 10), gridspec_kw={'height_ratios': height_ratios});
ax1.plot(x,height,'k-',linewidth=2);
ax1.set_title('DM phase');
ax1.set_ylabel('phase (rad)')

ax2.plot(x, np.real(phasor21),'r-',linewidth=3,label='true real part');
ax2.plot(x, np.imag(phasor21),'k-',linewidth=3,label='true imaginary part');
ax2.plot(x, np.abs(phasor21) ,'c-',linewidth=3,label='true absolute value');
ax2.plot(x, np.real(phasor33),'r:',linewidth=1,label=None)#'fitted real part');
ax2.plot(x, np.imag(phasor33),'k:',linewidth=1,label=None)#'fitted imaginary part');
ax2.plot(x, np.abs(phasor33) ,'c:',linewidth=1,label=None)#'fitted absolute value');
skip = 4
ax2.plot(x[::skip], np.real(phasor33[::skip]),'rd',linewidth=1,label='fitted real part');
ax2.plot(x[::skip], np.imag(phasor33[::skip]),'kd',linewidth=1,label='fitted imaginary part');
ax2.plot(x[::skip], np.abs(phasor33[::skip]) ,'cd',linewidth=1,label='fitted absolute value');
ax2.set_xlabel("'x' spatial coordinate")
ax2.set_ylabel('phasor value');
ax2.set_title('Phasor Interpolation');
ax2.legend(loc='lower center');
#plt.savefig(figdir + 'PhasorSplInerp_sig3.14.jpg')


co21 = (np.pi/3)*phrnd
height = B21.SplInterp(co21)
phasor21 = np.exp(1j*height)
co33 = B33.GetSplCoefs(phasor21);
phasor33 = B33.SplInterp(co33)

height_ratios = [1,2.5]
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 10), gridspec_kw={'height_ratios': height_ratios});
ax1.plot(x,height,'k-',linewidth=2);
ax1.set_title('DM phase');
ax1.set_ylabel('phase (rad)')

ax2.plot(x, np.real(phasor21),'r-',linewidth=3,label='true real part');
ax2.plot(x, np.imag(phasor21),'k-',linewidth=3,label='true imaginary part');
ax2.plot(x, np.abs(phasor21) ,'c-',linewidth=3,label='true absolute value');
ax2.plot(x, np.real(phasor33),'r:',linewidth=1,label=None)#'fitted real part');
ax2.plot(x, np.imag(phasor33),'k:',linewidth=1,label=None)#'fitted imaginary part');
ax2.plot(x, np.abs(phasor33) ,'c:',linewidth=1,label=None)#'fitted absolute value');
skip = 4
ax2.plot(x[::skip], np.real(phasor33[::skip]),'rd',linewidth=1,label='fitted real part');
ax2.plot(x[::skip], np.imag(phasor33[::skip]),'kd',linewidth=1,label='fitted imaginary part');
ax2.plot(x[::skip], np.abs(phasor33[::skip]) ,'cd',linewidth=1,label='fitted absolute value');
ax2.set_xlabel("'x' spatial coordinate")
ax2.set_ylabel('phasor value');
ax2.set_title('Phasor Interpolation');
ax2.legend(loc='lower right');
#plt.savefig(figdir + 'PhasorSplInerp_sig1.04.jpg')

#%%
