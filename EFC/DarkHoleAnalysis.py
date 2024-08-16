#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 18:09:24 2024
@author: rfrazin

This provides analysis based on EFC class in EFC.py. 
It's kind of a grabbag of code lines

"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import optimize
import EFC
fig = plt.figure
MakePixList = EFC.MakePixList;
ProbeIntensity = EFC.ProbeIntensity
NegLLPoisson = EFC.NegLLPoisson
CRB_Poisson = EFC.CRB_Poisson
EFC = EFC.EFC  # need this to load the pickle

#get EFC instance w/ dark hole command
with open('minus10HoleWithSpeckles33x33.pickle','rb') as filep: stuff = pickle.load(filep);
Cdh = stuff['DH command']  # Dark Hole command (phase values not DM height)
#Z = stuff['EFC class'];

extfac = np.sqrt(1.e-5) # linear polarizer (amplitude) extinction factor

######################################################################
# optimization to find good probes (assuming 10^-5 linear polarizer)
######################################################################

IZ0fx = Z.PolIntensity(Cdh,XorY='X',region='Full',DM_mode='phase',return_grad=False,SpeckleFactor=None)
IZ0fx = IZ0fx.reshape((256,256))
plt.figure(); plt.imshow(np.log10(1.e-13+IZ0fx),cmap='seismic',origin='lower');plt.colorbar(); plt.title('Ix')

cfcnIm = lambda a: A.CostCrossFieldWithDomPenalty(a, Cdh, return_grad=False, OptPixels=None,ReOrIm='Im',intthr=1.e-6,pScale=2.e-3)
cfcnRe = lambda a: A.CostCrossFieldWithDomPenalty(a, Cdh, return_grad=False, OptPixels=None,ReOrIm='Re',intthr=1.e-6,pScale=2.e-3)

Ntrials = 21
sol_re = []; sol_im = []; cost_re =[]; cost_im = []
for k in range(Ntrials):
    out = optimize.minimize(cfcnRe, .15*np.random.randn(1089), options={'disp':True,'maxiter':40}, method='Powell',jac=False)
    sol_re.append(out['x']); cost_re.append(out['fun'])
    out = optimize.minimize(cfcnIm, .15*np.random.randn(1089), options={'disp':True,'maxiter':40}, method='Powell',jac=False)
    sol_im.append(out['x']); cost_im.append(out['fun'])

sol = sol_re + sol_im; # '+' concats two lists here!
cost = cost_re + cost_im  # '+' concats two lists here!


#############################################################
# load some optimized probe solutions and analyze the CRBs  #
#############################################################
from EFC import CRB_Poisson
with open('stuff20240626.pickle','rb') as filep: stuff = pickle.load(filep)
A =stuff['EFCobj']; # EFC class object
sols=stuff['solutions'];  # DM command solutions
sIp = 0.003*stuff['sIp']; # sqrt (DOM intensity) for each solution (with sqrt(10^-5) polarizer)
#fyr=stuff['fyr']; fyi=stuff['fyi']; # Real and Imag CROSS field probe values for each solution
ftdom = A.Field(Cdh,'X','Hole','phase',False)  # true dominant field
ftcro = A.Field(Cdh,'Y','Hole','phase',False)  # true cross field
fmdom = A.Field(Cdh,'X','Hole','phase',False,SpeckleFactor=0.) # model field at dark hole 
fmcro = A.Field(Cdh,'Y','Hole','phase',False,SpeckleFactor=0.)
sInc  = 0.  # sqrt(incoherent intensity)

photons = 1.e15
metric = []; k1k2 = []; 
met2 = 1.e12*np.ones((len(sols),len(sols)))
for k1 in range(len(sols)):
  prdom1 = A.Field(Cdh + sols[k1],'X','Hole','phase',False,SpeckleFactor=None) - fmdom  # k1 dominant probe
  prcro1 = A.Field(Cdh + sols[k1],'Y','Hole','phase',False,SpeckleFactor=None) - fmcro
  for k2 in np.arange(k1+1, len(sols)):
      k1k2.append((k1,k2))
      prdom2 = A.Field(Cdh + sols[k2],'X','Hole','phase',False,SpeckleFactor=None) - fmdom
      prcro2 = A.Field(Cdh + sols[k2],'Y','Hole','phase',False,SpeckleFactor=None) - fmcro
      pk = np.zeros((len(A.HolePixels)))
      for k3 in range(len(A.HolePixels)):  # pixel index
          f0  = np.array( [ftdom[k3], ftcro[k3]])  # true field values
          pr1 = np.array( [prdom1[k3], prcro1[k3] ] )  # k1 probe 
          pr2 = np.array( [prdom2[k3], prcro2[k3] ] ) # k2 probe
          s0, gs0 = ProbeIntensity(f0, 0*pr1, 'Cross', True)
          s1, gs1 = ProbeIntensity(f0,   pr1, 'Cross', True)
          s2, gs2 = ProbeIntensity(f0,   pr2, 'Cross', True)
          S = np.array([s0, s1, s2])*photons
          Sg = np.stack( (gs0, gs1, gs2) )*photons
          crb = CRB_Poisson(S, Sg)
          pk[k3] = np.max(np.diag(crb))
      metric.append(pk)
    
metric = np.array(metric)
met2 = np.max(metric,axis=1); b = np.where(met2 == met2.min())[0][0]
sol1 = sols[k1k2[b][0]]; sol2 = sols[k1k2[b][1]]  # best solution 

#################################################################
# load pickle containing the two best probes and do some tests  #
#################################################################

CSQ = lambda a: np.real( a*np.conj(a) )

with open('minus10HoleWithSpeckles33x33.pickle','rb') as filep:
  stuff = pickle.load(filep)
Z = stuff['EFC class']  #  EFC class object  -  with aberrations
Cdh = stuff['DH command']  # Dark Hole command (phase values not DM height)  2116 pixels in dark hole
with open('stuff20240626.pickle','rb') as filep:
  stuff= pickle.load(filep)
A =stuff['EFCobj']; # EFC class object  -  no aberrations - 441 pixels in dark hole
with open('TwoDMsolutions.pickle','rb') as filep:  # these are probe solutions found with a CRB analysis (see above)
    stuff = pickle.load(filep)
sol1 = stuff['solution1']; sol2 = stuff['solution2'];  del stuff

#get the speckle fields from Z corresponding to the dark hole in A
sphx = np.zeros((len(A.HolePixels),)).astype('complex')
sphy = np.zeros((len(A.HolePixels),)).astype('complex')
for k in range(len(A.HolePixels)):
    sphx[k] = Z.spx[A.HolePixels[k]]  # dom speckles
    sphy[k] = Z.spy[A.HolePixels[k]]  # cross speckles

f0x = Z.Field(Cdh,'X','Hole','phase',False,None)  # known dom field
f0y = Z.Field(Cdh,'Y','Hole','phase',False,None)  # unknown cross field
fAhx0 = A.Field(Cdh,'X','Hole','phase',False,None);  # model field
fAhy0 = A.Field(Cdh,'Y','Hole','phase',False,None);  # model field
pm = 1
px1 = A.Field(Cdh + pm*sol1,'X','Hole','phase',False,0.) - fAhx0;  # probe 1
px2 = A.Field(Cdh + pm*sol2,'X','Hole','phase',False,0.) - fAhx0;  # probe 2
py1 = A.Field(Cdh + pm*sol1,'Y','Hole','phase',False,0.) - fAhy0;  # probe 1
py2 = A.Field(Cdh + pm*sol2,'Y','Hole','phase',False,0.) - fAhy0;  # probe 2

photons = 1.e15; sqphots = np.sqrt(photons);
S = np.zeros((len(A.HolePixels),2))  # array of true intensities
g0 = np.zeros(())  # real and imag parts of estimated cross fields
U = 1.0*S  # array of measured intensities
for k in range(len(A.HolePixels)):
    S[k,0], gSk0 = ProbeIntensity([extfac*sqphots*f0x[k], sqphots*f0y[k]],  # true dom intensity
                                  [extfac*sqphots*px1[k], sqphots*py1[k]], 'Cross', True)
    S[k,1], gsk1 = ProbeIntensity([extfac*sqphots*f0x[k], sqphots*f0y[k]],  # true cross intensity
                                  [extfac*sqphots*px2[k], sqphots*py2[k]], 'Cross', True)
    U[k,0] = np.random.poisson(2 + photons*S[k,0])
    U[k,1] = np.random.poisson(2 + photons*S[k,1])

    #Perform estimation of a = [ Re(f0y), Im(f0y) ]
    #This is a cost fcn to be optimized
    def CostNegllPoisson(a, Ncnt):
      I1, gI1 = ProbeIntensity( [extfac*sqphots*f0x[k], sqphots*(a[0] +1j*a[1])], [extfac*sqphots*px1[k], sqphots*py1[k]],'CrossDom', True)
      I2, gI2 = ProbeIntensity( [extfac*sqphots*f0x[k], sqphots*(a[0] +1j*a[1])], [extfac*sqphots*px2[k], sqphots*py2[k]],'CrossDom', True)
      gI1 = gI1[2:,2:]; gI2 = gI2[2:,2:]
      c, cg = NegLLPoisson( [U[k,0], U[k,1]], [I1, I2],Ig=[gI1, gI2] )
      return (c, cg)

#############################################################
#       plots for the DM commands sol1 and sol2             #
#############################################################

pm = 1.   # positive probes
IAhx0 = A.PolIntensity(Cdh,'X','Hole','phase',False,None)
fAhx0 = A.Field(Cdh,'X','Hole','phase',False,0.);
fAhy0 = A.Field(Cdh,'Y','Hole','phase',False,0.);
IAhx1 = A.PolIntensity(Cdh + pm*sol1,'X','Hole','phase',False,None)
IAhx2 = A.PolIntensity(Cdh + pm*sol2,'X','Hole','phase',False,None)
phx1p = A.Field(Cdh + pm*sol1,'X','Hole','phase',False,0.) - fAhx0;
phx2p = A.Field(Cdh + pm*sol2,'X','Hole','phase',False,0.) - fAhx0;
phy1p = A.Field(Cdh + pm*sol1,'Y','Hole','phase',False,0.) - fAhy0;
phy2p = A.Field(Cdh + pm*sol2,'Y','Hole','phase',False,0.) - fAhy0;


plt.figure(); #result of modulation with solution 1
plt.plot(extfac*np.sqrt(IAhx0),marker='s',color='black',ls='None');
plt.plot(extfac*np.sqrt(IAhx1),marker='s',color='tan',ls='None');
plt.plot(np.real(phy1p),marker='d',color='crimson',ls='None');
plt.plot(np.imag(phy1p),marker='p',color='dodgerblue',ls='None');
plt.title('Solution One');plt.xlabel('pixel index'); plt.ylabel('modulation field');

plt.figure();  # result of modulation with solution 2
plt.plot(extfac*np.sqrt(IAhx0),marker='s',color='black',ls='None');
plt.plot(extfac*np.sqrt(IAhx2),marker='s',color='tan',ls='None');
plt.plot(np.real(phy2p),marker='d',color='crimson',ls='None');
plt.plot(np.imag(phy2p),marker='p',color='dodgerblue',ls='None');
plt.title('Solution 2');plt.xlabel('pixel index'); plt.ylabel('modulation field')

if False:
  pm = -1.  # negative probes
  phx1m = A.Field(Cdh + pm*sol1,'X','Hole','phase',False,0.) - fAhx0;
  phx2m = A.Field(Cdh + pm*sol2,'X','Hole','phase',False,0.) - fAhx0;
  phy1m = A.Field(Cdh + pm*sol1,'Y','Hole','phase',False,0.) - fAhy0;
  phy2m = A.Field(Cdh + pm*sol2,'Y','Hole','phase',False,0.) - fAhy0;

  plt.figure(); # fields w/o modulation
  plt.plot(np.real(fAhy0),marker='d',color='crimson',ls='None');
  plt.plot(np.imag(fAhy0),marker='p',color='dodgerblue',ls='None');
  plt.title('Nominal Dark Hole Cross field');plt.xlabel('pixel index'); plt.ylabel('nominal field')






