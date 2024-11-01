#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 18:09:24 2024
@author: rfrazin

This provides analysis based on EFC class in EFC.py.
It's kind of a grabbag of code lines

"""
# %%
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy import optimize
import EFC
fig = plt.figure
MakePixList = EFC.MakePixList;
ProbeIntensity = EFC.ProbeIntensity
NegLLPoisson = EFC.NegLLPoisson
CRB_Poisson = EFC.CRB_Poisson
RobustPoisson = EFC.RobustPoissonRnd
# %%

with open('stuff20240905.pickle','rb') as filep:  stuffB= pickle.load(filep)
B = EFC.EFC(HolePixels=stuffB['HolePixels'], SpeckleFactor=stuffB['SpeckleFactor'])
Cdh = stuffB['DHcmd']  # dark hole command for dominant field
photons = stuffB['photons_unitycontrast']  # photons/pixel/exposure corresponding to unity contrast
extfac = stuffB['extfac_sqrt2eps']  # this polarizer extinction factor is a field (not intensity) multiplier, corresponding to 2 times the intensity exctinction factor of a single polarizer.
et = stuffB['pixel_extent']


# %%
######################################################################
# make PSF images #
######################################################################

IXnom = B.PolIntensity(0*Cdh,XorY='X',region='Full',DM_mode='phase',return_grad=False,SpeckleFactor=0).reshape((256,256))
IYnom = B.PolIntensity(0*Cdh,XorY='Y',region='Full',DM_mode='phase',return_grad=False,SpeckleFactor=0).reshape((256,256))
IXab  = B.PolIntensity(0*Cdh,XorY='X',region='Full',DM_mode='phase',return_grad=False,SpeckleFactor=None).reshape((256,256))
IYab  = B.PolIntensity(0*Cdh,XorY='Y',region='Full',DM_mode='phase',return_grad=False,SpeckleFactor=None).reshape((256,256))
IXabh = B.PolIntensity(  Cdh,XorY='X',region='Full',DM_mode='phase',return_grad=False,SpeckleFactor=None).reshape((256,256))
IYabh = B.PolIntensity(  Cdh,XorY='Y',region='Full',DM_mode='phase',return_grad=False,SpeckleFactor=None).reshape((256,256))
 # %%


SaveFigs = False

plt.figure(); plt.imshow(np.log10(1.e-7 + IXnom[et[2]:et[3],et[0]:et[1]]), extent=et,cmap='seismic',origin='lower');plt.colorbar();
plt.title('Nominal Dominant PSF (contrast units)'); plt.ylabel('pixel index'); plt.xlabel('pixel index');
if SaveFigs: plt.savefig('Figs/NomDomPSF.png', dpi=300, bbox_inches='tight')

plt.figure(); plt.imshow(np.log10(1.e-13 + IYnom[et[2]:et[3],et[0]:et[1]]), extent=et,cmap='seismic',origin='lower');plt.colorbar();
plt.title('Nominal Cross PSF (contrast units)'); plt.ylabel('pixel index'); plt.xlabel('pixel index');
if SaveFigs: plt.savefig('Figs/NomCrossPSF.png', dpi=300, bbox_inches='tight')

plt.figure(); plt.imshow(np.log10(1.e-7 + IXab[et[2]:et[3],et[0]:et[1]]), extent=et,cmap='seismic',origin='lower');plt.colorbar();
plt.title('Aberrated Dominant PSF (contrast units)'); plt.ylabel('pixel index'); plt.xlabel('pixel index');
if SaveFigs: plt.savefig('Figs/AbDomPSF.png', dpi=300, bbox_inches='tight')

plt.figure(); plt.imshow(np.log10(1.e-13+ IYab[et[2]:et[3],et[0]:et[1]]), extent=et,cmap='seismic',origin='lower');plt.colorbar();
plt.title('Aberrated Cross PSF (contrast units)'); plt.ylabel('pixel index'); plt.xlabel('pixel index');
if SaveFigs: plt.savefig('Figs/AbCrossPSF.png', dpi=300, bbox_inches='tight')

plt.figure(); plt.imshow(np.log10(1.e-7 + IXabh[et[2]:et[3],et[0]:et[1]]), extent=et,cmap='seismic',origin='lower');plt.colorbar();
plt.title('Aberrated Dominant PSF (contrast units) with Hole'); plt.ylabel('pixel index'); plt.xlabel('pixel index');
if SaveFigs: plt.savefig('Figs/AbDomPSFwHole.png', dpi=300, bbox_inches='tight')

plt.figure(); plt.imshow(np.log10(1.e-13+ IYabh[et[2]:et[3],et[0]:et[1]]), extent=et,cmap='seismic',origin='lower');plt.colorbar();
plt.title('Aberrated Cross PSF (contrast units) with Hole'); plt.ylabel('pixel index'); plt.xlabel('pixel index');
if SaveFigs: plt.savefig('Figs/AbCrossPSFwHole.png', dpi=300, bbox_inches='tight')

et2 = [ 145, 176, 145, 176]  #  for dark hole closeup
plt.figure(); plt.imshow(np.log10(1.e-13 + IXabh[et2[2]:et2[3],et2[0]:et2[1]]), vmax=-7, extent=et2,cmap='seismic',origin='lower');plt.colorbar();
plt.title('Aberrated Dominant PSF (contrast units) with Hole'); plt.ylabel('pixel index'); plt.xlabel('pixel index');
if SaveFigs: plt.savefig('Figs/CloseUpDomHole.png', dpi=300, bbox_inches='tight');

plt.figure(); plt.imshow(np.log10(1.e-13 + IYabh[et2[2]:et2[3],et2[0]:et2[1]]), extent=et2,cmap='seismic',origin='lower');plt.colorbar();
plt.title('Aberrated Cross PSF (contrast units) with Hole'); plt.ylabel('pixel index'); plt.xlabel('pixel index');
if SaveFigs: plt.savefig('Figs/CloseUpCrossHole.png', dpi=300, bbox_inches='tight');



# %%

######################################################################
# optimization to find good probes (assuming 10^-6 linear polarizer)
######################################################################

cfcn = lambda a: B.CostCrossFieldWithDomPenalty(a, Cdh, return_grad=False, OptPixels=None, mode='Int',intthr=1.e-6,pScale=2.e-3)
#cfcnRe = lambda a: A.CostCrossFieldWithDomPenalty(a, Cdh, return_grad=False, OptPixels=None, mode='Re',intthr=1.e-6,pScale=2.e-3)

Ntrials = 21
#sol_re = []; sol_im = []; cost_re =[]; cost_im = []
sol = []; cost = []
for k in range(Ntrials):
    #out = optimize.minimize(cfcnRe, .8*np.pi*(np.random.rand(1089)-.5), options={'disp':True,'maxiter':40}, method='Powell',jac=False)
    #sol_re.append(out['x']); cost_re.append(out['fun'])
    out = optimize.minimize(cfcn, .8*np.pi*(np.random.rand(1089)-.5), options={'disp':True,'maxiter':40}, method='Powell',jac=False)
    sol.append(out['x']);
    cost.append(out['fun'])
#sol = sol_re + sol_im; # '+' concats two lists here!
#cost = cost_re + cost_im  # '+' concats two lists here!


#############################################################
# load some optimized probe solutions and analyze the CRBs  #
#############################################################
# %%
from EFC import CRB_Poisson
with open('ProbeSolutions20240905.pickle','rb') as pfpp:
    soldict = pickle.load(pfpp);
sols = soldict['All_solutions']


ftdom = B.Field(Cdh,'X','Hole','phase',False,SpeckleFactor=None)  # true dominant field at dark hole
ftcro = B.Field(Cdh,'Y','Hole','phase',False,SpeckleFactor=None)  # true cross field at dark hole
fmcro = B.Field(Cdh,'Y','Hole','phase',False,SpeckleFactor=0.)  # model dark hole cross field
ftdom *= extfac

import time;  time_loopstart = time.time()
metric = []; indlist = [];
for k1 in range(len(sols)):
  print("loop",k1,"of",len(sols), "elapsed time is", (time.time()-time_loopstart)/60 ,"minutes")
  prdom1 = B.Field(Cdh + sols[k1],'X','Hole','phase',False,SpeckleFactor=None) - ftdom  # k1 dominant probe
  prdom1 *= extfac
  prcro1 = B.Field(Cdh + sols[k1],'Y','Hole','phase',False,SpeckleFactor=0.  ) - fmcro
  for k2 in np.arange(k1+1, len(sols)):
      prdom2 = B.Field(Cdh + sols[k2],'X','Hole','phase',False,SpeckleFactor=None) - ftdom
      prdom2 *= extfac
      prcro2 = B.Field(Cdh + sols[k2],'Y','Hole','phase',False,SpeckleFactor=0.  ) - fmcro
      for k3 in np.arange(k2+1, len(sols)):
        indlist.append( (k1,k2,k3) )
        pk = np.zeros((len(B.HolePixels)))
        prdom3 = B.Field(Cdh + sols[k3],'X','Hole','phase',False,SpeckleFactor=None) - ftdom
        prdom3 *= extfac
        prcro3 = B.Field(Cdh + sols[k3],'Y','Hole','phase',False,SpeckleFactor=0.  ) - fmcro
        for kp in range(len(B.HolePixels)):  # pixel index
           f0  = np.array( [ftdom[ kp], ftcro[ kp]])  # true field values
           pr1 = np.array( [prdom1[kp], prcro1[kp] ] ) # k1 probe
           pr2 = np.array( [prdom2[kp], prcro2[kp] ] ) # k2 probe
           pr3 = np.array( [prdom3[kp], prcro3[kp] ] ) # k3 probe
           s0, gs0 = ProbeIntensity(f0, 0*pr1, 'Cross', True)
           s1, gs1 = ProbeIntensity(f0,   pr1, 'Cross', True)
           s2, gs2 = ProbeIntensity(f0,   pr2, 'Cross', True)
           s3, gs3 = ProbeIntensity(f0,   pr3, 'Cross', True)
           S = np.array([s0, s1, s2, s3])*photons
           Sg = np.stack( (gs0, gs1, gs2, gs3) )*photons
           crb = CRB_Poisson(S, Sg)
           pk[kp] = np.max(np.diag(crb))
        metric.append(pk)

metric = np.array(metric)  #metric has the max CRB for each solution pair

met2 = np.max(metric,axis=1); b = np.where(met2 == met2.min())[0][0]
sol1 = sols[indlist[b][0]]; sol2 = sols[indlist[b][1]]; sol3 = sols[indlist[b][2]]  # best solution

# %%

#################################################################
# perform probing estimates  #
#################################################################
# %%

with open('ProbeSolutions20240905.pickle','rb') as pfpp:
    soldict = pickle.load(pfpp);
sol1 = soldict['Best3'][0]; sol2 = soldict['Best3'][1]; sol3 = soldict['Best3'][2]

f0x = B.Field(Cdh,'X','Hole','phase',False,None)*extfac  # true dom field with speckles
f0y = B.Field(Cdh,'Y','Hole','phase',False,None)  # true cross field with speckles
f0my = B.Field(Cdh,'Y','Hole','phase',False,0.)  # model field (no speckles)
pm = 1  #positive probes
px1  = B.Field(Cdh + pm*sol1,'X','Hole','phase',False,None) - f0x;  # tru dom probe 1
px2  = B.Field(Cdh + pm*sol2,'X','Hole','phase',False,None) - f0x;  # tru dom probe 2
px3  = B.Field(Cdh + pm*sol3,'X','Hole','phase',False,None) - f0x;  #               3
py1 = B.Field(Cdh + pm*sol1,'Y','Hole','phase',False,0.) - f0my;  # probe 1
py2 = B.Field(Cdh + pm*sol2,'Y','Hole','phase',False,0.) - f0my;  # probe 2
py3 = B.Field(Cdh + pm*sol3,'Y','Hole','phase',False,0.) - f0my   #       3
px1 *= extfac; px2 *= extfac; px3 *= extfac


# %%

sqphots = np.sqrt(photons);
S =   np.zeros((len(B.HolePixels),4))  # array of true intensities
#g0l = np.zeros((len(B.HolePixels), )).astype('complex')  # linearly estimated cross fields
g0n = np.zeros((len(B.HolePixels), )).astype('complex')  # nonlinearly estimated cross fields
cvg0n = np.zeros((len(B.HolePixels),2,2))  # estimate error covariance matrices
std0n = np.zeros((len(B.HolePixels),2))  #  corresponding error bars
U = 1.0*S  # array of measured intensities
# %%
for k in range(len(B.HolePixels)):
#
    S[k,0], gSk0 = ProbeIntensity([sqphots*f0x[k], sqphots*f0y[k]],
                                  [0., 0.],                         'Cross', True)  # unprobed intensity
    S[k,1], gSk1 = ProbeIntensity([sqphots*f0x[k], sqphots*f0y[k]],  #
                                  [sqphots*px1[k], sqphots*py1[k]], 'Cross', True)
    S[k,2], gsk2 = ProbeIntensity([sqphots*f0x[k], sqphots*f0y[k]],  #
                                  [sqphots*px2[k], sqphots*py2[k]], 'Cross', True)
    S[k,3], gsk3 = ProbeIntensity([sqphots*f0x[k], sqphots*f0y[k]],  #
                                  [sqphots*px3[k], sqphots*py3[k]], 'Cross', True)
    U[k,0] = RobustPoisson(2 + S[k,0])  #measured unprobed intensity
    U[k,1] = RobustPoisson(2 + S[k,1])  #measured intensity for probe 1 - the constant is for dark counts
    U[k,2] = RobustPoisson(2 + S[k,2])  #measured intensity for probe 2
    U[k,3] = RobustPoisson(2 + S[k,3])  #                             3
    measlist = [U[k,0], U[k,1], U[k,2], U[k,3] ]
    Probes = sqphots*np.array( [[0., 0.], [px1[k], py1[k]], [px2[k],py2[k]], [px3[k],py3[k]]] )

    #Perform estimation of a = [ Re(f0y), Im(f0y) ]
    #This has two modes.
    #a - len(a) = 2 , real and imag parts of fields to be estimated
    #mode = 'NegLL' - the output is a cost function and gradient corresponding to the negative log-likelihood
    #       'CRB' - the ouput is the Cramer Rao lower  bound matrix (no gradient)
    def Poisson_CostNegLL_Or_CRB(a, mode='NegLL'):
      if len(a) != 2:
          raise ValueError("a has two components. a[0] is the real part and a[1] is the imag part.")
      if mode not in ['NegLL', 'CRB']:
          raise ValueError("mode must be 'NegLL' or 'CRB'. ")
      intlist = []; gradintlist = []
      for kp in range(len(Probes)):
          Ikp, gIkp = ProbeIntensity( [sqphots*f0x[k], sqphots*(a[0] +1j*a[1])], Probes[kp,:],'CrossDom', True)
          gIkp = gIkp[2:]  # only need the last two components of the gradient output
          intlist.append(Ikp); gradintlist.append(gIkp)
      intlist = np.array(intlist);  gradintlist = np.array(gradintlist)
      if mode == 'NegLL':
          c, cg = NegLLPoisson( measlist, intlist, gradintlist )
          cg *= sqphots   # without this, the gradient is w.r.t. the field in sqphots units
          return (c, cg)
      else:  # mode = 'CRB'
          return CRB_Poisson(intlist, gradintlist)/sqphots  # gradintlist is in sqphots units
    #this performs a local minimization at each grid point
#
    def GridSearchMin():
#
        om = optimize.minimize
        fun = Poisson_CostNegLL_Or_CRB
        amps = [ np.sqrt( U[k,0]/photons) ]  #  with the dark hole + polarizers, the unprobed is all cross intensity
        angs = list(np.linspace(0, 2*np.pi*(23/24), 24) - np.pi)
        #amps = []; amps.append(np.abs(f0y[k]))  # append the true answer
        #angs = []; angs.append(np.angle(f0y[k]))
        nm = len(amps); ng = len(angs)
        cost = []
        sols = []
        for km in range(nm):
            for kg in range(ng):
                start = amps[km]*np.exp(1j*angs[kg]);
                a0 = np.array([np.real(start), np.imag(start)])
                out = om(fun,a0,args=(),method='CG',jac=True,options={'maxiter':90})
                cost.append(out['fun'])
                sols.append(out['x'])
        cost = np.array(cost); sols = np.array(sols)
        sol = sols[np.argmin(cost)]
#
        return sol[0] + 1j*sol[1]
    g0n[k] = GridSearchMin()
    cvg0n[k,:,:] = Poisson_CostNegLL_Or_CRB([np.real(g0n[k]), np.imag(g0n[k])], mode='CRB')/sqphots
    std0n[k,:] = np.sqrt(np.diag(cvg0n[k,:,:]))


# %%  plot estimates
plt.figure(figsize=(7,9))
plt.plot(np.arange(441), np.real(f0y), 'ks', markersize=8,label='Real part of true cross field');
plt.errorbar(np.arange(441), np.real(g0n), fmt='ro', markersize=4, yerr=std0n[:, 0], label='Real part of estimated cross field');
plt.title('Real part of cross field', fontsize=14);
plt.xlabel('pixel index (within dark hole)', fontsize=11);
plt.ylabel('electric field ($\sqrt{\mathrm{contrast}}$ units)',fontsize=11)
plt.legend(fontsize=12);
if False:
  plt.savefig('Figs/RealEstimate.png', dpi=300, bbox_inches='tight');

plt.figure(figsize=(7,9))
plt.plot(np.arange(441), np.imag(f0y), 'ks', markersize=8,label='Imaginary part of true cross field');
plt.errorbar(np.arange(441), np.imag(g0n), fmt='ro', markersize=4, yerr=std0n[:, 0], label='Imaginary part of estimated cross field');
plt.title('Imaginary part of cross field', fontsize=14);
plt.xlabel('pixel index (within dark hole)', fontsize=11);
plt.ylabel('electric field ($\sqrt{\mathrm{contrast}}$ units)', fontsize=11)
plt.legend(fontsize=12);
if False:
  plt.savefig('Figs/ImagEstimate.png', dpi=300, bbox_inches='tight');


# %%
#############################################################
#      probe field plots                                    #
#############################################################

with open('ProbeSolutions20240905.pickle','rb') as pfpp:
    soldict = pickle.load(pfpp);
sol1 = soldict['Best3'][0]; sol2 = soldict['Best3'][1]; sol2 = soldict['Best3'][2]

pm = 1.   # positive probes
IAhx0 = B.PolIntensity(Cdh,'X','Hole','phase',False,None)
fAhy0 = B.Field(Cdh,'Y','Hole','phase',False,0.);
IAhx1 = B.PolIntensity(Cdh + pm*sol1,'X','Hole','phase',False,None)
phy1p = B.Field(Cdh + pm*sol1,'Y','Hole','phase',False,0.) - fAhy0;


plt.figure(figsize=(10,5)); #result of modulation with solution 1
plt.plot(extfac*np.sqrt(IAhx0),label='Unprobed Dominant $\sqrt{Intensity}$',marker='s',color='black',ls='None');
plt.plot(extfac*np.sqrt(IAhx1),label='Probed Dominant $\sqrt{Intensity}$',marker='s',color='tan',ls='None');
plt.plot(np.real(phy1p),label='real part of cross probe', marker='d',color='crimson',ls='None');
plt.plot(np.imag(phy1p),label='imag part of cross probe',marker='p',color='dodgerblue',ls='None');
plt.title('Probe Fields',fontsize=12)
plt.xlabel('pixel index',fontsize=12);
plt.ylabel('field ($\sqrt{\mathrm{constrast}}$ units)',fontsize=12);
plt.legend();
if False:
  plt.savefig('Figs/ManyFields.png', dpi=300, bbox_inches='tight')


# %%

# %%  plot Ix as a function of probe amplitude
pramp = np.logspace(-3,0,33)
Ihole = np.zeros_like(pramp)
for k in range(len(pramp)):
    Ihole[k] = np.median(B.PolIntensity(Cdh + pramp[k]*sol1,'X','Hole','phase',False,None))

fig(); plt.plot(pramp,Ihole,'ko-'); plt.xscale('log'); plt.yscale('log');
plt.tick_params(axis='both', labelsize=12);
plt.xlabel('probe amplitude', fontsize=12);
plt.ylabel('median dark hole contrast', fontsize=12);
plt.title('Domintant Dark Hole Intensity vs. Probe Amplitude',fontsize=12)
if savefigs:
  plt.savefig('Figs/HoleIntVsProbeAmp.png', dpi=300, bbox_inches='tight')


# %%  linear estimator stuff


#pm = -1  #negative probes
#px1n = A.Field(Cdh + pm*sol1,'X','Hole','phase',False,0.) - fAhx0;  # probe 1
#px1n *= extfac
#px2n = A.Field(Cdh + pm*sol2,'X','Hole','phase',False,0.) - fAhx0;  # probe 2
#px2n *= extfac
#py1n = A.Field(Cdh + pm*sol1,'Y','Hole','phase',False,0.) - fAhy0;  # probe 1
#py2n = A.Field(Cdh + pm*sol2,'Y','Hole','phase',False,0.) - fAhy0;  # probe 2
#py3n = A.Field(Cdh + pm*sol3,'Y','Hole','phase',False,0.) - f0my   #       3
#fAhx0 = A.Field(Cdh,'X','Hole','phase',False,0.)  # model dom field (no speckles)


    # def LinearEstimator():
    #     sqp = sqphots/1.41421;  # this splits the exposure time between + and - probes
    #     a = [ np.real(f0y[k]) , np.imag(f0y[k]) ]
    #     I1p = ProbeIntensity( [sqp*f0x[k], sqp*(a[0] +1j*a[1])], [sqp*px1[ k], sqp*py1[ k]],'Cross', False)
    #     I2p = ProbeIntensity( [sqp*f0x[k], sqp*(a[0] +1j*a[1])], [sqp*px2[ k], sqp*py2[ k]],'Cross', False)
    #     I1n = ProbeIntensity( [sqp*f0x[k], sqp*(a[0] +1j*a[1])], [sqp*px1n[k], sqp*py1n[k]],'Cross', False)
    #     I2n = ProbeIntensity( [sqp*f0x[k], sqp*(a[0] +1j*a[1])], [sqp*px2n[k], sqp*py2n[k]],'Cross', False)
    #     U1p = np.random.poisson(2 + I1p) - ProbeIntensity( [sqp*f0x[k], sqp*(a[0] +1j*a[1])], [sqp*px1[ k], sqp*py1[ k]],'Cross', False, IdomOnly=True)
    #     U1n = np.random.poisson(2 + I1n) - ProbeIntensity( [sqp*f0x[k], sqp*(a[0] +1j*a[1])], [sqp*px1n[k], sqp*py1n[k]],'Cross', False, IdomOnly=True)
    #     U2p = np.random.poisson(2 + I2p) - ProbeIntensity( [sqp*f0x[k], sqp*(a[0] +1j*a[1])], [sqp*px2[ k], sqp*py2[ k]],'Cross', False, IdomOnly=True)
    #     U2n = np.random.poisson(2 + I2n) - ProbeIntensity( [sqp*f0x[k], sqp*(a[0] +1j*a[1])], [sqp*px2n[k], sqp*py2n[k]],'Cross', False, IdomOnly=True)
    #     q = np.array([(U1p - U1n), (U2p - U2n)])/2
    #     M = photons*np.array([[ - np.imag(py1[k]), np.real(py1[k])],[-np.imag(py2[k]), np.real(py2[k])]])
    #     fyhreim = np.linalg.pinv(M).dot(q)  # estimate
    #     return(fyhreim[0] + 1j*fyhreim[1])  # phasor for estimated cross field
