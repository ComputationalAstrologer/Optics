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
EFC = EFC.EFC  # need this to load the pickle
# %%

######################################################################
#  make image of original dark hole
######################################################################
with open('minus10HoleWithSpeckles33x33.pickle','rb') as filep: stuff = pickle.load(filep);
Cdh = stuff['DH command']  # Dark Hole command (phase values not DM height)
Z = stuff['EFC class'];
IZ0fx = Z.PolIntensity(Cdh,XorY='X',region='Full',DM_mode='phase',return_grad=False,SpeckleFactor=None)
IZ0fx = IZ0fx.reshape((256,256))
plt.figure(); plt.imshow(np.log10(1.e-13+IZ0fx),cmap='seismic',origin='lower');plt.colorbar(); plt.title('Ix')


with open('stuff20240902.pickle','rb') as filep: stuff = pickle.load(filep)
A = EFC(HolePixels=stuff['HolePixels'], SpeckleFactor=stuff['SpeckleFactor'])
Cdh = stuff['DHcmd']
sols = stuff['solutions']



######################################################################
# optimization to find good probes (assuming 10^-6 linear polarizer)
######################################################################

cfcn = lambda a: A.CostCrossFieldWithDomPenalty(a, Cdh, return_grad=False, OptPixels=None, mode='Int',intthr=1.e-6,pScale=2.e-3)
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
from EFC import CRB_Poisson

ftdom = A.Field(Cdh,'X','Hole','phase',False,SpeckleFactor=None)  # true dominant field
ftcro = A.Field(Cdh,'Y','Hole','phase',False,SpeckleFactor=None)  # true cross field
fmdom = A.Field(Cdh,'X','Hole','phase',False,SpeckleFactor=0.) # model field at dark hole 
fmcro = A.Field(Cdh,'Y','Hole','phase',False,SpeckleFactor=0.)

photons = 1.e15
metric = []; k1k2 = [];
for k1 in range(len(sols)):
  prdom1 = A.Field(Cdh + sols[k1],'X','Hole','phase',False,SpeckleFactor=None) - ftdom  # k1 dominant probe
  prcro1 = A.Field(Cdh + sols[k1],'Y','Hole','phase',False,SpeckleFactor=0.  ) - fmcro
  for k2 in np.arange(k1+1, len(sols)):
      k1k2.append((k1,k2))
      prdom2 = A.Field(Cdh + sols[k2],'X','Hole','phase',False,SpeckleFactor=None) - ftdom
      prcro2 = A.Field(Cdh + sols[k2],'Y','Hole','phase',False,SpeckleFactor=0.  ) - fmcro
      pk = np.zeros((len(A.HolePixels)))
      for k3 in range(len(A.HolePixels)):  # pixel index
          f0  = np.array( [ftdom[ k3], ftcro[ k3]])  # true field values
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
    
metric = np.array(metric)  #metric has the max CRB for each solution pair

met2 = np.max(metric,axis=1); b = np.where(met2 == met2.min())[0][0]
sol1 = sols[k1k2[b][0]]; sol2 = sols[k1k2[b][1]]  # best solution 

# %% 
kk = []; metric = [];  # find a third probe that helps
prdom1 = A.Field(Cdh + sol1,'X','Hole','phase',False,SpeckleFactor=None) - ftdom  # k1 dominant probe
prcro1 = A.Field(Cdh + sol1,'Y','Hole','phase',False,SpeckleFactor=0.  ) - fmcro
prdom2 = A.Field(Cdh + sol2,'X','Hole','phase',False,SpeckleFactor=None) - ftdom  # k1 dominant probe
prcro2 = A.Field(Cdh + sol2,'Y','Hole','phase',False,SpeckleFactor=0.  ) - fmcro
for km in range(len(sols)):
    if km == k1k2[b][0] or km == k1k2[b][1]: continue
    kk.append(km)
    sol3 = sols[km]
    prdom3 = A.Field(Cdh + sol3,'X','Hole','phase',False,SpeckleFactor=None) - ftdom
    prcro3 = A.Field(Cdh + sol3,'Y','Hole','phase',False,SpeckleFactor=0.) - fmcro
    pk = np.zeros((len(A.HolePixels)))  # store max(crb) for each pixel
    for kx in range(len(A.HolePixels)):  # pixel loop
        f0  = np.array([ftdom[kx], ftcro[kx]])  # true field values
        pr1 = np.array([prdom1[kx], prcro1[kx]])  # probe 1
        pr2 = np.array([prdom2[kx], prcro2[kx]])  # probe 2
        pr3 = np.array([prdom3[kx], prcro3[kx]])  # probe 3
        s0, gs0 = ProbeIntensity(f0, 0*pr1, 'Cross', True)
        s1, gs1 = ProbeIntensity(f0,   pr1, 'Cross', True)
        s2, gs2 = ProbeIntensity(f0,   pr2, 'Cross', True)
        s3, gs3 = ProbeIntensity(f0,   pr3, 'Cross', True)
        S = np.array([s0, s1, s2, s3])*photons
        Sg = np.stack((gs0, gs1, gs2, gs3))*photons
        crb = CRB_Poisson(S,Sg)
        pk[kx] = np.max(np.diag(crb))
    metric.append(pk)

metric = np.array(metric); met2 = np.max(metric, axis=1)
bb = np.where(met2 == met2.min())[0][0]
sol3 = sols[kk[bb]]

# %%

#################################################################
# load pickle containing the best probes and do some tests  #
#################################################################
# %%

with open('stuff20240902.pickle','rb') as filep:
  stuff= pickle.load(filep)
A = EFC(HolePixels=stuff['HolePixels'], SpeckleFactor=stuff['SpeckleFactor'])
Cdh = stuff['DHcmd']  # dark hole command for dominant field
sol1 = stuff['solution1']; sol2 = stuff['solution2']; sol3 = stuff['solution3']
del stuff



# %%
extfac = np.sqrt(9.99.e-6) # linear polarizer (amplitude) extinction factor
f0x = A.Field(Cdh,'X','Hole','phase',False,None)*extfac  # true dom field with speckles
f0y = A.Field(Cdh,'Y','Hole','phase',False,None)  # true cross field with speckles
f0my = A.Field(Cdh,'Y','Hole','phase',False,0.)  # model field (no speckles)
pm = 1  #positive probes
px1  = A.Field(Cdh + pm*sol1,'X','Hole','phase',False,None) - f0x;  # tru dom probe 1
px2  = A.Field(Cdh + pm*sol2,'X','Hole','phase',False,None) - f0x;  # tru dom probe 2
px3  = A.Field(Cdh + pm*sol3,'X','Hole','phase',False,None) - f0x;  #               3
py1 = A.Field(Cdh + pm*sol1,'Y','Hole','phase',False,0.) - f0my;  # probe 1
py2 = A.Field(Cdh + pm*sol2,'Y','Hole','phase',False,0.) - f0my;  # probe 2
py3 = A.Field(Cdh + pm*sol3,'Y','Hole','phase',False,0.) - f0my   #       3
px1 *= extfac; px2 *= extfac; px3 *= extfac


# %%
photons = 1.e15; sqphots = np.sqrt(photons);
S =   np.zeros((len(A.HolePixels),4))  # array of true intensities
#g0l = np.zeros((len(A.HolePixels), )).astype('complex')  # linearly estimated cross fields
g0n = np.zeros((len(A.HolePixels), )).astype('complex')  # nonlinearly estimated cross fields
cvg0n = np.zeros((len(A.HolePixels),2,2))  # estimate error covariance matrices
std0n = np.zeros((len(A.HolePixels),2))  #  corresponding error bars
U = 1.0*S  # array of measured intensities
# %%
for k in range(len(A.HolePixels)):
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
plt.savefig('Figs/RealEstimate.png', dpi=300, bbox_inches='tight');
# %%
plt.figure(figsize=(7,9))
plt.plot(np.arange(441), np.imag(f0y), 'ks', markersize=8,label='Imaginary part of true cross field');
plt.errorbar(np.arange(441), np.imag(g0n), fmt='ro', markersize=4, yerr=std0n[:, 0], label='Imaginary part of estimated cross field');
plt.title('Imaginary part of cross field', fontsize=14); 
plt.xlabel('pixel index (within dark hole)', fontsize=11);
plt.ylabel('electric field ($\sqrt{\mathrm{contrast}}$ units)', fontsize=11)
plt.legend(fontsize=12);
plt.savefig('Figs/ImagEstimate.png', dpi=300, bbox_inches='tight');

# %%  plot Ix as a function of probe amplitude
pramp = np.logspace(-3,0,33)
Ihole = np.zeros_like(pramp)
for k in range(len(pramp)):
    Ihole[k] = np.median(A.PolIntensity(Cdh + pramp[k]*sol1,'X','Hole','phase',False,None))

fig(); plt.plot(pramp,Ihole,'ko-'); plt.xscale('log'); plt.yscale('log');
plt.tick_params(axis='both', labelsize=12);


# %%
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





