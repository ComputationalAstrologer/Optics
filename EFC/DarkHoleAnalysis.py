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
from scipy import optimize
import EFC
fig = plt.figure
MakePixList = EFC.MakePixList;
ProbeIntensity = EFC.ProbeIntensity
CRB_Poisson = EFC.CRB_Poisson
EFC = EFC.EFC  # need this to load the pickle

#get EFC instance w/ dark hole command
stuff = pickle.load(open('minus10HoleWithSpeckles33x33.pickle','rb'))
Z = stuff['EFC class'];
Cdh = stuff['DH command']  # Dark Hole command (phase values not DM height)

#############################################################
# load some optimized probe solutions and analyze the CRBs  #
#############################################################
from EFC import CRB_Poisson
stuff = pickle.load(open('stuff20240626.pickle','rb'))
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
metric = np.zeros((len(sols),len(sols),len(A.HolePixels)))
for k1 in range(len(sols)):
  prdom1 = A.Field(Cdh + sols[k1],'X','Hole','phase',False,SpeckleFactor=None) - fmdom  # k1 dominant probe
  prcro1 = A.Field(Cdh + sols[k1],'Y','Hole','phase',False,SpeckleFactor=None) - fmcro
  for k2 in np.arange(k1+1, len(sols)):
      prdom2 = A.Field(Cdh + sols[k2],'X','Hole','phase',False,SpeckleFactor=None) - fmdom
      prcro2 = A.Field(Cdh + sols[k2],'Y','Hole','phase',False,SpeckleFactor=None) - fmcro
      for k3 in range(len(A.HolePixels)):  # pixel index
          f0  = np.array( [ftdom[k3], ftcro[k3], sInc ])  # true field values
          pr1 = np.array( [prdom1[k3], prcro1[k3] ] )  # k1 probe 
          pr2 = np.array( [prdom2[k3], prcro2[k3] ] ) # k2 probe
          s0, gs0 = ProbeIntensity(f0, 0*pr1, 'Cross', True)
          s1, gs1 = ProbeIntensity(f0,   pr1, 'Cross', True)
          s2, gs2 = ProbeIntensity(f0,   pr2, 'Cross', True)
          S = np.array([s0, s1, s2])*photons
          Sg = np.stack( (gs0, gs1, gs2) )*photons
          crb = CRB_Poisson(S, Sg)



######################################################################
# continuous region probe experiment (assuming 10^-5 linear polarizer)
######################################################################

extfac = np.sqrt(1.e-5) # linear polarizer (amplitude) extinction factor

lgabsr = lambda a: np.log10(np.abs(np.real(a)))
lgabsi = lambda a: np.log10(np.abs(np.imag(a)))

IZ0fx = Z.PolIntensity(Cdh,XorY='X',region='Full',DM_mode='phase',return_grad=False,SpeckleFactor=None)
IZ0fx = IZ0fx.reshape((256,256))
plt.figure(); plt.imshow(np.log10(1.e-13+IZ0fx),cmap='seismic',origin='lower');plt.colorbar(); plt.title('Ix')

pli = MakePixList([160,164,160,164], (256,256))
A = EFC(HolePixels=pli, SpeckleFactor=Z.SpeckleFactor)

cfcnIm = lambda a: A.CostCrossFieldWithDomPenalty(a, Cdh, return_grad=False, OptPixels=None,ReOrIm='Im',intthr=1.e-6,pScale=2.e-3)
cfcnRe = lambda a: A.CostCrossFieldWithDomPenalty(a, Cdh, return_grad=False, OptPixels=None,ReOrIm='Re',intthr=1.e-6,pScale=2.e-3)

Ntrials = 21
sol_re = []; sol_im = []; cost_re =[]; cost_im = []
for k in range(Ntrials):
    out = optimize.minimize(cfcnRe, .15*np.random.randn(1089), options={'disp':True,'maxiter':40}, method='Powell',jac=False)
    sol_re.append(out['x']); cost_re.append(out['fun'])
    out = optimize.minimize(cfcnIm, .15*np.random.randn(1089), options={'disp':True,'maxiter':40}, method='Powell',jac=False)
    sol_im.append(out['x']); cost_im.append(out['fun'])

sol = sol_re + sol_im; 
cost = cost_re + cost_im

sIx0 = np.sqrt(A.PolIntensity(Cdh,'X','Hole','phase',False,None))
fy0 = A.Field(Cdh,'Y','Hole','phase',False,0.);
sI_p = []
fyr = []; fyi = [];
for cmd in sol:
    sI_p.append( np.sqrt(A.PolIntensity(Cdh + cmd,'X','Hole','phase',False,None)) )
    fyr.append( np.real(A.Field(Cdh + cmd,'Y','Hole','phase',False,0.)) - np.real(fy0))
    fyi.append( np.imag(A.Field(Cdh + cmd,'Y','Hole','phase',False,0.)) - np.imag(fy0) )



##########################

filep = open('3_lambda_D_probe.pickle','rb'); stuff = pickle.load(filep); filep.close()
C_pre = stuff['probe_real']; C_pim = stuff['probe_imag'];

pm = 1.
IAhx0 = A.PolIntensity(Cdh,'X','Hole','phase',False,None)
fAhy0 = A.Field(Cdh,'Y','Hole','phase',False,0.);
IAhx_pre = A.PolIntensity(Cdh + pm*C_pre,'X','Hole','phase',False,None)
IAhx_pim = A.PolIntensity(Cdh + pm*C_pim,'X','Hole','phase',False,None)
fAhy_pre = A.Field(Cdh + pm*C_pre,'Y','Hole','phase',False,0.);
fAhy_pim = A.Field(Cdh + pm*C_pim,'Y','Hole','phase',False,0.);

plt.figure(); #result of modulation with 'Re' target
plt.plot(extfac*np.sqrt(IAhx0),marker='s',color='black',ls='None');
plt.plot(extfac*np.sqrt(IAhx_pre),marker='s',color='tan',ls='None');
plt.plot((np.real(fAhy_pre) - np.real(fAhy0)),marker='d',color='crimson',ls='None');
plt.plot((np.imag(fAhy_pre) - np.imag(fAhy0)),marker='p',color='dodgerblue',ls='None');
plt.title('Real Probe Target');plt.xlabel('pixel index'); plt.ylabel('modulation field')

plt.figure();  # result of modulation with 'Im' target
plt.plot(extfac*np.sqrt(IAhx0),marker='s',color='black',ls='None');
plt.plot(extfac*np.sqrt(IAhx_pim),marker='s',color='tan',ls='None');
plt.plot((np.real(fAhy_pim) - np.real(fAhy0)),marker='d',color='crimson',ls='None');
plt.plot((np.imag(fAhy_pim) - np.imag(fAhy0)),marker='p',color='dodgerblue',ls='None');
plt.title('Imag Probe Target');plt.xlabel('pixel index'); plt.ylabel('modulation field')

plt.figure(); # fields w/o modulation
plt.plot(np.real(fAhy0),marker='d',color='crimson',ls='None');
plt.plot(np.imag(fAhy0),marker='p',color='dodgerblue',ls='None');
plt.title('Nominal Dark Hole Cross field');plt.xlabel('pixel index'); plt.ylabel('nominal field')


######################################################################

#single pixel experiment
#this creates a modulating DM command for Ey in the Null space of Shx
singlepix = MakePixList([150,150,186,186],(256,256))
A = EFC(singlepix, SpeckleFactor=Z.SpeckleFactor)
Mx = A.MakeMmat(Cdh,'X',singlepix)
My = A.MakeMmat(Cdh,'Y',singlepix)
Ux, sx, Vx = np.linalg.svd(Mx);  Uy, sy, Vy = np.linalg.svd(My); Vx = Vx.T; Vy = Vy.T
t0 = Vy[:,0]; t1 = Vy[:,1];
a0 = np.linalg.pinv(Vx[:,2:])@t0; ht0 = Vx[:,2:]@a0; cmd0 = Vx[:,2:]@a0
a1 = np.linalg.pinv(Vx[:,2:])@t1; ht1 = Vx[:,2:]@a1; cmd1 = Vx[:,2:]@a1 
fig(); plt.plot(t1,'ko',ht1,'rx');plt.title('Match to 1st singular vector from Vx Null');
fig(); plt.plot(t0,'ko',ht0,'rx');plt.title('Match to 0th singular vector from Vx Null');
fig(); plt.plot(cmd0,'ko');plt.title('DM phase for 0th SV match');
fig(); plt.plot(cmd1,'ko');plt.title('DM phase for 1st SV match');

ey0 = lambda alpha : (A.Shy*np.exp(1j*Cdh))@np.exp(1j*alpha*cmd0)
ey1 = lambda alpha : (A.Shy*np.exp(1j*Cdh))@np.exp(1j*alpha*cmd1)
ey00 = ey0(0.)
ey10 = ey1(0.)
ey0 = lambda alpha : (A.Shy*np.exp(1j*Cdh))@np.exp(1j*alpha*cmd0) - ey00
ey1 = lambda alpha : (A.Shy*np.exp(1j*Cdh))@np.exp(1j*alpha*cmd1) - ey10



alpha = np.linspace(-5,5,151)
abex = np.zeros((len(alpha),2))
eyre = np.zeros((len(alpha),2))
eyim = np.zeros((len(alpha),2))
if len(A.HolePixels) > 1 : assert False  # this assumes a single pixel in the DH
for k in range(len(alpha)):
    abex[k,0] = np.sqrt(A.PolIntensity(alpha[k]*cmd0 + Cdh,'X','Hole','phase',False,None))
    abex[k,1] = np.sqrt(A.PolIntensity(alpha[k]*cmd1 + Cdh,'X','Hole','phase',False,None))
    eyre[k,0] = np.real(ey0(alpha[k]))
    eyre[k,1] = np.real(ey1(alpha[k]))
    eyim[k,0] = np.imag(ey0(alpha[k]))
    eyim[k,1] = np.imag(ey1(alpha[k]))

fig();
pl0 = plt.plot(alpha,abex[:,0],'ko-' ,label=r'$\sqrt{I_x}$')
pl1 = plt.plot(alpha,eyre[:,0],'rx:' ,label='Re(Ey)')
pl2 = plt.plot(alpha,eyim[:,0],'c^-.',label='Im(Ey)');
plt.ylabel(r'field $\sqrt{\mathrm{contrast}}$')
plt.xlabel('DM command scaling factor')
plt.title('Single Pixel Constrained Probe Command')
plt.legend();

fig();
pl0 = plt.plot(alpha,abex[:,1],'ko-' ,label=r'$\sqrt{I_x}$')
pl1 = plt.plot(alpha,eyre[:,1],'rx:' ,label='Re(Ey)')
pl2 = plt.plot(alpha,eyim[:,1],'c^-.',label='Im(Ey)');
plt.ylabel(r'field $\sqrt{\mathrm{contrast}}$')
plt.xlabel('DM command scaling factor')
plt.title('Single Pixel Constrained Probe Command')
plt.legend();


#not sure when it works and when it does not.

ff = lambda xy : np.ravel_multi_index((xy[1],xy[0]), (256,256))


#pl = [ (180,160), (181,161) ]  #(181,160) ,  (181,160),
#somepix = []
#for k in range(len(pl)): somepix.append(ff(pl[k]))

somepix = MakePixList( [180,182,160,162] , (256,256)); 
pl =[(180,160),(182,162),(180,162),(182,160) ]
for k in range(len(pl)): somepix.remove(ff(pl[k]))

somepix = [ff((181,156)), ff((181,161))]

A = EFC(somepix, SpeckleFactor=Z.SpeckleFactor)
Mx = A.MakeMmat(Cdh,'X',somepix)
My = A.MakeMmat(Cdh,'Y',somepix)
Ux, sx, Vx = np.linalg.svd(Mx);  Uy, sy, Vy = np.linalg.svd(My); Vx = Vx.T; Vy = Vy.T
t0 = Vy[:,0]; t1 = Vy[:,1];
a0 = np.linalg.pinv(Vx[:,len(sx):])@t0; ht0 = Vx[:,len(sx):]@a0; cmd0 = Vx[:,len(sx):]@a0
a1 = np.linalg.pinv(Vx[:,len(sx):])@t1; ht1 = Vx[:,len(sx):]@a1; cmd1 = Vx[:,len(sx):]@a1 
fig(); plt.plot(t1,'ko',ht1,'rx');plt.title('Match to 1st singular vector from Vx Null');
fig(); plt.plot(t0,'ko',ht0,'rx');plt.title('Match to 0th singular vector from Vx Null');
fig(); plt.plot(cmd0,'ko');plt.title('DM phase for 0th SV match');
fig(); plt.plot(cmd1,'ko');plt.title('DM phase for 1st SV match');





              
               
               
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
 








