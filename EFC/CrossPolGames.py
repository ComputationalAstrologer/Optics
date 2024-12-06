#!/usr/bin/env python3
"""
author: Richard Frazin  (rfrazin@umich.edu).  Please email if you have questions.

This is tutorial code designed to teach people how to use the tools I created
in course of writing  the article I sent to the Journal of Astronomical Telescopes,
Instruments and Systems (JATIS) on Oct. 3, 2024, entitled: A Laboratory Method for
Measuring the Cross-Polarizaion in High-Contrast Imaging.
This article is also publicly available at http://arxiv.org/abs/2410.03579

Before doing anything with this file, please see README.txt

"""
# %%
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.sparse.linalg import eigsh
import EFC  # this module is the main one
# %%
# generate and instance of the EFC class.  This will simulate a coronagraph without any
#   dark hole pixels specifies and no speckle field.  There are no aberrations, so this is the nominal model.
A = EFC.EFC(HolePixels=None, SpeckleFactor=0.)  # if this doesn't work, go back to README.txt

#create a DM command corresponding to a flat surface
C_flat = np.zeros((A.Sx.shape[1],))  # A.Sx is dominant field Jacobian (the variable name has no relationship to the musical note)

#get the dominant ('X') electric field at the detector for the nominal model - this array is complex-valued
fd = A.Field(C_flat, XorY='X', region='Full', DM_mode='phase',return_grad=False,SpeckleFactor=0.)
#        cross    ('Y")
fc = A.Field(C_flat, XorY='Y', region='Full', DM_mode='phase',return_grad=False,SpeckleFactor=0.)
fd = fd.reshape((256,256));  fc = fc.reshape((256,256))
#make images of the imag parts of the dominant and cross fields
plt.figure(); plt.imshow(np.imag(fd),cmap='seismic',origin='lower');plt.colorbar();
plt.figure(); plt.imshow(np.imag(fc),cmap='seismic',origin='lower');plt.colorbar();
# to get the nominal PSF images shown in the top row of Fig. 8, multiply fc and fd by their complex conjugates
# %%

#Make a random DM command (phase values, not heights) and look at the intensities.
#Note .PolIntensity uses the .Field function called above
C_rnd = 0.2*np.random.randn(len(C_flat))
Id_rnd = A.PolIntensity(C_rnd,'X','Full','phase',False,0.).reshape((256,256))
Ic_rnd = A.PolIntensity(C_rnd,'Y','Full','phase',False,0.).reshape((256,256))
plt.figure(); plt.imshow(np.log10(1.e-7  + Id_rnd),cmap='seismic',origin='lower');plt.colorbar();
plt.figure(); plt.imshow(np.log10(1.e-12 + Ic_rnd),cmap='seismic',origin='lower');plt.colorbar();
# %%
#Going back to a flat DM command, let's include the speckle fields from the files SpeckleFieldReducedFrom33x33PhaseScreen_Ex.npy, SpeckleFieldReducedFrom33x33PhaseScreen_Ey.npy
Id_sp = A.PolIntensity(C_flat,'X','Full','phase',False,SpeckleFactor=1.).reshape((256,256))
Ic_sp = A.PolIntensity(C_flat,'Y','Full','phase',False,SpeckleFactor=1.).reshape((256,256))
plt.figure(); plt.imshow(np.log10(1.e-7  + Id_sp),cmap='seismic',origin='lower');plt.colorbar();
plt.figure(); plt.imshow(np.log10(1.e-12 + Ic_sp),cmap='seismic',origin='lower');plt.colorbar();
# %%
#Now let's make up new speckle fields arising from random phases and amplitudes multiplying the CBS basis functions in the input plane
rndphasor = 1 + 0.3*np.random.randn(len(C_flat)) + 1j*np.random.randn(len(C_flat))
spfield_d = (A.Sx@rndphasor).reshape((256,256)) - fd  # dominant field
spfield_c = (A.Sy@rndphasor).reshape((256,256)) - fc  # cross field
plt.figure(); plt.imshow(np.imag(spfield_d),cmap='seismic',origin='lower');plt.colorbar();
plt.figure(); plt.imshow(np.imag(spfield_c),cmap='seismic',origin='lower');plt.colorbar();
# %%
#Let's recreate the bottom row of Fig.8 in the paper
with open('stuff20240905.pickle','rb') as filep:  stuffB= pickle.load(filep)
print(stuffB.keys())  # see what's in there!
#this sets up the instance of the EFC class that already contains a dark for the
#  speckle field read in by the .__init__  function
B = EFC.EFC(HolePixels=stuffB['HolePixels'], SpeckleFactor=stuffB['SpeckleFactor'])
et = stuffB['pixel_extent']  # et phone home!
Cdh = stuffB['DHcmd']  # dark hole command for dominant field
extfac = stuffB['extfac_sqrt2eps']

IXabh = B.PolIntensity(  Cdh,XorY='X',region='Full',DM_mode='phase',return_grad=False,SpeckleFactor=None).reshape((256,256))
IYabh = B.PolIntensity(  Cdh,XorY='Y',region='Full',DM_mode='phase',return_grad=False,SpeckleFactor=None).reshape((256,256))

plt.figure(); plt.imshow(np.log10(1.e-7 + IXabh[et[2]:et[3],et[0]:et[1]]), extent=et,cmap='seismic',origin='lower');plt.colorbar();
plt.title('Aberrated Dominant PSF (contrast units) with Hole'); plt.ylabel('pixel index'); plt.xlabel('pixel index');

plt.figure(); plt.imshow(np.log10(1.e-13+ IYabh[et[2]:et[3],et[0]:et[1]]), extent=et,cmap='seismic',origin='lower');plt.colorbar();
plt.title('Aberrated Cross PSF (contrast units) with Hole'); plt.ylabel('pixel index'); plt.xlabel('pixel index');
# %%
# Let's make Fig.6 from the paper.
with open('ProbeSolutions20240905.pickle','rb') as pfpp:
    soldict = pickle.load(pfpp);
print(soldict.keys())  # see what's in there!
sol1 = soldict['Best3'][0]; sol2 = soldict['Best3'][1]; sol3 = soldict['Best3'][2]

pm = 1.   # positive probes
IAhx0 = B.PolIntensity(Cdh,'X','Hole','phase',False,None)  # Unprobed dominant intensity in the dark hole
IAhx1 = B.PolIntensity(Cdh + pm*sol1,'X','Hole','phase',False,None) # dominant intensity in the dark hole with probe #1 applied
fAhy0 = B.Field(Cdh,'Y','Hole','phase',False,0.);  # unprobed cross field in the dark hole
phy1p = B.Field(Cdh + pm*sol1,'Y','Hole','phase',False,0.) - fAhy0  # field corresponding to probe #1


plt.figure(figsize=(10,5)); #result of modulation with solution 1
plt.plot(extfac*np.sqrt(IAhx0),label='Unprobed Dominant $\sqrt{Intensity}$',marker='s',color='black',ls='None');
plt.plot(extfac*np.sqrt(IAhx1),label='Probed Dominant $\sqrt{Intensity}$',marker='s',color='tan',ls='None');
plt.plot(np.real(phy1p),label='real part of cross probe', marker='d',color='crimson',ls='None');
plt.plot(np.imag(phy1p),label='imag part of cross probe',marker='p',color='dodgerblue',ls='None');
plt.title('Probe Fields',fontsize=12)
plt.xlabel('pixel index',fontsize=12);
plt.ylabel('field ($\sqrt{\mathrm{constrast}}$ units)',fontsize=12);
plt.legend();
# %%
#
#To recreate Fig.10 (bearning in mind that the measurements are Poisson random deviates),
#   run the code lines in DarkHoleAnalysis.py under the heading: perform probing estimates

"""
                        Jacobian Testing

All EFC approaches, apart from the model-free one (ref. 10 in the paper) rely
  on the validity of the linearized hybrid equation for the dominant field,
  which is eq.29 in the paper.  This approach relies on the nonlinear version
  in eq.27 and its sibling for the cross field, i.e., eq.28.


Let Dx and Dy be the model (i.e., idealized) Jacobians indicated in the paper:
Dmx = B.Shx   # this only includes the 441 rows corresponding to the dark hole pixels
Dmy = B.Shy   # this only includes the 441 rows corresponding to the dark hole pixels
In this paper optical aberrations are mimicked with a a phase and amplitude screen
  applied to the input field.  A convenient way to do this for (at least for
  aberrations with power at low spatial frequencies) is to take advantage of the
  33x33 spline basis to which the Jacobians correspond.
"""
# %%
with open('stuff20240905.pickle','rb') as filep:  stuffB= pickle.load(filep)
#see what's in there!
print(stuffB.keys())
with open('ProbeSolutions20240905.pickle','rb') as pfpp:
    soldict = pickle.load(pfpp);
print(soldict.keys())  # see what's in there!
sol1 = soldict['Best3'][0]; sol2 = soldict['Best3'][1]; sol3 = soldict['Best3'][2]
solnames = ["sol1", "sol2", "sol3"]

#this sets up the instance of the EFC class that already contains a dark for the
#  speckle field read in by the .__init__  function
B = EFC.EFC(HolePixels=stuffB['HolePixels'], SpeckleFactor=stuffB['SpeckleFactor'])
Cdh = stuffB['DHcmd']  # dark hole command for dominant field
# %%

#First, let's trim the various vectors and matrices to the interior part of the image.
goodpix = EFC.MakePixList([45,210,45,210] ,(256,256))  # the spline model has problems outside of this region  due to high frequency errors (my guess is that the Lyot stop does this)
Sx = np.zeros((len(goodpix),B.Sx.shape[1])).astype('complex')
Sy = np.zeros((len(goodpix),B.Sx.shape[1])).astype('complex')
spx = np.zeros((len(goodpix),)).astype('complex')
spy = np.zeros((len(goodpix),)).astype('complex')
count = -1
for k in range(B.Sx.shape[0]):  # loop over rows (each column is an image in vector form)
   if k in goodpix:
      count += 1
      Sx[count, :] = B.Sx[k,:]
      Sy[count, :] = B.Sy[k,:]
      spx[count] = B.spx[k]*B.SpeckleFactor  # speckle field in dom polarization
      spy[count] = B.spy[k]*B.SpeckleFactor  # speckle field in cross polarization


# %%
#Let's find a vector of spline coefficients that
# comes close to reproducing the speckle field (spx and spy).   The basic equation is that the
# speckle field, f, is given by f = S a, where a is the complex-valued
# vector that plays the role of aberrations and S is the system matrix.  Note this model is equivalent
# to f = S (a + one) - S one , where one is vector of ones, so a represents perturbations from unity.
#Thus, the aberrated system matrix responsible for this specklefield is S.dot(np.diag(a + one))

abx = np.linalg.pinv(Sx)@spx
aby = np.linalg.pinv(Sy)@spy
abxy = (abx + aby)/2  # I don't have better idea.  See below.

#For reasons I don't understand, this scheme of weighting the two systems by their largest
#  singular values leads to large values of abxy that are not useful.
#normx, _ = eigsh(np.conj(Sx.T).dot(Sx), k=1, which='LM'); normx = np.sqrt(normx[0])  # largest SV
#normy, _ = eigsh(np.conj(Sy.T).dot(Sy), k=1, which='LM'); normy = np.sqrt(normy[0])  # largest SV
#spxy = np.hstack((spx/normx, spy/normy))  # make a long vector of the speckle fields
#Sxy = np.vstack((Sx/normx, Sy/normy))  # stack the system matrices
#abxy = np.linalg.pinv(Sxy)@(spxy)  # the pinv takes a minute

#!!!!!!!!!
rspx = Sx@(abxy)  # reconstructions of speckle fields
rspy = Sy@(abxy)


# %%
plt.figure();plt.plot(np.imag(spx),'ko',np.imag(rspx),'rx');plt.title('imag part of dominant speckle field');
plt.figure();plt.plot(np.real(spx),'ko',np.real(rspx),'rx');plt.title('real part of dominant speckle field');
plt.figure();plt.plot(np.imag(spy),'ko',np.imag(rspy),'rx');plt.title('imag part of cross speckle field');
plt.figure();plt.plot(np.real(spy),'ko',np.real(rspy),'rx');plt.title('real part of cross speckle field');

print("dominant speckle field, real part:")
print('The ratio of the misfit error std to the std of the true value is', np.std(np.real(rspx-spx))/np.std(np.real(spx)),
      '.  Their correlation coefficient is', np.corrcoef(np.real(rspx),np.real(spx))[0,1]  ,  '.')
print("dominant speckle field, imag part:")
print('The ratio of the misfit error std to the std of the true value is', np.std(np.imag(rspx-spx))/np.std(np.imag(spx)),
      '.  Their correlation coefficient is', np.corrcoef(np.imag(rspx),np.imag(spx))[0,1]  ,  '.')

print("cross speckle field, real part:")
print('The ratio of the misfit error std to the std of the true value is', np.std(np.real(rspy-spy))/np.std(np.real(spy)),
      '.  Their correlation coefficient is', np.corrcoef(np.real(rspy),np.real(spy))[0,1]  ,  '.')
print("cross speckle field, imag part:")
print('The ratio of the misfit error std to the std of the true value is', np.std(np.imag(rspy-spy))/np.std(np.imag(spy)),
      '.  Their correlation coefficient is', np.corrcoef(np.imag(rspy),np.imag(spy))[0,1]  ,  '.')
# %%

#while this set of aberration coefficients does not provide a prefect match to the
#  the speckle fields, it provides speckles that are similar to the real ones and thus
#  should be adequate for testing the Jacobian.

#In the paper simulations, the speckle field the dark hole command does not
#  alter the speckle field.  This is OK for the purposes of showing the estimation
#  works because the choice of speckle field is fairly arbitrary.
Sx = Sx.dot(np.diag(np.exp(1j*Cdh)))  # apply the dark hole command to the system matrices
Sy = Sy.dot(np.diag(np.exp(1j*Cdh)))



Sx_ab = Sx.dot(np.diag(abxy + 1.))  # apply the aberration found above to the system matrices
Sy_ab = Sy.dot(np.diag(abxy + 1.))
# %%
for soln in solnames:
    sol = globals()[soln]  # Access the global variables dynamically.  Note: the function globals() returns a dictionary of the global name space
    modeldomprobe   = Sx@(    sol - 1.)
    truedomprobe    = Sx_ab@( sol - 1.)
    modelcrossprobe = Sy@(    sol - 1.)
    truecrossprobe  = Sy_ab@( sol - 1.)

    plt.figure();
    plt.plot(np.real(truedomprobe), 'bo', label = 'true dominant probe value');
    plt.plot(np.real(modeldomprobe), 'rx', label='model dominant probe value');
    plt.title(soln +': real part'); plt.legend()
    plt.figure();
    plt.plot(np.imag(truedomprobe), 'bo', label = 'true cross probe value');
    plt.plot(np.imag(modeldomprobe), 'rx', label='model cross probe value');
    plt.title(soln +': imaginary part'); plt.legend()

    print(soln,":")

    print("dominant probe, real part:")
    print('The ratio of the misfit error std to the std of the true value is', np.std(np.real(modeldomprobe-truedomprobe))/np.std(np.real(truedomprobe)),
         '.  Their correlation coefficient is', np.corrcoef(np.real(modeldomprobe),np.real(truedomprobe))[0,1]  ,  '.')
    print("dominant probe, imag part:")
    print('The ratio of the misfit error std to the std of the true value is', np.std(np.imag(modeldomprobe-truedomprobe))/np.std(np.imag(truedomprobe)),
         '.  Their correlation coefficient is', np.corrcoef(np.imag(modeldomprobe),np.imag(truedomprobe))[0,1]  ,  '.')

    print("cross probe, real part:")
    print('The ratio of the misfit error std to the std of the true value is', np.std(np.real(modelcrossprobe-truecrossprobe))/np.std(np.real(truecrossprobe)),
         '.  Their correlation coefficient is', np.corrcoef(np.real(modelcrossprobe),np.real(truecrossprobe))[0,1]  ,  '.')
    print("cross probe, imag part:")
    print('The ratio of the misfit error std to the std of the true value is', np.std(np.imag(modelcrossprobe-truecrossprobe))/np.std(np.imag(truecrossprobe)),
         '.  Their correlation coefficient is', np.corrcoef(np.imag(modelcrossprobe),np.imag(truecrossprobe))[0,1]  ,  '.')



print("Still Under Construction")
