#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 13:06:35 2024
@author: rfrazin

This performs EFC functions.
In this simple model, the polarized detector fields are given by a spline coefficient vector
  times the propagation matrix corresponding to the desired polarization.

"""

import numpy as np
from os import path as ospath  #needed for isfile(), join(), etc.
from sys import path as syspath
from scipy import optimize
import matplotlib.pyplot as plt
machine = "homeLinux"
#machine = "officeWindows"
if machine == "homeLinux":
    MySplineToolsLocation = "/home/rfrazin/Py/Optics"
    PropMatLoc = "/home/rfrazin/Py/EFCSimData/"
elif machine == 'officeWindows':
    MySplineToolsLocation = "E:/Python/Optics"
    PropMatLoc = "E:/MyOpticalSetups/EFC Papers/DataArrays"
syspath.insert(0, MySplineToolsLocation)
import Bspline3 as BS  # this module is in MySplineToolsLocation

Reduced = True
Sxfn = 'SysMat_LgOAPcg21x21_ContrUnits_Ex.npy'
Syfn = 'SysMat_LgOAPcg21x21_ContrUnits_Ey.npy'
SpecfieldXfn = 'SpeckleFieldFrom24x24screen_Ex.npy'
SpecfieldYfn = 'SpeckleFieldFrom24x24screen_Ey.npy'
fpsize = 512  # size of focal plane in pixels
fplength = 20. #length of detector in mm
if Reduced:  #stuff averaged over 2x2 pixels in the image plane
    fpsize //= 2 
    Sxfn = 'SysMatReduced_LgOAPcg21x21_ContrUnits_Ex.npy'
    Syfn = 'SysMatReduced_LgOAPcg21x21_ContrUnits_Ey.npy'
    SpecfieldXfn = 'SpeckleFieldReducedFrom24x24screen_Ex.npy'
    SpecfieldYfn = 'SpeckleFieldReducedFrom24x24screen_Ey.npy'


#===============================================================================
#                      EFC Class starts here
#==============================================================================
#This assumes that the input light is linearly polarized in the X-direction
# - holepixels list (or array) of 1D pixel indices of the pixels in the dark hole.
#    if None, there is no dark hole 
#    A list such as this can be created by the non-member MakePixList() function.

class EFC():
    def __init__(self, holepixels=None, SpeckleFactor=10.):
        self.lamb = 1.  # wavelength in microns
        self.ndm = 21  # number of actuators (1D)
        self.lamdpix = (fpsize/fplength)*5*800*(self.lamb*1.e-3)/(21*0.3367) # "lambda/D" in pixel units, i.e., (pixels per mm)*magnification*focal length*lambda/diameter
        self.SpeckleFactor = SpeckleFactor  # see self.PolIntensity.  This can be changed in its function call
        self.Sx = np.load(ospath.join(PropMatLoc, Sxfn))  # Sytem matrices
        self.Sy = np.load(ospath.join(PropMatLoc, Syfn))
        self.spx = np.load(ospath.join(PropMatLoc, SpecfieldXfn))  # X polarized speckles
        self.spy = np.load(ospath.join(PropMatLoc, SpecfieldYfn))  # y polarized speckles
        assert self.Sx.shape[1] == self.ndm**2
        assert self.Sy.shape == self.Sx.shape
        self.spx = self.spx.flatten()
        self.spy = self.spy.flatten()
        self.holepixels = holepixels
        if len(holepixels) == 4:
            print("Just so you know, the dark hole only has 4 pixels.")
        
        if holepixels is not None:  #trim the matrices and the speckle field to correspond to HoleBndy
            self.Shx  = np.zeros((len(holepixels), self.ndm**2)).astype('complex')  #trimmed system matrices
            self.Shy  = np.zeros((len(holepixels), self.ndm**2)).astype('complex')
            self.sphx = np.zeros((len(holepixels),)).astype('complex')
            self.sphy = np.zeros((len(holepixels),)).astype('complex')
            for k in range(len(holepixels)):
                self.Shx[k,:] = self.Sx[holepixels[k],:]
                self.Shy[k,:] = self.Sy[holepixels[k],:]
                self.sphx[k]  = self.spx[holepixels[k]]
                self.sphy[k]  = self.spy[holepixels[k]]
        else: pass
        return(None)
    
    #This returns the x- or y- polarized intensity as a function of the spline coefficient vector
    #The spline coefficient vector must have a shape (self.ndm**2,)
    #XorY - select the desired polarization 'X' or 'Y'
    #region - if 'Hole' only the intensity inside self.HoleBndy is computed.
    #            'Full' the intensity is calculated over the entire range
    #DMheight - if True the coefficient vector is interpreted as a DM height in microns and must be real-valued
    #          - if False it is simply a coefficient and can be complex valued
    #return_grad - return the gradient.  For now, only works when DMheight is True
    #SpeckleFactor - multiplier for additive speckle field.  Can be 0.  None corresponds to defaul (see __init__)
    def PolIntensity(self, coef, XorY='X', region='Hole', DMheight=True, return_grad=True,
                     SpeckleFactor=None):
        assert region == 'Hole' or region == 'Full'
        nc = self.ndm**2
        if SpeckleFactor is None: SpeckleFactor = self.SpeckleFactor
        assert coef.shape == (nc,)
        assert XorY == 'X' or XorY == 'Y'
        if XorY == 'X': 
            if region == 'Hole':
                Sys = self.Shx
                sp = self.sphx
            else:  # 'Full'
                Sys = self.Sx
                sp  = self.spx
        else:  # 'Y'
            if region == 'Hole':
                Sys = self.Shy
                sp  = self.sphy
            else:  #'Full'
                Sys = self.Sy
                sp  = self.spy

        if DMheight:
            assert np.iscomplexobj(coef) == False
            c = np.exp(1j*4.*np.pi*coef/self.lamb)
            if return_grad:
                dc = 1j*4.*np.pi*c/self.lamb
        else:
            c = 1.0*coef
            if return_grad:
                dc = np.ones(coef.shape).astype('complex')*1.0
        f = Sys.dot(c)
        f += SpeckleFactor*sp
        I = np.real(f*np.conj(f))
        if not return_grad:
            return I
        df = Sys*dc # speckles don't depend on c in this approximation.  This is the same as Sy.dot(diag(dc))
        dI = 2*np.real(np.conj(f)*df.T).T
        return (I, dI)
    
    #This is a cost function for a dark hole in the dominant polarization ('X')
    #c - DM command vector
    def CostHoleDominant(self, c, return_grad=True):
        assert np.iscomplexobj(c) is False
        assert c.shape == (self.Sx.shape[1],)
        if return_grad:
            I, dI = self.PolIntensity(c,XorY='X',region='Hole',DMheight=True,return_grad=True)
            cost = np.sum(I)
            dcost = np.sum(dI, axis=0)
            return (cost, dcost)
        else:
            I     = self.PolIntensity(c,XorY='X',region='Hole',DMheight=True,return_grad=False)
            cost = np.sum(I)
            return cost
        
    #This does the optimization over the dominant intensity to dig the dark hole
    #c0 - initial guess
    #method - options are 'NCG' (Newton Conj Grad), 'SLSQP'
    def DigHoleDominant(self, c0, method='NCG',maxiter=20):
        if method == 'NCG':  #Newton-CG set up.  leads to large command values without a penalty
           options={'disp':True, 'maxiter':maxiter}
           out = optimize.minimize(self.CostHoleDominant,c0,args=(),method='Newton-CG',options=options,jac=True)
        elif method == 'SLSQP':   #SLSQP set up
            options = {'disp': True, 'maxiter':10 }
            maxabs = np.pi/12
            conmat = np.eye(len(c0))
            ub =   maxabs*np.ones(c0.shape)
            lb = - maxabs*np.ones(c0.shape)
            constr = optimize.LinearConstraint(conmat, lb=lb, ub=ub)
            out = optimize.minimize(self.CostHoleDominant, c0, args=(),options=options,
                                    method='SLSQP',jac=True,constraints=(constr,))
        return out
    
    #This calculates something like signal-to-noise ratio for the cross polarization
    #metric - the various options correspond to:
    #  'A' - Iy/(const + Ix)
    def _CostCrossSNR(self,c, metric='A'):  # passed gradient test 2/5/24
        assert False # this approach doesn't seem to work
        const = 1.e-9
        Ix, gIx = self.PolIntensity(c,'X')
        Iy, gIy = self.PolIntensity(c,'Y',SpeckleFactor=0.)
        if metric == 'A':
            s = np.sum(Iy/(const + Ix))
            gs = gIy.T/(const + Ix) - gIx.T*Iy/( (const + Ix)**2 )
            gs = np.sum(gs,axis=1)
        else: assert False
        return (-1*s, -1*gs)  #make it loss function instead of a profit function
    
    #This uses the Newton Conjugate Grad method to find minima of a function
    #   from a number of random starting points.
    def _FindCrossMinima(self):
        assert False  # this approach doesn't seem to work
        funB = self._CostCrossSNR
        options={'disp':False, 'maxiter':15}
        rnd = lambda scale : scale*(2*np.pi*np.random.rand(self.ndm**2) - np.pi)
        nholes = 9
        command_dh = []
        cost_dh = []
        command_c = []
        cost_c = []
        for k in range(nholes):
          out = optimize.minimize(self.CostHoleDominant,rnd(1.),args=(),method='Newton-CG',options=options,jac=True)
          command_dh.append(out['x'])
          cost_dh.append(out['fun'])
          out = optimize.minimize(self.CostCrossSNR,out['x'],args=(),method='Newton-CG',options=options,jac=True)
          command_c.append(out['x'])
          cost_c.append(out['fun'])
          
        return {'dh_commands': command_dh, 'cross_commands': command_c, 'dh_cost': cost_dh, 'cross_cost': cost_c}

#===============================================================================
#                      EFC Class ends here
#==============================================================================

#this creates the b-spline basis coefficients corresponding to a linear phase,
#  which behaves as an off-axis source.
#For now, the off axis beam is displacesd on the x-axis
#   - Warning!  this code and BS.BivariateCubicSPline disagree on the
#   x and y directions
#nc - the number of cycles per pupil
#ndm - the DM is ndm x ndm
def SplCoefs4LinearPhase(nc, ndm):
   s = np.linspace(-np.pi, np.pi, 13*ndm)
   sx, sy = np.meshgrid(s,s)
   lp = np.exp(1j*nc*sy)  # linear phase function
   sx = sx.flatten()
   sy = sy.flatten()
   lp = lp.flatten()
   
   #now set up cubic b-spline
   cbs = BS.BivariateCubicSpline(sx, sy, ndm)
   spco = cbs.GetSplCoefs(lp)
   return spco


#This makes a list of the 1D pixel indices corresponding to a rectangular
# region (specified by its corners) within a flattened 2D array.
#corners - list (or array) corresponding to the 2D coords of the corners of the
# desired region in the following order [Xmin, Xmax, Ymin, Ymax].  The boundaries
# are inclusive.
#BigArrayShape a tuple (or whatever) corresponding the shape of the larger array 
def MakePixList(corners, BigArrayShape):
    assert len(BigArrayShape) == 2
    assert corners[2] <= BigArrayShape[0] and corners[3] <= BigArrayShape[0]
    assert corners[0] <= BigArrayShape[1] and corners[2] <= BigArrayShape[1]
    pixlist = []
    rows = np.arange(corners[2], corners[3]+1)  # 'y'
    cols = np.arange(corners[0], corners[1]+1)  # 'x'
    ff = lambda r,c : np.ravel_multi_index((r,c), (BigArrayShape[0],BigArrayShape[1]))
    for r in rows:
        for c in cols:
            pixlist.append(ff(r,c))
    return pixlist
    