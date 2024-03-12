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
from os import path as ospath
from sys import path
from scipy import optimize
import matplotlib.pyplot as plt
#machine = "homeLinux"
machine = "officeWindows"
if machine == "homeLinux":
    MySplineToolsLocation = "/home/rfrazin/Py/Optics"
    PropMatLoc = "/home/rfrazin/Py/EFCSimData/"
elif machine == 'officeWindows':
    MySplineToolsLocation = "E:/Python/Optics"
    PropMatLoc = "E:/MyOpticalSetups/EFC\ Papers/DataArrays"
path.insert(0, MySplineToolsLocation)
import Bspline3 as BS  # this module is in MySplineToolsLocation

HoleBndy = np.array([227, 285, 314, 372])
SysXfn = 'SysMat_LgOAPcg21x21_ContrUnits_Ex.npy'
SysYfn = 'SysMat_LgOAPcg21x21_ContrUnits_Ey.npy'
SpecfieldXfn = 'SpeckleFieldFrom24x24screen_Ex.npy'
SpecfieldYfn = 'SpeckleFieldFrom24x24screen_Ey.npy'
fpsize = 512  # size of focal plane in pixels
fplength = 20. #length of detector in mm
Reduced = True
if Reduced:
    fpsize //= 2
    HoleBndy //=2 
    SysXfn = 'SysMatReduced_LgOAPcg21x21_ContrUnits_Ex.npy'
    SysYfn = 'SysMatReduced_LgOAPcg21x21_ContrUnits_Ey.npy'
    SpecfieldXfn = 'SpeckleFieldReducedFrom24x24screen_Ex.npy'
    SpecfieldYfn = 'SpeckleFieldReducedFrom24x24screen_Ey.npy'



#This assumes that the input light is linearly polarized in the X-direction
#HoleBndy - Speficies the corners of the dark hole (inclusive).  It is a list or tuple with
#  4 pixel values: [minY, maxY, minX, maxX] (note the row,col order) 
class EFC():
    def __init__(self, HoleBndy=HoleBndy, SpeckleFactor=10.):
        if HoleBndy is not None:
            assert len(HoleBndy) == 4
            self.HoleShape = (HoleBndy[1]-HoleBndy[0]+1, HoleBndy[3]-HoleBndy[2]+1)
            assert self.HoleShape[0] > 0 and self.HoleShape[1] > 0 
        else: self.HoleShape = (0,0)
        self.lamb = 1.  # wavelength in microns
        self.ndm = 21  # number of actuators (1D)
        self.lamdpix = (fpsize/fplength)*5*800*(self.lamb*1.e-3)/(21*0.3367) # "lambda/D" in pixel units, i.e., (pixels per mm)*magnification*focal length*lambda/diameter
        self.SpeckleFactor = SpeckleFactor  # see self.PolIntensity.  This can be changed in its function call
        SysX = np.load(PropMatLoc + SysXfn )  # system matrices
        SysY = np.load(PropMatLoc + SysYfn )
        SpecX = np.load(PropMatLoc + SpecfieldXfn )  # X polarized speckles
        SpecY = np.load(PropMatLoc + SpecfieldYfn)  # y polarized speckles
        assert SysX.shape[1] == self.ndm**2
        assert SysY.shape == SysX.shape
        self.HoleBndy = HoleBndy
        
        if HoleBndy is not None:  #trim the matrices and the speckle field to correspond to HoleBndy
            sz = int(np.sqrt(SysX.shape[0]))
            self.lindex = []  # 1D pixel index of dark hole pixel
            self.duodex = []  # 2D pixel index of dark hole pixels
            self.SpecX = SpecX[HoleBndy[2]:HoleBndy[3] + 1, HoleBndy[0]:HoleBndy[1] +1]
            self.SpecY = SpecY[HoleBndy[2]:HoleBndy[3] + 1, HoleBndy[0]:HoleBndy[1] +1]
            for row in np.arange(HoleBndy[0], HoleBndy[1] + 1):
                for col in np.arange(HoleBndy[2], HoleBndy[3] +1):
                   self.duodex.append( (row,col) )
                   self.lindex.append( np.ravel_multi_index((row,col), (sz,sz)) )
            self.SysX = np.zeros((len(self.lindex), self.ndm**2)).astype('complex')
            self.SysY = np.zeros((len(self.lindex), self.ndm**2)).astype('complex')
            for k in range(len(self.lindex)):
                self.SysX[k,:] = SysX[self.lindex[k],:]
                self.SysY[k,:] = SysY[self.lindex[k],:]
        else:
            self.SpecX = SpecX
            self.SpecY = SpecY
            self.SysX = SysX
            self.SysY = SysY
         
        return(None)
    
    #This returns the x- or y- polarized intensity as a function of the spline coefficient vector
    #The spline coefficient vector must have a shape (self.ndm**2,)
    #XorY - select the desired polarization 'X' or 'Y'
    #DMheight - if True the coefficient vector is interpreted as a DM height in microns and must be real-valued
    #          - if False it is simply a coefficient and can be complex valued
    #return_grad - return the gradient.  For now, only works when DMheight is True
    #SpeckleFactor - multiplier for additive speckle field.  Can be 0.  None corresponds to defaul (see __init__)
    def PolIntensity(self, coef, XorY='X', DMheight=True, return_grad=True,
                     SpeckleFactor=None):
        nc = self.ndm**2
        if SpeckleFactor is None: SpeckleFactor = self.SpeckleFactor
        assert coef.shape == (nc,)
        assert XorY == 'X' or XorY == 'Y'
        if XorY == 'X': 
            Sys = self.SysX
            sp  = self.SpecY
        else:
            Sys = self.SysY
            sp  = self.SpecY
        if DMheight:
            assert np.iscomplexobj(coef) == False
            c = np.exp(2j*np.pi*coef/self.lamb)
            if return_grad:
                dc = 2j*np.pi*c/self.lamb
        else:
            c = 1.0*coef
            if return_grad:
                print('No gradient information provided when DMheight is False.')
                assert False

        f = Sys.dot(c)
        f += SpeckleFactor*sp.reshape(f.shape)
        I = np.real(f*np.conj(f))
        if not return_grad:
            return I
        df = Sys*dc # speckles don't depend on c in this approximation.  This is the same as Sys.dot(diag(dc))
        dI = 2*np.real(np.conj(f)*df.T).T
        return (I, dI)
    
    #This is a cost function for a dark hole in the dominant polarization ('X')
    #c - DM command vector
    def CostHoleDominant(self, c, return_grad=True):
        assert np.iscomplexobj(c) is False
        assert c.shape == (self.SysX.shape[1],)
        if return_grad:
            I, dI = self.PolIntensity(c,XorY='X',DMheight=True,return_grad=True)
            cost = np.sum(I)
            dcost = np.sum(dI, axis=0)
            return (cost, dcost)
        else:
            I     = self.PolIntensity(c,XorY='X',DMheight=True,return_grad=False)
            cost = np.sum(I)
            return cost
    
    #This calculates something like signal-to-noise ratio for the cross polarization
    #metric - the various options correspond to:
    #  'A' - Iy/(const + Ix)
    def CostCrossSNR(self,c, metric='A'):  # passed gradient test 2/5/24
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
    #c - a dark hole starting point
    def FindMaxima(self,c,npts=12,maxiter=30):
        fun = self.CostCrossSNR
        options={'disp':True, 'maxiter':30}
        out = optimize.minimize(fun,c,args=(),method='Newton-CG',options=options,jac=True)
        return out
    
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
    