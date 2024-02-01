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
from sys import path
from scipy import optimize
import matplotlib.pyplot as plt
machine = "homeLinux"
if machine == "homeLinux":
    MySplineToolsLocation = "/home/rfrazin/Py/Optics"
    PropMatLoc = "/home/rfrazin/Py/EFCSimData/"
else: assert False
path.insert(0, MySplineToolsLocation)
import Bspline3 as BS  # this module is in MySplineToolsLocation

#This assumes that the input light is linearly polarized in the X-direction
#HoleBndy - Speficies the corners of the dark hole (inclusive).  It is a list or tuple with
#  4 pixel values: [minX, maxX, minY, maxY] 

class EFC():
    def __init__(self, HoleBndy=[270, 370, 206, 306]):
        if HoleBndy is not None:
            assert len(HoleBndy) == 4
            self.HoleShape = (HoleBndy[3]-HoleBndy[2]+1, HoleBndy[1]-HoleBndy[0]+1)
            assert self.HoleShape[0] > 0 and self.HoleShape[1] > 0 
        else: self.HoleShape = (0,0)
        self.lamb = 1.  # wavelength in microns
        self.ndm = 21  # number of actuators (1D)
        SysX = np.load(PropMatLoc + 'SysMat_LgOAPcg21x21_ContrUnits_Ex.npy' )  # system matrices
        SysY = np.load(PropMatLoc + 'SysMat_LgOAPcg21x21_ContrUnits_Ey.npy' )
        SpecX = np.load(PropMatLoc + 'SpeckleFieldFrom24x24screen_Ex.npy')  # X polarized speckles
        SpecY = np.load(PropMatLoc + 'SpeckleFieldFrom24x24screen_Ey.npy')  # y polarized speckles
        assert SysX.shape[1] == self.ndm**2
        assert SysY.shape == SysX.shape
        self.HoleBndy = HoleBndy
        
        if HoleBndy is not None:  #trim the matrices and the speckle field to correspond to HoleBndy
            sz = int(np.sqrt(SysX.shape[0]))
            self.lindex = []  # 1D pixel index of dark hole pixel
            self.duodex = []  # 2D pixel index of dark hole pixels
            self.SpecX = SpecX[HoleBndy[0]:HoleBndy[1] + 1, HoleBndy[2]:HoleBndy[3] +1]
            self.SpecY = SpecY[HoleBndy[0]:HoleBndy[1] + 1, HoleBndy[2]:HoleBndy[3] +1]
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
    #InclSpeckles - Include additive speckle field
    #SpeckleFactor - multiplier for speckle field
    def PolIntensity(self, coef, XorY='X', DMheight=True, return_grad=True,
                     InclSpeckles=True, SpeckleFactor=1.0):
        nc = self.ndm**2
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
        if InclSpeckles: f += SpeckleFactor*sp.reshape(f.shape)
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
            I, dI = self.PolIntensity(c,XorY='X',DMheight=True,return_grad=True, InclSpeckles=True,SpeckleFactor=1.)
            cost = np.sum(I)
            dcost = np.sum(dI, axis=0)
            return (cost, dcost)
        else:
            I     = self.PolIntensity(c,XorY='X',DMheight=True,return_grad=False,InclSpeckles=True,Specklefactor=1.)
            cost = np.sum(I)
            return cost
    
    #This does the optimization over the dominant intensity to dig the dark hole
    #c0 - initial guess
    def DigHoleDominant(self, c0):
        #Newton-CG set up.  leads to large command values
        optionsNCG={'disp':True, 'maxiter':20}
        out = optimize.minimize(self.CostHoleDominant,c0,args=(),method='Newton-CG',options=optionsNCG,jac=True)
        #SLSQP set up
        maxabs = np.pi/12
        conmat = np.eye(len(c0))
        ub =   maxabs*np.ones(c0.shape)
        lb = - maxabs*np.ones(c0.shape)
        constraints = optimize.LinearConstraint(conmat, lb=lb, ub=ub)
        
        
        
        return out
    