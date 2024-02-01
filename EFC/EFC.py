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
import matplotlib.pyplot as plt
machine = "homeLinux"
if machine == "homeLinux":
    MySplineToolsLocation = "/home/rfrazin/Py/Optics"
    PropMatLoc = "/home/rfrazin/Py/EFCSimData/"
else: assert False
path.insert(0, MySplineToolsLocation)
import Bspline3 as BS  # this module is in MySplineToolsLocation


#HoleBdy - Speficies the corners of the dark hole (inclusive).  It is a list or tuple with
#  4 pixel values: [minX, maxX, minY, maxY] 

class EFC():
    def __init__(self, HoleBdy=None):
        if HoleBdy is not None:
            assert len(HoleBdy) == 4
        self.lamb = 1.  # wavelength in microns
        self.ndm = 21  # number of actuators (1D)
        SysX = np.load(PropMatLoc + 'SysMat_LgOAPcg21x21_ContrUnits_Ex.npy' )  # system matrices
        SysY = np.load(PropMatLoc + 'SysMat_LgOAPcg21x21_ContrUnits_Ey.npy' )
        SpecX = np.load(PropMatLoc + 'SpeckleFieldFrom24x24screen_Ex.npy')  # X polarized speckles
        SpecY = np.load(PropMatLoc + 'SpeckleFieldFrom24x24screen_Ey.npy')  # y polarized speckles
        assert SysX.shape[1] == self.ndm**2
        assert SysY.shape == SysX.shape
        
        
        if HoleBdy is not None:  #trim the matrices and the speckle field to correspond to HoleBdy
            sz = int(np.sqrt(SysX.shape[0]))
            self.lindex = []  # 1D pixel index of dark hole pixel
            self.duodex = []  # 2D pixel index of dark hole pixels
            self.SpecX = SpecX[HoleBdy[0]:HoleBdy[1] + 1, HoleBdy[2]:HoleBdy[3] +1]
            self.SpecY = SpecY[HoleBdy[0]:HoleBdy[1] + 1, HoleBdy[2]:HoleBdy[3] +1]
            for row in np.arange(HoleBdy[0], HoleBdy[1] + 1):
                for col in np.arange(HoleBdy[2], HoleBdy[3] +1):
                   self.duodex.append( (row,col) )
                   self.lindex.append( np.ravel_multi_index((row,col), (sz,sz)) )
            self.SysX = np.zeros((len(self.lindex), self.ndm**2)).astype('complex')
            self.SysX = np.zeros((len(self.lindex), self.ndm**2)).astype('complex')
            for k in range(len(self.lindex)):
                self.SysX[k,:] = SysX[self.lindex[k],:]
                self.SysY[k,:] = SysY[self.lindex[k],:]
        
        
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
                dc = 2j*np.i*c/self.lamb
        else:
            if return_grad:
                print('No gradient information provided when DMheight is False.')
                assert False
            c = 1.0*coef
            
        f = Sys.dot(c)
        if InclSpeckles: f += sp*SpeckleFactor
        I = np.real(f*np.conj(f))
        if not return_grad:
            return I
        df = Sys*dc # speckles don't depend on c in this approximation.  This is the same as Sys.dot(diag(dc))
        dI = 2*np.real(df*np.conj(f))
        return (I, dI)
    
    
    
        

        
        return(None)