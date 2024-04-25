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
if not Reduced: assert False  # the fplength is problematic
Sxfn = 'SysMat_LgOAPcg21x21_ContrUnits_Ex.npy'
Syfn = 'SysMat_LgOAPcg21x21_ContrUnits_Ey.npy'
SpecfieldXfn = 'SpeckleFieldFrom24x24screen_Ex.npy'
SpecfieldYfn = 'SpeckleFieldFrom24x24screen_Ey.npy'
fpsize = 512  # size of focal plane in pixels
fplength = 20. #length of detector in mm
if Reduced:  #stuff averaged over 2x2 pixels in the image plane
    fpsize //= 2 
    Sxfn = 'ThreeOAP20mmSquareApCgkn33x33_SystemMatrixReducedContrUnits_Ex.npy'  # 'SysMatReduced_LgOAPcg21x21_ContrUnits_Ex.npy'
    Syfn = 'ThreeOAP20mmSquareApCgkn33x33_SystemMatrixReducedContrUnits_Ey.npy'  # 'SysMatReduced_LgOAPcg21x21_ContrUnits_Ey.npy'
    SpecfieldXfn = 'SpeckleFieldReducedFrom33x33PhaseScreen_Ex.npy'  # 'SpeckleFieldReducedFrom24x24screen_Ex.npy'
    SpecfieldYfn = 'SpeckleFieldReducedFrom33x33PhaseScreen_Ey.npy'  # 'SpeckleFieldReducedFrom24x24screen_Ey.npy'


#===============================================================================
#                      EFC Class starts here
#==============================================================================
#This assumes that the input light is linearly polarized in the X-direction
# - HolePixels list (or array) of 1D pixel indices of the pixels in the dark hole.
#    if None, there is no dark hole 
#    A list such as this can be created by the non-member MakePixList() function.

class EFC():
    def __init__(self, HolePixels=None, SpeckleFactor=0.):
        if HolePixels is not None:
            print("The dark hole has", str(len(HolePixels)) ,"pixels.")
        self.HolePixels = HolePixels
        self.lamb = 1.  # wavelength in microns
        self.ndm = 33  # number of actuators (1D)
        #self.lamdpix = (fpsize/fplength)*5*800*(self.lamb*1.e-3)/(21*0.3367) # "lambda/D" in pixel units, i.e., (pixels per mm)*magnification*focal length*lambda/diameter
        self.lamdpix = 10.
        if Reduced: self.lamdpix = 5.
        self.SpeckleFactor = SpeckleFactor  # see self.PolIntensity.  This can be changed in its function call
        self.Sx = np.load(ospath.join(PropMatLoc, Sxfn))  # Sytem matrices
        self.Sy = np.load(ospath.join(PropMatLoc, Syfn))
        self.spx = np.load(ospath.join(PropMatLoc, SpecfieldXfn))  # X polarized speckles
        self.spy = np.load(ospath.join(PropMatLoc, SpecfieldYfn))  # y polarized speckles
        assert self.Sx.shape[1] == self.ndm**2
        assert self.Sy.shape == self.Sx.shape
        self.spx = self.spx.flatten()
        self.spy = self.spy.flatten()
        self.CostScale = None
        
        if HolePixels is not None:  #trim the matrices and the speckle field to correspond to HoleBndy
            self.Shx  = np.zeros((len(HolePixels), self.ndm**2)).astype('complex')  #trimmed system matrices
            self.Shy  = np.zeros((len(HolePixels), self.ndm**2)).astype('complex')
            self.sphx = np.zeros((len(HolePixels),)).astype('complex')
            self.sphy = np.zeros((len(HolePixels),)).astype('complex')
            for k in range(len(HolePixels)):
                self.Shx[k,:] = self.Sx[HolePixels[k],:]
                self.Shy[k,:] = self.Sy[HolePixels[k],:]
                self.sphx[k]  = self.spx[HolePixels[k]]
                self.sphy[k]  = self.spy[HolePixels[k]]
        else: pass
        return(None)
    
    #This makes the 'M matrix', which is linearized constraints to keep the dominant field dark
    #  while probing the cross field.
    #c0 - the linearization point 
    #XorY - 'X' or 'Y'
    #pixlist - the list of 1D pixel indices within the dark hole that are
    #  to be kept dark while the cross field is probed.
    #With U,S,V = np.linalg.svd(M)  the last rows of V, e.g., V[k,:], are basis vectors corresponding to the small SVs of M
    def MakeMmat(self, c0, XorY='X', pixlist=None):
        if pixlist is None: pixlist = self.HolePixels
        assert XorY in ['X', 'Y']
        lpl = len(pixlist); 
        spl = pixlist
        if XorY == 'X': 
            S = self.Sx
        else: 
            S = self.Sy
        cc0 = np.cos(c0); sc0 = np.sin(c0)
        M = np.zeros((2*lpl, self.Sx.shape[1]))
        for k in range(lpl):
            M[k      ] = np.real(S[spl[k],:])*cc0 - np.imag(S[spl[k],:])*sc0
            M[k + lpl] = np.real(S[spl[k],:])*sc0 + np.imag(S[spl[k],:])*cc0
        return M

    #This returns a matrix, V, whose columns for a basis of the M matrix (see self.MakeMmatrix()  )
    #c0 - reference DM command, probably corresponding to a dark hole
    #pixlist - list of pixels that define the M matrix, if None then self.HolePixels is used
    #sv_thresh the singular value ration used to define the effective null space of M
    #   the 10^-3 value is chosen by comparing the SVs of self.Shx to the norms of
    #   vectors obtained by self.Shy@VM[k,:] and examining the slopes of the two curves.
    #   If you do this you will see that the cross norms decay more slowly at first
    def GetNull(self, c0, pixlist=None, sv_thresh=1.e-3):
        if pixlist is None: pixlist = self.HolePixels
        M = self.MakeMmat(c0, pixlist)
        UM, SM, VM = np.linalg.svd(M)  # the ROWS of VM, e.g., VM[0,:] are the singular vectors 
        svals = np.zeros(len(c0))
        svals[:len(SM)] = SM
        isv = np.where(svals < SM[0]*sv_thresh)[0]
        VM = VM.T # now the columns of VM are the singular vectors
        V = np.zeros((VM.shape[0],len(isv)))
        for k in range(len(isv)):
            V[:,k] = VM[:,isv[k]]
        return V
    
   
    #This calculates Ey on self.HolePixels when the DM command is defined in
    #  terms of basis vectors
    #This does not include the speckle field.  In terms of the simulation, it is the model field
    #The idea is that the basis vectors in a useful effective null space for
    #  dominant field
    #a  - vector of null space coefficients. must have len(a) == Vn.shape[1]
    #c0 - initial (dark hole) DM command
    #Vn - the columns of Vn form the new basis.  See self.GetNull
    #return_grad - if True it returns the derivatives with respect to the coefficient vector a
    def CrossFieldNewBasis(self, a, c0, Vn, return_grad=True):
        assert len(a) == Vn.shape[1]
        Vna = Vn@a  # phasor = np.exp(1j*Vn@a)
        S = self.Shy*np.exp(1j*c0)  # equiv to self.Shy@np.diag(np.exp(1j*c0))
        #ey = S@np.exp(1j*Vna); ey_re = np.real(ey); ey_im = np.imag(ey)  # the code below is slightly faster
        ey_re = np.real(S)@np.cos(Vna) - np.imag(S)@np.sin(Vna)  # real part of field
        ey_im = np.real(S)@np.sin(Vna) + np.imag(S)@np.cos(Vna)  # imag part of field
        if not return_grad:
            return (ey_re, ey_im)
        else:
            Vnc = (Vn.T*np.cos(Vna)).T
            Vns = (Vn.T*np.sin(Vna)).T
            dr = - np.real(S)@Vns - np.imag(S)@Vnc
            di =   np.real(S)@Vnc - np.imag(S)@Vns
            return (ey_re, ey_im, dr, di)
        
    #Ithresh - the intensity level at which the penalty for X intensity turns on     
    def CostCrossDiffIntensityRatio(self, a, target, c0, Vn, return_grad=False, Ithresh = 1.e-9, scale=1.):
        def QuadThreshPenalty(s, thr, s_grad=None, return_grad=False, offset=1.e-15):  #  assumes s >=0.  offset is small intensity value
            wh = np.where(s > thr)[0]
            swt = s[wh] - thr
            pen = 0.5*np.sum( swt**2 ) + offset
            if not return_grad:
                return  pen
            else: 
                assert (s_grad is not None)
                sgw = s_grad[wh,:]
                gpen = np.sum( sgw.T*swt, axis=1 ) 
                return pen, gpen
        assert target in ['Re','Im']
        re0, im0 = self.CrossFieldNewBasis(np.zeros(a.shape),c0,Vn,return_grad=False)
        if not return_grad:
            Ix = self.PolIntensity(c0 + Vn@a,'X','Hole','phase',False,None)
            re, im = self.CrossFieldNewBasis(a,c0,Vn,return_grad=False)
            #cIx = self.CostHoleDominant(c0 + Vn@a, return_grad=False, scale=1.0)
            den = QuadThreshPenalty(Ix,Ithresh,None,False)         
        else:
            Ix, gIx = self.PolIntensity(c0 + Vn@a,'X','Hole','phase',True,None)
            re, im, dre, dim = self.CrossFieldNewBasis(a,c0,Vn,return_grad=True) 
            den, dden = QuadThreshPenalty(Ix,Ithresh,gIx,True) 
            dden = Vn.T@dden
        if target == 'Re':
            num = np.sum((re-re0)**2)
        else:
            num = np.sum((im-im0)**2)
        cost = - num/den  # we want to minimize this ratio
        if not return_grad: return cost*scale
        if target == 'Re':   
            dnum = 2*dre.T@(re - re0)
        elif target == 'Im':
            dnum = 2*dim.T@(im - im0)
        dcost = - dnum/den + (num/den**2)*dden
        return cost*scale, dcost*scale




    def CostCrossIntensityRatio(self, a, target, c0, Vn, return_grad=False):
        scale = 1.   # playing with this can help optimization sometimes
        assert target in ['Re','Im']
        if not return_grad:
            re, im = self.CrossFieldNewBasis(a,c0,Vn,return_grad=False)
            cIx = self.CostHoleDominant(c0 + Vn@a, return_grad=False, scale=1.0)
        else:
            re, im, dre, dim = self.CrossFieldNewBasis(a,c0,Vn,return_grad=True) 
            cIx, dcIx = self.CostHoleDominant(c0 + Vn@a, return_grad=True, scale=1.0)
            dcIx = Vn.T@dcIx
        if target == 'Re':
            num = np.sum(re*re)
        else:
            num = np.sum(im*im)
        cost = - num/cIx  # we want to maximize this ratio
        if not return_grad: return cost*scale
        if target == 'Re':   
            dnum = 2*dre.T@re
        elif target == 'Im':
            dnum = 2*dim.T@im
        dcost = - dnum/cIx + (num/cIx**2)*dcIx
        return cost*scale, dcost*scale


    #This is a cost function for minimizing the target (see below) corresponding
    #  to real or imag component of the cross field
    #a - coefficient vector of the null space basis, Vn
    #target - quantity to be minimized.  options are 'Re', '-Re', 'Im', '-Im'
    #c0 - dark hole command for the dominant field
    #Vn - matrix with null space vectors, see self.GetNull() 
    #return_grad (with respect to a)
    def CostCrossField(self, a, target, c0, Vn, return_grad=False):
        scale = 1.e7  # this helps some of the optimizers
        assert target in ['Re','-Re','Im','-Im']
        if not return_grad:
            re, im = self.CrossFieldNewBasis(a,c0,Vn,return_grad=False)
            if target == 'Re':
                cost = re.sum()
            elif target == '-Re':
                cost = -1.*re.sum()
            elif target == 'Im':
                cost = im.sum()
            else:  # '-Im'
                cost = -1.*im.sum()
            return cost*scale
        else:
            re, im, dre, dim = self.CrossFieldNewBasis(a,c0,Vn,return_grad=True)
            if target == 'Re':
                cost = re.sum()
                dcost = dre.sum(axis=0)
            elif target == '-Re':
                cost = -1.*re.sum()
                dcost = -1*dre.sum(axis=0)
            elif target == 'Im':
                cost = im.sum()
                dcost = dim.sum(axis=0)
            else:  # '-Im'
                cost = -1.*im.sum()
                dcost = -1.*dim.sum(axis=0)
            return cost*scale, dcost*scale
    
    #This sets up optimizations for minimizing the fuction self.CostCrossField
    #c0 - see self.CrossField
    #a0 - initial guess to start the optimizer
    #target - see self.CostCrossField
    #method - choices are 'CG', 'NCG'
    def OptCrossField(self, c0, a0=None, target='-Re', method='CG', maxiter=20):
        cfcn = self.CostCrossIntensityRatio
        Vn = self.GetNull(c0, pixlist=None, sv_thresh = 1.e-3)
        if a0 is None: a0 = np.zeros(Vn.shape[1])
        args = (target, c0, Vn, True)
        options = {'disp': True, 'maxiter': maxiter}
        init_cost = cfcn(a0,args[0],args[1],args[2],False)
        print('Starting Cost', init_cost)
        out = optimize.minimize(cfcn, a0, args=args, method=method, jac=True, options=options)
        return out
    
    
    #This returns the x- or y- polarized intensity as a function of the spline coefficient vector
    #The spline coefficient vector must have a shape (self.ndm**2,)
    #XorY - select the desired polarization 'X' or 'Y'
    #region - if 'Hole' only the intensity inside self.HoleBndy is computed.
    #            'Full' the intensity is calculated over the entire range
    #DM_mode - 'height': coef must be real-valued and the phasor phase is 4pi*coef/self.lambda 
    #        - 'phase' : coef must be real-valued and the phasor phase is coef itself
    #return_grad - return the gradient.  
    #SpeckleFactor - multiplier for additive speckle field.  Can be 0.  None corresponds to defaul (see __init__)
    def PolIntensity(self, coef, XorY='X', region='Hole', DM_mode='phase', return_grad=True, 
                     SpeckleFactor=None):
        assert region == 'Hole' or region == 'Full'
        assert DM_mode in ['height', 'phase']
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

        if DM_mode == 'height':
            assert np.iscomplexobj(coef) == False
            c = np.exp(1j*4.*np.pi*coef/self.lamb)
            if return_grad:
                dc = 1j*4.*np.pi*c/self.lamb
        elif DM_mode == 'phase':
            assert np.iscomplexobj(coef) == False
            c = np.exp(1j*coef)
            if return_grad:
                dc = 1j*c
        else:  # not an option   
            assert False
            
        f = Sys.dot(c)
        f += SpeckleFactor*sp
        I = np.real(f*np.conj(f))
        if not return_grad:
            return I
        df = Sys*dc #  This is the same as Sys.dot(diag(dc)). speckles don't depend on c in this approximation.
        dI = 2*np.real(np.conj(f)*df.T).T
        return (I, dI)
    
    #This is a cost function for a dark hole in the dominant polarization ('X')
    #c - DM command vector
    #scale - setting this to 10^6 seems to help the Newton-CG minimizer
    def CostHoleDominant(self, c, return_grad=True, scale=1.e6):
        self.CostScale = scale
        assert np.iscomplexobj(c) is False
        assert c.shape == (self.Sx.shape[1],)
        if return_grad:
            I, dI = self.PolIntensity(c,XorY='X',region='Hole',DM_mode='phase',return_grad=True)
            cost = np.sum(I)
            dcost = np.sum(dI, axis=0)
            return (scale*cost, scale*dcost)
        else:
            I     = self.PolIntensity(c,XorY='X',region='Hole',DM_mode='phase',return_grad=False)
            cost = np.sum(I)
            return scale*cost
        
    #This calculates the Hessian corresponding to CostHillCross.   See Frazin (2018) JOSAA v35, p594   
    #Note that the intensity of each detector pixel has its own Hessian, which would be a large object.
    #  This Hessian assumes that the cost is the sum of the pixel intensities in the dark hole
    def HessianCostHillCross(self, coef, SpeckleFactor=None):
        if self.CostScale is None:
            print("You must run CostHillCross to set self.CostScale first.")
            assert False
        if SpeckleFactor is None: SpeckleFactor = self.SpeckleFactor
        Sys = self.Shy  #  hole system matrix - see self.PolIntensity
        f = Sys.dot(np.exp(1j*coef))
        f += SpeckleFactor*self.sphy  # hole speckle field
        df = 1j*Sys*np.exp(1j*coef)  # same as Sys@np.diag(np.exp(1j*coef))
        H = np.zeros((len(coef),len(coef)))
        for m in range(Sys.shape[0]):  #loop over pixels
            H += -2*np.real( np.outer(df[m,:], df[m,:].conj()) )
            H += -2*np.real(np.diag( 1j*df[m,:]*f[m].conj() ))  # additional term for diagonal elements
        return H*self.CostScale

    #This calculates the Hessian corresponding to CostHoleDominant.   See Frazin (2018) JOSAA v35, p594   
    #Note that the intensity of each detector pixel has its own Hessian, which would be a large object.
    #  This Hessian assumes that the cost is the sum of the pixel intensities in the dark hole
    def HessianCostHoleDominant(self, coef, SpeckleFactor=None):
        if self.CostScale is None:
            print("You must run CostHoleDominant to set self.CostScale first.")
            assert False
        if SpeckleFactor is None: SpeckleFactor = self.SpeckleFactor
        Sys = self.Shx  #  hole system matrix - see self.PolIntensity
        f = Sys.dot(np.exp(1j*coef))
        f += SpeckleFactor*self.sphx  # hole speckle field
        df = 1j*Sys*np.exp(1j*coef)  # same as Sys@np.diag(np.exp(1j*coef))
        H = np.zeros((len(coef),len(coef)))
        for m in range(Sys.shape[0]):  #loop over pixels
            H += 2*np.real( np.outer(df[m,:], df[m,:].conj()) )
            H += 2*np.real(np.diag( 1j*df[m,:]*f[m].conj() ))  # additional term for diagonal elements
        return H*self.CostScale
        
    #This does the optimization over the dominant intensity to dig the dark hole
    #c0 - initial guess
    #method - options are 'NCG' (Newton-ConjGrad), 'CG' (ConjGrad), 'SLSQP'
    #NCG with the analytical Hessian is great, but applying CG afterwards is effective, too.
    def DigHoleDominant(self, c0, method='NCG',maxiter=20):
        if method == 'NCG':  #Newton-CG set up.  leads to large command values without a penalty
           options={'disp':True, 'maxiter':maxiter}
           out = optimize.minimize(self.CostHoleDominant,c0,args=(),method='Newton-CG',options=options,
                                   jac=True,hess=self.HessianCostHoleDominant)
        elif method == 'CG':
           options={'disp':True, 'maxiter':maxiter}
           out = optimize.minimize(self.CostHoleDominant,c0,args=(),method='CG',options=options,jac=True)

        elif method == 'SLSQP':   #SLSQP set up
            options = {'disp': True, 'maxiter':10 }
            maxabs = np.pi/12
            conmat = np.eye(len(c0))
            ub =   maxabs*np.ones(c0.shape)
            lb = - maxabs*np.ones(c0.shape)
            constr = optimize.LinearConstraint(conmat, lb=lb, ub=ub)
            out = optimize.minimize(self.CostHoleDominant, c0, args=(),options=options,
                                    method='SLSQP',jac=True,constraints=(constr,))
            ffvalue = self.CostHoleDominant(out['x'], return_grad=False)
            print("Final Dark Hole Cost = ", ffvalue)
        return out
    
    
    #==========================#
    #  scrapyard for EFC class #
    #==========================#
    
    #This cost fcn does modulate the cross field much because most of the gradient comes 
    #  from the dominant field
    def _CostCrossDiffIntensityRatio(self, a, target, c0, Vn, return_grad=False, scale=1.e8):
        assert False
        assert target in ['Re','Im']
        re0, im0 = self.CrossFieldNewBasis(np.zeros(a.shape),c0,Vn,return_grad=False)      
        if not return_grad:
            re, im = self.CrossFieldNewBasis(a,c0,Vn,return_grad=False)
            cIx = self.CostHoleDominant(c0 + Vn@a, return_grad=False, scale=1.0)
        else:
            re, im, dre, dim = self.CrossFieldNewBasis(a,c0,Vn,return_grad=True) 
            cIx, dcIx = self.CostHoleDominant(c0 + Vn@a, return_grad=True, scale=1.0)
            dcIx = Vn.T@dcIx
        if target == 'Re':
            num = np.sum((re-re0)**2)
        else:
            num = np.sum((im-im0)**2)
        cost = - num/cIx  # we want to minimize this ratio
        if not return_grad: return cost*scale
        if target == 'Re':   
            dnum = 2*dre.T@(re - re0)
        elif target == 'Im':
            dnum = 2*dim.T@(im - im0)
        dcost = - dnum/cIx + (num/cIx**2)*dcIx
        return cost*scale, dcost*scale

    #This is a cost function for a bright hill in the cross polarization ('Y')
    #c - DM command vector
    #scale - setting this to 10^? seems to help the Newton-CG minimizer
    def _CostHillCross(self, c, return_grad=True, scale=1.e11):
        assert False  # this approach doesn't work
        self.CostScale = scale
        assert np.iscomplexobj(c) is False
        assert c.shape == (self.Sx.shape[1],)
        if return_grad:
            I, dI = self.PolIntensity(c,XorY='Y',region='Hole',DM_mode='phase',return_grad=True)
            cost = - np.sum(I)
            dcost = - np.sum(dI, axis=0)
            return (scale*cost, scale*dcost)
        else:
            I     = self.PolIntensity(c,XorY='Y',region='Hole',DM_mode='phase',return_grad=False)
            cost = - np.sum(I)
            return scale*cost

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
    