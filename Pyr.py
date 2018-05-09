#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 15:22:49 2017
@author: rfrazin
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as so
import scipy.sparse as sparsemat
import time

# NDim 1 for 1D simulation, 2 for 2D simulation
# basis - basis set for phase reconstruction.  must be
#    supported by a member function
# npup - number of sample points across the pupil
# npad - size of the padded array for the 1st FFT
# pupil_geom - 'None' or 'circle', for square or circular pupil
# lam - wavelength in microns
# d - reduced beam diameter (feeding WFS)
# D - telescope diamter (after pupil mask)
# fl - focal length in units of d
# slope - angle of pyramid in deg
# indref - index of  refraction at lam
# rotate rotation angle of PyWFS (degrees)
# self.PointWiseOrthogonalBasis is True when the basis functions have no
#     overlapping support points
class Pyr():
    def __init__(self, NDim=2, basis='PhasePixel', RegMat='D1', npup=33,
                 npad=1024, pupil_geom='circle', lam=0.85, d=7.2e3, D=7.992e6,
                 fl=40, slope=3.734, indref=1.452, rotate=45.):
        slope = np.tan(slope*np.pi/180.)
        self.NDim = NDim
        self.basis_type = basis
        self.pupil_geom = 'square'
        self.PointWiseOrthogonalBasis = False  # property of basis functions
        self.npup = npup
        self.npad = npad
        self.pupil_geom = pupil_geom
        self.rotate = rotate
        self.x = np.linspace(-1, 1, npup)
        self.RegScale = None # needed for regularization (scale param)
        self.RegHess = None  #  needed for regularized Hessian
        self.AmpConst = None # needed for magnitude constraint
        fcl = fl*d
        if (npup == 129) and (npad == 4096):
            fscale = lam*fcl/d/32.  # scaling for dist in focal plane
            self.il0, self.il1 = 1833, 1961  # range of left pupil image
            self.ir0, self.ir1 = 2135, 2263  #         right
            if NDim == 1:
                self.nd = 2 + self.il1 - self.il0 + self.ir1 - self.ir0
            else:
                raise Exception('Arrays Too Big for 2D.')
        elif (npup == 33) and (npad == 1024):
            fscale = lam*fcl/d/31.
            if NDim == 2:
                if self.npup == 33:
                    nn = 62
                self.i0 = npad/2 - nn
                self.i1 = npad/2 + nn
                self.nd = (2*nn + 1)*(2*nn + 1)
                self.npix = 1 + 2*nn
            else:
                self.nd = 2*npup
        else:
            assert False
        xf = fscale*np.linspace(-npad/2, npad/2-1, npad)  # focal plane coord.
        if NDim == 1:
            zf = - np.abs(xf)*slope # height of prism surface
            self.pramp = np.exp(-1j*zf*(indref - 1)*2*np.pi/lam)  # phase ramp
        else:

            # index functions for pupil plane
            self.UIp = lambda i: np.unravel_index(i, (npup, npup))  # 1D -> 2D
            self.IUp = lambda i1, i2: np.ravel_multi_index((i1, i2), (npup, npup))
            # index functions for detector plane
            self.UId = lambda i: np.unravel_index(i, (1+2*nn, 1 + 2*nn))
            self.IUd = lambda i1, i2: np.ravel_multi_index((i1, i2), (self.npix, self.npix))
            self.pupil_to_detector = None  #1-to-4 mapping from entrance pupil to detector
            (xf, yf) = np.meshgrid(xf, xf)
            if rotate != 0.:
                rotmat = np.zeros((2,2))
                rotmat[0,0] = np.cos(rotate*np.pi/180)
                rotmat[0,1] = np.sin(rotate*np.pi/180)
                rotmat[1,1] = rotmat[0,0]
                rotmat[1,0] = - rotmat[0,1]
                for k1 in range(npad):
                    for k2 in range(npad):
                        xfyf = rotmat.dot(np.array((xf[k1,k2], yf[k1,k2])))
                        xf[k1,k2] = xfyf[0]
                        yf[k1,k2] = xfyf[1]
            zf = np.zeros((npad, npad))  # height of prism surface
            for k1 in range(npad):
                for k2 in range(npad):
                    angle = np.arctan2(yf[k1,k2], xf[k1,k2])
                    if (angle >= -np.pi/4) and (angle < np.pi/4):
                        zf[k1,k2] = - np.abs(xf[k1,k2])*slope
                    elif (angle >= np.pi/4) and (angle < 3*np.pi/4):
                        zf[k1,k2] = - np.abs(yf[k1,k2])*slope
                    elif (angle >= 3*np.pi/4) or (angle < -3*np.pi/4):
                        zf[k1,k2] = - np.abs(xf[k1,k2])*slope
                    else:
                        zf[k1,k2] = - np.abs(yf[k1,k2])*slope
            self.pramp = np.exp(-1j*zf*(indref - 1)*2*np.pi/lam)  # phase ramp
        # set up a basis set
        if basis == 'ReImPixel':
            self.ReImPixelBasis(RegMat=RegMat)
        elif basis == 'PhasePixel':
            self.PhasePixelBasis(RegMat=RegMat)
        else:
            raise Exception('unknown basis')
        if (pupil_geom == 'circle') and (self.NDim == 2):
            self.SquarePupilToCircle()
        return

    # pyramid operator.  returns the detector field given pupil field g
    # g - pupil plane field
    # FullImage - return entire field on detector plane
    # Block3Faces - allow light only through one face of the pyramid
    # facedex - (0, 1, 2 or 3) index of transmissive face (only matters if Block3Faces)
    def FieldToField(self, g, FullImage=False, Block3Faces=False, facedex=0):
        u = myfft(self.pramp*myfft(myzp(g, self.npad)))
        if Block3Faces:  #only allow light through a single pyramid face
            assert facedex in [0, 1, 2, 3], "facedex must be 0, 1, 2 or 3"
            xf = np.linspace(-self.npad/2, self.npad/2-1, self.npad)  # focal plane coord.
            (xf, yf) = np.meshgrid(xf, xf)
            zf = np.ones(xf.shape)
            if self.rotate != 0.:
                rotmat = np.zeros((2,2))
                rotmat[0,0] = np.cos(self.rotate*np.pi/180)
                rotmat[0,1] = np.sin(self.rotate*np.pi/180)
                rotmat[1,1] = rotmat[0,0]
                rotmat[1,0] = - rotmat[0,1]
                for k1 in range(self.npad):
                    for k2 in range(self.npad):
                        xfyf = rotmat.dot(np.array((xf[k1,k2], yf[k1,k2])))
                        xf[k1,k2] = xfyf[0]
                        yf[k1,k2] = xfyf[1]
            for k1 in range(self.npad):
                for k2 in range(self.npad):
                    angle = np.arctan2(yf[k1,k2], xf[k1,k2])
                    if facedex == 0 and (angle >= -np.pi/4) and (angle < np.pi/4):
                        zf[k1,k2] = 1.
                    elif facedex == 1 and (angle >= np.pi/4) and (angle < 3*np.pi/4):
                        zf[k1,k2] = 1.
                    elif facedex == 2 and ((angle >= 3*np.pi/4) or (angle < - 3*np.pi/4)):
                        zf[k1,k2] = 1.
                    elif facedex == 3 and (angle >= -3*np.pi/4) and (angle < -np.pi/4):
                        zf[k1,k2] = 1.
                    else:
                        zf[k1,k2] = 0.
            u = myfft(self.pramp*zf*myfft(myzp(g, self.npad)))
        if not FullImage:
            u = self.ExtractPupilImage(u)
            u = u.reshape((self.nd,))
        return(u)

    # this extracts the "important" pixels from
    #   the image on the detector
    def ExtractPupilImage(self, h):
        if self.NDim == 1:
            assert len(h) == self.npad
            return(np.hstack((h[self.il0:self.il1+1],
                              h[self.ir0:self.ir1+1])))
        else:
            assert h.shape[0] == self.npad and h.shape[1] == self.npad
            return(h[self.i0:self.i1+1, self.i0:self.i1+1])


    # least-squares solution, response at c0
    # y - data vector
    # wt - weight, can be None
    # c0 - coefficients for gradient evaluation
    # RegParam - regularization parameter, can be None (unregularized solution)
    # PupilImage - take only inensity values from the geometrical images of the pupils
    # slopes - use only 'slopes' instead of raw intesnity values (PupilImage only)
    # returns estimate of coeffienct vector and estimator covariance
    def LeastSqSol(self, y, wt, c0, RegParam=None, ZeroPhaseMean=True,
                   PupilImage=False, slopes=True):
        assert len(c0) == self.nc
        if wt is not None:
            assert len(wt) == len(y)
        if RegParam is not None:  # Cost sets value of self.RegScale
            self.set_RegScale(wt, PupilImage, slopes) 
            R = RegParam*self.RegScale*self.RegMat.T.dot(self.RegMat)
        else:
            R = 0.
        if PupilImage:
            if slopes:
                (I0, dI0, I0tot) = self.PupilImageIntensity(c0, g=None, slopes=True)
            else:
                (I0, dI0) = self.PupilImageIntensity(c0, g=None, slopes=False)            
        else:  # take pixels from "whole" detector
            (I0, dI0) = self.Intensity(c0)
        y0 = y - I0
        if ZeroPhaseMean:
            y0 = np.hstack((y0, 0))
            if wt is not None:
                wt0 = np.hstack((wt, np.max(wt)))
            newrow = np.zeros(self.nc)
            if self.basis_type == 'ReImPixel':
                rowrange = np.arange(self.nc/2) + self.nc/2
            elif self.basis_type == 'PhasePixel':
                rowrange = np.arange(self.nc)
            else:
                assert False
            maxval = np.max(np.abs(dI0))
            for k in rowrange:
                newrow[k] = maxval
            dI0 = np.vstack((dI0, newrow))
        if wt is None:
            cov = np.linalg.inv(dI0.T.dot(dI0) + R)
            c = cov.dot(dI0.T.dot(y0))
        else:
            W = sparsemat.dia_matrix((wt0, 0), shape=(len(wt0), len(wt0)))
            cov = np.linalg.inv(dI0.T.dot(W.dot(dI0)) + R)
            c = cov.dot(dI0.T.dot(W*y0))
        if len(c.shape) == 2:  # this sucks
            cnew = np.zeros(len(c0))
            if c.shape[0] == 1:
                for k in range(len(c0)):
                    cnew[k] = c[0, k]
            else:
                for k in range(len(c0)):
                    cnew[k] = c[k, 0]
            c = cnew
        c += c0
        return(c, cov)


    # find the coefficient vector that minimizes the weighted cost
    # c0 - initial guess, can be None if LsqStart is set to True
    # y - intensity measurements
    # wt - weights
    # RegParam - regularization parameter
    # method - 'Newton-CG' or 'BFGS'
    # MaxIt - maximum number of iterations
    # LsqStart - start with Linear Least Squares estimate about c0
    # AmpCon - amplitude constraint employed.  Don't do this.
    def FindMinCost(self, c0, y, wt=None, RegParam=None,  method='Newton-CG', MaxIt=10,
                    LsqStart=False, AmpCon=False, ):
        assert method in ['Newton-CG', 'BFGS'], "Invalid method"
        if LsqStart:
            cg, gcg = self.LeastSqSol(y, wt, c0, RegParam, ZeroPhaseMean=True)
        else:
            assert len(c0) == self.nc, "nc parameter set incorrectly"
            cg = c0
        fargs = (y, wt, RegParam, True, AmpCon)
        if method == 'Newton-CG':
            opt = {'disp': True, 'maxiter': MaxIt, 'xtol': 1.e-6, 'return_all': True}
        elif method == 'BFGS':
            opt = {'disp': True, 'maxiter': MaxIt, 'gtol': 1.e-7, 'return_all': True}
        res = so.minimize(self.Cost, cg, args=fargs, method=method,
                          options=opt, jac=True, hess=self.CostHessian)
        sol = res.x
        return(sol, res)

    # calculates the cost function for ReIm basis
    # c - coefficient vector
    # y - measured intensities
    # wt - weight corresponding to y, can be None
    # if ZeroPhaseMean - include ZeroPhaseMean penalty in cost
    # RegParam - bigger is more regularized, None turns it off
    # PupilImage - Consider only pixels in geometrical image of pupil
    #  slopes not implemented, only relevant for PupilImage
    # AmpCon - amplitude inequality penalty for ReImPixel basis,
    #       (assumes max allowed value is 1). This function is
    #       almost convex.   set to 'None' for no constraint
    # returns cost and gradient
    def Cost(self, c, y, wt=None, RegParam=None, ZeroPhaseMean=True,
             PupilImage=False, slopes=False, AmpCon=False):
        assert not PupilImage, "Doesn't work for PupilImage."
        if wt is None:
            wt0 = np.ones(y.shape)
        else:
            assert len(wt) == len(y)
            wt0 = wt
        if (RegParam is not None):
            self.set_RegScale(wt, PupilImage, slopes)
            self.RegHess = self.RegScale*RegParam*self.RegMat.T.dot(self.RegMat)
        else:  # make sparse matrix of zeros 
            self.RegHess = sparsemat.dia_matrix((self.RegMat.shape[1], self.RegMat.shape[1]))
        (I, dI) = self.Intensity(c)
        wtIy = wt0*(I - y)
        cost = np.sum(np.power(wtIy, 2))/2.
        dcost = np.zeros(self.nc)
        for k in range(self.nc):
            dcost[k] = np.sum(wtIy*dI[:, k])
        if ZeroPhaseMean:
            if self.basis_type == 'ReImPixel':
                cost += np.max(np.abs(y))*np.max(wt0)*np.power(np.mean(c[self.nc/2:]), 2)/2.
                for k in np.arange(self.nc/2) + self.nc/2:
                    dcost[k] += np.max(np.abs(y))*np.max(wt0)*c[k]/(self.nc/2.)
            elif self.basis_type == 'PhasePixel':
                cost += np.max(np.abs(y))*np.max(wt0)*np.power(np.mean(c), 2)/2.
                for k in np.arange(self.nc):
                    dcost[k] += np.max(np.abs(y))*np.max(wt0)*c[k]/(self.nc)
            else:
                assert False
        if RegParam is not None:
            assert RegParam > 0.
            v = self.RegMat.dot(c)
            cost += RegParam*self.RegScale*v.dot(v)
            dcost += 2*RegParam*self.RegScale*self.RegMat.T.dot(v)
        if not AmpCon:
            return(cost, dcost)
        # implement maximum magnitude inequality penalty - only for ReImPixel basis
        assert self.basis_type == 'ReImPixel'
        MaxSqMag = 1.  # max squared magnitude
        scale = np.power(np.max(wt0)*np.max(y), 2)
        for k in range(self.nc/2):
            l = k + self.nc/2
            smag = c[k]*c[k] + c[l]*c[l]
            if smag > MaxSqMag:  # turn on constraint
                cost += .5*scale*(smag - MaxSqMag)
                dcost[k] += scale*c[k]
                dcost[l] += scale*c[l]
#        if AmpVal is not None:  # this is a non-convex cost -- not useful
#            assert AmpVal > 0.
#            const = self.RegScale*AmpFac/AmpVal/AmpVal
#            self.AmpConst = const
#            for k1 in range(self.nc/2):
#                k2 = k1 + self.nc/2
#                cost += const*np.power(c[k1]*c[k1] + c[k2]*c[k2] - AmpVal, 2)
#                dcost[k1] += 4.*const*c[k1]*(c[k1]*c[k1] + c[k2]*c[k2] - AmpVal)
#                dcost[k1] += 4.*const*c[k2]*(c[k1]*c[k1] + c[k2]*c[k2] - AmpVal)
        return(cost, dcost)

    # sets the value of self.RegScale
    # slopes only has meaning if PupilImage == True  
    def set_RegScale(self, wt=None, PupilImage=False, slopes=False):
        if self.basis_type == 'ReImPixel':
            c0 = np.hstack((np.ones(self.nc/2), np.zeros(self.nc/2)))
        elif self.basis_type == 'PhasePixel':
            c0 = np.zeros(self.nc)
        else:
            assert False, "Cost: not set up for this basis."
        if PupilImage and slopes:
            I, dI, Itot = self.PupilImageIntensity(c0, g=None, slopes=True)
            if wt is not None:
                wt0 = wt
            else:
                wt0 = 1.
            self.RegScale = np.sum(np.power(wt0*Itot, 2))/self.nc
            return
        if PupilImage and not slopes:
            I, dI = self.PupilImageIntensity(c0, g=None, slopes=False)
        else:  # not PupilImage
            I, dI = self.Intensity(c0, g=None)
            if wt is None:
                wt0 = 1.
            else:
                wt0 = wt
                assert wt.shape == I.shape
            self.RegScale = np.sum(np.power(wt0*I, 2))/self.nc
            return


    def CostHessian(self, c, y, wt=None, RegParam=None, ZeroPhaseMean=True,
                    AmpCon=False, verbose=True):
        assert len(y) == self.nd
        if wt is None:
            wt0 = np.ones(y.shape)
        else:
            assert len(wt) == self.nd
            wt0 = wt
        if verbose:
            tt0 = time.time()
        (I, dI) = self.Intensity(c, g=None)
        wtIy = wt0*(I - y)

        hes = np.zeros((self.nc, self.nc))
        for k1 in range(self.nc):
            for k2 in range(self.nc):
                if k2 > k1:
                    continue
                hes[k1, k2] = np.sum(wt0*dI[:, k1]*dI[:, k2])
                d2I = self.IntensityHessian([k1, k2], c, g=None)
                hes[k1, k2] += np.sum(wtIy*d2I)
        for k1 in range(self.nc):
            for k2 in range(self.nc):
                if k2 > k1:
                    hes[k1, k2] = hes[k2, k1]

        if ZeroPhaseMean:
            if self.basis_type == 'ReImPixel':
                for k in np.arange(self.nc/2) + self.nc/2:
                    hes[k, k] += np.max(np.abs(y))*np.max(wt0)/(self.nc/2.)
            elif self.basis_type == 'PhasePixel':
                for k in np.arange(self.nc):
                    hes[k, k] += np.max(np.abs(y))*np.max(wt0)/(self.nc)
            else:
                assert False
        if RegParam is not None:
            if self.RegHess is None:  # set it up
                self.Cost(c, y, wt, RegParam=RegParam, ZeroPhaseMean=ZeroPhaseMean, AmpCon=AmpCon)
            hes += self.RegHess
        if verbose:
            print("Hessian calculation time: ", (time.time() - tt0)/60, " minutes.")
        if not AmpCon:
            return(hes)
        # implement maximum magnitude inequality penalty for ReImPixel
        assert self.basis_type == 'ReImPixel'
        MaxSqMag = 1.  # max squared magnitude
        scale = np.power(np.max(wt0)*np.max(y), 2)
        for k in range(self.nc/2):
            l = k + self.nc/2
            smag = c[k]*c[k] + c[l]*c[l]
            if smag > MaxSqMag:  # turn on constraint
                hes[k, k] += scale
                hes[l, l] += scale
#      # non-convex penalty for ReImPixel, so not useful
#            if self.AmpConst is None:  # set it up
#                self.ReImCost(c, y, wt, RegParam=RegParam, AmpVal=1., AmpFac=.1,
#                              ZeroPhaseMean=ZeroPhaseMean)
#            for k1 in range(self.nc/2):
#                k2 = k1 + self.nc/2
#                hes[k1, k1] += 4.*self.AmpConst*(3.*c[k1]*c[k1] + c[k2]*c[k2] - AmpVal)
#                hes[k2, k2] += 4.*self.AmpConst*(c[k1]*c[k1] + 3.*c[k2]*c[k2] - AmpVal)
#                hes[k1, k2] += 8.*self.AmpConst*c[k1]*c[k2]
#                hes[k2, k1] += hes[k1, k2]
        return(hes)

    def CostHessianComponent(self, ind, return_value=False):
        k = self.dct['indlist'][ind]
        k1, k2 = np.unravel_index(k, (self.nc, self.nc))
        print(k, k1, k2)
        d2I = self.IntensityHessian([k1, k2], self.dct['c'], g=None)
        th = np.sum(self.dct['wt0']*self.dct['dI'][:, k1]*self.dct['dI'][:, k2])
        th += np.sum(self.dct['wtIy']*d2I)
        self.temphes[k1, k2] = th
        if return_value:
            return(th)
        else:
            return

    # c is the coefficient vector in the ReImPixels basis.
    # no FullImage capability
    # returns detector itensity and gradient w.t.t. c
    # g argument needed for consistency
    def ReImIntensity(self, c, g=None):
        assert len(c) == self.nc
        ud = np.zeros(self.nd).astype('complex')  # detector field
        for k in range(self.nc):
            ud += c[k]*self.field_grad[:, k]
        I = np.real(ud*np.conj(ud))
        dI = np.zeros((self.nd, self.nc))
        for k in range(self.nc):
            dI[:, k] = 2.*np.real(np.conj(ud)*self.field_grad[:, k])
        return(I, dI)

    # ind - 2-element coefficient index
    #   returns the component of the Intensity Hessian
    #   for all pixels in extraction region
    # c,g arguments needed for consistency
    def ReImIntensityHessian(self, ind, c=None, g=None):
        assert g is None, "Must modify to support field multiplier g."
        assert len(ind) == 2
        hes = self.field_grad[:, ind[0]]*np.conj(self.field_grad[:, ind[1]])
        hes = 2*np.real(hes)
        return(hes)

    # pix - index of desired pixel (can be 1 or 2 numbers for 2D case)
    #   returns the Hessian matrix corresponding to pixel on detector
    # c,g arguments needed for consistency
    def ReImIntensityHessianAtPixel(self, pix, c=None, g=None):
        assert g is None, "Must modify to support field multiplier g."
        if self.NDim == 1:
            assert np.ndim(pix) == 0
            pixel = pix
        else:
            if len(pix) == 2:
                pixel = np.ravel_multi_index((pix[0],pix[1]),(self.npix,self.npix))
        pixhess = np.zeros((self.nc, self.nc))
        for k1 in range(self.nc):
            for k2 in range(self.nc):
                if k2 > k1:
                    continue
                pixhess[k1,k2] = self.ReImIntensityHessian([k1,k2])[pixel]
        for k1 in range(self.nc):
            for k2 in range(self.nc):
                if k2 > k1:
                    pixhess[k1,k2] = pixhess[k2,k1]
        return(pixhess)

    # compute detector field and its gradient given c,
    #  where c is the vector of coefficients in the PhasePixel basis.
    # c - coefficient vector
    # g - field multiplier, can be None
    # In order to avoid recalcuation of gradients, it checks to see if
    #    it has already been calculated for the current value of c
    def PhasePixelToField(self, c, g=None):
        if g is not None:
            gg = g.reshape((self.nc,))
            if np.array_equal(self.c_current, c) and np.array_equal(gg, self.g_current):
                return(self.field_current, self.field_grad_current)
        else:
            if np.array_equal(self.c_current, c) and self.g_current is None:
                return(self.field_current, self.field_grad_current)
        f = np.zeros(self.nd).astype('complex')  # detector field
        gf = np.zeros((self.nd, self.nc)).astype('complex')  # its grad
        for k in range(self.nc):
            s = np.exp(1j*c[k])*self.field_grad_factor[:, k]
            if g is not None:
                s *= gg[k]
            f += s
            gf[:, k] = 1j*s
        self.c_current = 1.*c  # store deep copies
        self.field_current = 1.*f
        self.field_grad_current = 1.*gf
        if g is not None:
            self.g_current = 1.*gg
        return(f, gf)

    # Intensity and its grad for phase-only basis functions
    # c is coefficient vector
    # g is an optional field multiplier
    def PhasePixelIntensity(self, c, g=None):
        field, gfield = self.PhasePixelToField(c, g=g)
        if self.NDim == 2:
            field = field.reshape((self.nd,))
        I = np.real(field*np.conj(field))
        gI = np.zeros(gfield.shape)
        for k in range(self.nc):
            gI[:,k] = 2*np.real(gfield[:,k]*np.conj(field))
        return(I, gI)

    # ind - 2-element coefficient index
    #   returns d^2I/(dc[ind[0]] dc[ind[1]]) for all pixels in extraction region
    def PhasePixelIntensityHessian(self, ind, c, g=None):
        assert g is None, "Must modify to support field multiplier g."
        f, df = self.PhasePixelToField(c, g=g)
        assert len(ind) == 2
        hes = df[:, ind[0]]*np.conj(df[:, ind[1]])
        if ind[0] == ind[1]:
            hes -= np.conj(f)*np.exp(1j*c[ind[0]])*self.field_grad_factor[:,ind[0]]
        hes = 2*np.real(hes)
        return(hes)

    #Returns the intensity Hessian matrix at detector pixel specified by pixind
    #In 2D, pixind can be a 1D or 2D index.
    #c - coef vector
    #g - optional field multiplier
    def PhasePixelIntensityHessianAtPixel(self, pixind, c, g=None):
        assert g is None, "Must modify to support field multiplier g."
        assert self.PointWiseOrthogonalBasis  # otherwise this is MUCH worse
        pix = np.array(pixind).astype('int')  # deep copy
        if self.NDim == 2:
            if len(pix) == 2:
                pix = self.IUd(pix[0],pix[1])
        hesmat = np.zeros((self.nc, self.nc)).astype('complex')
        f, gf = self.PhasePixelToField(c, g)
        for k1 in range(self.nc):
            for k2 in range(self.nc):
                if k2 > k1:
                    continue
                hesmat[k1,k2] = np.conj(gf[pix,k1])*gf[pix,k2]
                if k1 == k2:
                    hesmat[k1,k1] -= np.conj(f[pix])*np.exp(1j*c[k1])*self.field_grad_factor[pix,k1]
        hesmat = 2*np.real(hesmat)
        for k1 in range(self.nc):
            for k2 in range(self.nc):
                if k2 > k1:
                    hesmat[k1,k2] = hesmat[k2,k1]
        return(hesmat)

    # This cuts the pupil grid from a square to a circle
    def SquarePupilToCircle(self):
        assert self.basis_type == 'PhasePixel', "SquareToCircle: Bad basis choice."
        self.pupil_geom = 'circle'
        x = np.linspace(-1, 1, self.npup)
        xg, yg = np.meshgrid(x, x)
        rg = np.sqrt(xg*xg + yg*yg)
        inside = []
        outside = []
        self.pixel_map = dict()  # pixel locations for coefficients
        self.inv_pixel_map = dict()
        count = -1
        for k1 in range(self.npup):
            for k2 in range(self.npup):
                m = self.IUp(k1, k2)
                if rg[k1, k2] > 1.:
                    outside.append(m)
                else:
                    inside.append(m)
                    count += 1
                    self.pixel_map[count] = (k1, k2)
                    self.inv_pixel_map[(k1, k2)] = count
        inside = np.array(inside)
        outside = np.array(outside)
        nin = len(inside)
        basis = np.zeros((self.npup, self.npup, nin))
        field_grad_factor = np.zeros((self.nd, nin)).astype('complex')
        for m in range(len(inside)):
            basis[:, :, m] = self.basis[:, :, inside[m]]
            field_grad_factor[:, m] = self.field_grad_factor[:, inside[m]]

        # splice regularization matrix
        nrow, ncol = self.RegMat.shape
        assert ncol == self.nc
        assert np.mod(nrow, self.nc) == 0
        lmat = nrow/self.nc
        newreg = np.zeros((lmat*nin, nin))  # new RegMat
        rcount = -1
        for k1 in inside:
            rcount += 1
            ccount = -1
            for k2 in inside:
                ccount += 1
                for nm in range(lmat):
                    newreg[rcount + nm*nin, ccount] = RetrieveMatrixElement((k1 + self.nc*nm, k2), self.RegMat)
        del self.RegMat, self.basis, self.field_grad_factor
        self.RegMat = sparsemat.coo_matrix(newreg)
        self.nc = nin
        self.basis = basis
        self.field_grad_factor = field_grad_factor
        if self.rotate == 45.:  # set up pupil-to-detector mapping
            self.pupil_to_detector = dict()
            ph = np.zeros(self.nc)
            for k1 in range(self.nc):
                amp = 0.*ph
                amp[k1] = 1.
                Id, dI = self.PhasePixelIntensity(ph, g=amp)
                Id = Id.reshape(self.npix, self.npix)
                mask = np.zeros((self.npix, self.npix))
                mask[0:self.npix/2, 0:self.npix/2] = 1.
                nw = self.UId(np.argmax(mask*Id))
                mask = 0*mask
                mask[self.npix/2:self.npix, 0:self.npix/2] = 1.
                sw = self.UId(np.argmax(mask*Id))
                mask = 0*mask
                mask[0:self.npix/2, self.npix/2:self.npix] = 1.
                ne = self.UId(np.argmax(mask*Id))
                mask = 0*mask
                mask[self.npix/2:self.npix, self.npix/2:self.npix] = 1.
                se= self.UId(np.argmax(mask*Id))
                self.pupil_to_detector[k1] = [nw, sw, ne, se]
        return

    # return intensities only in the pixel corresponding to the pupil images
    # if slopes, it returns (unnormalized) "slope" values from the intensity
    # normalized only applies to slope measurements, otherwise it does not matter
    def PupilImageIntensity(self, paramvec, g=None, slopes=True, normalized=False):
        assert g==None, "Not implemented for nonzero multiplier g."
        assert self.pupil_to_detector is not None, "must set self.pupil_to_detector"
        sz = len(self.pupil_to_detector)  # = self.nc for PhasePixel
        I, dI = self.Intensity(paramvec, g=g)
        I = I.reshape(self.npix, self.npix)
        dI = dI.reshape(self.npix, self.npix, self.nc)
        if slopes:
            S = np.zeros((2*sz,))
            Itot = np.zeros((2*sz,))  #for calculating shot noise
            dS = np.zeros((2*sz, self.nc))
        else:
            S = np.zeros((4*sz,))
            dS = np.zeros((4*sz, self.nc))
        for k1 in range(sz):
            p2d = self.pupil_to_detector[k1]
            nw = I[p2d[0]]
            sw = I[p2d[1]]
            ne = I[p2d[2]]
            se = I[p2d[3]]
            if slopes:
                S[2*k1 + 0] = nw + sw - ne - se 
                S[2*k1 + 1] = nw + ne - sw - se
                Itot[2*k1 + 0] = nw + sw + ne + se
                Itot[2*k1 + 1] = nw + sw + ne + se
                if normalized:
                    S[2*k1 + 0] /= Itot[2*k1 + 0]
                    S[2*k1 + 1] /= Itot[2*k1 + 1]
            else:
                S[4*k1 + 0] = nw
                S[4*k1 + 1] = sw
                S[4*k1 + 2] = ne
                S[4*k1 + 3] = se
            nw = dI[p2d[0][0], p2d[0][1], :].reshape((1, self.nc))
            sw = dI[p2d[1][0], p2d[1][1], :].reshape((1, self.nc))
            ne = dI[p2d[2][0], p2d[2][1], :].reshape((1, self.nc))
            se = dI[p2d[3][0], p2d[3][1], :].reshape((1, self.nc))
            if slopes:
                if normalized:
                    normgrad =  nw + sw + ne + se
                    dS[2*k1 + 0, :] = (nw + sw - ne - se)/Itot[2*k1 + 0] - normgrad*(S[2*k1 + 0]/Itot[2*k1 + 0]**2)
                    dS[2*k1 + 1, :] = (nw + ne - sw - se)/Itot[2*k1 + 1] - normgrad*(S[2*k1 + 1]/Itot[2*k1 + 1]**2)
                else:
                    dS[2*k1 + 0, :] = nw + sw - ne - se
                    dS[2*k1 + 1, :] = nw + ne - sw - se
            else:
                dS[4*k1 + 0, :] = nw
                dS[4*k1 + 1, :] = sw
                dS[4*k1 + 2, :] = ne
                dS[4*k1 + 3, :] = se
        if slopes:
            return(S, dS, Itot)
        else:
            return(S, dS)


    # This function gives the intensity that results from allowing the light
    #  into only of the pyramid faces (at a time) in order to evaluate the
    #  importance of interference.  Assumes cicurlar pupi.
    # phase - the vector of pupil plane phases
    def IntensityBlock3Faces(self, phase, show_indiv=False):
        assert len(phase) == self.nc
        assert self.pupil_geom == 'circle'
        g = np.zeros((self.npup, self.npup)).astype('complex')
        for k in range(len(phase)):
            ki, kj = self.pixel_map[k]
            g[ki, kj] = np.exp(1j*phase[k])
        intensity = 0.*np.real(self.FieldToField(g, FullImage=False, Block3Faces=False, facedex=-1))
        for face in [0, 1, 2, 3]:
            field = self.FieldToField(g, FullImage=False, Block3Faces=True, facedex=face)
            intensity += np.real(field*np.conj(field))
            if show_indiv:
                plt.figure(10+face)
                f = field.reshape(self.npix, self.npix)
                plt.imshow(np.sqrt(np.real(f*np.conj(f))));
                plt.colorbar()
        return(intensity)

    def PhasePixelBasis(self, RegMat='D1'):
        self.Intensity = self.PhasePixelIntensity
        self.IntensityHessian = self.PhasePixelIntensityHessian
        self.IntensityHessianAtPixel = self.PhasePixelIntensityHessianAtPixel
        self.PointWiseOrthogonalBasis = True
        self.c_current = None  # these current values avoid repeat calculation
        self.g_current = None
        self.field_current = None
        self.field_grad_current = None
        if self.NDim == 1:
            self.nc = self.npup
            self.basis = np.zeros((self.npup, self.nc))
            for k in range(self.npup):
                self.basis[k, k] = 1.
            if RegMat == 'ID':
                self.RegMat = sparsemat.dia_matrix((np.ones(self.nc), 0), shape=(self.nc, self.nc))
            elif RegMat == 'D1':
                self.RegMat = sparsemat.dia_matrix((np.ones(self.nc),0), shape=(self.nc, self.nc))
                self.RegMat += sparsemat.dia_matrix((-np.ones(self.nc),1), shape=(self.nc, self.nc))
            elif RegMat == 'D2':
                self.RegMat = sparsemat.dia_matrix((2*np.ones(self.nc),0), shape=(self.nc, self.nc))
                self.RegMat += sparsemat.dia_matrix((-np.ones(self.nc),1), shape=(self.nc, self.nc))
                self.RegMat += sparsemat.dia_matrix((-np.ones(self.nc),-1), shape=(self.nc, self.nc))
            else:
                raise Exception("unknown regularization option.")
        else:  # self.NDim = 2
            self.nc = self.npup*self.npup
            self.basis = np.zeros((self.npup, self.npup, self.nc))
            for k in range(self.nc):
                (i1, i2) = self.UIp(k)
                self.basis[i1, i2, k] = 1
            if RegMat == 'ID':
                self.RegMat = sparsemat.dia_matrix((np.ones(self.nc), 0), shape=(self.nc, self.nc))
            elif RegMat == 'D1':
                Dv = np.zeros((self.nc, self.nc)) # vertical deriv
                for k1 in range(self.npup):  # horizontal index
                    for k2 in np.arange(self.npup-1):  # vertical index
                        ks = self.IUp(k2, k1)  # self index
                        kn = self.IUp(k2+1, k1) # neighbor index
                        Dv[ks, ks] = 1.
                        Dv[ks, kn] = -1.
                Dv = sparsemat.dia_matrix(Dv)
                Dh = np.zeros((self.nc, self.nc)) # horizontal deriv
                for k1 in range(self.npup-1):  # horizontal index
                    for k2 in np.arange(self.npup):  # vertical index
                        ks = self.IUp(k2, k1)  # self index
                        kn = self.IUp(k2, k1+1) # neighbor index
                        Dh[ks, ks] = 1.
                        Dh[ks, kn] = -1.
                Dh = sparsemat.dia_matrix(Dh)
                self.RegMat = sparsemat.vstack((Dv, Dh))
            elif RegMat == 'DelSq':  # del squared regularizer
                assert self.NDim == 2
                R = np.zeros((self.nc, self.nc))
                for k1 in 1 + np.arange(self.npup - 2):
                    for k2 in 1 + np.arange(self.npup - 2):
                        ks = self.IUp(k1, k2)  # self pixel
                        ki = self.IUp(k1-1, k2)  # above neighbor
                        km = self.IUp(k1+1, k2)  # below neighbor
                        kj = self.IUp(k1, k2-1)  # left neighbor
                        kl = self.IUp(k1, k2+1)  # right neighbor
                        R[ks,ks] = 4.
                        R[ks,ki] = -1.
                        R[ks,km] = -1.
                        R[ks,kj] = -1.
                        R[ks,kl] = -1.
                self.RegMat = sparsemat.dia_matrix(R)
            else:
                raise Exception('Unknown Regularization Matrix Type')
        # calculate field gradient factors using PointWiseOrthogonality
        #    actual field gradient is 1j*c[k]*self.field_grad_factor[:,k]
        self.field_grad_factor = np.zeros((self.nd, self.nc)).astype('complex')
        for k in range(self.nc):
            if self.NDim == 2:
                self.field_grad_factor[:,k] = self.FieldToField(self.basis[:,:,k], FullImage=False)
            else:
                self.field_grad_factor[:,k] = self.FieldToField(self.basis[:,k], FullImage=False)
        return

    # pixel basis functions for Re and Im parts of the pupil field.
    # builds self.basis containing basis functions
    # RegMat specifies the type of regularization matrix (self.RegMat)
    def ReImPixelBasis(self, RegMat='ID'):
        self.Intensity = self.ReImIntensity
        self.IntensityHessian = self.ReImIntensityHessian
        self.IntensityHessianAtPixel = self.ReImIntensityHessianAtPixel
        self.PointWiseOrthogonalBasis = True
        if self.NDim == 1:
            self.nc = 2*self.npup
            self.basis = np.zeros((self.npup, self.nc)).astype('complex')
            for k in range(self.npup):
                self.basis[k, k] = 1
                self.basis[k, k + self.npup] = 1j
            self.field_grad = np.zeros((self.nd, self.nc)).astype('complex')
            for k in range(self.nc):  # these are constant!
                self.field_grad[:, k] = self.FieldToField(self.basis[:, k], FullImage=False)
            if RegMat == 'ID':  # regularization matrix
                self.RegMat = sparsemat.dia_matrix((np.ones(self.nc), 0),
                                                   shape=(self.nc, self.nc))
            elif RegMat == 'D1':
                zer = sparsemat.dia_matrix((np.zeros(self.nc/2),0), shape=(self.nc/2, self.nc/2))
                dma = sparsemat.dia_matrix((np.ones(self.nc/2),0), shape=(self.nc/2, self.nc/2))
                dma += sparsemat.dia_matrix((-np.ones(self.nc/2),1), shape=(self.nc/2, self.nc/2))
                dma[self.nc/2 - 1, self.nc/2 - 1] = 0
                top = sparsemat.hstack((dma, zer))
                bot = sparsemat.hstack((zer, dma))
                self.RegMat = sparsemat.vstack((top, bot))
            return()
        # 2D case
        self.nc = 2*self.npup*self.npup
        self.basis = np.zeros((self.npup, self.npup, self.nc)).astype('complex')
        for k in range(self.nc/2):
            (i1, i2) = self.UIp(k)
            self.basis[i1, i2, k] = 1
            self.basis[i1, i2, k + self.nc/2] = 1j
        self.field_grad = np.zeros((self.nd, self.nc)).astype('complex')
        for k in range(self.nc):
            fgrad = self.FieldToField(self.basis[:, :, k], FullImage=False)
            self.field_grad[:, k] = fgrad
        if RegMat == 'ID':  # regularization matrix
            self.RegMat = sparsemat.dia_matrix((np.ones(self.nc), 0),
                                                   shape=(self.nc, self.nc))
        elif RegMat == 'D1':
            assert self.NDim == 2
            Dv = np.zeros((self.nc, self.nc)) # vertical deriv
            for k1 in range(self.npup):  # horizontal index
                for k2 in np.arange(self.npup-1):  # vertical index
                    ks = self.IUp(k2, k1)  # self index
                    kn = self.IUp(k2+1, k1) # neighbor index
                    Dv[ks, ks] = 1.
                    Dv[ks, kn] = -1.
                    ks += self.nc/2
                    kn += self.nc/2
                    Dv[ks, ks] = 1.
                    Dv[ks, kn] = -1.
            Dv = sparsemat.dia_matrix(Dv)
            Dh = np.zeros((self.nc, self.nc)) # horizontal deriv
            for k1 in range(self.npup-1):  # horizontal index
                for k2 in np.arange(self.npup):  # vertical index
                    ks = self.IUp(k2, k1)  # self index
                    kn = self.IUp(k2, k1+1) # neighbor index
                    Dh[ks, ks] = 1.
                    Dh[ks, kn] = -1.
                    ks += self.nc/2
                    kn += self.nc/2
                    Dh[ks, ks] = 1.
                    Dh[ks, kn] = -1.
            Dh = sparsemat.dia_matrix(Dh)
            self.RegMat = sparsemat.vstack((Dv, Dh))

        elif RegMat == 'DelSq':  # del squared regularizer
            assert self.NDim == 2
            R = np.zeros((self.nc, self.nc))
            for k1 in 1 + np.arange(self.npup - 2):
                for k2 in 1 + np.arange(self.npup - 2):
                    ks = self.IUp(k1, k2)  # self pixel
                    ki = self.IUp(k1-1, k2)  # above neighbor
                    km = self.IUp(k1+1, k2)  # below neighbor
                    kj = self.IUp(k1, k2-1)  # left neighbor
                    kl = self.IUp(k1, k2+1)  # right neighbor
                    R[ks,ks] = 4.
                    R[ks,ki] = -1.
                    R[ks,km] = -1.
                    R[ks,kj] = -1.
                    R[ks,kl] = -1.
                    ks += self.nc/2
                    ki += self.nc/2
                    km += self.nc/2
                    kj += self.nc/2
                    kl += self.nc/2
                    R[ks,ks] = 4.
                    R[ks,ki] = -1.
                    R[ks,km] = -1.
                    R[ks,kj] = -1.
                    R[ks,kl] = -1.
            self.RegMat = sparsemat.dia_matrix(R)
        else:
            raise Exception('Unknown Regularization Matrix Type')
        return


# SciPy.sparse matrices don't allow element access with the
#   usual bracket notation, so I had to write this function.
# index_tuple - index pair tuple
# matrix - duh
def RetrieveMatrixElement(index_tuple, matrix):
    assert matrix.ndim == 2
    assert len(index_tuple) == 2
    s0, s1 = matrix.shape
    r = index_tuple[0]
    c = index_tuple[1]
    assert (r >= 0) and (r < s0) and (c >= 0) and (c < s1)
    v = np.zeros(s1)
    v[c] = 1
    col = matrix.dot(v)
    return(col[r])


def myfft(g):  # for centered arrays, custom normalization
    if g.ndim == 1:
        return np.fft.fftshift(np.fft.fft(np.fft.fftshift(g)))/np.sqrt(len(g))
    else:
        return(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(g)))/np.sqrt(g.shape[0]*g.shape[1]))


#  this zero pad function gives rise to purely real myfft with symm. input
def myzp(f, npad):  # zero-pad function for pupil fields
    if f.ndim == 1:
        nfield = len(f) + 1
        if np.mod(nfield, 2) != 0:
            raise Exception("len(f) must be odd.")
        if npad <= len(f):
            raise Exception("npad must be greater than len(f)")
        if np.mod(npad, 2) == 1:
            raise Exception("npad must be even.")
        g = np.zeros(npad).astype('complex')
        g[npad/2 - nfield/2 + 1:npad/2 + nfield/2] = f
        return(g)
    else:
        if f.shape[0] != f.shape[1]:
            raise Exception("f must be square (if 2D).")
        nfield = f.shape[0] + 1
        if np.mod(nfield, 2) != 0:
            raise Exception("len(f) must be odd.")
        if npad <= f.shape[0]:
            raise Exception("npad must be greater than len(f)")
        if np.mod(npad, 2) == 1:
            raise Exception("npad must be even.")
        g = np.zeros((npad, npad)).astype('complex')
        g[npad/2 - nfield/2 + 1:npad/2 + nfield/2,
          npad/2 - nfield/2 + 1:npad/2 + nfield/2] = f
        return(g)

def SigmaToStrehl(sigma):
    sig = np.array(sigma)
    strehl = np.exp(-sig*sig)
    return(strehl)

def StrehlToSigma(strehl):
    st = np.array(strehl)
    sigma = np.sqrt(- np.log(st))
    return(sigma)

