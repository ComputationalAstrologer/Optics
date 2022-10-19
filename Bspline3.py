#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 14:41:44 2022
@author: Richard Frazin

Cubic B-Spline utilities

I could not get scipy.interpolate.LSQBivariateSpline to work
"""


import numpy as np
import matplotlib.pyplot as plt

"""
cubic b-spline functions
See M. Unser, IEEE Sig. Proc. Mag., Nov. 1999
This assumes the knots are uniformly spaced,
  which makes them cardinal b-splines.

this evaluates the 1D cubic b-spline at x
delta is the spacing between the spline knots.
The resulting function is nonzero for |x/delta| < 2
"""
def Bspline3_1D(x, delta):
    assert np.isscalar(x)
    assert np.isscalar(delta)
    q = np.abs(x/delta)
    if q >= 2.:
        return(0.)
    if q >= 1.:
        return( ((2.-q)**3)/6.  )
    else:
        return(2./3 - q*q + q*q*q/2.)
    
#See Bspline3_1D
def Bspline3_2D(x, y, delta):
    bx = Bspline3_1D(x, delta)
    by = Bspline3_1D(y, delta)
    return(bx*by)

"""
Class for bivariate b-spline regression.
This assumes the same spacing of the spline knots in both directions on a rectanglular
  grid of knots.  There is a spline centered on every knot.  Note that in 1D, a given spline
  fcn centered at knot n, located at x_n, has support between x_{n-2} and x_{n+2}.  
  Therefore, a point located bewteen x_n and x_{n+1} is within the support of the splines
  [n-1, n, n+1, n+2].  Thus, in 2D, there are up to 16 splines to consider for a given pont.

z - 1D vector of function values to be fit (not needed for __init__() )
x - 1D vector of x coords corresponding to z.  must have len(x) == len(z)
y - 1D vector of y coords corresponding to z.  must have len(y) == len(x)
Nx - number of spline knots in x-direction
Xmin - location (x coordinate) of leftmost spline knot
Delta - distance (1D) between spline knots (same in both directions)
Ny - number of splne knots in y-direction. if None -> Ny = Nx
Ymin - location (y coordinate) of bottom spline knot. if None -> Ymin = Xmin
reg - regularization parameter (Tikhonov).  if None -> truncated SVD is applied

"""
class BivariateCubicSpline():  # 
    def __init__(self, x, y, Nx, Xmin, Delta, Ny=None, Ymin=None, reg=None):
        assert np.array(x).ndim == np.array(y).ndim ==  1
        assert len(x) == len(y)
        self.Delta = Delta
        self.x = x; self.y = y
        if Ny == None: Ny = Nx
        if Ymin == None: Ymin = Xmin
        self.Xmin = Xmin; self.Ymin = Ymin; self.Nx=Nx; self.Ny=Ny
        self.Nsp = Nx*Ny  # total number of splines
        #xkn = Xmin + np.arange(Nx)*Delta  # x knot locations
        #ykn = Ymin + np.arange(Ny)*Delta  # y knot locations
        self.twoTOone = dict()  # dictionary from 2D index to 1D index of spline knots
        self.oneTOtwo = dict()  # dictionary from 1D index to 2D index of spline knots
        self.splcen   = dict()  # dictionary of spline centers (note the two key types, as per below)
        k = -1
        for kx in range(Nx):
            for ky in range(Ny):
                k += 1
                self.oneTOtwo[k] = (kx, ky)
                self.twoTOone[(kx, ky)] = k
                self.splcen[k] = (Xmin + kx*Delta, Ymin + ky*Delta)
                self.splcen[(kx, ky)] = (Xmin + kx*Delta, Ymin + ky*Delta)

        mat = np.zeros((len(x), self.Nsp)).astype('float32')  # the spline coefficients come from inverting this matrix
        for k in range(len(x)):
            mx = np.floor( (x[k] - Xmin)/Delta ).astype('int')  # x[k] is between mx and mx+1
            my = np.floor( (y[k] - Ymin)/Delta ).astype('int')  # y[k] is between my and my+1
            for ky in range(my-1, my+3):
                if (ky > -1) and (ky < Ny):
                    for kx in range(mx-1, mx+3):
                        if (kx > -1 ) and (kx < Nx):
                            kn = self.twoTOone[(kx,ky)]  # 1D knot index
                            dx = x[k] - self.splcen[(kx,ky)][0]
                            dy = y[k] - self.splcen[(kx,ky)][1]
                            mat[k, kn] = Bspline3_2D(dx, dy, Delta)
                        else:  pass  # condition on kx
                else: pass  # condition on ky

        #w is a vector of eigenvalues, the columns of V are the corresponding eigenvectors
        #V.dot(np.diag(w)).dot(V.T) = mat.T.dot(mat)
        #V.dot(V.T) is the identity
        #V.dot(np.inv(np.diag(w))).dot(V.T) = np.inv(mat.T.dot(mat))  if there are no zero eigenvalues
        w, V = np.linalg.eigh(mat.T.dot(mat))
        self.eigvals = w
        nz = len(w) - np.count_nonzero(w)
        if nz == 0:  #no zero eigenvalues
            print("The regression matrix has no zero eigenvalues, and the condition number is ",
                  np.max(w)/np.min(w), "  The eigenvalues are stored in .eigvals" )
        else:
            print("The condition matrix has ", nz, " zero eigenvalues.  The eigenvalues are stored in .eigvals")

        print("The pseudo-inverse is stored in .imat")
        if reg is None:
            print("You have chosen not to apply regularization.")
            if nz == 0:
                self.imat = V.dot(np.diag(1/w)).dot(V.T).dot(mat.T)
            else:
                self.imat = np.linalg.pinv(mat)
        else:
            print("You have chosen to apply regularization with parameter ", reg)
            self.imat = np.linalg.inv(mat.T.dot(mat) + reg*np.eye(self.Nsp)).dot(mat.T)
        return(None)

    #This gets the spline coefficients for a given set of observations
    # z is flattened array of observations
    def GetSplCoefs(self, z): 
        assert np.array(z).ndim == 1
        assert len(z) == len(self.x)
        return(self.imat.dot(z))

    #Given a set of spline coefs (see GetSplCoefs() ), this provides the interpolation.
    #xi - a flattened array (or list) of x coordinates
    #yi - a flattened array (or list) of y coordinates
    #coefs - a vector of spline coefficients (See .GetSplCoefs)
    def SplInterp(self, xi, yi, coefs):
        assert np.array(xi).ndim == np.array(yi).ndim == 1
        assert len(xi) == len(yi)
        out = np.zeros((len(xi), ))  # output values
        for k in range(len(xi)):
            mx = np.floor( (xi[k] - self.Xmin)/self.Delta ).astype('int')  # xi[k] is between mx and mx+1
            my = np.floor( (yi[k] - self.Ymin)/self.Delta ).astype('int')  # yi[k] is between my and my+1
            for ky in range(my-1, my+3):
                if (ky > -1) and (ky < self.Ny):
                    for kx in range(mx-1, mx+3):
                        if (kx > -1 ) and (kx < self.Nx):
                            kn = self.twoTOone[(kx,ky)]  # 1D knot index
                            dx = xi[k] - self.splcen[(kx,ky)][0]
                            dy = yi[k] - self.splcen[(kx,ky)][1]
                            out[k] += coefs[kn]*Bspline3_2D(dx, dy, self.Delta)
                        else:  pass  # condition on kx
                else: pass  # condition on ky
        return(out)





















