#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 18:03:51 2020

@author: rfrazin
"""

import numpy as np
from scipy.optimize import minimize as MIZE

#This function produces a dictionary that goes from a 1D index corresponding
#  to pixels in circular pupil to the 2D index of a square array containing
#  said pupil.
#N - square array is N-by-N
#pixrad - radius of pupil in pixels
#return_inv - also return inverse map
def PupilMap(N=50, pixrad=25, return_inv=False):
    x = (np.arange(0,N,1) - (N/2)) + 0.5
    X,Y = np.meshgrid(x,x)
    R = np.sqrt(X*X + Y*Y)
    pupil_array = np.float32(R<=pixrad)

    pmap = dict()
    if return_inv:
        ipmap = dict()
    pind = -1
    for k in range(N):
        for l in range(N):
            if pupil_array[k,l] > 0:
                pind += 1
                pmap[pind] = (l,k)
                if return_inv:
                    ipmap[(l,k)] = pind
    if return_inv:
        return((pmap, ipmap))
    else:
        return(pmap)

#This takes 1D vector representing pixels in a circular pupil and places them
#  into a square 2D array.
#pvec - 1D array of input pixels
#pmap - dict of mappings from PupilMap function
#N - output array is N-by-N
def EmbedPupilVec(pvec, pmap, N):
    assert pvec.ndim == 1
    square_array = np.zeros((N,N)).astype('complex')
    for k in range(len(pvec)):
        square_array[pmap[k]] = pvec[k]
    return(square_array)

#This extracts a 1D vector of circular pupil pixels from a square array.
#square_array - 2D (square) array of pupil pixels
#ipmap - inverse pupil map from PupilMap function called with return_inv flag
def ExtractPupilVec(square_array, ipmap):
    assert square_array.shape[0] == square_array.shape[1]
    N = square_array.shape[0]
    pvec = []
    for k in range(N):
        for l in range(N):
            if (l,k) in ipmap:
                pvec.append(square_array[l,k])
    return(np.array(pvec))


#np.sign has the annoying property that np.sign(0) = 0
#only works on scalars
#this returns a float, i.e., 1.0 or -1.0
def MySign(x):
    assert np.isscalar(x)
    if x >= 0. : return(1.0)
    else: return(-1.0)

#This is called FindPerfectAngle
#It returns the misfit metric and its derivatives
#v - vector of params to be fit: v[0]- amplitude, v[1] - std dev, v[2],v[3] - center
#y - vector of amplitudes
#angles - list of touples containing angles
def ChiSqGauss(v, y, angles):
    assert len(v) == 4
    assert len(y) == len(angles)
    assert len(angles[0]) == 2  # each angle has 2 components
    assert len(y) >= 4  # the function being fitted has 4 free params
    sign_0 = MySign(v[0])
    v0 = v[0]*sign_0  # removes positivity constraint
    ch = 0.
    dchd0 = 0.
    dchd1 = 0.
    dchd2 = 0.
    dchd3 = 0.
    for k in range(len(y)):
        num = -0.5*( (v[2] - angles[k][0])**2 + (v[3] - angles[k][1])**2 )
        Q =  num/(v[1]*v[1])
        yf = v0*np.exp(Q)
        diff = yf - y[k]
        ch += 0.5*diff**2  # chi-squared value
        dchd0 += sign_0*diff*np.exp(Q)
        dchd1 += diff*yf*(-2.*Q/v[1])
        dchd2 += -1.*diff*yf*(v[2] - angles[k][0])/(v[1]*v[1])
        dchd3 += -1.*diff*yf*(v[3] - angles[k][1])/(v[1]*v[1])
    return(ch, dchd0, dchd1, dchd2, dchd3)


#This finds the slope of pupil plane phase that centers the image of a point
#  source exactly over the center of a focal plane pixel.
#This works by fitting a Gaussian to the |field(angle)| function.
#(alpha_x, alpha_y) angle for initial phasor (must have len == 2)
#It is assumed that (alpha_x, alpha_y) are in units of radians, so that
#  (2pi, 0) will create a spot at 1 lambda/D away from the center.
#D - the complex-valued system matrix (focal pixes X pupil pixels)
#szp - linear size of pupil in pixels - needed for ExtractPupilVec
#ipmap inverse pupil map from PupilMap - needed for ExtractPupilVec
#returns a refinement of 'ang' that centers it on a focal plane pixel
def FindPerfectAngle(ang, D, szp, ipmap):
    assert len(ang) == 2
    nn = 6  # number of additional points used for fit
    rr = np.pi  # = lambda/D/2 change in angle
    s = np.linspace(-.5, .5, szp)
    (xx, yy) = np.meshgrid(s,s,indexing='xy'); del s
    af = []  # |field| values at desired pixel
    phi = []  # angles corresponding to values in 'af'
    for k in np.arange(-1, nn):
        ph = np.zeros((szp, szp))
        if k == -1:  # -1 corresponds to the initial guess
            a0 = 0.; a1 = 0.
        else:
            th = k*2.*np.pi/nn
            a0 = rr*np.cos(th); a1 = rr*np.sin(th)
        alpha0 = ang[0] + a0
        alpha1 = ang[1] + a1
        phi.append((alpha0, alpha1))
        for ky in range(szp):
            for kx in range(szp):
                ph[ky, kx] = alpha0*xx[ky, kx] + alpha1*yy[ky, kx]
        u = np.exp(ph)
        v = ExtractPupilVec(u, ipmap)  # pupil field vector
        w = D.dot(v)  #  focal plane field vector
        if k == -1:
            N = np.argmax(np.abs(w))  # pixel where |field| is max
        af.append(np.abs(w[N]))
    af = np.array(af)
    af /= np.max(af)

    mm = np.argmax(af)
    guess = [1., 1., phi[mm][0], phi[mm][1]]
    out = MIZE(ChiSqGauss, guess, args=(af, phi), method='CG', jac=True)
    perfang = (out['x'][2], out['x'][3])
    return(perfang)

