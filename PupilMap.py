#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 18:03:51 2020

@author: rfrazin
"""

import numpy as np
import matplotlib.pyplot as plt
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

#this evaluates a chi-squared
#v - input vector (center_x, center_y, 2xVariance, amplitude)
#  center_x and center_y are pixel index values consisten with 'img' indicies
#img - 2D image over which the Gaussian is compared
def ChiSq2DGauss(v, img, return_derivs=True, debug=False):
    vv = 1.*np.array(v)  # deep copy
    assert len(vv) == 4
    assert img.ndim == 2
    ny = img.shape[0]; nx = img.shape[1]
    s2 = MySign(vv[2])  # this removes positivity constraint
    vv[2] *= s2
    s3 = MySign(vv[3])  # this removes the positivity constraint
    vv[3] *= s3
    c = 0.  # cost
    if debug: yy = np.zeros(img.shape)
    for ky in range(ny):
        for kx in range(nx):
            arg = (vv[0] - ky)**2 + (vv[1] - kx)**2
            arg *= -1./vv[2]
            ym = vv[3]*np.exp(arg)
            c += 0.5*(ym - img[ky, kx])**2
            if debug: yy[ky,kx] = ym
    if debug:
        #plt.figure(); plt.imshow(yy ); plt.colorbar(); plt.title('yy');
        #plt.figure(); plt.imshow(img); plt.colorbar(); plt.title('img');
        plt.figure(); plt.imshow(yy-img); plt.colorbar(); plt.title('diff')
    if not return_derivs:
        return(c)
    #calculate derivatives
    dcd0 = 0.
    dcd1 = 0.
    dcd2 = 0.
    dcd3 = 0.
    for ky in range(ny):
        for kx in range(nx):
            arg = (vv[0] - ky)**2 + (vv[1] - kx)**2
            arg *= -1./vv[2]
            ym = vv[3]*np.exp(arg)
            d = (ym - img[ky, kx])
            dcd1 += -1.*d*ym*2.*(vv[1] - kx)/vv[2]
            dcd0 += -1.*d*ym*2.*(vv[0] - ky)/vv[2]
            dcd2 += -1.*s2*d*ym*arg/vv[2]
            dcd3 += d*s3*np.exp(arg)
    return(c, np.array([dcd0, dcd1, dcd2, dcd3]))

#the amplitude is fixed in this one
def ChiSq2DGaussConstAmp(v, img, return_derivs=True, amp=1.0, debug=False):
    vv = 1.*np.array(v)  # deep copy
    assert len(vv) == 3
    assert img.ndim == 2
    ny = img.shape[0]; nx = img.shape[1]
    s2 = MySign(vv[2])  # this removes positivity constraint
    vv[2] *= s2
    c = 0.  # cost
    if debug: yy = np.zeros(img.shape)
    for ky in range(ny):
        for kx in range(nx):
            arg = (vv[0] - ky)**2 + (vv[1] - kx)**2
            arg *= -1./vv[2]
            ym = amp*np.exp(arg)
            c += 0.5*(ym - img[ky, kx])**2
            if debug: yy[ky,kx] = ym
    if debug:
        #plt.figure(); plt.imshow(yy ); plt.colorbar(); plt.title('yy');
        #plt.figure(); plt.imshow(img); plt.colorbar(); plt.title('img');
        plt.figure(); plt.imshow(yy-img); plt.colorbar(); plt.title('diff')
    if not return_derivs:
        return(c)
    #calculate derivatives
    dcd0 = 0.
    dcd1 = 0.
    dcd2 = 0.
    for ky in range(ny):
        for kx in range(nx):
            arg = (vv[0] - ky)**2 + (vv[1] - kx)**2
            arg *= -1./vv[2]
            ym = amp*np.exp(arg)
            d = (ym - img[ky, kx])
            dcd1 += -1.*d*ym*2.*(vv[1] - kx)/vv[2]
            dcd0 += -1.*d*ym*2.*(vv[0] - ky)/vv[2]
            dcd2 += -1.*s2*d*ym*arg/vv[2]
    return(c, np.array([dcd0, dcd1, dcd2]))


def FindPerfectAngle(ang, D, szp, ipmap):
    assert len(ang) == 2
    gridl = 3*np.pi  # linear size of angle grid
    ng = 5  # angle grid has shape = (ng, ng)
    ndpix = int(np.round(np.sqrt(D.shape[0])))  # linear dimension of detector
    s = np.linspace(-.5, .5, szp)  # pupil 1D coord
    g = np.linspace(-gridl/2, gridl/2, ng); delta_g = g[1] - g[0]
    (xx, yy) = np.meshgrid(s,s,indexing='xy'); del s  # pupil grid
    (gx, gy) = np.meshgrid(g,g,indexing='xy'); del g  # angle grid
    gim = np.zeros((ng, ng))
    ph = np.zeros((szp, szp))
    #find the pixel we want to be centered on
    for ky in range(szp):
        for kx in range(szp):
            ph[ky, kx] = ang[0]*yy[ky, kx] + ang[1]*xx[ky, kx]
    u = np.exp(1j*ph)
    v = ExtractPupilVec(u, ipmap)  # pupil field vector
    w = D.dot(v)  #  focal plane field vector
    M = np.argmax(np.abs(w))
    dpix = np.unravel_index(M, (ndpix, ndpix), 'C')
    for my in range(ng):
        for mx in range(ng):
            sang = ( ang[0] + gy[my, mx], ang[1] + gx[my,mx] )  #sky angle
            for ky in range(szp):
                for kx in range(szp):
                    ph[ky, kx] = sang[0]*yy[ky,kx] + sang[1]*xx[ky,kx]
            u = np.exp(1j*ph)
            v = ExtractPupilVec(u, ipmap)
            w = D.dot(v)
            gim[my, mx] = np.abs(w[M])

    #fit a 2D Gaussian to gim to find the center
    guess = [ng/2., ng/2., 3.]
    out = MIZE(ChiSq2DGaussConstAmp, guess, args=(gim), method='CG', jac=True, options={'gtol': 0.01})
    if not out['success']:
        plt.figure(); plt.imshow(gim); plt.colorbar();
        print()
        print("out = ", out)
        print("ang = ", ang)

    pang = np.array((ang[0] + out['x'][0]*delta_g, # perfect angle
                     ang[1] + out['x'][1]*delta_g))
    return(pang, dpix, out['success'])


#this works by finding the center of the images and regressing on the centers
def DoNotUseFindPerfectAngle(ang, D, szp, ipmap):
    assert len(ang) == 2
    assert D.ndim == 2
    nn = 4 # number of additional angles
    nd = int(np.round(np.sqrt(D.shape[0])))
    rr = 1.9*np.pi  # magnitude of change in angle for regression points
    s = np.linspace(-.5, .5, szp)
    (xx, yy) = np.meshgrid(s, s, indexing='xy'); del s
    centers = []; angles = []
    ang0 = np.array([0.,0.])
    ph = np.zeros((szp, szp))
    out = None
    for k in np.arange(-1,nn):  #-1 corresponds to uperturbed center
        if k == -1:
            ang0[0] = ang[0]; ang0[1] = ang[1]
        else:
            th = 2.*np.pi*(.1 + k/nn)
            ang0[0] = ang[0] + rr*np.cos(th)
            ang0[1] = ang[1] + rr*np.sin(th)
        for ky in range(szp):
            for kx in range(szp):
                ph[ky, kx] = ang0[0]*yy[ky, kx] + ang0[1]*xx[ky, kx]
        u = np.exp(1j*ph)
        v = ExtractPupilVec(u, ipmap)  # pupil field vector
        w = D.dot(v)  #  focal plane field vector
        (my, mx) = np.unravel_index(np.argmax(np.abs(w)), (nd, nd), order='C')
        if k == -1:
            mymx = (my, mx)  # desired center
        W = w.reshape(nd,nd)
        if False:  # k > -1:
            guess = [my, mx, out['x'][2], out['x'][3]]
        else:
            guess = [my, mx, 3., np.max(np.abs(w))]
        opts = {'gtol': 0.1}
        out = MIZE(ChiSq2DGauss, guess, args=(np.abs(W)), method='CG', jac=True, options=opts)
        if out['success']:
            centers.append((out['x'][0], out['x'][1]))
            angles.append((ang0))
        else:  pass
            #print('FindPerfectAngle: Optimization Failure.')
            #plt.figure(); plt.imshow(np.abs(W)); 
            #plt.title("abs(W)"); plt.colorbar();
            #print("angles = ", ang0)
            #print("out = "); print(out);
            #assert False

    # given angles and centers, do linear fits 
    angles = np.array(angles)
    centers = np.array(centers)
    px = np.polyfit(angles[0,:], centers[0,:], 1)
    py = np.polyfit(angles[1,:], centers[1,:], 1)
    pang = np.array((px[0]*mymx[0] + px[1], py[0]*mymx[1] + py[1] ))  # perfect angle
    return(pang)


#---  Do not use.  It doesn't work very well.
#This finds the slope of pupil plane phase that centers the image of a point
#  source exactly over the center of a focal plane pixel.
#This works by fitting a Gaussian to the |field(angle)| function.
#(alpha_x, alpha_y) angle for initial phasor (must have len == 2)
#It is assumed that (alpha_x, alpha_y) are in units of radians, so that
#  (2pi, 0) will create a spot at 1 lambda/D away from the center.
#D - the complex-valued system matrix (focal pixels X pupil pixels)
#szp - linear size of pupil in pixels - needed for ExtractPupilVec
#ipmap inverse pupil map from PupilMap - needed for ExtractPupilVec
#returns a refinement of 'ang' that centers it on a focal plane pixel
def DoNotUseXXFindPerfectAngle(ang, D, szp, ipmap):
    assert len(ang) == 2
    nn = 7  # number of additional points used for fit
    rr = 1.9*np.pi  # = lambda/D/2 change in angle
    s = np.linspace(-.5, .5, szp)
    (xx, yy) = np.meshgrid(s,s,indexing='xy'); del s
    itcount = -1
    while True:
        af = []  # |field| values at desired pixel
        phi = []  # angles corresponding to values in   'af'
        itcount += 1
        if itcount == 0:
            ang0 = (ang[0], ang[1])
        for k in np.arange(-1, nn):
            ph = np.zeros((szp, szp))
            if k == -1:  # -1 corresponds to the initial guess
                a0 = 0.; a1 = 0.
            else:
                th = k*2.*np.pi/nn + np.pi/11
                a0 = rr*np.cos(th); a1 = rr*np.sin(th)
                if itcount > 0:
                    a0 /= 2.; a1 /= 2.
            alpha0 = ang0[0] + a0
            alpha1 = ang0[1] + a1
            phi.append((alpha0, alpha1))
            for ky in range(szp):
                for kx in range(szp):
                    ph[ky, kx] = alpha0*yy[ky, kx] + alpha1*xx[ky, kx]
            u = np.exp(1j*ph)
            v = ExtractPupilVec(u, ipmap)  # pupil field vector
            w = D.dot(v)  #  focal plane field vector
            if k == -1:
                N = np.argmax(np.abs(w))  # pixel where |field| is max
            af.append(np.abs(w[N]))
        af = np.array(af)
        if itcount == 0:
            normf = np.max(af)
        af /= normf

        if itcount == 0:
            mm = np.argmax(af)
            guess = [1.5, 2., phi[mm][0], phi[mm][1]]
        out = MIZE(ChiSq1DGauss, guess, args=(af, phi), method='CG', jac=True)
        if out['success']:
            ang0 = (out['x'][0], out['x'][1])
            guess = out['x']
            if itcount > 1: break
        else:
            ang0 = (ang[0] + np.random.rand()/np.pi/2., ang[1] + np.random.randn()/np.pi/2. )
            if itcount > 2:
                break
    perfang = (out['x'][2], out['x'][3])  # perfang is the optimal angle (hopefully)

    #find the corresponding detector pixel
    for ky in range(szp):
        for kx in range(szp):
            ph[ky, kx] = perfang[0]*yy[ky, kx] + perfang[1]*xx[ky, kx]
    u = np.exp(1j*ph)
    v = ExtractPupilVec(u, ipmap)
    mm = np.argmax(np.abs(D.dot(v)))  # find pixel of max |field| on detector

    #print()
    #print(ang)
    #print(out)

    return( (perfang, mm, out['success']) )

