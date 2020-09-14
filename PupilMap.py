#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 18:03:51 2020

@author: rfrazin
"""

import numpy as np

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
    square_array = np.zeros((N,N))
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