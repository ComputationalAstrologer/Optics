#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 16:06:24 2020

@author: rfrazin
"""

import numpy as np


nn = 100
s = np.linspace(-1., 1., nn)
[xx, yy] = np.meshgrid(s, s, indexing='xy')



#This returns a 2D function that is 1 inside the polygon and 0 otherwise
#mesh - tuple (two 2D arrays) output of np.meshgrid, e.g.,
#   s = np.linspace(-1.1,1.1,243);  mesh = np.meshgrid(s,s);
#center - (len=2) center of circle containing polygon
#radius - radius of circle in which polygon is inscribed
#rot0 - (degrees) allows rotation of polygon
#N - number of sides of polygon
def RegPolygon(mesh, radius=1., center=[0,0], rot0=0, N=6):
    assert isinstance(N, int) and N > 2
    assert mesh[0].ndim == 2 and mesh[1].ndim == 2
    assert mesh[1].shape == mesh[0].shape
    assert len(center) == 2
    norms = np.zeros((2,N))  # matrix of normal vectors
    sidedist = radius*np.cos(np.pi/N)  # distance from center to side
    angles = np.linspace(rot0, rot0 + 360.*(N-1)/N, N)  # angles of normal vectors
    #circle_points = angles - (angles[1] - angles[0])/2.  # (degrees) angles at which vertices intersect the unit circle
    #circle_points = np.mod(circle_points, 360.)
    #circle_points *= np.pi/180
    angles = np.mod(angles, 360.)
    constraintViolation = np.zeros((N,))  # 1. if in violation, 0. if not
    in_or_out = np.zeros(mesh[0].shape)

    for k in range(N):
        rm = PlaneRotationMatrix(angles[k], units='degrees')
        norms[:, k] = rm.dot(np.array([0.,-1.]))

    for k in range(mesh[0].shape[0]):
        for l in range(mesh[0].shape[1]):
            constraintViolation = 0.*constraintViolation
            x = mesh[0][k,l] - center[0]
            y = mesh[1][k,l] - center[1]
            for m in range(N):
                d = norms[0,m]*x + norms[1,m]*y
                if np.sign(d -sidedist) >= 0:
                    constraintViolation[m] = 1.
            if np.sum(constraintViolation) < 1.:
                in_or_out[k,l] = 1.
                
    return(in_or_out)

def PlaneRotationMatrix(theta, units='degrees'):
    assert units == 'degrees' or units == 'radians'
    if units == 'degrees' :
        s = theta*np.pi/180
        rm = np.array([[np.cos(s), np.sin(s)], [np.sin(-s), np.cos(s)]])
    return(rm)
