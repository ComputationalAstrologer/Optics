#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 16:06:24 2020

@author: rfrazin
"""

import numpy as np


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

#This returns square mask array that cuts a circle out of a square
#square_size - 1D size of square array (pixels)
#circ_diam - diameter of circle in pixels
#gtORgeq - boundary condition for circle, 'gt' = '>', 'geq' = '>='
def CutCircle(square_size, circ_diam, gtORgeq = 'gt'):
    assert circ_diam <= square_size
    mask = np.ones((square_size, square_size))
    s = np.linspace(-1, 1, square_size)
    x, y = np.meshgrid(s, s, indexing='xy')
    r = np.sqrt(x*x + y*y)
    rmax = circ_diam/square_size
    if gtORgeq == 'gt':  q = np.where(r > rmax)
    elif gtORgeq == 'geq':  q = np.where(r >= rmax)
    else:  assert False
    for k in range(len(q[0])):
        mask[q[0][k], q[1][k]] = 0.
    return(mask)

"""
This is a class that handles various functions for a regular hexagon tiling.
R - the radius of the circle that inscribes a hex segment
EdgeWidth - the width (within a given segment) of the transition to the neighboring segment (same units as R!)
  If set to 0, there is no transition.  If > 0, at the edge of the segment the height will be the 
  mean of the segment and its neighbor
Nx - the number of tiles in the x-direction
Ny - the number of tiles in the y-direction
"""


class HexagonTile():
    def __init__(self, R=1.0, EdgeWidth=0., Nx=3, Ny=3):
        assert EdgeWidth >= 0.
        self.R = R
        self.ew = EdgeWidth
        self.Nhex = Nx*Ny
        self.Index = {'xy2lin': {}, 'lin2xy': {}}  # 2d and 1d index maps
        self.Seg = {}  # info about individual segments, indexed by the values in self.Index['linind']
        
        #find the center for each hex in the grid
        if np.mod(Nx, 2) > 0:
            xind = np.arange(-(Nx-1)/2, (Nx-1)/2 + 1).astype('int')
        else:
            xind = np.arange(-Nx/2 + 1, Nx/2 + 1).astype('int')
        if np.mod(Ny, 2) > 0:
            yind = np.arange(-(Ny-1)/2, (Ny-1)/2 + 1).astype('int')
        else:
            yind = np.arange(-Ny/2 + 1, Ny/2 + 1).astype('int')
        idval = -1
        for kx in range(Nx):  # fill out the 'id' and 'revid' dicts
            for ky in range(Ny):
                idval += 1
                self.Index['xy2lin'][(xind[kx], yind[ky])] = idval
                self.Index['lin2xy'][idval] = (xind[kx], yind[ky])
                self.Seg[idval] = dict()

        for k in self.Index['lin2xy']:  # fill out the center values
            xyind = self.Index['lin2xy'][k]
            xcen = xyind[0]*3*R/2
            if np.mod(xyind[0], 2) > 0:  # note: e.g., np.mod(-5,2) = 1
                ycen = xyind[1]*3*R/2 + R
            else:
                ycen = xyind[1]*3*R/2
            self.Seg[k]['center'] = (xcen, ycen)
            self.Seg[k]['neighbors'] = np.zeros(6).astype('int')  # one neighbor for each sector

        for k in self.Index['lin2xy']:
            xyind = self.Index['lin2xy'][k]

            for ks in range(6):  # loop over sectors
                mx = xyind[0]
                my = xyind[1]


                if ks == 0:  # upper right
                        mx += 1
                        if np.mod(mx, 2) == 1:
                            my += 1
                elif ks == 1:  # above
                        my += 1
                elif ks == 2: # upper left
                        mx -= 1
                        if np.mod(mx, 2) == 1:
                            my += 1
                elif ks == 3: # lower left
                        mx -= 1
                        if np.mod(mx,2) == 0:
                            my -= 1
                elif ks == 4:  # below
                        my -= 1
                elif ks == 5:          # lower right
                    mx += 1
                    if np.mod(mx,2) == 0:
                        my -= 1
                else:
                    print('case error');
                    assert False
                if (mx, my) in self.Index['xy2lin']:
                    self.Seg[k]['neighbors'][ks] = self.Index['xy2lin'][(mx, my)]
                else:  # no neighbor for that sector
                    self.Seg[k]['neighbors'][ks] = -1  # no neighbor code
        return(None)