#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 16:06:24 2020

@author: rfrazin
"""

import numpy as np
import matplotlib.pyplot as plt


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
Nx - the number of tiles in the x-direction
Ny - the number of tiles in the y-direction
EdgeWidth - the width (within a given segment) of the transition to the neighboring segment (same units as R!)
  If set to 0, there is no transition.  If > 0, at the edge of the segment the height will be the 
  mean of the segment and its neighbor
HeightFile - filename containing height map.  If None, random heights will be assigned
"""

#=======================Start HexagonTile Class================================
class HexagonTile():
    def __init__(self, R=1.0, Nx=3, Ny=3, EdgeWidth=0., HeightFile=None):
        assert EdgeWidth >= 0. and EdgeWidth < np.sqrt(3)*R/2
        self.R = R
        self.ew = EdgeWidth
        self.Nhex = Nx*Ny
        self.HeightFile = HeightFile
        self.Index = {'xy2lin': {}, 'lin2xy': {}}  # 2d and 1d index maps
        self.Seg = {}  # info about individual segments, indexed by the values in self.Index['linind']
        self.Rmax = 0.  # maximum radius to be on grid
        
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
            if self.Rmax < np.sqrt(xcen*xcen + ycen*ycen):
                self.Rmax = np.sqrt(xcen*xcen + ycen*ycen)
            self.Rmax = self.R + self.Rmax

            #initialize the other dicts with info about each segment
            self.Seg[k]['neighbors'] = np.zeros(6).astype('int')  # one neighbor for each sector
            self.Seg[k]['border'] = False  # true for border pixels at edges of the mask
            self.Seg[k]['height'] = 0.  # height of the segment (excluding the edge)

        self.GetHexHeights()  # get the height of each segment

        #fill out the neighbors dict
        for k in self.Index['lin2xy']:
            xyind = self.Index['lin2xy'][k]

            for ks in range(6):  # loop over sectors
                mx = xyind[0]
                my = xyind[1]
                if ks == 0:  # upper right
                     if np.mod(mx, 2) > 0: my += 1  # this step must be done before modifying mx!
                     mx += 1
                elif ks == 1:  # above
                     my += 1
                elif ks == 2: # upper left
                     if np.mod(mx, 2) > 0: my += 1
                     mx -= 1
                elif ks == 3: # lower left
                     if np.mod(mx,2) < 1: my -= 1
                     mx -= 1
                elif ks == 4:  # below
                     my -= 1
                elif ks == 5:          # lower right
                     if np.mod(mx,2) < 1: my -= 1
                     mx += 1
                else:
                     print('case error');
                     assert False
                if (mx, my) in self.Index['xy2lin']:
                     self.Seg[k]['neighbors'][ks] = self.Index['xy2lin'][(mx, my)]
                else:  # no neighbor for that sector
                     self.Seg[k]['neighbors'][ks] = -1  # no neighbor code
                     self.Seg[k]['border'] = True  # if it has less than 6 neighbors, it is a border pixel
        return(None)

    #This calculates the height corresponding to a given point, including the edge
    #point - any 2-element array-like object, e.g., (x,y), [x,y], where x and y
    #  specify the point's location.  Same units as the 'R' and 'EdgeWidth' 
    #  arguments of the constructor
    #Unless a point is closest to the center of a border segment, it is inside the segment
    #  with the closest center (due to convexity)
    #I WILL VECTORIZE THIS LATER
    def CalcPointHeight(self, point):
        hs = self.R*np.sqrt(3)/2  # distance from segment center to a side
        height = 0.
        pt = np.array(point)
        if np.sqrt(pt[0]**2 + pt[1]**2) >= self.Rmax:  # point is outside the grid
            return(height)
        #find the segment center to which pt is closest
        segid = -1  #
        d2 = self.Rmax
        for k in self.Index['lin2xy']:
            ct = self.Seg[k]['center']
            dt2 = np.sqrt((pt[0] - ct[0])**2 + (pt[1] - ct[1])**2)
            if dt2 < d2: 
                d2 = dt2
                segid = k  # the point is inside segment sedig, unless segment segid is on the border

        #first, figure out which sector it's in - needed for border pixels and treating edges
        xc = pt[0] - self.Seg[segid]['center'][0]
        yc = pt[1] - self.Seg[segid]['center'][1]
        dc = np.sqrt(xc*xc + yc*yc)
        ang = np.arctan2(yc, xc)
        if ang < 0: ang += 2*np.pi
        sector = np.floor( ang/(np.pi/3) ).astype('int')
        
        #now determine where the point is in relationship to the sector boundary
        cbisang = np.cos(ang - sector*np.pi/3 - np.pi/6)  # cosine of the angle between center-point line and sector bisector line
        dproj = dc*cbisang  # projected distance along sector bisector
        if dproj > hs:
            height = 0.  # point is outside of segment, which only happens on border segments because segid is the one closest to the point
        elif dproj <= (hs - self.ew):  # point is in interior of segment
            return(self.Seg[segid]['height'])
        else: #complicated edge region case!
            nabe = self.Seg[segid]['neighbors'][sector]
            dq = dproj - (hs - self.ew)
            if nabe == -1:  # no neighbor
                nheight = 0.
            else:
                nheight = self.Seg[nabe]['height']  # height of neighbor
                mheight = 0.5*(nheight + self.Seg[segid]['height'])  # height at sector boundary
                height = self.Seg[segid]['height'] + (dq/self.ew)*(mheight - self.Seg[segid]['height'])
        return(height)


    #This reads in the HeightFile or assigns random heights
    def GetHexHeights(self):
        if self.HeightFile is None:  # fill this out randomly
            for k in self.Index['lin2xy']:
                self.Seg[k]['height'] = np.random.rand()
        else:
            print('HeightFile kwarg under construction!')
            assert False
        return(None)

    #This debugging code plots the hex index numbers on a 2D grid.
    def PlotIndexMap(self):
        plt.figure();
        for k in self.Index['lin2xy']:
            c = self.Seg[k]['center']
            plt.plot(c[0],c[1],'rh')
        for k in self.Index['lin2xy']:
            c = self.Seg[k]['center']
            plt.text(c[0], c[1], str(k))
        return(None)

#================= end of HexagonTile Class  ==================================

#This provides an example of using the HexagonTile Class
def HexTileExample(N=5, R=1/3):  # full width of grid is about 2*R*N
    EdgeWidth = 0.
    HT =  HexagonTile(R=R, Nx=N, Ny=N, EdgeWidth=EdgeWidth, HeightFile=None)
    s = np.linspace(-1.1*R*N, 1.1*R*N, 100*N)
    ext = [min(s), max(s), min(s), max(s)]
    (xx, yy) = np.meshgrid(s,-s)
    hmap = np.zeros(xx.shape)
    for kx in range(xx.shape[0]):
        for ky in range(xx.shape[1]):
            hmap[kx,ky] = HT.CalcPointHeight((xx[kx,ky], yy[kx, ky]))
    plt.figure();
    plt.imshow(hmap, extent=ext); plt.colorbar(); plt.xlabel('x'); plt.ylabel('y'); plt.title('HexagonTile Example (height map)');
    return((HT, hmap))
    