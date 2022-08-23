#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 16:06:24 2020

@author: rfrazin
"""

import numpy as np
import matplotlib.pyplot as plt
from os.path import join as pathjoin
from astropy.io import fits


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
HeightFilePath - path to the directory containing HeightFile.  None corresponds to "./"
"""

#=======================Start HexagonTile Class================================
class HexagonTile():
    def __init__(self, R=1.0, Nx=3, Ny=3, EdgeWidth=0., HeightFile=None, HeightFilePath=None):
        assert EdgeWidth >= 0. and EdgeWidth < np.sqrt(3)*R/2
        self.R = R
        self.ew = EdgeWidth
        self.Nhex = Nx*Ny
        self.HeightFile = HeightFile
        self.HFPath = HeightFilePath
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
                ycen = xyind[1]*np.sqrt(3)*R + R*np.sqrt(3)/2
            else:
                ycen = xyind[1]*np.sqrt(3)*R
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
        pt = np.array(point)
        #first, figure out which (if any) segment pt belongs
        rpt = np.sqrt(pt[0]**2 + pt[1]**2)
        if rpt >= self.Rmax: return(0.0)

        edgedist = self.R*np.sqrt(3)/2  # this is the distance from the segment center to the center of one of its edges
        #determine grid spacing parameter.  this only works if the hex segments have 2 edged parallel to the x-axis.
        k = self.Index['xy2lin'][(0,0)]; c0 = self.Seg[k]['center']; k = -1
        dgridX = self.R*3/2; dgridY = self.R*np.sqrt(3)

        #determine the 'pixel number' of pt as if the grid were square
        px = np.rint( (pt[0] - c0[0])/dgridX ).astype('int')
        py = np.rint( (pt[1] - c0[1])/dgridY ).astype('int')

        #see which segment (if any) pt is in
        searchind = [(0,  0), ( 0, -1), ( 1, -1),  # put (0,0) first, since that is the most likely
                     (-1, 0), (-1, -1), ( 1,  0),
                     (-1, 1), ( 0,  1), ( 1,  1)]
        searchind = np.array(searchind) + np.array((px,py))
        seg = -1  # corresponds to pt not being inside any segment
        for si in searchind:
            if (si[0], si[1]) in self.Index['xy2lin']:
                k = self.Index['xy2lin'][(si[0], si[1])]
            else: 
                continue  # (si[0], si[1]) is not a valid segment id
            #  pt is inside segment k if it is not too far from the center within its sector
            xc = pt[0] - self.Seg[k]['center'][0]
            yc = pt[1] - self.Seg[k]['center'][1]
            dc = np.sqrt(xc*xc + yc*yc)  # distance from segment center
            if dc > self.R:
                continue
            ang = np.arctan2(yc, xc)
            if ang < 0: ang += 2*np.pi
            sector = np.floor( ang/(np.pi/3) ).astype('int')
            cbisang = np.cos(ang - sector*np.pi/3 - np.pi/6)  # cosine of the angle between center-point line and sector bisector line
            dproj = dc*cbisang  # projected distance along sector bisector
            if dproj > edgedist:
                seg = -1
            else:  # it is inside segment k!  But is it within the edge region?
                seg = k
                if dproj <= edgedist - self.ew:  # not within the edge region
                    return(self.Seg[seg]['height'])
                else: # within the edge region
                    nabe = self.Seg[seg]['neighbors'][sector]
                    if nabe == -1:
                        nheight = 0. # no neighbor
                    else:
                        nheight = self.Seg[nabe]['height']  # height of neighbor
                mheight = 0.5*(nheight + self.Seg[seg]['height'])  # height at sector boundary
                dq = dproj - (edgedist - self.ew)
                height = self.Seg[seg]['height'] + (dq/self.ew)*(mheight - self.Seg[seg]['height'])
                return(height)
        if seg == -1:
            return(0.0)

    #This reads in the HeightFile to get the segment heights or assigns random heights
    def GetHexHeights(self):
        if self.HeightFile is None:  # fill this out randomly
            for k in self.Index['lin2xy']:
                self.Seg[k]['height'] = np.random.rand()
            return(None)
        elif self.HeightFile == 'cmcMask_v6_200nm_sag.fits':
            if self.HFPath is None:
                fname = self.HeightFile
            else:
                fname = pathjoin(self.HFPath, self.HeightFile)
            hm = fits.getdata(fname)
            assert hm.shape == (2056, 2056)
            s = np.linspace(-205.6, 205.6, 2056)  # 1D spatial coord (um), 0.2 um per pixel
            dgrid = s[1] - s[0]  # grid spacing

            for k in self.Index['lin2xy']:
                hc = self.Seg[k]['center']  # hex center
                px = np.rint( (hc[0] - s[0])/dgrid ).astype('int')
                py = np.rint( (hc[1] - s[0])/dgrid ).astype('int')
                py = hm.shape[1]-1 - py  # reverse the up/down orientation
                if (px < 0) or (px > hm.shape[0]-1) or (py < 0) or (py > hm.shape[1]-1):
                    self.Seg[k]['height'] = 0.0
                else:
                    self.Seg[k]['height'] = hm[py, px]
        else:
            print('I dont know how to handle the file: ', self.HeightFile)
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
#res - a resolution scaling factor
def HexTileExample(res=100, N=5, R=1/3, EdgeWidth=0., HeightFile=None, HeightFilePath=None):  # full width of grid is about 2*R*N
    HT =  HexagonTile(R=R, Nx=N, Ny=N, EdgeWidth=EdgeWidth, HeightFile=HeightFile, HeightFilePath=HeightFilePath)
    s = np.linspace(-1.1*R*N, 1.1*R*N, res*N)
    ext = [min(s), max(s), min(s), max(s)]
    (xx, yy) = np.meshgrid(s,-s)
    hmap = np.zeros(xx.shape)
    for kx in range(xx.shape[0]):
        for ky in range(xx.shape[1]):
            hmap[kx,ky] = HT.CalcPointHeight((xx[kx,ky], yy[kx, ky]))
    plt.figure();
    plt.imshow(hmap, extent=ext); plt.colorbar(); plt.xlabel('x'); plt.ylabel('y'); plt.title('HexagonTile Example (height map)');
    return((HT, hmap))

#This is for the specific case of cmcMask_v6_200nm_sag.fits
#  the width of the file is 2056*(0.2 um) = 411.2 um
#N - grid is N-by-N segments.  N=27 is needed to cover the whole grid, but a smaller number can be used
#res - a parameter that is proportional to the number of points in the height map
#    res=40 is good enough to see the hex segments, but resolving the transition between segments
#    requires res > 400 (800 is nice!), but the run time is proportional to res^2.  One option is reduce N and 
#    only model the central part of the mask.  Note that N=5, res=800 takes about 8.5 min on my machine
def cmcMask_v6(N=27, res=40):
    R = 15.3/np.sqrt(3)  # [um] edge-to-opposite edge distance is 15.3 um
    EdgeWidth = 0.1 # [um]
    fn = 'cmcMask_v6_200nm_sag.fits'
    pth = './'
    (HT, hmap) = HexTileExample(res=res, N=N, R=R, EdgeWidth=EdgeWidth, HeightFile=fn, HeightFilePath=pth)
    return((HT, hmap))

