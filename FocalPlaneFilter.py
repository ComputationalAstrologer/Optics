#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 13:33:09 2020
@author: rfrazin

This code performs Fourier filtering of a collimated beam by bringing it
to a focus, applying a field stop and then recollimating with a second
lens.  If the focal length of the 2nd lens is smaller than the first, then
the beam diamter is reduced.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import decimate as dc
import FourierOptics
from PolygonTools import CutCircle

fparams= dict()
fparams['beam diameter'] = 1.e4 # input beam diameter (microns)
fparams['wavelength'] = 1.1 # wavelength (microns)
fparams['f1'] = 500.e3 # focal length of lens #1 (focuses light on plane with stop)
fparams['stop_diam_lam/D'] = 16.  #  diameter (not radius!) of field stop in lambda/D units
fparams['max_chirp_step_deg'] = 90  # maximum allowed value (degrees) in chirp step for Fresnel prop
fparams['max_lens_step_deg'] = 20 # maximum step size allowed for lens phase screen
fparams['interp_style'] = 'linear'  # type of 2d interpolator for field resampling
fparams['resize factor'] = 1.2 # > 1 implies image is smaller than original

#Note that the mask = CutCircle(51,52) has 1976 nonzero values.
#g - input field, g.shape = (m,m)
#params - a dict()
#len_out - size of output array (1D)
def FocalPlaneFilter(g, params=fparams, len_out=52):
    F = FourierOptics.FourierOptics(params=params)
    if g.shape[0] != 52:
        print("This filter is desinged for a 52x52 input field.")
    assert g.shape[0] == g.shape[1]
    m = g.shape[0]
    diam0 = params['beam diameter']
    fl2 = params['f1']/params['resize factor']
    lod = params['f1']*params['wavelength']/params['beam diameter']  # lambda/D at focal plane
    stop_diam = lod*params['stop_diam_lam/D']

    x = np.linspace(-diam0/2, diam0/2, m)
    #apply lens1
    g1, x1 = F.ApplyThinLens2D(g, x, [0,0], params['f1'], return_derivs=False)
    #prop to focal plane, this cuts field at the 
    g2, x2 = F.ConvFresnel2D(g1, x1, stop_diam, params['f1'], set_dx=5.)
    #prop to lens2
    g3, x3 = F.ConvFresnel2D(g2, x2, params['beam diameter'], fl2, set_dx=True)
    g3 /= np.max(np.abs(g3))
    #apply lens2
    g4, x4 = F.ApplyThinLens2D(g3, x3, [0,0], fl2, set_dx=4.80769)

    #bin down to needed ouputsize
    if len_out != 52 or len(x4) != 2080:
        print("len_out = ", len_out, ", len(x4) = ", len(x4), ", dx = ", x4[1]-x4[0])
        print("this decimation step only works when the above values are 52 and 2912, respectively.")
        plt.figure(); plt.imshow(np.log10(1.e-2*np.max(np.abs(g4)) + np.abs(g4))); plt.colorbar();
        assert False
    g5 = dc(g4, 5, ftype='fir', axis=0, zero_phase=True)
    g5 = dc(g5, 8, ftype='fir', axis=0, zero_phase=True)
    g5 = dc(g5, 5, ftype='fir', axis=1, zero_phase=True)
    g5 = dc(g5, 8, ftype='fir', axis=1, zero_phase=True)
    return(g5)


#This builds the spatial filter for the circle containing 1976 pixels
#output is a sz-by-sz complex array
def BuildFilterMatrix(sz=1976):
    mat = np.zeros((sz, sz)).astype('complex')  # output matrix
    sb = 52; ss = 51
    mask = CutCircle(sb,ss)
    assert np.sum(mask) == sz

    k = -1
    for kk in range(sb):
        for kl in range(sb):
            if mask[kl, kk] > 0:
                k += 1
                if np.mod(k,10) == 0:
                    print("k = ", k, "of ", sz)
                g = np.zeros((sb,sb))
                g[kl,kk] = 1.
                h = FocalPlaneFilter(g, params=fparams,len_out=52)  # h is a 52x52 array
                m = -1
                for mk in range(sb):
                    for ml in range(sb):
                        if mask[ml,mk] > 0:
                            m += 1
                            mat[k,m] = h[ml, mk]
    print("done.  k  = ", k, ", m = ", m)
    return(mat)
