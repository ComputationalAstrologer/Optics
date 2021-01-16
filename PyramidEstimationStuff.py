#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 10:23:51 2021

@author: rfrazin
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.signal import decimate as dc
import PupilMap as PM
import LabPyr as LP


szp = 51  # linear size of pupil plane
prad = 15  # radius of entrance pupil within pupil plane

parms = dict()
parms['beam_diameter'] = 1.e3 # input beam diameter (microns)
parms['wavelength'] = 0.6328 # wavelength (microns)
parms['pyramid_slope_deg'] = 0.4 # slope of pyramid faces relative to horizontal (degrees)
parms['n_starting_points'] = szp  # number of resolution elements in initital beam diameter
parms['f1'] = 40.e3 # focal length of lens #1 (focuses light on pyramid tip)
parms['D_e_to_l1'] = 2*parms['f1'] # nominal distance from entrance pupil to lens1 (microns)
parms['lens1_fill_diameter'] = parms['beam_diameter']*1.5  #  computational beam width at lens#1.  This matters!
parms['D_l1_to_pyr'] = parms['f1'] # nominal distance from lens1 to pyramid tip
parms['D_l1_to_detector'] = 2*parms['f1']  # distance from lens to detector in 4f system
parms['apex_diam'] = 20 # diameter of focal spot at pyramid apex in units of lambda_over_D (set by stop)
parms['beam_diameter_at_pyramid'] = parms['apex_diam']*parms['f1']*parms['wavelength']/parms['beam_diameter']
parms['detector_width'] = None # width of detector
parms['max_chirp_step_deg'] = 120  # maximum allowed value (degrees) in chirp step for Fresnel prop
parms['max_lens_step_deg'] = 20 # maximum step size allowed for lens phase screen
parms['max_plane_step_deg'] = 20  # (degrees) maximum phase step allowed for planar phase screen
parms['max_pyramid_phase_step'] = 20  # maximum step (degrees) allowed for pyramid phase ramp
parms['max_fractional_change'] = 1.e-5  # maximum change tolerated in certain finite differences
parms['interp_style'] = 'linear'  # type of 2d interpolator for field resampling

WOM = LP.WorkingOpticalModels(params=parms)


#create initial pupil plane
pmap, ipmap = PM.PupilMap(N=szp, pixrad=prad, return_inv=True)
nppix = len(pmap)

MakeSysmat = False
if MakeSysmat:
    #create the system matrix for the WFS
    sysmat = np.zeros((175*175, nppix)).astype('complex')
    lng = 350
    xu = np.linspace(- parms['beam_diameter']/2, parms['beam_diameter']/2, szp)
    for k in range(nppix):
        if np.mod(k, 50) == 0: print(k, " of 709")
        #set up field
        u = np.zeros((nppix,)).astype('complex')
        u[k] = 1.0 
        uu =  PM.EmbedPupilVec(u, pmap, szp)
    
        result = WOM.PropF4ReflectiveNSidedPyramid(g=uu, x=xu, SlopeDeviations=None, FaceCenterAngleDeviations=None, pyr_dist_error=0.,
                                          N=3, NominalSlope=None, return_derivs=False, print_stuff=False)
        df = result['field'][45:45+lng, 75:75+lng]
        #make it smaller.  more filtering destroys the phase information
        dfr = dc(df, 2, ftype='fir', axis=0, zero_phase=True)
        dfr = dc(dfr, 2, ftype='fir', axis=1, zero_phase=True)
        dfr /= 2.e-6
        sysmat[:,k] = dfr.reshape(175**2,)
    print('sysmat is done.')

LoadSysmat = True
if LoadSysmat:
    fn = 'Pickles/PyWFSsysmat3sides709_175x175.pickle'
    fid = open(fn, 'rb')
    sysmat = pickle.load(fid); fid.close()


















