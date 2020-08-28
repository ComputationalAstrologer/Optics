#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 16:18:28 2020

@author: rfrazin
"""

import numpy as np
import matplotlib.pyplot as plt
import FourierOptics
import LabPyr
import PolygonTools as PT

samparams = dict()
samparams['beam_diameter'] = 0.90625e3 # input beam diameter (microns) - this is 1/8 of 7.25 mm
samparams['wavelength'] = 1.65 # wavelength (microns)
samparams['indref'] = None # pyramid index of refraction
samparams['pyramid_slope_deg'] = 0.5# 10.5  # slope of pyramid faces relative to horizontal (degrees)
samparams['pyramid_roofsize'] = None # length of one side of square pyramid roof (microns)
samparams['pyramid_height'] = None # (microns)  height of pyramid
samparams['n_starting_points'] = 150  # number of resolution elements in initital beam diameter
samparams['D_e_to_l1'] = 66.89091e3 # nominal distance from entrance pupil to lens1 (microns)
samparams['f1'] = 7.8e3 # focal length of lens #1 (focuses light on pyramid tip)
samparams['lens1_fill_diameter'] = None  #  computational beam width at lens#1.  This matters!
samparams['beam_diameter_at_pyramid'] = None # size of domain at pyramid tip
samparams['D_l1_to_pyr'] = 7.8e3 # distance from lens1 to pyramid tip
samparams['D_l1_to_detector'] = 8.8296e3  # distance from lens to detector in 4f system
samparams['apex_diam'] = 8 # diameter of focal spot at pyramid apex in units of lambda_over_D (set by stop)
samparams['D_foc_to_l2'] = None # distance from focus to lens #2 (includes effective OPL thru prism)
samparams['f2'] = None # focal length of lens #2
samparams['diam_lens2'] =  None # effective diameter of lens2 (set by a stop)
samparams['D_l2_to_detector'] = None # distrance from lens2 to detector
samparams['detector_width'] = None # width of detector
samparams['max_chirp_step_deg'] = 90  # maximum allowed value (degrees) in chirp step for Fresnel prop
samparams['max_lens_step_deg'] = 20 # maximum step size allowed for lens phase screen
samparams['max_plane_step_deg'] = 20  # (degrees) maximum phase step allowed for planar phase screen
samparams['max_pyramid_phase_step'] = 5  # maximum step (degrees) allowed for pyramid phase ramp
samparams['max_finite_diff_phase_change_deg'] = 5  # maximum change in phase tolerated in finite difference
samparams['interp_style'] = 'linear'  # type of 2d interpolator for field resampling

#According to Sam, the entrace pupil is at 
def SamPyr1(g=None, x=None, Reflective=True, params=samparams, plots=True):
    assert Reflective  # this is reflective pyramid WFS
    nsides = 4  # how many sides to the pyramid
    # beta is the angle between beams for transmissive pyramid, see Eq. 21 in Sec. 4.1 of Depierraz' thesis
    indref = 2.27 # bk7 glass near 1 micron
    transmit_slope_deg = 1/1.6
    beta =  2*(indref - 1)*transmit_slope_deg  # degree units
    #mimic this slope with reflective geometry
    nomslope = 0.5*beta  # degrees, for reflective geometry
    bd = 7.25e3  #params['beam_diameter']
    f1 = 417.e3  # lens1 focal length, beam f/56.5
    mag_tot = 0.96/7.25  # final pupil image diam is 0.96e3 = 0.1324*bd
    ob1 = 0
    #im1 = 1./(1/f1 - 1/ob1)
    #mag1 = im1/ob1
    #mag2 = mag_tot/mag1
    f2 = 50.4e3
    #ob2 = f2*(mag2 + 1)/mag2
    #im2 = mag2*ob2
    #d_pyr_to_l2 = ob2 + im1 -f1

    if x is None: assert g is None
    if g is None:
        assert x is None
        nx = 300  #  params['n_starting_points']        
        x = np.linspace(-bd/2, bd/2, nx)
        mesh = np.meshgrid(x, x, indexing='xy')
        g = PT.RegPolygon(mesh, radius=bd/2, center=[0,0], rot0=0, N=6)
        del mesh
        #cr = np.where(xx*xx + xy*xy > bd*bd/4)
        #g = np.ones((nx,nx))
        #g[cr[0],cr[1]] = 0.
        #del xx, xy, cr

    FO = FourierOptics.FourierOptics(params)

    # prop to lens1
#    lens_supp = 2*bd  #  important!  support at lens
#    gl1, xl1 = FO.ConvFresnel2D(g, x, lens_supp, ob1, set_dx=True, return_derivs=False)
#    if plots:
#        plt.figure(); plt.imshow(np.abs(gl1)); plt.colorbar(); plt.title('at lens1');
#        plt.figure(); plt.plot(xl1,np.abs(gl1[len(gl1)//2,:]),'bx-'); plt.title('at lens1');
#        print('at lens1: dx = ', xl1[1] - xl1[0], ' microns, field size = ' , gl1.shape)
    
    
    #thin lens phase screen
#    gl1p, xl1p = FO.ApplyThinLens2D(gl1, xl1, [0,0], f1, return_derivs=False)
    gl1p, xl1p = FO.ApplyThinLens2D(g, x, [0,0], f1, return_derivs=False)

    #prop to focus (pyramid tip)
    l_over_d = f1*params['wavelength']/bd  # focal spot size
    gt, xt = FO.ConvFresnel2D(gl1p, xl1p, 20*l_over_d, f1, set_dx=10., return_derivs=False)
    if plots:
        plt.figure(); plt.imshow(np.abs(gt)); plt.colorbar(); plt.title('at tip');
        plt.figure(); plt.plot(xt,np.abs(gt[len(gt)//2,:]),'bx-'); plt.title('at tip');
        print('at tip: dx = ', xt[1] - xt[0], ' microns, field size = ' , gt.shape)

    #pyramid phase screen (reflective)
    gtp, xtp = FO.ApplyNSidedPyrPhase2D(gt, xt, SlopeDeviations=None,
                FaceCenterAngleDeviations=None, N=nsides, set_dx=True,
                NominalSlope=nomslope, rot0=None, reflective=True)

    p_to_l2 = 20.45e3
    diam_l2 = 2.8e3  # beam diam at lens2
    gl2, xl2 = FO.ConvFresnel2D(gtp, xtp, diam_l2 , p_to_l2, set_dx=6., return_derivs=False)
    if plots:
        plt.figure(); plt.imshow(np.abs(gl2)); plt.colorbar(); plt.title('at lens2');
        plt.figure(); plt.plot(xl2,np.abs(gl2[len(gl2)//2,:]),'bx-'); plt.title('at lens2');

    #thin lens phase screen

    gl2p, xl2p = FO.ApplyThinLens2D(gl2, xl2, [0,0], f2, return_derivs=False)

    #prop to detector plane
    l2_to_d = 50.e3
    gd, xd = FO.ConvFresnel2D(gl2p, xl2p, 4*mag_tot*bd, l2_to_d, set_dx=8., return_derivs=False)
    if plots:
        plt.figure(); plt.imshow(np.abs(gd)); plt.colorbar(); plt.title('at detector');
        plt.figure(); plt.plot(xd,np.abs(gd[len(gd)//2,:]),'bx-'); plt.title('at detector');

    
    return(gd,xd)
