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


def SamPyr0(g=None, x=None, Reflective=True, params=samparams):

    bd = params['beam_diameter']

    assert Reflective  # this is reflective pyramid WFS
    if x is None: assert g is None
    if g is None:
        assert x is None
        nx = params['n_starting_points']
        x = np.linspace(-bd/2, bd/2, nx)
        # make a circular input field
        [xx, xy] = np.meshgrid(x, x, indexing='xy')
        cr = np.where(xx*xx + xy*xy > bd*bd/4)
        g = np.ones((nx,nx))
        g[cr[0],cr[1]] = 0.
        del xx, xy, cr

    FO = FourierOptics.FourierOptics(params)

    ff = 7800.  #focal length
    ob = params['D_e_to_l1']
    im = params['D_l1_to_detector']
    nomslope = 4. # params['pyramid_slope_deg']
    Nsides = 4
    mag = im/ob

    lens_supp = 1.5*params['beam_diameter']  #  important!  support at lens
    # max angle between beams (approx)
    beta =  2*params['pyramid_slope_deg']*np.pi/180  #angle between beams
    detector_diam = 2*(im - ff)*np.tan(beta) + 4.5*mag*params['beam_diameter']

    #prop to lens and apply lens phase screen
    g, x = FO.ConvFresnel2D(g, x, lens_supp, ob, set_dx=12., return_derivs=False)
    g, x = FO.ApplyThinLens2D(g, x, [0,0], ff, return_derivs=False)

    #propagate to pyramid tip
    diam0 = ff*12*params['wavelength']/params['beam_diameter']
    g, x = FO.ConvFresnel2D(g, x, diam0, ff, set_dx=0.3, return_derivs=False)

    #apply paramid phase mask
    g, x = FO.ApplyNSidedPyrPhase2D(g, x, SlopeDeviations=None,
                FaceCenterAngleDeviations=None, N=Nsides, set_dx=True,
                NominalSlope=nomslope, rot0=None, reflective=True)

    g, x = FO.ConvFresnel2D(g, x, detector_diam, im - ff, set_dx=0.5, return_derivs=False)
    return((g,x))

def SamPyr1(g=None, x=None, Reflective=True, params=samparams, plots=True):
    assert Reflective  # this is reflective pyramid WFS
    bd = 7.25e3  #params['beam_diameter']
    f1 = 409.625e3  # lens1 focal length, beam f/56.5
    ob = 20*f1
    mag = 0.1324 # final pupil image diam is 0.96e3 = 0.1324*bd
    d_l1_to_det = mag*ob
    f_tot  = 1./(1./ob + 1/d_l1_to_det) # focal length of compound (lens1 + lens2)
    nsides = 4
    nomslope = 0.5

    if x is None: assert g is None
    if g is None:
        assert x is None
        nx = 512  #  params['n_starting_points']        
        x = np.linspace(-bd/2, bd/2, nx)
        # make a circular input field
        [xx, xy] = np.meshgrid(x, x, indexing='xy')
        cr = np.where(xx*xx + xy*xy > bd*bd/4)
        g = np.ones((nx,nx))
        g[cr[0],cr[1]] = 0.
        del xx, xy, cr

    FO = FourierOptics.FourierOptics(params)

    # prop to first focus
    lens_supp = 2*bd  #  important!  support at lens
    gl1, xl1 = FO.ConvFresnel2D(g, x, lens_supp, ob, set_dx=30., return_derivs=False)
    
    if plots:
        plt.figure(); plt.imshow(np.abs(gl1)); plt.colorbar(); plt.title('at lens1');
        plt.figure(); plt.plot(xl1,np.abs(gl1[len(gl1)//2,:]),'bx-'); plt.title('at lens1');
    
    #thin lens phase screen
    gl1p, xl1p = FO.ApplyThinLens2D(gl1, xl1, [0,0], f1, return_derivs=False)

    #prop to focus (pyramid tip)
    l_over_d = f1*params['wavelength']/bd  # focal spot size
    gt, xt = FO.ConvFresnel2D(gl1p, xl1p, 20*l_over_d, f1, set_dx=10., return_derivs=False)
    if plots:
        plt.figure(); plt.imshow(np.abs(gt)); plt.colorbar(); plt.title('at tip');
        plt.figure(); plt.plot(xt,np.abs(gt[len(gt)//2,:]),'bx-'); plt.title('at tip');

    #pyramid phase screen (reflective)
    gtp, xtp = FO.ApplyNSidedPyrPhase2D(gt, xt, SlopeDeviations=None,
                FaceCenterAngleDeviations=None, N=nsides, set_dx=True,
                NominalSlope=nomslope, rot0=None, reflective=True)

    #prop to final image plane
    gf, xf = FO.ConvFresnel2D(gtp, xtp, ?? , d12, set_dx=True, return_derivs=False)
    if plots:
        plt.figure(); plt.imshow(np.abs(gt)); plt.colorbar(); plt.title('at tip');
        plt.figure(); plt.plot(xt,np.abs(gt[len(gt)//2,:]),'bx-'); plt.title('at tip');




    
    return(g,x)
    
    
    
    
    
    
    
    
