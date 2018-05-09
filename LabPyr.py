#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  9 14:55:31 2018
@author: rfrazin

This code is a descendant of Pyr.py.   This version is designed to accurately
reproduce lab measurements, and so allows for different paddings of the two 
FFTs and treats alignments (and possibly other) errors with free parameters.  

params is a dictionary of the basic parameters of the numerical model

"""
pyrparams = {}
pyrparams['lambda'] = 0.8 # wavelength (microns)
pyrparams['indref'] = 1.45 # pyramid index of refraction
pyrparams['pslope'] = 3. # slope of pyramid faces relative to horizontal (degrees)  
pyrparams['beamd'] = 2.e4 # input beam diameter (microns)
pyrparams['f1'] = 1.e6 # focal length of lens #1 (focuses light on pyramid tip)
pyrparams['f2'] = 1.e6 # focal length of lens #2 (makes 4 pupil images)
pyrparams['npup'] = 33  # number of pixels in entrace pupil
pyrparams['npad1'] = 2048 # number of points in first FFT
pyrparams['npad2'] = 2048 # number of points in second FFT

class Pyr():
    def __init__(self, params=pyrparams):
        return