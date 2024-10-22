#!/usr/bin/env python3
"""
author: Richard Frazin  (rfrazin@umich.edu)

This is tutorial code designed to teach people how to use the tools I created
in course of writing  the article I sent to the Journal of Astronomical Telescopes,
Instruments and Systems (JATIS) on Oct. 3, 2024, entitled: A Laboratory Method for 
Measuring the Cross-Polarizaion in High-Contrast Imaging.
This article is also publicly available at http://arxiv.org/abs/2410.03579

Before doing anything with this file, please see README.txt

"""

import numpy as np  
import matplotlib.pyplot as plt
import pickle
import EFC  # this module is the main one

# generate and instance of the EFC class.  This will simulate a coronagraph without any
#   dark hole pixels specifies and no speckle field.  There are no aberrations, so this is the nominal model.
A = EFC.EFC(HolePixels=None, SpeckleFactor=0.)  # if this doesn't work, go back to README.txt

#create a DM command corresponding to a flat surface
C_flat = np.zeros((A.Sx.shape[1],))  # A.Sx is dominant field Jacobian

#get the dominant ('X') electric field at the detector for the nominal model - this array is complex-valued 
fd = A.Field(C_flat, XorY='X', region='Full', DM_mode='phase',return_grad=False,SpeckleFactor=0.)
#        cross    ('Y")
fc = A.Field(C_flat, XorY='Y', region='Full', DM_mode='phase',return_grad=False,SpeckleFactor=0.)
fd = fd.reshape((256,256));  fc = fc.reshape((256,256))

#make images of the imag parts of the dominant and cross fields
plt.figure(); plt.imshow(np.imag(fd),cmap='seismic');plt.colorbar();
plt.figure(); plt.imshow(np.imag(fc),cmap='seismic');plt.colorbar();

print("Still Under Construction")