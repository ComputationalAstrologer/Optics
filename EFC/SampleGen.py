# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 11:50:18 2024
@author: Richard Frazin
"""

import numpy as np
import EFC


defaultsetup = {}
defaultsetup['corners'] = [115,145,138,168]
defaultsetup['amp_std'] = 0.06
defaultsetup['phase_std'] = 0.11


def GenerateFieldPairs(Npairs, setup=defaultsetup):
   keylist = ['corners', 'amp_std', 'phase_std']
   if not all(key in setup for key in keylist):
      print ("Needed keys:", keylist)
      raise ValueError("Missing keys in the setup dictionary.")

   pl = EFC.MakePixList(setup['corners'],(256,256))
   imshape = ( 1+ setup['corners'][3] - setup['corners'][2], 1+ setup['corners'][1] - setup['corners'][0] )
   A = EFC.EFC(HolePixels=pl, SpeckleFactor=0.)
   Czero = np.zeros((A.Shx.shape[1],))
   f0x = A.Field(Czero,'X','Hole','phase',False,0.)  # get nominal field the region
   f0y = A.Field(Czero,'Y','Hole','phase',False,0.)

   input_imgs  = np.zeros((Npairs,2,imshape[0],imshape[1]))
   target_imgs = np.zeros(input_imgs.shape)
   ampp = setup['amp_std'] # amplitude std
   phap = setup['phase_std'] # phase std
   md_dom = []; md_cro = []
   for ks in range(Npairs):
       phasor_coef = (1. + ampp*np.random.randn(len(Czero)))*np.exp(1j*phap*np.random.randn(len(Czero)))
       f1x = A.Shx@phasor_coef - f0x; md_dom.append(np.median(np.abs(f1x)))
       f1y = A.Shy@phasor_coef - f0y; md_cro.append(np.median(np.abs(f1y)))
       f1x = f1x.reshape(imshape)
       f1y = f1y.reshape(imshape)
       input_imgs[ks,0,:,:] = np.real(f1x)
       input_imgs[ks,1,:,:] = np.imag(f1x)
       target_imgs[ks,0,:,:] = np.real(f1y)
       target_imgs[ks,1,:,:] = np.imag(f1y)
   md_dom = np.mean(md_dom)
   md_cro = np.mean(md_cro)
   return( (input_imgs, target_imgs, (md_dom,md_cro)) )
