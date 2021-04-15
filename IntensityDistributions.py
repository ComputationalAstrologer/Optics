#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 13:44:27 2021

@author: Richard A. Frazin
"""

import numpy as np
from scipy.signal import convolve as conv

#This provides the modified Rician probability distribution, P(x)
#x - output intensity (list or array)
#a - length of constant phasor, a^2 is its intensity
#sig - the mean intensity of the random phasor is 2sig^2
#n_angles - the number of points used to calculate the integral over angle
#return_derivs - if True, also returns derivs w.r.t. 'a' and sig
def ModifiedRician(x, a, sig, n_angles=360, return_derivs=False):
    x = np.array(x)
    assert x.ndim == 1
    assert all(x >= 0.)
    dth = 2*np.pi/n_angles
    th = np.linspace(-np.pi + dth/2, np.pi - dth/2, n_angles)
    cth = np.cos(th)
    Icg = 2.*sig*sig
    b = 2*np.sqrt(x)*a/Icg
    ModBes = np.zeros((len(x),))
    for k in range(len(x)):
        ModBes[k] = dth*np.sum(np.exp(b[k]*cth))/(2.*np.pi)  # modified Bessel function of the first kind of order 0
    px = ModBes*np.exp(-1*(x + a*a)/Icg)/Icg
    if not return_derivs:
        return(px)
    else:
        integral = np.zeros((len(x),))
        dpxda = np.zeros((len(x),))
        dpxdsig = np.zeros((len(x),))
        for k in range(len(x)):
            integral[k] = dth*np.sum(cth*np.exp(b[k]*cth))
            bb = np.sqrt(x[k])*a/(np.pi*Icg*Icg*Icg)
            q  = np.exp( -1*(x[k] + a*a)/Icg )
            dpxdsig[k] = px[k]*(x[k] + a*a - Icg)/(Icg*Icg) - bb*q*integral[k]  # deriv w.r.t Icg
            dpxdsig[k] *= 4*sig  # deriv w.r.t. sig
            bb  = (np.sqrt(x[k])/a)/(2*np.pi*Icg*Icg)
            dpxda[k] = 2*a*(integral[k]*bb*q - px[k]/Icg)
        return((px,dpxda,dpxdsig))


#Similar to Modified Rician, except generalized to non-circular statistics.
#x - output intensity (list or array)
#a - length of constant phasor, a^2 is its intensity
#sigr - std dev of real part of random phasor
#sigi -            imag
#cc - correlation coefficient of real and imag parts of random phasor
#n_angles - the number of points used to calculate the integral over angle
#ignore_cc - treat the correlation coef as zero don't return its deriv.  However, cc needs to be in the calling sequence
#return_derivs - if True, also returns derivs w.r.t. 'a' and sigr, sigi and cc (if ignore_cc == False)
def Frazinian(x, a, sigr, sigi, cc, n_angles=360, ignore_cc=True, return_derivs=False):
    x = np.array(x)
    assert all(x >= 0.)
    assert x.ndim == 1
    assert (a >=0.) and (sigi > 0.) and (sigr > 0.)
    if ignore_cc:
        cc = 0.0
    else:
        #assert (cc > -1.) and (cc < 1.)  # rectification removes this constraint
        rectfactor = 0.98*(np.pi/2)*np.cos(cc*np.pi/2)
        cc = 0.98*np.sin(cc*np.pi/2)  # this way no boundary constraint on cc is needed (I call this procedure "recitification")
    px = np.zeros(x.shape)  # output probability
    dth = 2*np.pi/n_angles
    th = np.linspace(-np.pi + dth/2, np.pi - dth/2, n_angles)
    cth = np.cos(th)
    sth = np.sin(th)
    A = 4*np.pi*sigr*sigi*np.sqrt(1. - cc*cc)
    B = 1./(cc*cc - 1.)
    for k in range(len(x)):
        sx = np.sqrt(x[k])
        C1 = ( (sx*cth - a)**2 )/(2*sigr*sigr)
        C2 = sth*sth*x[k]/(2*sigi*sigi)
        C3 = cc*(a - sx*cth)*sx*sth/(sigr*sigi)
        px[k] = dth*np.sum(np.exp(B*(C1 + C2 + C3)))
    px /= A
    if not return_derivs:
        return(px)
    
    #derivative caclulations (see notes)
    dpxda    = np.zeros(x.shape)
    dpxdsigr = np.zeros(x.shape)
    dpxdsigi = np.zeros(x.shape)
    dpxdcc   = np.zeros(x.shape)
    dAdsigr =  4*np.pi*sigi*np.sqrt(1 - cc*cc)
    dAdsigi =  4*np.pi*sigr*np.sqrt(1 - cc*cc)
    dAdcc   = -4*np.pi*sigr*sigi*cc/np.sqrt(1 - cc*cc)
    dBdcc   = -2*cc/( (cc*cc - 1)**2 )
    for k in range(len(x)):
        sx = np.sqrt(x[k])
        C1 = ( (sx*cth - a)**2 )/(2*sigr*sigr)
        dC1da = (a - sx*cth)/(sigr*sigr)
        dC1dsigr = - 2.*C1/sigr
        C2 = sth*sth*x[k]/(2*sigi*sigi)
        dC2dsigi = - 2.*C2/sigi
        dC3dcc = (a - sx*cth)*sx*sth/(sigr*sigi)
        C3 = cc*dC3dcc
        dC3da = cc*sx*sth/(sigr*sigi)
        dC3dsigr = - C3/sigr
        dC3dsigi = - C3/sigi
        C = C1 + C2 + C3
        eBC = np.exp(B*C)

        # calculate dpxda
        dpxda[k] = dth*np.sum((dC1da + dC3da)*B*eBC)/A

        # calculate dpxdsigr
        dpxdsigr[k] = - dAdsigr*px[k]/A
        integ2 = dth*np.sum( (dC1dsigr + dC3dsigr)*B*eBC )/A
        dpxdsigr[k] += integ2

        # calculate dpxdsigi
        dpxdsigi[k] = - dAdsigi*px[k]/A
        integ2 = dth*np.sum( (dC2dsigi + dC3dsigi)*B*eBC )/A
        dpxdsigi[k] += integ2

        if ignore_cc:
            return((px, dpxda, dpxdsigr, dpxdsigi))

        # calculate dpxdcc
        dpxdcc[k] = - dAdcc*px[k]/A
        integ1 = dth*np.sum( dBdcc*C*eBC )/A
        integ2 = dth*np.sum( dC3dcc*B*eBC )/A
        dpxdcc[k] += integ1 + integ2
        dpxdcc *= rectfactor
        return((px, dpxda, dpxdsigr, dpxdsigi, dpxdcc))


#this adds a constant (incoherent) intensity, c**2, to the ModifiedRician
#the easiest way to do this is to convolve the MR with a narrow normal (or half normal)
#x - the intensity values  -- must be evenly spaced!  and in ascending order.
#c - c**2 is the constant intensity added to the MR
#the other paramers are passed to Modified Rician.
def ModifiedRicianPlusConstant(x, c, a, sig, n_angles=360, return_derivs=False):
    assert np.isclose(x[1]-x[0], x[-1]-x[-2])  # check for uniform spacing
    assert x[1] - x[0] > 0.  #x must be increasing
    if not return_derivs:
        mr = ModifiedRician(x, a, sig, n_angles=n_angles, return_derivs=False)
    else:
        (mr, dmrda, dmrds) = ModifiedRician(x, a, sig, n_angles=n_angles, return_derivs=True)
        
    space = x[1] - x[0]
    s2 = ( space/2.35 )**2  # sigma^2 of normal
    cen = (x[0] + x[-1])/2. + c*c
    #indic = np.where(x - cen < 0.)[0]  # indicator function
    f = np.exp( (- 0.5/s2)*(x-cen)**2  )
    #f[indic] = 0.  # make into a half-Gaussian
    f /= np.sum(f)
    g = conv(mr, f, mode='same', method='fft')
    if not return_derivs:
        return(g)
    else:
        dmrda = conv(dmrda, f, mode='same', method='fft')
        dmrds = conv(dmrds, f, mode='same', method='fft')
        dfdcen = f*(x-cen)/s2
        dmrdcen = conv(mr, dfdcen, mode='same', method='fft')
        return((mr, dmrdcen, dmrda, dmrds))

