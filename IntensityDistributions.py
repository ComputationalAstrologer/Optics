#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 13:44:27 2021

@author: Richard A. Frazin
"""

import numpy as np

#This provides the modified Rician probability distribution, P(x)
#x - output intensity (list or array)
#a - length of constant phasor, a^2 is its intensity
#sig - the mean intensity of the random phasor is 2sig^2
#n_angles - the number of points used to calculate the integral over angle
#return_derivs - if True, also returns derivs w.r.t. 'a' and sig
def ModifiedRician(x, a, sig, n_angles=360, return_derivs=False):
    x = np.array(x)
    assert x.ndim == 1
    assert (a >=0.) and (sig >= 0.) and all(x >= 0.)
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
#return_derivs - if True, also returns derivs w.r.t. 'a' and sigr, sigi and cc
def Frazinian(x, a, sigr, sigi, cc, n_angles=360, return_derivs=False):
    x = np.array(x)
    assert x.ndim == 1
    assert (a >=0.) and (sigi > 0.) and (sigr > 0.)
    assert (cc > -1.) and (cc < 1.)
    assert all(x >= 0.)
    px = np.zeros(x.shape)  # output probability
    dth = 2*np.pi/n_angles
    th = np.linspace(-np.pi + dth/2, np.pi - dth/2, n_angles)
    cth = np.cos(th)
    sth = np.sin(th)
    A = 4*np.pi*sigr*sigi*np.sqrt()
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

        # calculate dpxdcc
        dpxdcc[k] = - dAdcc*px[k]/A
        integ1 = dth*np.sum( dBdcc*C*eBC )/A
        integ2 = dth*np.sum( dC3dcc*B*eBC )/A
        dpxdcc += integ1 + integ2

    return((px, dpxda, dpxdsigr, dpxdsigi, dpxdcc))