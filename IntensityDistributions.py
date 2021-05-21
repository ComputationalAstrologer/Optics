#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 13:44:27 2021

@author: Richard A. Frazin
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve as conv

#This provides the modified Rician probability distribution, P(x)
#x - output intensity (list or array)
#a - length of constant phasor, a^2 is its intensity
#sig - the mean intensity of the random phasor is 2sig^2
#n_angles - the number of points used to calculate the integral over angle
#return_derivs - if True, also returns derivs w.r.t. 'a' and sig
#
#  This version sometimes has floating overflow issues.  Hence the 'Old' suffix
#
def ModifiedRicianOld(x, a, sig, n_angles=360, return_derivs=False):
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
        return(px, np.array([dpxda,dpxdsig]))

#This version seeks to avoid floating point overflows by combining the exponential
def ModifiedRician(x, a, sig, n_angles=360, return_derivs=False):
    x = np.array(x)
    assert x.ndim == 1
    assert all(x >= 0.)
    dth = 2*np.pi/n_angles
    th = np.linspace(-np.pi + dth/2, np.pi - dth/2, n_angles)
    cth = np.cos(th)
    Icg = 2.*sig*sig
    px = np.zeros(x.shape)
    for k in range(len(x)):
        u = (2*np.sqrt(x[k])*a*cth - x[k]  - a*a)/Icg
        px[k] = dth*np.sum(np.exp(u))/(2.*np.pi*Icg)
    if not return_derivs:
        return(px)
    else:
        dpxda = np.zeros((len(x),))
        dpxdsig = np.zeros((len(x),))
        for k in range(len(x)):
            u = (2.*np.sqrt(x[k])*a*cth - x[k]  - a*a)/Icg
            dpxda[k] = dth*np.sum( (2.*np.sqrt(x[k])*cth - 2.*a)*np.exp(u)/Icg )/(2.*np.pi*Icg)
            dpxdsig[k] = - px[k]/Icg
            dpxdsig[k] += dth*np.sum( -(u/Icg)*np.exp(u) )/(2.*np.pi*Icg)
        dpxdsig *= 4.*sig   #this is because the derivative calculated so far is with respect to Icg
        return(px, np.array([dpxda,dpxdsig]))



#np.sign has the annoying property that np.sign(0) = 0
#only works on scalars
#this returns a float, i.e., 1.0 or -1.0
def MySign(x):
    assert np.isscalar(x)
    if x >= 0. : return(1.0)
    else: return(-1.0)

#Similar to Modified Rician, except generalized to non-circular statistics.
#x - output intensity (list or array)
#a - length of constant phasor, a^2 is its intensity
#sigr - std dev of real part of random phasor
#sigi -            imag
#cc - correlation coefficient of real and imag parts of random phasor
#n_angles - the number of points used to calculate the integral over angle
#ignore_cc - treat the correlation coef as zero don't return its deriv.  However, cc needs to be in the calling sequence
#return_derivs - if True, also returns derivs w.r.t. 'a' and sigr, sigi and cc (if ignore_cc == False)
def Frazinian(x, a, sigr, sigi, cc, n_angles=360, ignore_cc=False, return_derivs=False):
    x = np.array(x)
    assert all(x >= 0.)
    assert x.ndim == 1
    sign_a = MySign(a)  # this step removes positivity constraints
    sign_i = MySign(sigi)
    sign_r = MySign(sigr)
    a *= sign_a
    sigr *= sign_r
    sigi *= sign_i
    if ignore_cc:
        cc = 0.0
    else:
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


        # calculate dpxdcc
        if not ignore_cc:
            dpxdcc[k] = - dAdcc*px[k]/A
            integ1 = dth*np.sum( dBdcc*C*eBC )/A
            integ2 = dth*np.sum( dC3dcc*B*eBC )/A
            dpxdcc[k] += integ1 + integ2
            dpxdcc[k] *= rectfactor

    dpxda    *= sign_a
    dpxdsigr *= sign_r
    dpxdsigi *= sign_i
    if ignore_cc:
        return(px, np.array([dpxda, dpxdsigr, dpxdsigi]))
    else:
        return(px, np.array([dpxda, dpxdsigr, dpxdsigi, dpxdcc]))


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
    ss = ( space/2.35 )  # sigma of normal  -
    cen = (x[0] + x[-1])/2. + c*c
    #indic = np.where(x - cen < 0.)[0]  # indicator function
    f = space*np.exp( (- 0.5/ss**2)*(x-cen)**2 )/(2*np.pi*ss)  # this integrates to unity with good sampling
    sf = np.sum(f)
    if sf > 1.e-4:
        f /= sf
    g = conv(mr, f, mode='same', method='fft')
    if not return_derivs:
        return(g)
    else:
        dmrda = conv(dmrda, f, mode='same', method='fft')
        dmrds = conv(dmrds, f, mode='same', method='fft')
        dfdcen = f*(x-cen)/ss**2
        dmrdcen = conv(mr, dfdcen, mode='same', method='fft')
        return(mr, np.array([dmrdcen, dmrda, dmrds]))

#this calculates the misfit of the fitted function to the observed histogram.
#  It also returns the gradient.  
#v - vector of parameters fed to func
#y - vector of histogram frequencies (or probability densities)
#centers - centers of histogram bins (these are intensity values)
#func - the function used to model the histogram.  Must take the 'return_derivs' argument
#scale - scaling of the chi-squared metric, if None np.median(y)**2 is used
#ignore_cc - forces correlation coef to be zero, due to optimization difficulties (only valid for Frazinian)
def ChiSqHist(v, y, centers, func, scale=None, ignore_cc=False):
    assert y.shape == centers.shape
    if scale == None: scale = np.median(y)**2
    if func == Frazinian: 
        assert ( (len(v) == 4) or (len(v) == 3) )
    elif func == ModifiedRician: assert len(v) == 2
    elif func == ModifiedRicianPlusConstant: assert len(v) ==3
    else: raise Exception("Error: 'func' is not implemented.")
    if func == Frazinian:
        if ignore_cc:
            assert len(v) == 3
            Q = Frazinian(centers, v[0], v[1], v[2], 0., ignore_cc=True, return_derivs=True)
        else:
            assert len(v) == 4
            Q = Frazinian(centers, v[0], v[1], v[2], v[3], ignore_cc=False, return_derivs=True)
    elif func == ModifiedRician:
        Q = ModifiedRician(centers, v[0], v[1], return_derivs=True)
    elif func == ModifiedRicianPlusConstant:
        Q = ModifiedRicianPlusConstant(centers, v[0], v[1], v[2], return_derivs=True)

    ym = Q[0]  # modeled histogram values
    ch = 0.5*np.sum( (ym - y)**2 )/scale
    g = np.zeros((len(v),))  # gradient values
    for k in range(len(v)):
        g[k] = np.sum( (ym - y)*Q[1][k] )/scale

    return((ch, g))


def MyGaussian(length,fwhm):  #for wavelet analysis with scipy.signal.cwt
     x = np.arange(length)
     c = (length-1)/2
     std = fwhm/2.355
     f = np.exp( - ((x-c)**2)/2/std**2)
     f /= np.sum(f)
     return(f)  