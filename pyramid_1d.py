#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 18:08:07 2017
@author: Richard Frazin

This is a 1D simulation of an optical pyramid.  Note that the pyramid
output is the convolution of this result with the pupil plane field.
This treats the pyramid as a thin phase screen.   Now, the pupil is
focused (Fourier transformed) onto the pyramid tip, so the pyramid
should be wide enough to see N diffraction orders of the focal spot.
"""


import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as so

# global variables
lam = 0.85  # sensing wavelength (microns)
d = 7.2e3  # beam diameter (microns)
D = 7.992e6 # telescope diameter (microns)
angres = lam/D  # angular resolution (radians)
arcsecrad = 3600*180/np.pi  # arcsec/radian

def SigmaToStrehl(sigma):
    strehl = np.exp(-sigma*sigma)
    return(strehl)

def StrehlToSigma(strehl):
    sigma = np.sqrt(- np.log(strehl))
    return(sigma)

def myfft(g):  # for centered arrays
    return np.fft.fftshift(np.fft.fft(np.fft.fftshift(g)))/np.sqrt(len(g))
#  this zero pad function gives rise to purely real myfft with symm. input
def myzp(f, npad):  # zero-pad function for pupil fields
    nfield = len(f) + 1
    if np.mod(nfield, 2) != 0:
        raise Exception("len(f) must be odd.")
    if npad <= len(f):
        raise Exception("npad must be greater than len(f)")
    if np.mod(npad, 2) == 1:
        raise Exception("npad must be even.")
    g = np.zeros(npad).astype('complex')
    g[npad/2 - nfield/2 + 1:npad/2 + nfield/2] = f
    return g

# Note the returns vary greatly depending on what you ask for!
# g the input field (not phase!), len(g) must be npup.
#     the input phase is given by np.angle(g)
# npup - number of points in pupil (must be odd!)
# npad - number of points in array
#   if npup=31 and npad=1024, first 0 is at 31 pixels from center.
#   So, the the spatial frequency corresponding to n is
#       is  (n/31)*(lambda*f/d) where d is the pupil diameter.
# Note: zero padding doesn't make a lot of difference in the
#   FFT of the phase ramp, so there is only zero-padding of the pupil
#
# If no_derivs, return only intensity and spatial grid
# If (not no_derivs) and (not pHes), return intensity gradients 
# If pHes and (not aHes), return
#   cost, d_cost/d_phase + d2_cost/d2_phase
# if aHes, assert pHes and return
#   cost, d_cost/d_x, d2_cost/d2_x where x = [phase, amplitude]
#
# If Hessians are wanted, it is assumed gradients and Hessians of
#  a least-squares cost function are to be returned, so it needs:
#       a vector of measurement weights, set by weight parameter
#       a vector of (noisy) intensity measurements, set by y parameter
#  This is done in order to not store a Hessian matrix for every pixel.
#  pHess == True: return grad and Hessian w.r.t. phase
#  aHess == True: also return grand and Hessian w.r.t amplitude 
#          (+ mixed 2nd deriv)
# ZeroPhaseCenter: the cost function, grad, and Hessian use the condition
#    that the phase in the center of the pupil is 0
#
def pyramid(g=None, no_derivs=False, pHes=False, aHes=False, 
            y=None, weight=None, npup=129, npad=4096, ZeroPhaseCenter=True):
    if aHes:
        assert pHes
    fl = 40*d  # focal length  (microns)
    slope = 3.734  # slope of pyramid in deg
    indref = 1.452  # index of refraction
    if (npup == 33) and (npad == 128):
        fscale = lam*fl/d/4.  # scaling for dist in focal plane
    elif (npup == 33) and (npad == 1024):
        fscale = lam*fl/d/31.  # scaling for dist in focal plane
    elif (npup == 33) and (npad == 4096):
        fscale = lam*fl/d/132.  # scaling for dist in focal plane
    elif (npup == 129) and (npad == 4096):
        fscale = lam*fl/d/32.  # scaling for dist in focal plane
    elif (npup == 33) and (npad == 8192):
        fscale = lam*fl/d/264.3  # scaling for dist in focal plane
    elif (npup == 33) and (npad == 16384):
        fscale = lam*fl/d/528.6  # scaling for dist in focal plane
    else:
        raise Exception("Invalide choice for npad.")
    # spatial coordinate in pupil plane
    pscale = (d/npup)*np.linspace(-npad/2, npad/2-1, npad)
    # focal plane spatial transverse coordinate
    xf = fscale*np.linspace(-npad/2, npad/2-1, npad)
    # height of prism surface
    yf = - np.abs(xf)*np.tan(slope*np.pi/180)
    # phase ramp caused by prism
    pramp = np.exp(-1j*yf*(indref - 1)*2*np.pi/lam)
    # pupil plane field
    up = np.ones(npup).astype('complex')
    if g is not None:
        assert len(g) == npup
        up *= g
    uf = myfft(myzp(up, npad))  # focal plane field
    uf = myfft(uf*pramp)  # detector plane field
    ipp = np.real(uf*np.conj(uf))  # intensity on detector
    if no_derivs:
        return(ipp, pscale)
        
    if g is None:
        g = np.ones(npup).astype('complex')

    assert (npup == 129) and (npad == 4096)
    il0, il1 = 1833, 1961  # range of left pupil image
    ir0, ir1 = 2135, 2263  #         right
    npp2 = il1 - il0 + 1  # number of pixels per pupil image
    npp = 2*npp2  # number of points in Jacobian
    u = 1j*np.zeros(npp)  # field values at important pixels
    pjac = np.zeros((npp, npup))  # d_intensity/d_phase
    ajac = np.zeros((npp, npup))  # d_intensity/d_amp
    dufdp = 1j*np.zeros((npp, npup))  # d_field/d_phase, needed for Hessians
    dufda = 1j*np.zeros((npp, npup))  # d_field/d_amp, needed for Hessians

    u[0:npp2] = uf[il0:il1+1]
    u[npp2:] = uf[ir0:ir1+1]
    ival = np.real(u*np.conj(u))  # intensity values in important pixels
    for k in range(npup):
        dup = 0*up
        dup[k] = 1j*np.abs(g[k])*np.exp(1j*np.angle(g[k]))  # d_(field at pupil)
        duf = myfft(pramp*myfft(myzp(dup, npad)))  # d_(field at detector)
        dufdp[0:npp2, k] = duf[il0:il1+1]  # needed for Hessians
        dufdp[npp2:, k] = duf[ir0:ir1+1]  # needed for Hessians
        dipp = 2*np.real(np.conj(uf)*duf)
        pjac[0:npp2, k] = dipp[il0:il1+1]
        pjac[npp2:, k] = dipp[ir0:ir1+1]

        # calculate ajac, need it or not.
        dup = 0*up
        dup[k] = np.exp(1j*np.angle(g[k]))  # d_(field at pupil)
        duf = myfft(pramp*myfft(myzp(dup, npad)))  # d_(field at detector)
        dufda[0:npp2, k] = duf[il0:il1+1]  # needed for Hessians
        dufda[npp2:, k] = duf[ir0:ir1+1]  # needed for Hessians
        dipp = 2*np.real(np.conj(uf)*duf)
        ajac[0:npp2, k] = dipp[il0:il1+1]
        ajac[npp2:, k] = dipp[ir0:ir1+1]
    if not pHes:
        return(ival, pjac, ajac)

    assert y is not None
    assert len(y) == npp
    if weight is None:
        weight = np.ones(npp)
    else:
        assert len(weight) == npp

    # calculate cost fcn and derivs
    cost = 0.
    pgrad = np.zeros(npup)  # 1st deriv of cost
    pphes = np.zeros((npup, npup))  # 2nd deriv matrix of cost
    duf2dp2 = 1j*dufdp  # d2_(detector field)/d2_(phase)- diag matrix
    for l in range(npp2):  # loop over detector pixels
        m = l + npp2
        cost += .5*np.power(ival[l] - y[l], 2)*weight[l]  # left pupil image
        cost += .5*np.power(ival[m] - y[m], 2)*weight[m]  # right
        for k1 in range(npup):
            pgrad[k1] += (ival[l] - y[l])*pjac[l, k1]*weight[l]
            pgrad[k1] += (ival[m] - y[m])*pjac[m, k1]*weight[m]
            for k2 in range(npup):
                if k2 > k1:
                    continue  # fill this in using symmetry later
                # add contribution from left pupil image
                pph = pjac[l, k1]*pjac[l, k2]
                s = dufdp[l, k1]*np.conj(dufdp[l, k2])
                if k2 == k1:
                    s += duf2dp2[l, k1]*np.conj(u[l])
                pph += (ival[l] - y[l])*2*np.real(s)
                pph *= weight[l]
                pphes[k1, k2] += pph
                # add contribution from right pupil image
                pph = pjac[m, k1]*pjac[m, k2]
                s = dufdp[m, k1]*np.conj(dufdp[m, k2])
                if k2 == k1:
                    s += duf2dp2[m, k1]*np.conj(u[m])
                pph += (ival[m] - y[m])*2*np.real(s)
                pph *= weight[m]
                pphes[k1, k2] += pph

    # use symmetry property to fill in rest
    for k1 in range(npup):
        for k2 in range(npup):
            if k2 > k1:
                pphes[k1, k2] = pphes[k2, k1]

    if ZeroPhaseCenter:  # add penalty enforcing zero phase at center 
        cost += 50.*np.max(weight)*np.power(np.angle(g[npup/2 + 1]), 2)
        pgrad[npup/2 + 1] += 100*np.max(weight)*np.angle(g[npup/2 + 1])
        pphes[npup/2 + 1, npup/2 + 1] += 100*np.max(weight)

    if not aHes:
        return (cost, pgrad, pphes)

    jac = np.hstack((pjac, ajac))
    grad = np.hstack((pgrad, np.zeros(npup)))
    hes = np.zeros((2*npup, 2*npup))
    hes[0:npup, 0:npup] = pphes
    del pjac, ajac, pgrad, pphes
    duf2dpda = 1j*dufda  # d2_(detector field)/d2_(phase)- diag matrix
    for l in range(npp2):  # loop over detector pixels
        m = l + npp2
        for k1 in range(2*npup):
            if k1 >= npup:
                grad[k1] += (ival[l] - y[l])*jac[l, k1]*weight[l]
                grad[k1] += (ival[m] - y[m])*jac[m, k1]*weight[m]
            for k2 in range(2*npup):
                if (k1 < npup) and (k2 < npup):
                    continue  # this is the pphes
                if k2 > k1:
                    continue  # fill this in using symmetry later

                if (k1 >= npup) and (k2 < npup):  # d_amp d_phase case
                    pph = jac[l, k1]*jac[l, k2]
                    s = dufdp[l, k1 - npup]*np.conj(dufda[l, k2])
                    if k2 == k1 - npup:
                        s += duf2dpda[l, k2]*np.conj(u[l])
                    pph += (ival[l] - y[l])*2*np.real(s)
                    pph *= weight[l]
                    hes[k1, k2] += pph
                    # add contribution from right pupil image
                    pph = jac[m, k1]*jac[m, k2]
                    s = dufdp[m, k1 - npup]*np.conj(dufda[m, k2])
                    if k2 == k1 - npup:
                        s += duf2dpda[l, k2]*np.conj(u[l])
                    pph += (ival[m] - y[m])*2*np.real(s)
                    pph *= weight[m]
                    hes[k1, k2] += pph

                if (k1 >= npup) and (k2 >= npup):  # d2_amplitude case
                    pph = jac[l, k1]*jac[l, k2]
                    s = dufda[l, k1 - npup]*np.conj(dufda[l, k2 - npup])
                    if k2 == k1:
                        s += 0
                    pph += (ival[l] - y[l])*2*np.real(s)
                    pph *= weight[l]
                    hes[k1, k2] += pph
                    # add contribution from right pupil image
                    pph = jac[m, k1]*jac[m, k2]
                    s = dufda[m, k1 - npup]*np.conj(dufda[m, k2 - npup])
                    if k2 == k1:
                        s += 0
                    pph += (ival[m] - y[m])*2*np.real(s)
                    pph *= weight[m]
                    hes[k1, k2] += pph

    # use symmetry property to fill in rest
    for k1 in range(2*npup):
        for k2 in range(2*npup):
            if (k1 < npup) and (k2 < npup):
                continue
            if k2 > k1:
                hes[k1, k2] = hes[k2, k1]
    return (cost, grad, hes)


# a is the coefficient vector (or list) of the modes (for now, only tilt, defocus)
# g is the field to which the modes are applied
# the returned values depend on the inputs
def pyramid_modal(a, g=None, y=None, weight=None, npup=129, npad=4096):
    if g is not None:
        assert len(g) == npup
    assert len(a) == 2
    fl = 40*d  # focal length  (microns)
    slope = 3.734  # slope of pyramid in deg
    indref = 1.452  # index of refraction
    if (npup == 129) and (npad == 4096):
        fscale = lam*fl/d/32.  # scaling for dist in focal plane
    else:
        assert False
    # spatial coordinate in pupil plane
    # pscale = (d/npup)*np.linspace(-npad/2, npad/2-1, npad)
    # focal plane spatial transverse coordinate
    xf = fscale*np.linspace(-npad/2, npad/2-1, npad)
    # height of prism surface
    yf = - np.abs(xf)*np.tan(slope*np.pi/180)
    # phase ramp caused by prism
    pramp = np.exp(-1j*yf*(indref - 1)*2*np.pi/lam)
    pyr = lambda ff: myfft(pramp*myfft(myzp(ff, npad)))  # pyramid operator
    xxp = np.linspace(-1, 1, npup)  # normalized pupil coord
    f0 = np.pi*xxp  # tilt fcn (see ao_pyr_sim.py)
    f1 = 3.31*xxp*xxp  # defocus fcn (see ao_pyr_sim.py)
    ph = a[0]*f0
    ph += a[1]*f1

    up = np.exp(1j*ph)
    if g is not None:
        up *= g
    dupda0 = 1j*f0*up  # 1st derivs
    dupda1 = 1j*f1*up
    d2upd2a0 = 1j*f0*dupda0  # 2nd derivs
    d2upd2a1 = 1j*f1*dupda1
    d2upda1da0 = 1j*f1*dupda0

    uf = pyr(up)
    dufda0 = pyr(dupda0)
    dufda1 = pyr(dupda1)
    d2ufd2a0 = pyr(d2upd2a0)
    d2ufd2a1 = pyr(d2upd2a1)
    d2ufda1da0 = pyr(d2upda1da0)

    assert (npup == 129) and (npad == 4096)
    il0, il1 = 1833, 1961  # range of left pupil image
    ir0, ir1 = 2135, 2263  #         right
    uf = np.hstack((uf[il0:il1+1], uf[ir0:ir1+1]))
    dufda0 = np.hstack((dufda0[il0:il1+1], dufda0[ir0:ir1+1]))
    dufda1 = np.hstack((dufda1[il0:il1+1], dufda1[ir0:ir1+1]))
    d2ufd2a0 = np.hstack((d2ufd2a0[il0:il1+1], d2ufd2a0[ir0:ir1+1]))
    d2ufd2a1 = np.hstack((d2ufd2a1[il0:il1+1], d2ufd2a1[ir0:ir1+1]))
    d2ufda1da0 = np.hstack((d2ufda1da0[il0:il1+1], d2ufda1da0[ir0:ir1+1]))

    intf = np.real(uf*np.conj(uf))
    dintfda0 = 2*np.real(np.conj(uf)*dufda0)
    dintfda1 = 2*np.real(np.conj(uf)*dufda1)
    d2intfd2a0 = np.conj(dufda0)*dufda0 + d2ufd2a0*np.conj(uf)
    d2intfd2a0 = 2*np.real(d2intfd2a0)
    d2intfd2a1 = np.conj(dufda1)*dufda1 + d2ufd2a1*np.conj(uf)
    d2intfd2a1 = 2*np.real(d2intfd2a1)
    d2intfda1da0 = np.conj(dufda1)*dufda0 + d2ufda1da0*np.conj(uf)
    d2intfda1da0 = 2*np.real(d2intfda1da0)

    if y is None:
        return(intf)
        #  return (intf, dintfda0, dintfda1)  # test derivs

    assert len(y) == 2*npup
    assert len(weight) == len(y)

    # note that the linearity range of the cost is MUCH smaller than
    #  that of the intensity.
    cost = 0.
    dcost = np.zeros(2)  # cost gradient
    hcost = np.zeros((2, 2))  # cost Hessian
    for k in range(2*npup):
        cost += .5*weight[k]*(intf[k] - y[k])*(intf[k] - y[k])
        dcost[0] += weight[k]*(intf[k] - y[k])*dintfda0[k]
        dcost[1] += weight[k]*(intf[k] - y[k])*dintfda1[k]
        hcost[0, 0] += weight[k]*(dintfda0[k]*dintfda0[k] + (intf[k] - y[k])*d2intfd2a0[k])
        hcost[1, 1] += weight[k]*(dintfda1[k]*dintfda1[k] + (intf[k] - y[k])*d2intfd2a1[k])
        hcost[1, 0] += weight[k]*(dintfda1[k]*dintfda0[k] + (intf[k] - y[k])*d2intfda1da0[k])
    hcost[0, 1] = hcost[1, 0]
    return(cost, dcost, hcost)


# x is the vector being minimized.  y and wt are needed for the cost fcn.
# returns (cost, gradient)
def WrapPyramid(x, y, wt, Ampl=False, retHes=False):
    #print "retHes = ", retHes
    if Ampl:
        nx = len(x)
        phase = x[0:nx/2]
        ampli = x[nx/2:]
        g = ampli*np.exp(1j*phase)
        (f, df, ddf) = pyramid(g=g, no_derivs=False, pHes=True, aHes=True, 
                                  y=y, weight=wt, ZeroPhaseCenter=True)
    else:
        g = np.exp(1j*x)
        (f, df, ddf) = pyramid(g=g, no_derivs=False, pHes=True, aHes=False, 
                                  y=y, weight=wt, ZeroPhaseCenter=True)
    if retHes:
        return(ddf)
    else:
        return(f, df)

# the dummy argument is needed so it takes the same number of
#  arguments as WrapPyramid
def WrapPyramidHessian(x, y, wt, Ampl=False, dummy=None):
    return(WrapPyramid(x, y, wt, Ampl, retHes=True))


# a = [tilt, defocus] is the coefficient list
# g is the complex-valued current estimate of the field
def WrapPyramidModal(a, g, y, wt, retHes=False):
    (f, df, ddf) = pyramid_modal(a, g=g, y=y, weight=wt)
    if retHes:
        return(ddf)
    else:
        return(f, df)


# the dummy argument is needed so it takes the same number of
#  arguments as WrapPyramidModal
def WrapPyramidModalHessian(a, g, y, wt, dummy=None):
    return(WrapPyramidModal(a, g, y, wt, retHes=True))


# given the coefficient vector, this gives the addition phase
def CoefToPhase(a, npup=129):
    assert len(a) == 2
    xxp = np.linspace(-1, 1, npup)  # normalized pupil coord
    f0 = np.pi*xxp  # tilt fcn (see ao_pyr_sim.py)
    f1 = 3.31*xxp*xxp  # defocus fcn (see ao_pyr_sim.py)
    return(a[0]*f0 + a[1]*f1)


# this fits tilt and defocus coefficients to the phases
def FitLowOrder(phase):
    npup = len(phase)
    xxp = np.linspace(-1, 1, npup)  # normalized pupil coord
    f0 = np.pi*xxp  # tilt fcn (see ao_pyr_sim.py)
    f1 = 3.31*xxp*xxp  # defocus fcn (see ao_pyr_sim.py)
    D = np.zeros((2,2))
    y = np.zeros((2,1))
    D[0, 0] = np.sum(f0*f0)
    D[0, 1] = np.sum(f0*f1)
    D[1, 0] = D[0, 1]
    D[1, 1] = np.sum(f1*f1)
    y[0] = np.sum(phase*f0)
    y[1] = np.sum(phase*f1)
    c = np.linalg.inv(D).dot(y)
    return(np.array([c[0], c[1]]))


# This function sets up the minimization problem to find:
#  not Ampl: only solve for phase vector
#  Ampl: solve for [phase, amplitude) vector
def find_solution(Ampl=False, MaxIt=10, display=True, Strehl=.9, aStrehl=.9, pcount=1.e9):
    npup = 129
    psigma = np.sqrt(-np.log(Strehl)) 
    ph = psigma*np.random.randn(npup)
    ph -= ph[npup/2 + 1]  # ZeroPhaseCenter 
    amp = np.ones(npup)
    if Ampl:
        asigma = np.sqrt(-np.log(aStrehl))
        amp += asigma*np.random.randn(npup)

    (ytrue, pj, aj) = pyramid(g=amp*np.exp(1j*ph), no_derivs=False, pHes=False)
    del aj, pj

    # put measured y (intensity) in units photon counts
    norm = pcount/np.mean(ytrue)
    ym = ytrue*norm
    ym += np.sqrt(ym)*np.random.randn(len(ytrue))
    ym /= norm
    var = ytrue/norm
    wt = np.divide(1, var)
    wt /= np.mean(wt)

    if Ampl:
        x0 = np.hstack((np.zeros(npup), np.ones(npup)))
        fargs = (ym, wt, True)
    else:
        x0 = np.zeros(npup)
        fargs = (ym, wt, False)

    opts = {'disp': display, 'maxiter': MaxIt, 'xtol': 1.e-6, 'return_all': True}
    result = so.minimize(wrap, x0, args=fargs, method='Newton-CG',
                         options=opts, jac=True, hess=wrap_hessian)
    plt.plot(ph, 'k.')
    plt.plot(result.x, 'gx')
    
    if Ampl:
        return(ph, amp, result)
    else:
        return(ph, result)


def strehl_plot():
    strehl = [1., .8, .3]
    npup = 129
    plt.figure()
    for st in strehl:
        g = np.ones(npup)*np.exp(1j*np.sqrt(-np.log(st))*np.random.randn(npup))
        ig, ps, jac = pyramid(g, jac=True)
        sv = np.linalg.svd(jac[0:npup,:]-jac[npup:2*npup,:],
                           compute_uv=False)
        plt.plot(sv,'ko', markersize=3)
    return(0)
        

# fold over center and subtract one half of the pyramid output from other
# ip is the input pyramid intensity.
# This looses any information about the symmetric part of the field.
# Shift and subtract preserves it.
# x corresonding is the spatial coordinate (microns)
def pyrfoldsubtract(ip, x, divide='no'):
    assert len(ip) == len(x)
    assert np.mod(len(ip),2) == 0
    if (divide != 'yes') and (divide != 'no'):
        raise Exception("divide must be 'yes' or 'no'.")
    n2 = len(ip)/2
    rgt = ip[n2:]  # intensity on right side
    rx = x[n2:]
    lft = ip[1:n2+1]  #              left
    lft = lft[::-1]   # reverse array
    if divide == 'yes':
        dff = np.divide(2*(rgt - lft), lft + rgt)
    else:
        dff = rgt - lft
    return((dff, rx))

# makes a Gaussian bump in phase
# amp - amplitude of bump in radians
# cen - center of bump (width of pupil is 1) 
# std - stdev of bump 
# npup - number of points
def makegaussian(amp=angres, cen = 0, std=.15, npup=129):
    x = np.linspace(-.5, .5, npup)
    phase = np.exp(- (x - cen)*(x - cen)/2./std/std)
    phase *= amp*D*2*np.pi/lam
    # analytical derviative
    #dphase = - phase*(x - cen)/std/std
    bump = np.exp(1j*phase)
    return(bump, phase)

# make a random phase screen symmetric about center
# amp - tilt of wavefront in radians
def makerandomsymmetric(amp=angres, npup=129):
    assert np.mod(npup,2) == 1
    a = np.random.randn(npup/2)
    a = np.convolve(a, np.ones(npup/10), mode='same')
    b = np.hstack((a,0,a[::-1]))
    b *= amp*D*2*np.pi/lam
    pramp = np.exp(1j*b)
    return(pramp, b)

# amp - amplitude in radians
def maketriangle(amp=angres, npup=129):
    assert np.mod(npup,2) == 1
    px = np.linspace(-d/2, d/2, npup)  #x coord in pupil plane
    ramp = np.abs(px)*(D/d)*np.tan(amp)*2*np.pi/lam
    ramp = np.max(ramp) - ramp
    pramp = np.exp(1j*ramp)
    return(pramp, ramp)

# amp - tilt of wavefront in radians
def maketilt(amp=angres, npup=129):
    assert np.mod(npup,2) == 1
    px = np.linspace(-d/2, d/2, npup)  #x coord in pupil plane
    ramp = px*(D/d)*np.tan(amp)*2*np.pi/lam
    pramp = np.exp(1j*ramp)
    return(pramp, ramp)

def makefigs(npup=129, npad=4096, savefigs=None):
    # show pyramid with flat input
    im0, x0 = pyramid(g=None, npup=npup, npad=npad)
    plt.figure()
    plt.plot(x0, im0)
    plt.axis([-2.e4, 2.e4, 0, 1.42])
    plt.xlabel('spatial coord. ($\mu$)', fontsize='large')
    plt.ylabel('intensity', fontsize='large')
    plt.title('intensity with flat input', fontsize='large')
    if savefigs is not None:
        title = 'figs/pyramid/PyramidFlatInput.jpg'
        plt.savefig(title)

    pscale = np.linspace(-d/2, d/2, npup)
 
    plt.figure()
    pramp, ramp = maketilt(amp=2*angres, npup=npup)
    plt.plot(pscale,ramp/np.pi/2)
    plt.xlabel('pupil coord. ($\mu$)', fontsize='large')
    plt.ylabel('phase/(2$\pi$)', fontsize='large')
    plt.title('pure tilt (2$\lambda$/D)', fontsize='large')
    if savefigs is not None:
        title = 'figs/pyramid/Tilt.jpg'
        plt.savefig(title)
        
#%%        
    plt.figure()
    im0, x0 = pyramid(g=None, npup=npup, npad=npad)
    h0, = plt.plot(x0, im0,'k', label='flat')
    plt.axis([-2.e4, 2.e4, 0, 1.42])
    plt.xlabel('spatial coord. ($\mu$)', fontsize='large')
    plt.ylabel('intensity', fontsize='large')
    plt.title('intensity with tilt input', fontsize='large')
    pramp, ramp = maketilt(amp=angres, npup=npup)
    imtilt, crap = pyramid(g=pramp, npup=npup, npad=npad)
    h1, = plt.plot(x0, imtilt,'b',lw=2, label='tilt = $\lambda$/D')
    pramp, ramp = maketilt(amp=angres/2, npup=npup)
    imtilt, crap = pyramid(g=pramp, npup=npup, npad=npad)
    h2, = plt.plot(x0, imtilt,'g',lw=2, label='tilt = $\lambda$/2D')
    pramp, ramp = maketilt(amp=angres/4, npup=npup)
    imtilt, crap = pyramid(g=pramp, npup=npup, npad=npad)
    h3, = plt.plot(x0, imtilt,'r',lw=2, label='tilt = $\lambda$/4D')
    pramp, ramp = maketilt(amp=angres/8, npup=npup)
    imtilt, crap = pyramid(g=pramp, npup=npup, npad=npad)
    h4, = plt.plot(x0, imtilt,'c',lw=2, label='tilt = $\lambda$/8D')
    pramp, ramp = maketilt(amp=angres/16, npup=npup)
    imtilt, crap = pyramid(g=pramp, npup=npup, npad=npad)
    h5, = plt.plot(x0, imtilt,'m',lw=2, label='tilt = $\lambda$/16D')
    plt.legend(handles=[h0,h1,h2,h3,h4,h5], loc=2)
    if savefigs is not None:
        title = 'figs/pyramid/PyramidTiltInput.jpg'
        plt.savefig(title)
   
#%%        
    plt.figure()
    im0, x0 = pyramid(g=None, npup=npup, npad=npad)
    h0, = plt.plot(x0, im0,'k:', lw=2, label='raw flat (no diff)')
    plt.axis([4000, 13000, 0, 1.4])
    plt.xlabel('spatial coord. ($\mu$)', fontsize='large')
    plt.ylabel('folded intensity difference', fontsize='large')
    plt.title('folded and subtracted intensities', fontsize='large')
    pramp, ramp = maketilt(amp=angres, npup=npup)
    imtilt, crap = pyramid(g=pramp, npup=npup, npad=npad)
    imdf, xdf = pyrfoldsubtract(imtilt, x0, divide='no')
    h1, = plt.plot(xdf, imdf,'b',lw=2, label='tilt = $\lambda$/D')
    pramp, ramp = maketilt(amp=angres/2, npup=npup)
    imtilt, crap = pyramid(g=pramp, npup=npup, npad=npad)
    imdf, xdf = pyrfoldsubtract(imtilt, x0, divide='no')
    h2, = plt.plot(xdf, imdf,'g',lw=2, label='tilt = $\lambda$/2D')
    pramp, ramp = maketilt(amp=angres/4, npup=npup)
    imtilt, crap = pyramid(g=pramp, npup=npup, npad=npad)
    imdf, xdf = pyrfoldsubtract(imtilt, x0, divide='no')
    h3, = plt.plot(xdf, imdf,'r',lw=2, label='tilt = $\lambda$/4D')
    pramp, ramp = maketilt(amp=angres/8, npup=npup)
    imtilt, crap = pyramid(g=pramp, npup=npup, npad=npad)
    imdf, xdf = pyrfoldsubtract(imtilt, x0, divide='no')
    h4, = plt.plot(xdf, imdf,'c',lw=2, label='tilt = $\lambda$/8D')
    pramp, ramp = maketilt(amp=angres/16, npup=npup)
    imtilt, crap = pyramid(g=pramp, npup=npup, npad=npad)
    imdf, xdf = pyrfoldsubtract(imtilt, x0, divide='no')
    h5, = plt.plot(xdf, imdf,'m',lw=2, label='tilt = $\lambda$/16D')
    plt.legend(handles=[h0,h1,h2,h3,h4,h5], loc=1) 
    if savefigs is not None:
        title = 'figs/pyramid/PyramidFoldedDiffTilt.jpg'
        plt.savefig(title)
 
#%%        
    plt.figure()
    pramp, ramp = maketriangle(amp=angres/4, npup=npup)
    plt.plot(pscale, ramp/np.pi/2)
    plt.xlabel('pupil coord. ($\mu$)', fontsize='large')
    plt.ylabel('phase/(2$\pi$)', fontsize='large')
    plt.title('phase triangle ($\lambda$/4D)', fontsize='large')
    if savefigs is not None:
        title = 'figs/pyramid/Triangle.jpg'
        plt.savefig(title)
            
    plt.figure()
    imtri, crap = pyramid(g=pramp, npup=npup, npad=npad)
    plt.plot(x0, imtri)
    plt.axis([-2.e4, 2.e4, 0, 1.42])
    plt.xlabel('spatial coord. ($\mu$)', fontsize='large')
    plt.ylabel('intensity', fontsize='large')
    plt.title('intensity with ($\lambda$/4D) triangle', fontsize='large')
    if savefigs is not None:
        title = 'figs/pyramid/PyramidTriangleInput.jpg'
        plt.savefig(title)

    plt.figure()
    imdf, xdf = pyrfoldsubtract(imtilt, x0, divide='no')
    h1, = plt.plot(xdf, imdf, label='$\lambda$/4D triangle')
    plt.axis([4000, 13000, 0, 1.e-14])
    plt.xlabel('spatial coord. ($\mu$)', fontsize='large')
    plt.ylabel('folded intensity difference', fontsize='large')
    plt.title('folded and subtracted intensity', fontsize='large')
    plt.legend(handles=[h1], loc=2)
    if savefigs is not None:
        title = 'figs/pyramid/PyramidFoldedDiffTriangle.jpg'
        plt.savefig(title)

 
#%%        
    plt.figure()
    pramp, ramp = makegaussian(amp=angres/4, npup=npup)
    plt.plot(pscale, ramp/np.pi/2)
    plt.xlabel('pupil coord. ($\mu$)', fontsize='large')
    plt.ylabel('phase/(2$\pi$)', fontsize='large')
    plt.title('phase Gaussian ($\lambda$/4D)', fontsize='large')
    if savefigs is not None:
        title = 'figs/pyramid/Gaussian.jpg'
        plt.savefig(title)
            
    plt.figure()
    imtri, crap = pyramid(g=pramp, npup=npup, npad=npad)
    plt.plot(x0, imtri)
    plt.axis([-2.e4, 2.e4, 0, 1.42])
    plt.xlabel('spatial coord. ($\mu$)', fontsize='large')
    plt.ylabel('intensity', fontsize='large')
    plt.title('intensity with ($\lambda$/4D) Gaussian', fontsize='large')
    if savefigs is not None:
        title = 'figs/pyramid/PyramidGaussianInput.jpg'
        plt.savefig(title)

    plt.figure()
    imdf, xdf = pyrfoldsubtract(imtilt, x0, divide='no')
    h1, = plt.plot(xdf, imdf, label='$\lambda$/4D Gaussian')
    plt.axis([4000, 13000, 0, 1.e-14])
    plt.xlabel('spatial coord. ($\mu$)', fontsize='large')
    plt.ylabel('folded intensity difference', fontsize='large')
    plt.title('folded and subtracted intensity', fontsize='large')
    plt.legend(handles=[h1], loc=2)
    if savefigs is not None:
        title = 'figs/pyramid/PyramidFoldedDiffGaussian.jpg'
        plt.savefig(title)

#%%        
    plt.figure()
    pramp, ramp = makerandomsymmetric(amp=angres/4, npup=npup)
    plt.plot(pscale, ramp/np.pi/2)
    plt.xlabel('pupil coord. ($\mu$)', fontsize='large')
    plt.ylabel('phase/(2$\pi$)', fontsize='large')
    plt.title('phase RandomSymm ($\lambda$/4D)', fontsize='large')
    if savefigs is not None:
        title = 'figs/pyramid/RandomSymmetric.jpg'
        plt.savefig(title)
            
    plt.figure()
    imtri, crap = pyramid(g=pramp, npup=npup, npad=npad)
    plt.plot(x0, imtri)
    plt.axis([-2.e4, 2.e4, 0, 1.42])
    plt.xlabel('spatial coord. ($\mu$)', fontsize='large')
    plt.ylabel('intensity', fontsize='large')
    plt.title('intensity with ($\lambda$/4D) RandomSymm', fontsize='large')
    if savefigs is not None:
        title = 'figs/pyramid/PyramidRandomSymmInput.jpg'
        plt.savefig(title)

    plt.figure()
    imdf, xdf = pyrfoldsubtract(imtilt, x0, divide='no')
    h1, = plt.plot(xdf, imdf, label='$\lambda$/4D RandomSymm')
    plt.axis([4000, 13000, 0, 1.e-14])
    plt.xlabel('spatial coord. ($\mu$)', fontsize='large')
    plt.ylabel('folded intensity difference', fontsize='large')
    plt.title('folded and subtracted intensity', fontsize='large')
    plt.legend(handles=[h1], loc=2)
    if savefigs is not None:
        title = 'figs/pyramid/PyramidFoldedDiffRandomSymm.jpg'
        plt.savefig(title)
