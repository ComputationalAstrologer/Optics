# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 15:28:31 2022
@author: Richard Frazin

This is a module containing functions for sampling from a
power spectral density (PSD).  There are tools for 1D and 2D.
This is intended for modeling surface errors on the optics.
It will be assumed that there are no amplitude effects, so
the height variations are real-valued, making the Fourier
transforms Hermitian.
"""

import numpy as np
from scipy.integrate import quad

"""
Normal PSD fucntion (1D)
Fmax - the largest spatial frequency - the frequency grid goes from
    0 to +Fmax  (Kmax = 2piFmax)
h - the height of Gaussian
c - center 
S - the width (std) of the Gaussian
n - the number of sampling points
""" 
def MyGaussian1D(Fmax, h, c, sig, n):
    s = np.linspace(0, Fmax, n)
    f = np.zeros(s.shape)
    for k in range(len(s)):
        f[k] = h*np.exp(- (s[k] - c)**2/(2*sig*sig) )
    return(f)

"""
This surface roughness function is from Jhen Lumbres' thesis.
sigSR has units of nm
The default values for Kmax and Kmin are:
   Kmax = (1/2.5 um) = 4.e5 m^-1 and Kmin = (1/85 um) = 1.1765e4 m^-1, resp.
The output units are nm^2m^2
"""
def BetaSR(sigSR, Kmin=1.1765e4, Kmax=4.0e5):
    return( (sigSR*sigSR/np.pi)/(Kmax*Kmax - Kmin*Kmin) )

"""
Jhen's surface PSDs are given my sums of von Karman functions
k - spatial wavenumber (1/m)
beta - not sure what it means, but we need it.
inS - inner scale length (l0)
otS - outer scale length (L0)
sigSR - see BetaSR function
Kmin - see BetaSR function
Kmax - see BetaSR function
"""
def VonKarmanPSD(k, beta, inS, otS, alpha, sigSR, Kmin=1.1765e4, Kmax=4.0e5):
    ak = np.abs(k)
    betasr = BetaSR(sigSR, Kmin, Kmax)
    ex = np.exp(- (ak*inS)**2 )
    denom = ( 1/(otS*otS) + ak*ak )**(alpha/2)
    return(betasr + beta*ex/denom)

"""
Jhen's surface PSDs are given by sums of von Karman functions.
This function calls VonKarmanPSD in order to sum them up.
k - spatial wavenumber (units m^-1)
amp - list of a_0 values (usually 1)
beta - list of beta values
inS - list of inner scale values
otS - list of outer scale values
alpha - list of alpha values
sigSr - list of sigSR values
"""
def sumVonKarmanPSD(k, amp, beta, inS, otS, alpha, sigSR, base=1.e-9):
    assert (len(amp) == len(beta) == len(inS) == len(otS) == len(alpha) == len(sigSR))
    psd = base
    for l in range(len(beta)):
        psd += amp[l]*VonKarmanPSD(k, beta[l], inS[l], otS[l], alpha[l], sigSR[l])
    return(psd)
"""
This integrates the 1D or 2D PSD to get the RMS as a function of the maximum spatial
  frequency considered [Kmax - this has nothing to with Kmax in the function 
  VonKarmanPSD()].  Kmin sets the lower limit of the integration
Note that in 1D: RMS(Kmax) = 2*int_Kmin^Kmax PSD(k) dk  (Kmin, Kmax  > 0)
                           = 2*int_{log(Kmin)}^{log(Kmax)} PSD(exp(q)) exp(q) dq,
    where q = log(k), k>0 and log is the natural log
    In 2D:  RMS(Kmax) = 2pi*int_Kmin^Kmax PSD(k) k dk
                      = 2pi*int_{log(Kmin)}^{log(Kmax)} PSD^2(exp(q)) exp(2q) dq
        The 2D PSD is the square of the 1D PSD for isotropic media (I think)
Since k ranges over orders of magnitude, the log form the integral seems
    more suited to the problem.
Note that the units of k are [1/meters] so 1/(2.5 cm) = (1/0.025 m) = 40 m^-1
fcn  is a function handle that provides access to the function to be integrated.
    It is probably most convenient to make this a lambda function
npts - the number of Kmax points
maxKmax - is the Kmax value for the last integration
Kmin - smallest spatial frequency considered in the integral
"""
def integrateLog(fcn, npts=100, maxKmax=2.e4, Kmin=40., ndim=1):
    assert ( (ndim == 1) or (ndim == 2) )
    lowlim = np.log(Kmin)  #  lower limit of integration
    logKmax = np.linspace(np.log(Kmin), np.log(maxKmax), npts)  # integration end points
    if ndim == 1:
        f = lambda t:       2*np.exp(  t)*fcn(np.exp(t))
    else:
        f = lambda t: 2*np.pi*np.exp(2*t)*(fcn(np.exp(t))**2)
    RMS = np.zeros((npts,))
    natLog2Log10 = np.log10(np.e)  # factor to convert natural logs to 10-based logs
    for nn in (1 + np.arange(npts-1)):
        RMS[nn] = quad(f, lowlim, logKmax[nn])[0]

    return((RMS, natLog2Log10*logKmax))


"""
This samples a PSD from an input sequence assuming an
  exponential pdf: p(s) = (1/q)exp(-s/q) where q is the mean
psd - the input PSD -- only positive spatial frequencies!
L - length of the segment (centered on zero)
Fmax - max spatial frequency of psd
n - the number of points in the segment
"""
def SampleExpPSD1D(psd, Fmax, L, n):
    lp = len(psd)
    f = np.linspace(0, Fmax, lp)  # spatial freq grid
    x = np.linspace(-L/2, L/2, n)
    surf = np.zeros(x.shape)
    for k in (1 + np.arange(lp-1)): #don't include 0 spatial freq
        pwr = np.random.exponential(psd[k])
        camp = np.sqrt(pwr)*np.exp(1j*2*np.pi*np.random.rand())
        ramp = np.real(camp)
        iamp = np.imag(camp)
        kf = 2*np.pi*f[k]  # spatial wavenumber
        surf += 2*( ramp*np.cos(kf*x) + iamp*np.sin(kf*x) )
    return(surf)

"""
This loads Jhen's PSD parameters
"""
#OAP PSDs - output units are nm^2m^2
amp   = [1., 1., 1.]
alpha = [3.029, -3.103, 0.668]
beta  = [329.3, 1.6e-12, 3.5e-5]  # units are nm*m
otS   = [0.019, 16., 0.024]  # units are m
inS   = [3.e-6, 4.29e-3, 1.32e-4]  # units are m
sigSR = [5.e-6, 5.e-6, 5.5e-1]  # units are nm




