#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 14:41:44 2022
@author: Richard Frazin

cubic b-spline stuff.
See M. Unser, IEEE Sig. Proc. Mag., Nov. 1999

This assumes the knots are uniformly spaced,
  which makes them cardinal b-splines.

"""

import numpy as np

#this evaluates the 1D cubic b-spline at x
#delta is the spacing between the spline knots.
#The resulting function is nonzero for |x/delta| < 2
def Bspline3_1D(x, delta):
    assert np.isscalar(x)
    assert np.isscalar(delta)
    q = np.abs(x/delta)
    if q >= 2.:
        return(0.)
    if q >= 1.:
        return( ((2.-q)**3)/6.  )
    else:
        return(2./3 - q*q + q*q*q/2.)
    

#See Bspline3_1D
def Bspline3_2D(x, y, delta):
    bx = Bspline3_1D(x, delta)
    by = Bspline3_1D(y, delta)
    return(bx*by)
