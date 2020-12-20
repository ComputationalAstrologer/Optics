#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 17:52:13 2020
@author: rfrazin

Anderson-Darling Test for univariate normality.
See the wikipedia article.

This implements the single sample test for case 3 (i.e., unknown mean and var).
Note that the critical values of the test statistic (called A2 here) have
only a very weak dependence on the number of samples, making it particularly
good for non-independent samples for which the degrees of freedom are hard to
determine.  However, the effective sample size should be greater than 8.
Generally speaking, the normality is questionable when A2 is greater than
about 0.6 and should be rejected when greater than about 1.

"""

import numpy as np
from scipy.stats import norm

#x - is an array of samples from a univariate process
def ADtest(x):
    assert x.ndim == 1
    u = np.sort(x)
    n = len(u)
    mu = np.mean(u)
    var = np.var(u, ddof=1)
    y = (u - mu)/np.sqrt(var)
    s = 0.
    for k in range(n):
        cdfk = norm.cdf(y[k])
        s +=  (2*k - 1)*np.log(cdfk) + (2*(n-k) + 1)*np.log(1 - cdfk)
    s /= (-1*n)
    A2 = s - n
    return(A2)
