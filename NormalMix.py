#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 17:20:06 2019
@author: rfrazin

These codes perform various functions relating to mixtures
  models of normal densities.

"""

import numpy as np
import matplotlib.pyplot as plt


#Use the EM method to estimate the component frequencies and each (mean, covariance).
#  Reference: ch2 of Mixture Model Classification by Paul D. McNicholas
#The component probabilities and the initial guess for the mean and variance
#  of each component #  must be provided.
#data - input array of shape (nd, ns) where ns is the number of samples
#cat_probs - a list (or vector) initial guess of mixture probabilities, s.t. their sum is 1.0
#normal_params - a list of tuples specifying the initial guess of the
#  means and covariances of the various normal components, i.e. (mu, sigma) where
#      mu is a vector with m components
#      sigma is a m-by-m pos (semi-)def matrix
#  All of the mu vectors must be of the same size
#The dimension of the output samples is given by the size of mu (sigma must be consistent)
def FitNormalMix(data, cat_probs, normal_params, n_steps=20, print_iterations=False):
    assert data.ndim == 2
    data = np.matrix(data).astype('float')
    assert data.shape[1] > data.shape[0]  # samples along axis=1
    ns = data.shape[1]  # number of samples
    nc = len(cat_probs)
    cprob = np.array(cat_probs)
    assert np.isclose(np.sum(cprob), 1.)
    if nc != len(normal_params):
        raise Exception('category probabilities and normal_params must be consistent.')
    if type(normal_params[0]) is not tuple:
        raise Exception('normal_params must be a list of tuples')
    #prepare to deal with small numbers!
    #finfo = np.finfo('float')

    params = []
    for k in range(len(normal_params)):
        mu = np.matrix(normal_params[k][0]).astype('float')
        if mu.shape[0] == 1:  #mu needs to be a column vector
            mu = mu.T
        C = np.matrix(normal_params[k][1]).astype('float')
        if k == 0:
            m = np.size(mu)  # dimension of the normal
        assert np.size(mu) == m  # all normals must have the same dimensions
        assert C.shape == (m,m)
        params.append((mu, C))
    if data.shape[0] != m:
        raise Exception('input data must have the same dimension as the normal densities')

    for step in range(n_steps):
        #update estimate of membership for each sample (note this is a soft estimate)
        memest = np.zeros((nc, ns))
        for m in range(ns):
            stuff = np.zeros(nc)
            for k in range(nc):
                stuff[k] = cprob[k]*NormalEval(data[:,m], params[k][0], params[k][1])
            for k in range(nc):
                memest[k,m] = stuff[k]/np.sum(stuff)
        ng = np.sum(memest, axis=1)  # estimated number of members in each cat

        cprob = ng/ns  #update cat_probs
        params_new = []  # tuples don't support in-place assignment
        for k in range(nc):
            mu_new = 0.*params[0][0]
            C_new = 0.*params[0][1]
            for m in range(ns):  # update mu
                mu_new += memest[k,m]*data[:,m]/ng[k]
            for m in range(ns):  # update C
                C_new += memest[k,m]*((data[:,m] - mu_new)*(data[:,m] - mu_new).T)/ng[k]
            params_new.append((mu_new, C_new))
        params = params_new

        #calculate complete-data log-likelihood (cdll) and the observed data likelihood (odll)
        #     with new estimates.
        #Note that the EM method does NOT minimize the cdll, rather it minimizes the
        #  observed-data (marginal) likelihood of the observations ('y' in the textbooks)
        cll = 0.
        oll = 0.
        for m in range(ns):
            qq = 0.
            for k in range(nc):
                qp = NormalEval(data[:,m], params[k][0], params[k][1])
                cll += memest[k,m]*(np.log(cprob[k]) + np.log(qp))
                qq += memest[k,m]*qp
            oll += np.log(qq)

        if (print_iterations) or (step == n_steps-1):
            print('')
            print('Iteration ' + str(step) + ':')
            print('log-likelihood of observed data = ' + str(oll))
            print('log-likelihood of complete data = ' + str(cll))
            print('category probabilities: ' + str(cprob))
            print('Normal means and covariance matrices:')
            for k in range(nc):
                print(str(k) + ', mean = ' + str(np.array(params[k][0].T)) )
                print(str(k) + ', covar = ' + str(params[k][1]))

    return((cprob, params, memest))

#This evaluates the normal probability density at a point x
#  with a normal with mean mu and covariance matrix C.
#detC (optional) is the determinant of C
#If inverse_cov, replace C with its matrix inverse
def NormalEval(x, mu, C, detC=None, inverse_cov=False):
    x = np.matrix(x).astype('float')
    mu = np.matrix(mu).astype('float')
    C = np.matrix(C).astype('float')
    assert mu.shape == x.shape
    assert (mu.shape[1] == 1) or (mu.shape[0] == 1)
    if mu.shape[0] == 1:
        mu = mu.T
        x = x.T
    assert C.shape == (mu.shape[0], mu.shape[0])
    if detC is None:
        if inverse_cov:
            dC = np.linalg.det(np.linalg.inv(C))
        else:
            dC = np.linalg.det(C)
    else:
        dC = detC
    m = len(mu)
    if inverse_cov:
        iC = C
    else:
        iC = np.linalg.inv(C)
    q = -0.5*(x - mu).T*iC*(x - mu)
    r = np.sqrt( dC*(2*np.pi)**m )
    return( float(np.exp(q)/r) )


#This returns categorically distributed random values.
#p - 1D array (or list) of categorical probailities
#n - number of random samples deseried in output.
def CategoricalRandVar(p, n):
    p = np.array(p)
    assert(p.ndim == 1)
    assert(np.min(p) >= 0.)
    assert(np.isclose(np.sum(p), 1.))
    out = list()
    for k in range(n):
        t = np.random.rand()
        s = 0.
        cat = -1  # category
        while t >= s:
            cat += 1
            s += p[cat]
        out.append(cat)
    return(np.array(out))


#This generates data from a mixture of multivariate normals.
#ns - the number of samples to be generated.
#cat_probs - a list (or vector) of mixture probabilities, such that their sum is 1.
#normal_params - a list of tuples specifying the means and covariances of the various normal components, i.e. (mu, sigma) where
#      mu is a vector with m components
#      sigma is a m-by-m pos (semi-)def matrix
#  All of the mu vectors must be of the same size
#The dimension of the output samples is given by the size of mu (sigma must be consistent)
def GenerateNormalMix(ns, cat_probs, normal_params):
    probs = np.array(cat_probs)
    assert len(cat_probs) == len(normal_params)
    assert np.min(probs) >= 0
    assert np.isclose(np.sum(probs), 1.)
    assert type(normal_params[0]) == tuple
    params = []
    for k in range(len(normal_params)):
        mu = np.array(normal_params[k][0])
        assert mu.ndim == 1  # the mean of the normal is a 1D array
        C = np.array(normal_params[k][1])
        assert C.ndim == 2  # the covariance is 2D
        if k == 0:
            m = np.size(mu)  # dimension of the normal
        assert np.size(mu) == m
        assert np.size(C) == m*m
        params.append((mu, C))

    #generat RVs
    out = None
    cats = CategoricalRandVar(probs, ns)  # first generate the categoricals
    for k in cats:
        r = np.random.multivariate_normal(params[k][0],params[k][1])
        r = np.matrix(r).T
        if out is None:
            out = r
        else:
            out = np.hstack((out,r))
    return(out)


