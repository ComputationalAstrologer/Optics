#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 17:03:54 2020

@author: rfrazin
"""
import numpy as np
import scipy.optimize as mize


#This calculates the Frobenius cost of the upper triangular portion, excluding the diagonal
#H is the target matrix (pos def)
#G is the current approximation (can be a diag matrix)
#u is the update vector
def TriuFrobeniusCost(u, G, H, derivs=False):
    M = H.shape[0]
    assert G.shape == H.shape
    assert H.shape[0] == H.shape[1]
    assert u.shape == (M,)

    #updated G
    Gu = G + np.outer(u,u)
    #difference matrix
    D = Gu - H
    D = np.triu(D,1)
    cost = np.sum(D*D)/2.

    if not derivs:
        return(cost)

    #now, get derivative of cost w.r.t. u
    dcost = np.zeros((M,))
    P = np.zeros((M,M))
    for m in range(M):
        P = 0.*P
        for i in range(M): # rows
            for j in range(i+1,M): # columns
                if i == m:
                    P[i,j] = u[j]
                if j == m:
                    P[i,j] = u[i]
        dcost[m] += np.sum(D*P)  # D is already upper triangular
    return(dcost)
def TriuFrobeniusCostGrad(u, G, H):
    return(TriuFrobeniusCost(u, G, H, derivs=True))


M = 5
F = np.random.randn(M,M)
H = F.dot(F.T)

#this iterative updates G to approximate H

#get initial diagonal approximation
G = np.diag(np.diag(H))

u0 = np.random.randn(5,)
options = {'maxiter': 40, 'return_all': True}
mize.minimize(TriuFrobeniusCost, u0, args=(G, H), method='BFGS',
              jac=TriuFrobeniusCostGrad, options=options)




