#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 17:10:16 2018

"""

import numpy as np
import matplotlib.pyplot as plt

#this compares sequential least-squares estimation with an initial prior
#  to standard batch estimation with an initial prio
def CompareSequentialEst(sigma_noise=0.2, sigma_C0=100.):
    A = np.matrix(np.random.randn(5,2)) # system matrix
    xt = np.matrix([2,1]).T # true value of x
    x0 = np.matrix([0,3]).T #  mean of of initial prior
    C0 = sigma_C0*sigma_C0*np.matrix(np.eye(2)) # initial prior covariance
    iC0 = np.linalg.inv(C0)
    Cy = sigma_noise*sigma_noise*np.matrix(np.eye(5))
    iCy = np.linalg.inv(Cy)
    yt = A.dot(xt) # noiseless measurement values
    ym = yt + sigma_noise*np.random.randn(5,1)  # measured values of y
    z = A.T.dot(iCy.dot(ym)) + iC0.dot(x0)
    #standard estimate
    xcov = np.linalg.inv( A.T.dot(iCy.dot(A)) + iC0)
    xhat = xcov.dot(z)
    #sequential estimate, one element of y at a time
    Cx = C0  #initial covariance of x
    xc = x0  #initial estimate of x
    for k in range(5):
        B = A[k,:]
        D = Cy[k,k]
        yy = ym[k]
        icov_old = np.linalg.inv(Cx)
        Cx = np.linalg.inv(B.T.dot(B)/D + icov_old)
        xc = Cx.dot(B.T*yy/D +  icov_old*xc)
    print("standard xhat = \n" + str(xhat) )
    print("sequential xhat = \n" + str(xc))
    print("standard cov = \n" + str(xcov))
    print("sequential cov = \n" + str(Cx))


#Scalar regression with noise in both observation and independent variable.
#Consider the scalar regression problem y = q*b + n:
#    y = observation
#    b = regressand (what we want to estimate), b >0 
#    q = independent variable (measured with error), q >0
#    n = noise in observation
def ScalarProb(b_true=1, q_true=.5, y_std =0.04, q_std=0.15, Nruns=1000000):

    q_true = np.abs(q_true)  # make sure we are working in the ++ quadrant
    b_true = np.abs(b_true)
    y_true = q_true*b_true
    dispsize = 512
    q_grid = np.linspace(0,2,512)
    b_grid = np.linspace(0,2,512)

    #likelihood
    like = np.zeros((dispsize, dispsize))
    for k in range(dispsize):
        for l in range(dispsize):
            like[k,l]  = (1/(2*y_std*y_std))*(q_grid[k]*b_grid[l] - y_true)**2
            like[k,l] += (1/(2*q_std*q_std))*(q_grid[k] - q_true)**2
    like = np.exp(- like)
    like /= np.sum(like)  # normalize


    plt.figure()
    plt.imshow(like, extent=[np.min(b_grid), np.max(b_grid), np.max(q_grid), np.min(q_grid)])
    plt.colorbar()
    plt.title('joint likelihood, q_true= ' + str(q_true) + ', b_true = ' + str(b_true))
    plt.xlabel('b')
    plt.ylabel('q')

    #marginal likelihood - not useful
    #marlike = np.sum(like,axis=0)
    #maxmarlike = b_grid[np.argmax(marlike)]
    #print('marginal likelihood maximum is at b= ' + str(maxmarlike))
    #plt.figure()
    #plt.plot(b_grid, marlike)
    #plt.title('marginal likelihood, max at b= ' + str(maxmarlike))
    #plt.xlabel('b')

    #monte carlo simulation of naive estimates
    b_naive = np.zeros(Nruns)
    for k in range(Nruns):
        #print(k)
        q_meas = q_std*np.random.randn() + q_true
        #print('q_meas = ' + str(q_meas))
        y_meas = y_true + y_std*np.random.randn()
        #print('y_meas = ' + str(y_meas))
        b_naive[k] = y_meas/q_meas
        #print('b_naive = ', str(b_naive[k]))
    mean_naive = np.mean(b_naive)
    print('b_true=' + str(b_true) + ', mean naive estimate of b is ' + str(mean_naive))

    plt.figure()
    plt.title('histogram of naive estimates')
    plt.hist(b_naive, bins=np.linspace(-b_true/2,3*b_true, 50))