#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 17:10:16 2018

"""

import numpy as np
import matplotlib.pyplot as plt

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
    plt.hist(b_naive, bins=np.linspace(-b_true/2,3*b_true, 50))

    
    

    
    
