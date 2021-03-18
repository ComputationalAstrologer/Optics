#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 15:57:40 2021

@author: Richard Frazin

This plots some histograms

"""

import numpy as np
import matplotlib.pyplot as plt

#create a histogram corresponding to two exponential decay rates
#the probability of waiting a time t given rate r is r*exp(-rt).
#r could also be the intensity of photon source

r1 = 0.5 # rate of process 1 (proportional to intensity)
r2 = 5.  # rate of process 2
ttime1 = 1.e5  # time spent at rate 1
ttime2 = 2.e5  # time spent at rate 2

timelist = []

total_time = 0.
while total_time < ttime2:
    dt = np.random.exponential(scale=1/r2, size=1)
    timelist.append(dt)
    total_time += dt
timelist.pop()  # remove last event since it went past ttime2

total_time = 0.
while total_time < ttime1:
    dt = np.random.exponential(scale=1/r1, size=1)
    timelist.append(dt)
    total_time += dt
timelist.pop()  # remove last event since it went past ttime1



timelist = np.array(timelist)
#set up histogram bins
nbins = 300
tmax =  10./np.min([r1,r2])  # gives a probability of 4.5e-5 for the slow one
edges = np.linspace(0, tmax, 1 + nbins)
centers = np.linspace(edges[1]/2, (edges[-1] + edges[-2])/2 , nbins)
binwidth = edges[1] - edges[0]

hist = np.histogram(timelist,bins=edges, density=False)


#now, make a theoretical pdf based on the two exponentials
#p(t) = p(t|r1)p(r1) + p(t|r2)*p(r2)

#note that p(r1) is NOT the fraction of time the system spends at r1.
#Rather it is the probability that events occur at a time when the rate
#is r1.  Since there are more events when rates are higher, p(r1) is
#proportional to r1

pr1 = r1*ttime1/(r1*ttime1 + r2*ttime2)
pr2 = 1. - pr1
ptr1 = r1*np.exp(-r1*centers)
ptr2 = r2*np.exp(-r2*centers)
pt = ptr1*pr1 + ptr2*pr2

plt.figure()
plt.semilogy(centers, hist[0]/timelist.size, 'bo-',
             centers, pt*binwidth,'rx:');










