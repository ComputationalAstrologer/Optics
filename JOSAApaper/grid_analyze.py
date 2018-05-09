#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 12:03:03 2017

@author: frazin
"""
import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
from glob import glob
import Pyr as pyr



#this code is paired with pyramid_grid_solve2.py

strehl_list = [.005, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
strehl_list = np.array(strehl_list)
no_gain = pyr.StrehlToSigma(strehl_list)*180/np.pi

keys = ['c4n0', 'c4n4', 'c5n0', 'c5n4', 'c7n0', 'c7n4']

lin_score = dict()  # mean rms error of the trials
lin_std = dict() #  std of the rms error over the trials
nl_score = dict()  # mean rms error of the trials
nl_std = dict() #  std of the rms error over the trials
for key in keys:
    lin_score[key] = np.zeros(len(strehl_list))
    lin_std[key] = np.zeros(len(strehl_list))
    nl_score[key] = np.zeros(len(strehl_list))
    nl_std[key] = np.zeros(len(strehl_list))


#%% load results

files = glob('pickles/nonlin_*.p')
for f in files:
    fp = open(f,'r')
    g = pickle.load(fp)
    fp.close()
    pv = str(np.round(np.log10(g['pcount'])))[0]
    nv = str(g['det_noise'])[0]
    ix = np.where(strehl_list == g['strehl'])[0][0]
    key = 'c' + pv + 'n' + nv
    nl_score[key][ix] = np.mean(g['nl_score'])*180/np.pi
    nl_std[key][ix] = np.std(g['nl_score'])*180/np.pi
    lin_score[key][ix] = np.mean(g['lin_score'])*180/np.pi
    lin_std[key][ix] = np.std(g['lin_score'])*180/np.pi


#%%

markersize = 10
markeredgewidth = 2
xlims = [(.08, 0.52), (.48, .92)]
ylims = [(0, 110), (0,50)]
for k_lim in range(len(xlims)):
    
    plt.figure()
    fig_handles = []
    
    
    key = 'c4n0'
    offset = [0, .005]; marker = ['s','s']; color = ['r', 'orange']
    phc = "$10^" + key[1] + "$"  # photon count string
    label = "linear: " + phc + " ph, $\sigma_{\mathrm{r}}$ " + key[3]
    hand, = plt.plot(strehl_list + offset[0], lin_score[key], marker=marker[0], fillstyle='full', markeredgecolor=color[0], markersize=markersize, markeredgewidth=markeredgewidth,
                     linestyle='dotted', color=color[0],lw=2, label=label)
    fig_handles.append(hand)
    yerr = lin_std[key]
    plt.errorbar(strehl_list + offset[0], lin_score[key], yerr=yerr, ecolor=color[0], elinewidth=3, fmt='none')
    label = "nonlinear: " + phc + " ph, $\sigma_{\mathrm{r}}$ " + key[3]
    hand, = plt.plot(strehl_list + offset[1], nl_score[key], marker=marker[1], fillstyle='none', markeredgecolor=color[1], markersize=markersize, markeredgewidth=markeredgewidth,
                     linestyle='dotted', color=color[1], lw=2, label=label)
    fig_handles.append(hand)
    yerr = nl_std[key]
    plt.errorbar(strehl_list + offset[1], nl_score[key], yerr=yerr, ecolor=color[1], elinewidth=3, fmt='none')


    key = 'c7n4'
    offset = [.01, -.005]; marker = ['o','o']; color = ['k', 'brown']
    phc = "$10^" + key[1] + "$"  # photon count string
    label = "linear: " + phc + " ph, $\sigma_{\mathrm{r}}$ " + key[3]
    hand, = plt.plot(strehl_list + offset[0], lin_score[key], marker=marker[0], fillstyle='full', markeredgecolor=color[0], markersize=markersize, markeredgewidth=markeredgewidth,
                     linestyle='solid',color=color[0],lw=2, label=label)
    fig_handles.append(hand)
    yerr = lin_std[key]
    plt.errorbar(strehl_list + offset[0], lin_score[key], yerr=yerr, ecolor=color[0], elinewidth=3, fmt='none')
    label = "nonlinear: " + phc + " ph, $\sigma_{\mathrm{r}}$ " + key[3]
    hand, = plt.plot(strehl_list + offset[1], nl_score[key], marker=marker[1], fillstyle='none', markeredgecolor=color[1], markersize=markersize, markeredgewidth=markeredgewidth,
                     linestyle='solid',color=color[1], lw=2, label=label)
    fig_handles.append(hand)
    yerr = nl_std[key]
    plt.errorbar(strehl_list + offset[1], nl_score[key], yerr=yerr, ecolor=color[1], elinewidth=3, fmt='none')


    hand, = plt.plot(strehl_list, no_gain, lw=4, color='black', label="no gain")
    fig_handles.append(hand)
    plt.legend(handles=fig_handles, loc='upper right')
    plt.xlabel("input Strehl ratio",FontSize='large')
    plt.ylabel("RMS phase error (deg)", FontSize='large')
    plt.xlim(xlims[k_lim])
    plt.ylim(ylims[k_lim])


    plt.figure()
    fig_handles=[]


    key = 'c5n0'
    offset = [0, .005]; marker = ['D','D']; color = ['b', 'cyan']
    phc = "$10^" + key[1] + "$"  # photon count string
    label = "linear: " + phc + " ph, $\sigma_{\mathrm{r}}$ " + key[3]
    hand, = plt.plot(strehl_list + offset[0], lin_score[key], marker=marker[0], fillstyle='full', markeredgecolor=color[0], markersize=markersize, markeredgewidth=markeredgewidth,
                     linestyle='solid', color=color[0],lw=2, label=label)
    fig_handles.append(hand)
    yerr = lin_std[key]
    plt.errorbar(strehl_list + offset[0], lin_score[key], yerr=yerr, ecolor=color[0], elinewidth=3, fmt='none')
    label = "nonlinear: " + phc + " ph, $\sigma_{\mathrm{r}}$ " + key[3]
    hand, = plt.plot(strehl_list + offset[1], nl_score[key], marker=marker[1], fillstyle='none', markeredgecolor=color[1], markersize=markersize, markeredgewidth=markeredgewidth,
                     linestyle='solid', color=color[1], lw=2, label=label)
    fig_handles.append(hand)
    yerr = nl_std[key]
    plt.errorbar(strehl_list + offset[1], nl_score[key], yerr=yerr, ecolor=color[1], elinewidth=3, fmt='none')

    key = 'c5n4'
    offset = [.01, -.01]; marker = ['p','p']; color = ['magenta', 'purple']
    phc = "$10^" + key[1] + "$"  # photon count string
    label = "linear: " + phc + " ph, $\sigma_{\mathrm{r}}$ " + key[3]
    hand, = plt.plot(strehl_list + offset[0], lin_score[key], marker=marker[0], fillstyle='full', markeredgecolor=color[0], markersize=markersize, markeredgewidth=markeredgewidth,
                     linestyle=':', color=color[0],lw=2, label=label)
    fig_handles.append(hand)
    yerr = lin_std[key]
    plt.errorbar(strehl_list + offset[0], lin_score[key], yerr=yerr, ecolor=color[0], elinewidth=3, fmt='none')
    label = "nonlinear: " + phc + " ph, $\sigma_{\mathrm{r}}$ " + key[3]
    hand, = plt.plot(strehl_list + offset[1], nl_score[key], marker=marker[1], fillstyle='none', markeredgecolor=color[1], markersize=markersize, markeredgewidth=markeredgewidth,
                     linestyle=':', color=color[1], lw=2, label=label)
    fig_handles.append(hand)
    yerr = nl_std[key]
    plt.errorbar(strehl_list + offset[1], nl_score[key], yerr=yerr, ecolor=color[1], elinewidth=3, fmt='none')

    hand, = plt.plot(strehl_list, no_gain, lw=4, color='black', label="no gain")
    fig_handles.append(hand)
    plt.legend(handles=fig_handles, loc='upper right')
    plt.xlabel("input Strehl ratio",FontSize='large')
    plt.ylabel("RMS phase error (deg)", FontSize='large')
    plt.xlim(xlims[k_lim])
    plt.ylim(ylims[k_lim])



#%%
#    # log plots with error bars
#    plt.figure(20)
#    fig_handles = []
#    for k in range(len(keys)):
#        key = keys[k]
#        phc = "$10^" + key[1] + "$"  # photon count string
#        label = "linear: " + phc + " ph, $\sigma_{\mathrm{r}}$ " + key[3]
#        hand, = plt.plot(strehl_list + offset[k], np.log10(lin_score[key]), marker=marker[k], fillstyle='none', markeredgecolor=color[k], markersize=markersize, markeredgewidth=markeredgewidth,
#                         label=label)
#        fig_handles.append(hand)
#        ulim = np.log10(lin_score[key] + lin_std[key]) - np.log10(lin_score[key])
#        llim = lin_score[key] - lin_std[key]
#        lt0 = np.where(llim < 1.e-8)[0]
#        llim[lt0] = 1.e-8
#        llim = np.log10(lin_score[key]) - np.log10(llim)
#        yerr = np.vstack((ulim, llim))
#        plt.errorbar(strehl_list + offset[k], np.log10(lin_score[key]), yerr=yerr, ecolor=color[k], elinewidth=3, fmt='none')
#    hand, = plt.plot(strehl_list, np.log10(no_gain), label="no gain")
#    fig_handles.append(hand)
#    plt.legend(handles=fig_handles)
#    plt.xlabel("input Strehl ratio",FontSize='large')
#    plt.ylabel("$\log_{10}$[RMS phase error (deg)]", FontSize='large')



