#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 19:54:50 2018

@author: qliu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

def BPT_border(log_NII_Ha,linetype = "Kauffmann"):
	if linetype=="Kauffmann": return 1.3 + 0.61 / (log_NII_Ha-0.05)
	elif linetype=="Kewley": return 1.19 + 0.61 / (log_NII_Ha - 0.47)

def BPT_set(xls=(-2.5,0.5), yls=(-3.5,1.5),c="k"):
    NII_plot = np.linspace(-3.,0.0,100)
    NII_plot2 = np.linspace(-3.,0.45,100)
    plt.plot(NII_plot,BPT_border(NII_plot, 'Kauffmann'),c=c)
    plt.plot(NII_plot2,BPT_border(NII_plot2, 'Kewley'),c=c,ls='--')
    plt.xlim(xls); plt.ylim(yls)
    plt.xlabel(r"log([NII]6583/H$\alpha$)",fontsize="large")
    plt.ylabel(r"log([OIII]5007/H$\beta$)",fontsize="large")

def BPT_plot_kde(NII_Ha, OIII_Hb,
                 xls=(-2.5,0.5), yls=(-3.5,1.5), 
                 bw=None, cmap='hot'):
    
    xx, yy = np.mgrid[xls[0]:xls[1]:200j, yls[0]:yls[1]:200j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([NII_Ha, OIII_Hb])
    kernel = st.gaussian_kde(values, bw_method=bw)
    f = np.reshape(kernel(positions).T, xx.shape)
    
    plt.figure(figsize=(8,6))
    im = plt.imshow(f.T,cmap=cmap,origin='lower',aspect='auto',
                    extent=[xls[0],xls[1],yls[0],yls[1]])
    
    cb = plt.colorbar(mappable=im)
    cb.set_label('Density',fontsize="large")
    BPT_set(xls, yls, c="w")
    plt.tight_layout()

def BPT_plot_2Dhist(NII_Ha, OIII_Hb, weight,
                    xls=(-2.5,0.5), yls=(-3.5,1.5), 
                    dx=0.05, dy=0.1, cmap='rainbow'):
    
    # adjust weight by timestep
    d_age = np.append(weight[1:], weight.iloc[-1]) - weight
    d_age[d_age<=0] = 0.5e6 
    w = d_age/0.5e6
    
    H, xbins, ybins = np.histogram2d(NII_Ha, OIII_Hb, weights=w,
                                     bins=(np.arange(xls[0], xls[1]+dx, dx),
                                           np.arange(yls[0], yls[1]+dy, dy)))
    
    plt.figure(figsize=(8,6))
    im = plt.imshow(np.log10(H).T, cmap=cmap, 
                    extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]],
                    origin='lower', aspect='auto',alpha=0.7)
    
    cb = plt.colorbar(mappable=im)
    cb.set_label('log Num',fontsize="large")
    BPT_set(xls, yls, c="k")
    plt.tight_layout()


# =============================================================================
# Read
# =============================================================================

tab_lines = pd.read_table("object_tracks-lines.dat")

tab_shell = tab_lines[tab_lines.comp=="shell"]

Ha = tab_shell["H  1 6562.81A"]         #Ha
Hb = tab_shell["H  1 4861.33A"]         #Hb
NII = tab_shell["N  2 6583.45A"]        #NII
OIII = tab_shell["O  3 5006.84A"]       #OIII
OII = tab_shell["O  2 3726.03A"]        #OII
SIIa = tab_shell["S  2 6716.44A"]       #SIIa
SIIb = tab_shell["S  2 6730.82A"]       #SIIb

OIII_Hb = np.log10(OIII/Hb)
NII_Ha = np.log10(NII/Ha)

# =============================================================================
# Plot
# =============================================================================

BPT_plot_2Dhist(NII_Ha, OIII_Hb, weight=tab_shell.age_year)

BPT_plot_kde(NII_Ha, OIII_Hb,bw=0.1)
