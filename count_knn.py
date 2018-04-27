#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 14:42:19 2018

@author: Q.Liu
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from scipy.spatial import ConvexHull
from astropy.io import ascii
from astropy.table import Table, vstack, hstack
import seaborn as sns

import collections
import BPT_query as B

def Plot_grid(j=1,q_med,q_25,q_75):
    plt.figure(figsize=(16,5))
    ax0 = plt.subplot(131)
    plt.scatter(Q.data.log_NII6583_Ha,Q.data.log_OIII5007_Hb,
                c="gray",s=5,lw=0,alpha=0.3)
    
    s = ax0.scatter(positions[:,0][~np.isinf(den_pos)],
                    positions[:,1][~np.isinf(den_pos)],
                    c=(qk_75-qk_25)[:,j][~np.isinf(den_pos)],
                    cmap='gnuplot2',edgecolors="k",
                    linewidths=0.5,s=20,alpha=0.8)
    plt.colorbar(s)
    B.BPT_set()
    
    ax1 = plt.subplot(132)
    plt.scatter(Q.data.log_NII6583_Ha,Q.data.log_OIII5007_Hb,
                c="gray",s=5,lw=0,alpha=0.3)
    
    s=ax1.scatter(positions[:,0][~np.isinf(den_pos)],
                  positions[:,1][~np.isinf(den_pos)],
                  c=qk_med[:,j][~np.isinf(den_pos)],
                  cmap='gnuplot2',edgecolors="k",
                  linewidths=0.5,s=20,alpha=0.8)
    plt.colorbar(s)
    B.BPT_set()
    
    ax2 = plt.subplot(133)
    s = ax2.scatter(Q.data.log_NII6583_Ha,Q.data.log_OIII5007_Hb,
                    c=Q.data[Q.param[j]],cmap='gnuplot2',s=20,lw=0,alpha=0.5)
    
    plt.colorbar(s)
    B.BPT_set()
    
    plt.tight_layout()
#plt.savefig("M6_Grid_M",dpi=250)


# =============================================================================
# Line used
# =============================================================================
line_used_all = ["log_NII6583_Ha","log_OIII5007_Hb","log_OIII5007_NII6583","log_OII3727_Hb",
                 "log_OII3727_NII6583","log_OII3727_OIII5007_Hb","log_OIII5007_OII3727", 
                 "log_SII6716_6731_Ha","log_SII6716_6731_NII6583","log_SII6716_SII6731"]

line_used0 = ["log_NII6583_Ha","log_OIII5007_Hb"]
line_used0_err = [l+"_err" for l in line_used0]

line_used_noOII = ["log_NII6583_Ha","log_OIII5007_Hb","log_OIII5007_NII6583", 
                   "log_SII6716_6731_Ha","log_SII6716_6731_NII6583","log_SII6716_SII6731"]
line_used_noOII_err = [l+"_err" for l in line_used_noOII]

line_used = line_used0
line_used_err = line_used0_err

  
# =============================================================================
# Read data
# =============================================================================

tab_mod = B.read_data("table/reddened_2d_fluxes.dat")
tab_mod = tab_mod.dropna(subset=line_used)

table_obs = ascii.read("table/SITELLE-NGC628.dat")
table_sym = table_obs[(table_obs["category"]==1.)|(table_obs["category"]==2.)]     # Symmetric obj
table_M = table_sym[table_sym["L_Ha"]>1e37]        # Massive obj
table_M['log_SII6716_SII6731']= np.log10(table_M['SII6716_SII6731'])
table_M['log_SII6716_SII6731_err']= 0.434*table_M['SII6716_SII6731_err']
table = table_M.to_pandas()
table1 = table[table.category==1]
table2 = table[table.category==2]  

# =============================================================================
# Train
# =============================================================================
params = ["age_Myr","M_cl","SFE","hden0"]
Q = B.BPT_query(data=tab_mod, param=params)
Q.set_data(line_used = ["log_NII6583_Ha","log_OIII5007_Hb"],table_obs=table1, use_pca=False)
Q.regression("RF-single", n_estimators=200,min_samples_leaf=10)

# =============================================================================
# KNN
# =============================================================================
from sklearn.neighbors import KNeighborsRegressor

positions = np.vstack([X.ravel(), Y.ravel()]).T

knn = KNeighborsRegressor(n_neighbors=10)

knn.fit(Q.X_train,Q.y_train)

# RUN
dist,ind = knn.kneighbors(positions)

qk_med = np.zeros((positions.shape[0],len(Q.param)))
qk_25 = np.zeros((positions.shape[0],len(Q.param)))
qk_75 = np.zeros((positions.shape[0],len(Q.param)))
for k in range(len(positions)):
    k_samples = pd.DataFrame(Q.y_train.iloc[ind[k]])
    qk_med[k] = k_samples.quantile(0.5)
    qk_25[k] = k_samples.quantile(0.25)
    qk_75[k] = k_samples.quantile(0.75)

Plot_grid(j=1,q_med=qk_med,q_25=qk_25,q_75=qk_75)
#plt.savefig("Grid_M_KNN",dpi=200)

# =============================================================================
# KNN Bagging
# =============================================================================
from sklearn.utils import resample

qK_med = np.zeros((positions.shape[0],len(Q.param)))
qK_25 = np.zeros((positions.shape[0],len(Q.param)))
qK_75 = np.zeros((positions.shape[0],len(Q.param)))

for k in range(len(positions)):
    samples = collections.defaultdict(list)
    k_samples = pd.DataFrame()
    for i in range(n_estimators):
        X_re,y_re = resample(Q.X_train,Q.y_train)
        
        knn = KNeighborsRegressor(n_neighbors=10)
        
        knn.fit(X_re,y_re)
        
        dist,ind = knn.kneighbors(positions)
        samples[i] = ind

        k_samples = pd.concat([k_samples, y_re.iloc[samples[i][k]]])
        
    qK_med[k] = k_samples.quantile(0.5)
    qK_25[k] = k_samples.quantile(0.25)
    qK_75[k] = k_samples.quantile(0.75)

    if np.mod(k+1,50)==0:
        print "Grid: %d/%d"%(k+1,len(positions))

# Plot Grid        
Plot_grid(j=1,q_med=qK_med,q_25=qK_25,q_75=qK_75)