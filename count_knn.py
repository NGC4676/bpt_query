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
from astropy.io import ascii
from astropy.table import Table, vstack, hstack
import seaborn as sns
from sklearn.neighbors import KNeighborsRegressor
from sklearn.utils import resample

import collections
import BPT_query as B

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

def Plot_grid(q_m,q_25,q_75,den_pos,j=1,clim=[None,None,None],data=tab_mod):
    plt.figure(figsize=(16,5))
    ax0 = plt.subplot(131)
    plt.scatter(data.log_NII6583_Ha,data.log_OIII5007_Hb,
                c="gray",s=5,lw=0,alpha=0.3)
    
    s = ax0.scatter(positions[:,0][(den_pos)>-15],
                    positions[:,1][(den_pos)>-15],
                    c=(q_75-q_25)[:,j][(den_pos)>-15],
                    cmap='hot',edgecolors="k",
                    linewidths=0.5,s=20,alpha=0.8)
    cb1=plt.colorbar(s)
    cb1.set_clim(clim[0])
    B.BPT_set()
    
    ax1 = plt.subplot(132)
    plt.scatter(data.log_NII6583_Ha,data.log_OIII5007_Hb,
                c="gray",s=5,lw=0,alpha=0.3)
    
    s=ax1.scatter(positions[:,0][(den_pos)>-15],
                  positions[:,1][(den_pos)>-15],
                  c=q_m[:,j][(den_pos)>-15],
                  cmap='gnuplot2',edgecolors="k",
                  linewidths=0.5,s=20,alpha=0.8)
    cb2=plt.colorbar(s)
    cb2.set_clim(clim[1])
    B.BPT_set()
    
    ax2 = plt.subplot(133)
    s = ax2.scatter(data.log_NII6583_Ha,data.log_OIII5007_Hb,
                    c=data[Q.param[j]],cmap='gnuplot2',
                    edgecolors="k",s=15,linewidths=0.2,alpha=0.5)
    
    cb3=plt.colorbar(s)
    cb3.set_clim(clim[2])
    B.BPT_set()
    
    plt.tight_layout()

def plot_KDE(den_mod,shape):
    plt.figure(figsize=(7,6))
    im = plt.imshow(den_mod.reshape(shape[0],shape[1]).T,cmap="viridis",
                    origin='lower',aspect='auto',extent=[-2.5,0.5,-3.5,1.5])
    plt.scatter(Q.data.log_NII6583_Ha,Q.data.log_OIII5007_Hb,
                c="gray",s=5,lw=0,alpha=0.3)
    cb = plt.colorbar(mappable=im)
    cb.set_label('log density',fontsize="large")
    B.BPT_set()
# =============================================================================
# Train
# =============================================================================
params = ["age_Myr","M_cl","SFE","hden0"]
Q = B.BPT_query(data=tab_mod, param=params)
Q.set_data(line_used = ["log_NII6583_Ha","log_OIII5007_Hb"],table_obs=table1, use_pca=False)
Q.regression("RF-single", n_estimators=200,min_samples_leaf=5)

n_neighbors = 10
n_estimators = 200
n_MC = 200

# =============================================================================
# KDE
# =============================================================================
from sklearn.neighbors.kde import KernelDensity
xx_mod, yy_mod = np.mgrid[-2.5:0.5:100j, -3.5:1.5:100j]
pos_mod = np.vstack([xx_mod.ravel(), yy_mod.ravel()]).T
values = np.vstack([tab_mod.log_NII6583_Ha,tab_mod.log_OIII5007_Hb]).T
kde = KernelDensity(kernel='tophat',bandwidth=0.05).fit(values)

X_mod = np.array(zip(pos_mod[:,0],pos_mod[:,1]))
den_mod = kde.score_samples(X_mod)

plot_KDE(den_mod, shape=xx_mod.shape)

# =============================================================================
# KNN
# =============================================================================
Y, X = np.mgrid[0.7:-3.4:42j,-2.25:-.25:41j]
xx,yy = X[0,:],Y[:,0]
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

# =============================================================================
# KNN Bagging
# =============================================================================
Y, X = np.mgrid[0.7:-3.4:42j,-2.25:-.25:41j]
xx,yy = X[0,:],Y[:,0]
positions = np.vstack([X.ravel(), Y.ravel()]).T
den_pos = kde.score_samples(positions)

K_samples = collections.defaultdict(list)
K_models = collections.defaultdict(list)
for i in range(n_estimators):
    X_re,y_re = resample(Q.X_train,Q.y_train)
    
    knn = KNeighborsRegressor(n_neighbors=n_neighbors)  
    knn.fit(X_re,y_re)   
    dist,ind = knn.kneighbors(positions)

    k_samples = np.concatenate([y_re.iloc[ind[k]] for k in range(len(positions))])
    k_models = np.concatenate([X_re[ind[k]] for k in range(len(positions))])
    # Each line of K_samples_MC is a bagging
    K_samples[i] = k_samples
    K_models[i] = k_models
    if np.mod(i+1,20)==0:
        print "Bagging: %d/%d"%(i+1,n_estimators)

Samples = np.zeros((len(positions),n_estimators*knn.n_neighbors,len(Q.param)))
Models = np.zeros((len(positions),n_estimators*knn.n_neighbors,2))
qK_med = np.zeros((positions.shape[0],len(Q.param)))
qK_25 = np.zeros((positions.shape[0],len(Q.param)))
qK_75 = np.zeros((positions.shape[0],len(Q.param)))
Cov_K = np.zeros((positions.shape[0],2,2))
for k in range(len(positions)):
    # Reshape K_samples_MC, each line a grid point
    Samples[k] = np.concatenate([K_samples[i][k*knn.n_neighbors:(k+1)*knn.n_neighbors] for i in range(n_estimators)])  
    qK_med[k] = pd.DataFrame(Samples[k]).quantile(0.5)
    qK_25[k] = pd.DataFrame(Samples[k]).quantile(0.25)
    qK_75[k] = pd.DataFrame(Samples[k]).quantile(0.75)
    
    Models[k] = np.concatenate([K_models[i][k*knn.n_neighbors:(k+1)*knn.n_neighbors] for i in range(n_estimators)])  
    Cov_K[k] = np.cov(((10**Models[k]-10**positions[k])**2).T) / ((10**positions[k])*10**positions[k,None].T)
    
    if np.mod(k+1,100)==0:
        print "Grid: %d/%d"%(k+1,len(positions))
        
# Plot Grid        
Plot_grid(q_m=qK_med,q_25=qK_25,q_75=qK_75,den_pos=den_pos,
          j=1,clim=[[0.,2.],[5.,7.25],[5.,7.25]])
#plt.savefig("Grid_M_KNN_bag.png",dpi=200)
Plot_grid(q_m=qK_med,q_25=qK_25,q_75=qK_75,den_pos=den_pos,
          j=2,clim=[[0.,9.],[1.,10.],[1.,10.]])
#plt.savefig("Grid_SFE_KNN_bag.png",dpi=200)
Plot_grid(q_m=qK_med,q_25=qK_25,q_75=qK_75,den_pos=den_pos,
          j=3,clim=[[0.,400.],[100.,500.],[100.,500.]])
#plt.savefig("Grid_nH_KNN_bag.png",dpi=200)

Plot_grid(q_m=qK_med,q_25=qK_25,q_75=qK_75,den_pos=den_pos,
          j=0,clim=[[0.,30.],[0.,30.],[0.,30.]])
#plt.savefig("Grid_Age_KNN_bag.png",dpi=200)


# sigma x&y
plt.figure(figsize=(11,5))
for i in range(2):
    ax=plt.subplot(1,2,i+1)
    plt.scatter(Q.data.log_NII6583_Ha,Q.data.log_OIII5007_Hb,
                c="gray",s=5,lw=0,alpha=0.3)
    s = plt.scatter(positions[:,0][(den_pos)>-15],
                    positions[:,1][(den_pos)>-15],
                    c=0.5*np.log10(Cov_K[:,i,i])[(den_pos)>-15],
                    cmap='hot',edgecolors="k",
                    linewidths=0.5,s=20,alpha=0.8)
    cb=plt.colorbar(s)
    if i==0:    cb.set_label("log $\sigma_x/x$")
    else:   cb.set_label("log $\sigma_y/y$")
    B.BPT_set()
plt.tight_layout()

# =============================================================================
# Point MC
# =============================================================================
point = [-0.6,-0.5]
distance = np.sqrt((point[0]-Q.X_train[:,0])**2+(point[1]-Q.X_train[:,1])**2)
if np.min(distance)>0.05:
    print "No models exist around!"
else:
    plt.figure(figsize=(6,5))
    plt.scatter(tab_mod.log_NII6583_Ha,tab_mod.log_OIII5007_Hb,
                        c="gray",edgecolors="k",s=15,linewidths=0.2,alpha=0.3)
    pos = np.zeros((n_MC,len(point)))
    pos[0] = point
    num=1
    while num<n_MC:
        p = point+np.random.normal(0, 0.05*np.ones_like(point))
        den = kde.score_samples(np.atleast_2d(p))
        if den>-15:
            pos[num] = p
            num+=1
        else:
            plt.scatter(p[0],p[1],s=5,alpha=0.5)
    plt.scatter(point[0],point[1],c="gold",marker="*",s=60)
    B.BPT_set()
           
K_samples_MC = collections.defaultdict(list)

for i in range(n_estimators):
    X_re,y_re = resample(Q.X_train,Q.y_train)
    
    knn = KNeighborsRegressor(n_neighbors=n_neighbors) 
    knn.fit(X_re,y_re)  
    dist,ind = knn.kneighbors(pos)

    k_samples = np.concatenate([y_re.iloc[ind[k]] for k in range(n_MC)])
    # Each line of K_samples_MC is a bagging
    K_samples_MC[i] = k_samples
    if np.mod(i+1,40)==0:
        print "Bagging: %d/%d"%(i+1,n_estimators)

Samples_MC = np.zeros((n_MC,n_estimators*knn.n_neighbors,len(Q.param)))
for k in range(n_MC):
    # Reshape K_samples_MC to have each line to be a MC
    Samples_MC[k] = np.concatenate([K_samples_MC[i][k*knn.n_neighbors:(k+1)*knn.n_neighbors] for i in range(n_estimators)])
X_all = Samples_MC.reshape((n_MC*n_estimators*knn.n_neighbors,len(Q.param)))

#Plot
plt.figure(figsize=(7,6))
bins = [10,20,15,10]
for j,(b,t,lab) in enumerate(zip(bins,Q.param,["Age(Myr)","log $M_{cloud}$","SFE","nH"])):
    ax=plt.subplot(2,2,j+1)
    
    for k in range(n_MC):
        sns.distplot(Samples_MC[k][:,j], bins=b,color='gray',
                 hist_kws={"histtype":"step","alpha":0.1},
                 kde=False,label="MC",ax=ax)
    sns.distplot(pd.Series(X_all[:,j]).sample(n_estimators*knn.n_neighbors), color="k",bins=b,
                 hist_kws={"histtype":"step","linestyle": "-",
                           "linewidth": 3,"alpha":.7},
                           kde=False,label="Center",ax=ax)
    plt.xlabel(lab)
plt.suptitle("MC KNN Distribution for Point (%.2f,%.2f)"%(point[0],point[1]),fontsize=12)
plt.tight_layout(rect=(0,0,1,0.95)) 

# =============================================================================
# LOO test
# =============================================================================
X_all = np.vstack([tab_mod[line] for line in line_used]).T
y_all = np.vstack([tab_mod[t] for t in Q.param]).T

metric = np.zeros((len(tab_mod),len(Q.param)))
for k in range(len(tab_mod)):
    
    X_tr = np.delete(X_all,k,axis=0)
    X_te = [X_all[k]]
    y_tr = np.delete(y_all,k,axis=0)
    y_te = y_all[k]
    
    K_samples = collections.defaultdict(list)
    K_models = collections.defaultdict(list)
    for i in range(n_estimators):
        X_re,y_re = resample(X_tr,y_tr)
        
        knn = KNeighborsRegressor(n_neighbors=n_neighbors)  
        knn.fit(X_re,y_re)   
        dist,ind = knn.kneighbors(X_te)
    
        k_samples = y_re[ind[0]]
        k_models = X_re[ind[0]]
        K_samples[i] = k_samples
        K_models[i] = k_models
        
    samples = np.concatenate([K_samples[i] for i in range(n_estimators)])  
    qt_med = pd.DataFrame(samples).quantile(0.5)
    qt_25 = pd.DataFrame(samples).quantile(0.25)
    qt_75  = pd.DataFrame(samples).quantile(0.75)
    
    metric[k] = ((y_te>=qt_25)&(y_te<=qt_75))

    if np.mod(k+1,500)==0:
        print "Test: %d/%d"%(k+1,len(Q.X_test))

print 1.0*metric.sum(axis=0)/len(metric)
[ 0.49842333  0.6346078   0.71344107  0.8355341 ]   
# =============================================================================
# Local
# =============================================================================
d=4800
print Q.y_test.iloc[d]
print Q.X_test[d]
positions=np.array([Q.X_test[d]])

plt.figure(figsize=(7,6))
s = plt.scatter(Q.data.log_NII6583_Ha,Q.data.log_OIII5007_Hb,
            c=Q.data.M_cl,s=40,lw=0,alpha=0.5,cmap='gnuplot')
plt.scatter(positions[0,0],positions[0,1],c="r",marker="*",s=100)
plt.scatter(Models[0][:,0],Models[0][:,1],edgecolors="lime",marker="o",c="none",lw=.5,s=80,alpha=0.1)
cb = plt.colorbar(s)
cb.set_label("M_cl")
B.BPT_set()
plt.ylim(-1.4,-1.1)
plt.xlim(-1.2,-.8)
#plt.ylim(-0.15,0.1)
#plt.xlim(-.8,-.6)

plt.figure(figsize=(8,7))
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.hist(Samples[0][:,i])
    plt.xlabel(Q.param[i])
        