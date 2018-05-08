#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 14:58:18 2018

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
Q.regression("RF-single", n_estimators=200,min_samples_leaf=2)

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
# LOO test
# =============================================================================
X_all = np.vstack([tab_mod[line] for line in line_used]).T
y_all = np.vstack([tab_mod[t] for t in Q.param]).T

metric = np.zeros((len(tab_mod),len(Q.param)))
for k,ind in enumerate(tab_mod.index):
    Q = B.BPT_query(data=tab_mod.drop(ind), param=params)
    Q.set_data(line_used = ["log_NII6583_Ha","log_OIII5007_Hb"],table_obs=table1, use_pca=False)
    Q.regression("RF-single", n_estimators=200,min_samples_leaf=2)
    
    
    leaf_nodes_test = Q.regressor.apply([X_all[k]])
    leaf_nodes_mod = Q.regressor.apply(Q.X_train)
    
    nodes = collections.defaultdict(list)
    
    for i in range(n_estimators):
        node_ = leaf_nodes_test[0][i]
        nodes[i] = np.where(leaf_nodes_mod[:,i] == node_)
    
    leaf_samples = np.concatenate([Q.y_train.iloc[nodes[i]] for i in range(n_estimators)])
    
    qt_med = pd.DataFrame(leaf_samples).quantile(0.5)
    qt_25 = pd.DataFrame(leaf_samples).quantile(0.25)
    qt_75 = pd.DataFrame(leaf_samples).quantile(0.75)
    
    metric[k] = ((y_all[k]>=qt_25)&(y_all[k]<=qt_75))
    if np.mod(k+1,200)==0:
        print "Test: %d/%d"%(k+1,len(Q.X_test))

print 1.0*metric.sum(axis=0)/len(metric)
     
# =============================================================================
# New test
# =============================================================================
leaf_nodes_test = Q.regressor.apply(Q.X_test)
leaf_nodes_mod = Q.regressor.apply(Q.X_train)

Leaf_samples = collections.defaultdict(list)
for k in range(len(Q.X_test)):

    nodes = collections.defaultdict(list)

    for i in range(n_estimators):
        node_ = leaf_nodes_test[k][i]
        nodes[i] = np.where(leaf_nodes_mod[:,i] == node_)

    leaf_samples = np.concatenate([Q.y_train.iloc[nodes[i]] for i in range(n_estimators)])
    Leaf_samples[k] = leaf_samples
    
    if np.mod(k+1,500)==0:
        print "Test: %d/%d"%(k+1,len(Q.X_test))
    
qt_med = np.zeros((Q.X_test.shape[0],len(Q.param)))
qt_25=np.zeros((Q.X_test.shape[0],len(Q.param)))
qt_75=np.zeros((Q.X_test.shape[0],len(Q.param)))
for k in range(len(Q.X_test)):
    leaf_samples = Leaf_samples[k]
    qt_med[k] = pd.DataFrame(leaf_samples).quantile(0.5)
    qt_25[k] = pd.DataFrame(leaf_samples).quantile(0.25)
    qt_75[k] = pd.DataFrame(leaf_samples).quantile(0.75)
    if np.mod(k+1,500)==0:
        print "Test: %d/%d"%(k+1,len(Q.X_test))
        
metric = ((Q.y_test>=qt_25)&(Q.y_test<=qt_75)).sum(axis=0)/len(Q.y_test)

# =============================================================================
# Point MC
# =============================================================================
point = [-0.6,-0.5]
pos = np.tile(point,(n_MC,1))
noise = np.random.normal(0, 0.05*np.ones_like(pos))
pos += noise

pos[-1] = point         

leaf_nodes = Q.regressor.apply(pos)
leaf_nodes_mod = Q.regressor.apply(Q.X_train)

Leaf_samples_MC = collections.defaultdict(list)    
for k in range(n_MC):
    
    nodes = collections.defaultdict(list)
    
    for i in range(n_estimators):
        node_ = leaf_nodes[k][i]
        nodes[i] = np.where(leaf_nodes_mod[:,i] == node_)
        
    leaf_samples = np.concatenate([Q.y_train.iloc[nodes[i]] for i in range(n_estimators)])
    Leaf_samples_MC[k] = leaf_samples   
     
    if np.mod(k+1,50)==0:
        print "MC: %d/%d"%(k+1,n_MC)
        
# Plot 1
plt.figure(figsize=(7,6))
bins = [10,20,15,10]
for k in range(n_MC):
    leaf_samples = Leaf_samples_MC[k]
    for j,(b,t,lab) in enumerate(zip(bins,Q.param,["Age(Myr)","log $M_{cloud}$","SFE","nH"])):
        ax=plt.subplot(2,2,j+1)
        sns.distplot(leaf_samples[:,j], bins=b,color='gray',
             hist_kws={"histtype":"step","alpha":0.1},
             kde=False,label="MC",ax=ax)
for j,(b,t,lab) in enumerate(zip(bins,Q.param,["Age(Myr)","log $M_{cloud}$","SFE","nH"])):
    ax=plt.subplot(2,2,j+1)
    sns.distplot(leaf_samples[:,j], color="k",bins=b,
                 hist_kws={"histtype":"step","linestyle": "--",
                               "linewidth": 3,"alpha":.7},
                               kde=False,label="Center",ax=ax)
    X_all = np.concatenate([Leaf_samples_MC[k] for k in range(n_MC)])
    n_med = np.median([len(Leaf_samples_MC[i]) for i in range(n_MC)]).astype(int)
    sns.distplot(pd.Series(X_all[:,j]).sample(n_med), bins=b,
                 hist_kws={"histtype":"step","linestyle": "-",
                           "linewidth": 3,"alpha":.7},
                           color="k",kde=False,label="MC",ax=ax)
    plt.xlabel(lab)
plt.suptitle("MC Leaf Distribution for Point (%.2f,%.2f)"%(point[0],point[1]),fontsize=12)
plt.tight_layout(rect=(0,0,1,0.95))
#plt.savefig("Point_Dist_MC",dpi=200)


#Plot 2
plt.figure(figsize=(8,7))
y_pred=Q.regressor.predict(pos)
for k in range(n_MC):
    leaf_samples = Leaf_samples_MC[k]
    for j,(b,t) in enumerate(zip(bins,Q.param)):
        ax=plt.subplot(2,2,j+1)
        ax.scatter(leaf_samples[:,j],(leaf_samples[:,j]-y_pred[k,j]),color="gray",s=3,alpha=0.05)
for j,(b,t) in enumerate(zip(bins,Q.param)):
    ax=plt.subplot(2,2,j+1)    
    plt.scatter(leaf_samples[:,j],(leaf_samples[:,j]-y_pred[k,j]),color="orange",s=6,alpha=.7)
    plt.ylabel("$%s_{true}-%s_{pred}$"%(t,t))
    plt.xlabel(t)
plt.tight_layout()
    
# =============================================================================
# Grid
# =============================================================================
#Y, X = np.mgrid[0.25:-1.15:15j,-1.:-.4:13j]
#Y, X = np.mgrid[0.5:-1.5:21j,-1.4:-.4:21j]
Y, X = np.mgrid[0.7:-3.4:42j,-2.25:-.25:41j]
xx,yy = X[0,:],Y[:,0]
positions = np.vstack([X.ravel(), Y.ravel()]).T

den_pos = kde.score_samples(positions)

# RUN
Models = collections.defaultdict(list)
Samples = collections.defaultdict(list)
leaf_nodes = Q.regressor.apply(positions)
leaf_nodes_mod = Q.regressor.apply(Q.X_train)
    
for k in range(len(positions)): 
    nodes = collections.defaultdict(list)
    for i in range(n_estimators):
        node_ = leaf_nodes[k][i]
        nodes[i] = np.where(leaf_nodes_mod[:,i] == node_)
    
    leaf_models = np.concatenate([Q.X_train[nodes[i]] for i in range(n_estimators)])
    Models[k] = leaf_models
    leaf_samples = np.concatenate([Q.y_train.iloc[nodes[i]] for i in range(n_estimators)])
    Samples[k] = leaf_samples
    
    if np.mod(k+1,100)==0:
        print "Grid: %d/%d"%(k+1,len(positions))    
        

q_med = np.zeros((positions.shape[0],len(Q.param)))
#q_mean = np.zeros((positions.shape[0],len(Q.param)))
q_25 = np.zeros((positions.shape[0],len(Q.param)))
q_75 = np.zeros((positions.shape[0],len(Q.param)))
Cov_leaf = np.zeros((positions.shape[0],2,2))
for k in range(len(positions)):
    leaf_samples = Samples[k]
#    q_mean[k] = leaf_samples.mean()
    q_med[k] = pd.DataFrame(leaf_samples).quantile(0.5)
    q_25[k] = pd.DataFrame(leaf_samples).quantile(0.25)
    q_75[k] = pd.DataFrame(leaf_samples).quantile(0.75)
    Cov_leaf[k] = np.cov(((10**Models[k]-10**positions[k])**2).T) / ((10**positions[k])*10**positions[k,None].T)
    if np.mod(k+1,100)==0:
        print "Grid: %d/%d"%(k+1,len(positions))

Plot_grid(q_m=q_med,q_25=q_25,q_75=q_75,den_pos=den_pos,
          j=1,clim=[[0.,2.],[5.,7.25],[5.,7.25]])
#plt.savefig("Grid_M_RF.png",dpi=200)
Plot_grid(q_m=q_med,q_25=q_25,q_75=q_75,den_pos=den_pos,
          j=2,clim=[[0.,9.],[1.,10.],[1.,10.]])
#plt.savefig("Grid_SFE_RF.png",dpi=200)
Plot_grid(q_m=q_med,q_25=q_25,q_75=q_75,den_pos=den_pos,
          j=3,clim=[[0.,400.],[100.,500.],[100.,500.]])
#plt.savefig("Grid_nH_RF.png",dpi=200)

Plot_grid(q_m=q_med,q_25=q_25,q_75=q_75,den_pos=den_pos,
          j=0,clim=[[0.,30.],[0.,30.],[0.,30.]])
#plt.savefig("Grid_Age_RF.png",dpi=200)


# Sigma x&y
plt.figure(figsize=(11,5))
for i in range(2):
    ax=plt.subplot(1,2,i+1)
    plt.scatter(Q.data.log_NII6583_Ha,Q.data.log_OIII5007_Hb,
                c="gray",s=5,lw=0,alpha=0.3)
    s = plt.scatter(positions[:,0][(den_pos)>-15],
                    positions[:,1][(den_pos)>-15],
                    c=0.5*np.log10(Cov_leaf[:,i,i])[(den_pos)>-15],
                    cmap='hot',edgecolors="k",
                    linewidths=0.5,s=20,alpha=0.8)
    cb=plt.colorbar(s)
    if i==0:    cb.set_label("log $\sigma_x/x$")
    else:   cb.set_label("log $\sigma_y/y$")
    B.BPT_set()
plt.tight_layout()

cond=(positions[:,0]>-2.5)&(positions[:,1]>-3.5)
plt.figure(figsize=(11,5))
for i in range(2):
    ax=plt.subplot(1,2,i+1)
    plt.scatter(Q.data.log_NII6583_Ha,Q.data.log_OIII5007_Hb,
                c="gray",s=5,lw=0,alpha=0.3)
    s = plt.scatter(positions[:,0][cond],
                    positions[:,1][cond],
                    c=0.5*np.log10(Cov_leaf[:,i,i])[cond],
                    cmap='hot',edgecolors="k",
                    linewidths=0.5,s=20,alpha=0.8)
    cb=plt.colorbar(s)
    if i==0:    cb.set_label("log $\sigma_x/x$")
    else:   cb.set_label("log $\sigma_y/y$")
    B.BPT_set()
plt.tight_layout()

# =============================================================================
# Mass constrain
# =============================================================================
params = ["age_Myr","M_cl","SFE","hden0"]
Q = B.BPT_query(data=tab_mod, param=params)
Q.set_data(line_used = ["log_NII6583_Ha","log_OIII5007_Hb","M_t"],table_obs=table1, use_pca=False)
Q.regression("RF-single", n_estimators=n_estimators,min_samples_leaf=10)

positions = np.vstack([X.ravel(), Y.ravel(),np.ones(1722)*6.0+np.random.normal(0,0.1,1722)]).T

#KDE
xx_mod, yy_mod,zz_mod = np.mgrid[-2.5:0.5:100j, -3.5:1.5:100j, 5.:7.25:10j]
pos_mod = np.vstack([xx_mod.ravel(), yy_mod.ravel(),zz_mod.ravel()]).T
values = np.vstack([tab_mod.log_NII6583_Ha,tab_mod.log_OIII5007_Hb,tab_mod.M_t]).T
kde = KernelDensity(kernel='tophat',bandwidth=0.1).fit(values)

X_mod = np.array(zip(pos_mod[:,0],pos_mod[:,1],pos_mod[:,2]))
den_mod = kde.score_samples(X_mod)
den_pos = kde.score_samples(positions)

plt.figure(figsize=(7,6))
im = plt.imshow(den_mod.reshape(xx_mod.shape[0],xx_mod.shape[1],xx_mod.shape[2])[:,:,4].T,
                cmap="viridis",origin='lower',aspect='auto',extent=[-2.5,0.5,-3.5,1.5])
plt.scatter(Q.data.log_NII6583_Ha,Q.data.log_OIII5007_Hb,
            c="gray",s=5,lw=0,alpha=0.3)
cb = plt.colorbar(mappable=im)
cb.set_label('log density',fontsize="large")
B.BPT_set()

#RUN
leaf_nodes = Q.regressor.apply(positions)
leaf_nodes_mod = Q.regressor.apply(Q.X_train)

Leaf_samples = collections.defaultdict(list)   
for k in range(len(positions)):
    
    nodes = collections.defaultdict(list)
    for i in range(n_estimators):
        node_ = leaf_nodes[k][i]
        nodes[i] = np.where(leaf_nodes_mod[:,i] == node_)
    leaf_samples = np.concatenate([Q.y_train.iloc[nodes[i]] for i in range(n_estimators)])
    Leaf_samples[k] = leaf_samples
    
    if np.mod(k+1,100)==0:
        print "Grid: %d/%d"%(k+1,len(positions))
        
#q_mean = np.zeros((positions.shape[0],len(Q.param)))
q_med = np.zeros((positions.shape[0],len(Q.param)))
q_25 = np.zeros((positions.shape[0],len(Q.param)))
q_75 = np.zeros((positions.shape[0],len(Q.param)))
for k in range(len(positions)):
    leaf_samples = Leaf_samples[k]
#    q_mean[k] = leaf_samples.mean()
    q_med[k] = pd.DataFrame(leaf_samples).quantile(0.5)
    q_25[k] = pd.DataFrame(leaf_samples).quantile(0.25)
    q_75[k] = pd.DataFrame(leaf_samples).quantile(0.75)
    
    if np.mod(k+1,100)==0:
        print "Grid: %d/%d"%(k+1,len(positions))


#Plot    
Plot_grid(q_m=q_med,q_25=q_25,q_75=q_75,den_pos=den_pos,j=1,
          clim=[[0.,2.],[5.,7.25],[5.,7.25]],data=Q.data[abs(Q.data.M_t-6)<0.1])
#plt.savefig("M6_Grid_M_RF.png",dpi=200)
Plot_grid(q_m=q_med,q_25=q_25,q_75=q_75,den_pos=den_pos,j=2,
          clim=[[0.,9.],[1.,10.],[1.,10.]],data=Q.data[abs(Q.data.M_t-6)<0.1])
#plt.savefig("M6_Grid_SFE_RF.png",dpi=200)
Plot_grid(q_m=q_med,q_25=q_25,q_75=q_75,den_pos=den_pos,j=3,
          clim=[[0.,400.],[100.,500.],[100.,500.]],data=Q.data[abs(Q.data.M_t-6)<0.1])
#plt.savefig("M6_Grid_nH_RF.png",dpi=200)

Plot_grid(q_m=q_med,q_25=q_25,q_75=q_75,den_pos=den_pos,j=0,
          clim=[[0.,30.],[0.,30.],[0.,30.]],data=Q.data[abs(Q.data.M_t-6)<0.1])
#plt.savefig("M6_Grid_Age_RF.png",dpi=200)

# =============================================================================
# Partition Visual
# =============================================================================
plt.figure(figsize=(7,6))
plt.scatter(Q.data.log_NII6583_Ha,Q.data.log_OIII5007_Hb,
            c="gray",s=5,lw=0,alpha=0.3)
for n in np.unique(leaf_nodes[:,2]):
    pos_temp = positions[~np.isinf(den_pos)]
    ln=Q.regressor.apply(pos_temp)[:,2]
    s = plt.plot(pos_temp[:,0][ln==n],pos_temp[:,1][ln==n],"s",ms=6,alpha=0.7)
B.BPT_set()

# =============================================================================
# Local regions
# =============================================================================
d=4800
print Q.y_test.iloc[d]
print Q.X_test[d]
positions=np.array([Q.X_test[d]])

plt.figure(figsize=(7,6))
s = plt.scatter(Q.data.log_NII6583_Ha,Q.data.log_OIII5007_Hb,
            c=Q.data.M_cl,s=40,lw=0,alpha=0.5,cmap='gnuplot')
plt.scatter(positions[k][0],positions[k][1],c="r",marker="*",s=100)
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
    plt.hist(leaf_samples[:,i])
    plt.xlabel(Q.param[i])
    