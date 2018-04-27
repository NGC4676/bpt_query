#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 17:55:31 2018

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
#tab_mod_noB_rot = read_data("a_noB_rot-object_tracks-lines.dat")
#tab_mod = ascii.read("shell_object_tracks-lines.dat")
#tab_mod_Z = ascii.read("Zmock_object_tracks-lines.dat")
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

plt.figure(figsize=(8,7))
B.BPT_contours(tab_mod.log_NII6583_Ha,tab_mod.log_OIII5007_Hb,weights=tab_mod.age_Myr)
plt.errorbar(table1.log_NII6583_Ha, table1.log_OIII5007_Hb,
             xerr=table1.log_NII6583_Ha_err,yerr=table1.log_OIII5007_Hb_err,
             c="gray",ls="",marker="o",ms=4,mfc="gold",mec="k",lw=0.5,alpha=0.7,
             label=r"SITELLE: symmetric + L$\rm_{H\alpha}>10^{37}$",zorder=3)
#plt.errorbar(table2.log_NII6583_Ha, table2.log_OIII5007_Hb,
#             xerr=table2.log_NII6583_Ha_err,yerr=table2.log_OIII5007_Hb_err,
#             c="gray",ls="",marker="^",ms=4,mfc="orange",mec="k",lw=0.5,alpha=0.3,
#             label=r"SITELLE: symmetric + L$\rm_{H\alpha}>10^{37}$",zorder=2)
plt.scatter(tab_mod.log_NII6583_Ha,tab_mod.log_OIII5007_Hb,c=tab_mod.d_age,
            cmap='gnuplot2',s=5,lw=0,alpha=0.7)
cb = plt.colorbar()
cb.set_label(r'$\rm\Delta{\ }Age(Myr)$',fontsize="large")
plt.clim(0.,2.)
pos = [[-0.6,-0.5]]
plt.scatter(pos[0][0],pos[0][1],c="lime",s=100,edgecolor='gray',alpha=1.,zorder=4)
plt.scatter(pos[0][0],pos[0][1],marker='x',c="k",s=40,lw=1,alpha=1.,zorder=5)
   
# =============================================================================
#  Query Tool example
# =============================================================================
#Q = BPT_query(data=tab_mod,param=["log_age","M_cloud","SFE"])
#
#Q.set_data(line_used = ["log_NII6583_Ha","log_OIII5007_Hb"])
#
#Q.regression("SVR", gamma=50.,C=100)
#p2 = Q.predict([[-0.6,-0.2]])
#
#Q.regression("Linear")
#p3 = Q.predict([[-0.6,-0.2]])
#
#Q.regression("RF", n_estimators=100)
#p1, p1_vote = Q.predict([[-0.6,-0.2]])
#p1_vote = Q.predict_draw([[-0.6,-0.2]], "log_age", "M_cloud")


## Comparsion between methods
#with sns.axes_style("whitegrid"):
#    plt.figure(figsize=(8,7))
#    sns.violinplot(data=p1)
#    xx = np.array([0.,1.,2.])
#    plt.scatter(xx,p1_vote.mean(),color="k",marker="o",s=100,label="RF (mean)",alpha=.9,zorder=4)
#    plt.scatter(xx-0.1,p2,color="r",marker="s",s=80,label="SVM",alpha=.9)
#    plt.scatter(xx+0.1,p3,color="m",marker="p",s=120,label="Linear",alpha=.9)
#    plt.title("Prediction using different ML methods",fontsize=15)
#    plt.legend(loc="best",fontsize="large")
 
# =============================================================================
# Normalize
# =============================================================================
# Start the machine
params = ["log_age","M_cl","SFE","hden0"]
Q = B.BPT_query(data=tab_mod, param=params)
Q.set_data(line_used = ["log_NII6583_Ha","log_OIII5007_Hb"])
Q.regression("RF", n_estimators=100)

data_obs = pd.DataFrame(np.vstack([table_M["log_NII6583_Ha"],
                                   table_M["log_OIII5007_Hb"]]).T).dropna()

# Check Prediction
P1,P1_vote = Q.predict(data_obs,vote=True)
fig = plt.figure(figsize=(10,8))
for i, xlab in enumerate(["log Age","M cloud","SFE","nH"]):
    ax = plt.subplot(2,2,i+1)
    sns.distplot(P1[:,i])
    plt.xlabel(xlab,fontsize=12)
 
    
## Normalize by PDF
#from sklearn.neighbors import KernelDensity
#x_plot = np.vstack([np.linspace(5.,7.65,1000),
#                    np.linspace(5.,7.5,1000),
#                    np.linspace(2.,20.,1000),
#                    np.linspace(100.,500,1000)]).T
#bws = [Q.y_true.iloc[:,i].std()*len(tab_mod)**(-0.2) for i in range(4)]
#
#for i in range(4):
#    X = P1[:,i][:,None]
#    kde = KernelDensity(kernel='tophat', bandwidth=bws[i]).fit(X)
#    log_dens = kde.score_samples(x_plot[:,i][:,None])
#    plt.figure(figsize=(10,3))
#    ax = plt.subplot(131)
#    sns.distplot(P1[:,i])
#    plt.plot(x_plot[:,i],np.exp(log_dens),label="Predition")
#    plt.legend()
#    Y = Q.y_true.iloc[:,i][:,None]
#    kde2 = KernelDensity(kernel='gaussian', bandwidth=3*bws[i]).fit(Y)
#    log_dens2 = kde2.score_samples(x_plot[:,i][:,None])
#    ax2 = plt.subplot(132)
#    sns.distplot(Q.y_true.iloc[:,i])
#    plt.plot(x_plot[:,i],np.exp(log_dens2),label="Model")
#    plt.legend()
#    ax2 = plt.subplot(133)
#    plt.plot(x_plot[:,i],np.exp(log_dens)/np.exp(log_dens2),label="Normalized")
#    plt.legend()
#

## Normalize by number    
#bin_range = [np.linspace(4.75,7.75,7),
#             [1.,3.,9.,15,20.],
#             [50,150,250,550]]
#xplot = [np.linspace(5.,7.5,6),
#         [2.,5.,10.,20.],
#         [100.,200.,500.]]
# 
#for i in range(3):               
#    Hx,binx = np.histogram(P1_vote.iloc[:,(i+1)],bins=bin_range[i])
#    Hy,biny = np.histogram(Q.y_true.iloc[:,(i+1)],bins=bin_range[i])
#    
#    plt.figure(figsize=(10,3))
#    for j,(H,tit) in enumerate(zip([Hx,Hy,Hx*1.0/Hy],
#                                  ["Predition","Model","Normalized Prediction"])):
#        ax = plt.subplot(1,3,j+1)
#        sns.barplot(xplot[i], H, alpha=0.8,label=lab)
#        plt.title(param_pred[i+1]+": %s"%tit,fontsize=12)
#    plt.tight_layout()
   
    
    
# =============================================================================
# MC on error
# =============================================================================
data_obs_lin,data_obs_err_lin = B.get_data_err_linear(table_M,line_used,line_used_err)

params = ["log_age","M_cl","SFE","hden0"]
Q = B.BPT_query(data=tab_mod, param=params)
Q.set_data(line_used = ["log_NII6583_Ha","log_OIII5007_Hb"])
Q.regression("RF-single", n_estimators=100,min_samples_leaf=30,oob_score=True)


fig = plt.figure(figsize=(10,8))
for k in range(100):
    noise = np.random.normal(0, data_obs_err_lin)
    data_prtb = np.log10(data_obs_lin+noise).dropna()
    P1 = Q.predict(data_prtb,vote=False)

    for i, xlab in enumerate(["log Age","M cloud","SFE","nH"]):
        ax = plt.subplot(2,2,i+1)
        plt.xlabel(xlab,fontsize=12)
        sns.distplot(P1[:,i],
                     hist_kws={"histtype":"step","alpha":0.1},
                     kde_kws={"alpha":0.1},ax=ax)
plt.tight_layout()
    
# =============================================================================
# Possibility Histogram
# =============================================================================
data_obs_lin,data_obs_err_lin = B.get_data_err_linear(table_M,line_used,line_used_err)

params = ["Z_sol","M_cloud","SFE","nH"]
Q = B.BPT_query(data=tab_mod, param=params)
Q.set_data(line_used = line_used)
Q.regression("RF-single", n_estimators=100)

## Scatter Sands
#sand_tot = np.zeros((1,4))
#for k in range(302):
#    data_obs_lin_one = np.tile(data_obs_lin.iloc[k],(250,1))
#    noise_one = np.random.normal(0, data_obs_err_lin.iloc[k],(250,2))
#    data_prtb_one = pd.DataFrame(np.log10(data_obs_lin_one + noise_one)).dropna()
#    P1, P1_vote = Q.predict(data_prtb_one,vote=True)
#
#    sand = np.random.uniform(P1_vote.quantile(0.1),P1_vote.quantile(0.9),size=(10000,4))
#    sand_tot = np.vstack([sand_tot,sand])
#sand_tot = sand_tot[1:]
#
#fig = plt.figure(figsize=(10,8))
#for i, xlab in enumerate(["Z","M cloud","SFE","nH"]):
#    ax = plt.subplot(2,2,i+1)
#    sns.distplot(sand_tot[:,i],kde=False,bins=15)
#    plt.xlabel(xlab,fontsize=12)

# Histo bins
bin_range = [np.linspace(4.825,7.375,11),
            [0.5,1.5,3.0,5.0,7.0,9.0,15.],
            [50,150,300,700]]

xplot = [np.linspace(5.,7.25,10),
        [1.,2.,4.,6.,8.,10.],
        [100.,200.,500.]]

fig = plt.figure(figsize=(10,8))
for i, xlab in enumerate(["Z","M cloud","SFE","nH"]):
    ax = plt.subplot(2,2,i+1)
    n = len(xplot[i])
    stone_tot = np.zeros(n)
    for k in range(len(data_obs)):
        data_obs_lin_one = np.tile(data_obs_lin.iloc[k],(250,1))
        noise_one = np.random.normal(0, data_obs_err_lin.iloc[k],(250,len(line_used)))
        data_prtb_one = pd.DataFrame(np.log10(data_obs_lin_one + noise_one)).dropna()
        P1, P1_vote = Q.predict(data_prtb_one,vote=True)
        
        
        bin_num,_ = np.histogram(P1_vote[params[i]],bins=bin_range[i])
        stone = (bin_num>0.2/n*len(P1_vote)).astype("float")
        print "%s : "%xlab,stone
    
        stone_tot = stone_tot + stone
    sns.barplot(xplot[i], stone_tot, alpha=0.8)
    plt.xlabel(xlab)
    
# =============================================================================
# Grid Prediction for Model : Slice
# =============================================================================
import BPT_query as B
params = ["log_age","M_cl","SFE","hden0"]
cond = (tab_mod.log_NII6583_Ha>-2.5) & (tab_mod.log_OIII5007_Hb>-3.5)
Q = B.BPT_query(data=tab_mod, param=params)
Q.set_data(line_used = ["log_NII6583_Ha","log_OIII5007_Hb"],table_obs=table1, use_pca=False)
Q.regression("RF-single", n_estimators=200,min_samples_leaf=30)

Q.draw_pred_MC_hist([[-.65,-.5]],xlabels=['log age','M cloud','SFE','nH'])
Q.draw_pred_MC_sca([[-.8,0.]],labels=['log age','M cloud','SFE','nH'])

[[-0.8,0.1]]
[[-0.7,-0.2]]
[[-0.6,-0.5]]
[[-0.5,-0.8]]


Y, X = np.mgrid[0.25:-1.15:15j,-1.:-.4:13j]
xx,yy = X[0,:],Y[:,0]
positions = np.vstack([X.ravel(), Y.ravel()]).T

par_guess,std_guess,up_guess,down_guess = Q.pred_MC_grid(positions)

ax0 = Q.draw_MC_grid_slice(xx,yy,7,axis='col')
#ax0.pcolormesh(X,Y,np.ones_like(X),cmap='gray',facecolor='none', 
#               linewidth=1.,edgecolor='k', alpha=0.7, zorder=1)

ax0.errorbar(table.log_NII6583_Ha, table.log_OIII5007_Hb,
             xerr=table.log_NII6583_Ha_err,yerr=table.log_OIII5007_Hb_err,
             c="gray",ls="",marker="o",ms=4,mfc="gold",mec="k",lw=.8,alpha=0.7,
             label=r"SITELLE: symmetric + L$\rm_{H\alpha}>10^{37}$",zorder=1)


from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
fig = plt.figure(figsize=(9,7))
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, par_guess[:,3].reshape((15,11)),linewidth=0,alpha=0.8)
ax.plot_surface(X, Y, up_guess[:,3].reshape((15,11)),linewidth=0,alpha=0.8)
ax.plot_surface(X, Y, down_guess[:,3].reshape((11,15)),linewidth=0,alpha=0.8)
#ax.view_init(0,0)


# =============================================================================
# Classification
# =============================================================================
import BPT_query as B

def binning(x,a1,a2):
    if x<=a1: y=0
    elif x<=a2: y=1
    else: y=2
    return y

params = ["M_cl","SFE","hden0"]
tab_mod["M_bin"] = [binning(m,5.75,6.5) for m in tab_mod.M_cl]
tab_mod["SFE_bin"] = [binning(m,3,7) for m in tab_mod.SFE]
tab_mod["nH_bin"] = [binning(m,200.,500.) for m in tab_mod.hden0]


cond = (tab_mod.log_NII6583_Ha>-2.5) & (tab_mod.log_OIII5007_Hb>-3.5)

Q = B.BPT_query(data=tab_mod, param=["M_bin","SFE_bin","nH_bin"])
Q.set_data(line_used = line_used0, table_obs=table1, use_pca=False)
#Q.classification("RF",n_estimators=200,min_samples_leaf=20)
Q.classification("SVM",C=1,gamma=10,probability=True)


# =============================================================================
# 
# =============================================================================
params = ["log_age","SFE","hden0"]
Q = B.BPT_query(data=tab_mod, param=params)
Q.set_data(line_used = ["log_NII6583_Ha","log_OIII5007_Hb","M_t"],use_pca=False,table_obs=table1)
Q.regression("RF-single", n_estimators=200)

from sklearn.ensemble import ExtraTreesRegressor
regr_multi = ExtraTreesRegressor(200)
regr_multi.fit(Q.X_train, Q.y_train)

y_pred = regr_multi.predict(Q.X_test)
y_pred2 = pd.DataFrame({t:col for (t,col) in zip(Q.param,y_pred.T)},columns=Q.param)

[B.r2_score(Q.y_test[t], y_pred2[t]) for t in Q.param]
