#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 11:44:18 2018

@author: mac
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.pyplot as plt
import scipy.stats as st
import seaborn as sns

def BPT_set(xls=(-2.5,0.5), yls=(-3.5,1.5),c="k"):
    NII_plot = np.linspace(-2.5,0.0,100)
    NII_plot2 = np.linspace(-2.5,0.45,100)
    plt.plot(NII_plot,BPT_border(NII_plot, 'Kauffmann'),c=c,ls='--',label="Kauffmann 03")
    plt.plot(NII_plot2,BPT_border(NII_plot2, 'Kewley'),c=c,label="Kewley 01")
    plt.xlim(xls); plt.ylim(yls)
    plt.xlabel(r"log([NII]6583/H$\alpha$)",fontsize="large")
    plt.ylabel(r"log([OIII]5007/H$\beta$)",fontsize="large")

def BPT_border(log_NII_Ha,linetype = "Kauffmann"):
	if linetype=="Kauffmann": return 1.3 + 0.61 / (log_NII_Ha-0.05)
	elif linetype=="Kewley": return 1.19 + 0.61 / (log_NII_Ha - 0.47)
    
def BPT_plot_dens(NII_Ha, OIII_Hb, xls=(-2.5,0.5), yls=(-3.,1.5),bins=100,border=True):
    plt.figure(figsize=(8,6))
    if border:
        NII_plot=np.linspace(-3.,0.0,100)
        NII_plot2=np.linspace(-3.,0.45,100)
        plt.plot(NII_plot,BPT_border(NII_plot, 'Kauffmann'),c='k')
        plt.plot(NII_plot2,BPT_border(NII_plot2, 'Kewley'),c='k',ls='--')
    H, xbins, ybins = np.histogram2d(NII_Ha, OIII_Hb,
		bins=(np.linspace(xls[0]-0.5, xls[1], bins), np.linspace(yls[0]-0.5, yls[1], bins)))
    C = plt.contourf(np.log(H).T,aspect="auto",origin="lower", cmap='jet',
             extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]])
    plt.xlim(xls); plt.ylim(yls)
    plt.xlabel(r"log([NII]5007/H$\alpha$)",fontsize="large")
    plt.ylabel(r"log([OIII]5007/H$\beta$)",fontsize="large")
    cb = plt.colorbar(mappable=C)
    cb.set_label('Density',fontsize="large")

tab_lines = pd.read_table("object_tracks-lines.dat")
#tab_lines = pd.read_table("emp-pop-highres-t-object_tracks-lines.dat")

tab_shell = tab_lines[tab_lines.comp=="shell"]
tab_shell["log_age"] = np.log10(tab_shell.age_year)
tab_shell["age_Myr"] = tab_shell.age_year/1e6

Ha = tab_shell["H  1 6562.81A"]         #Ha
Hb = tab_shell["H  1 4861.33A"]         #Hb
NII = tab_shell["N  2 6583.45A"]        #NII
OIII = tab_shell["O  3 5006.84A"]       #OIII
OII = tab_shell["O  2 3726.03A"]+tab_shell['O  2 3728.81A']        #OII
SIIa = tab_shell["S  2 6716.44A"]       #SIIa
SIIb = tab_shell["S  2 6730.82A"]       #SIIb

tab_shell["OIII_Hb"] = np.log10(OIII/Hb)
tab_shell["NII_Ha"] = np.log10(NII/Ha)

tab_shell["log_OIII5007_Hb"] = np.log10(OIII/Hb)
tab_shell["log_OIII5007_NII6583"] = np.log10(OIII/NII)
tab_shell["log_OIII5007_OII3727"] = np.log10(OIII/OII)

tab_shell["log_OII3727_Hb"] = np.log10(OII/Hb)
tab_shell["log_OII3727_NII6583"] = np.log10(OII/NII)

tab_shell["log_OII3727_OIII5007_Hb"] = np.log10((OIII+OII)/Hb)

tab_shell["log_NII6583_Ha"] = np.log10(NII/Ha)

tab_shell["log_SII6716_6731_Ha"] = np.log10((SIIa+SIIb)/Ha)
tab_shell["log_SII6716_6731_NII6583"] = np.log10((SIIa+SIIb)/NII)
tab_shell["log_SII6716_SII6731"] = np.log10(SIIa/SIIb)


# =============================================================================
# # Compare BPT
# =============================================================================

#BPT_plot_dens(tab_shell["OII_Hb"],tab_shell["OIII_Hb"],
#              xls=(-1.5,2.5),yls=(-2.5,1.5),border=False)
#plt.xlabel(r"log([OII]3727/H$\beta$)",fontsize="large")
#plt.ylabel(r"log([OIII]5007/H$\beta$)",fontsize="large")
#
#
#BPT_plot_dens(tab_shell["SII_Ha"],tab_shell["OIII_Hb"],
#              xls=(-1.5,0.5),yls=(-3.25,1.25),border=False)
#plt.xlabel(r"log([SII]6716,6731/H$\alpha$)",fontsize="large")
#plt.ylabel(r"log([OIII]5007/H$\beta$)",fontsize="large")

# =============================================================================
# # Tranining set
# =============================================================================
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
from sklearn import svm

    
t1 = "log_age"
t2 = "M_cloud"

y_true = pd.DataFrame({t1:tab_shell[t1],t2:tab_shell[t2]},columns=[t1,t2])

tab_train, tab_test, y_train, y_test = train_test_split(tab_shell, y_true, test_size=0.25)

y_test = y_test.sort_index()
tab_test = tab_test.sort_index()

line_used = ["log_NII6583_Ha","log_OIII5007_Hb","log_OIII5007_NII6583","log_OII3727_Hb",
             "log_OII3727_NII6583","log_OII3727_OIII5007_Hb","log_OIII5007_OII3727", 
             "log_SII6716_6731_Ha","log_SII6716_6731_NII6583","log_SII6716_SII6731"]

X_train_0 = np.vstack([tab_train[line] for line in line_used]).T
X_test_0 = np.vstack([tab_test[line] for line in line_used]).T

X_shell = np.vstack([tab_shell[line] for line in line_used]).T

# =============================================================================
# Preprocessing
# =============================================================================

scale_flag = False
if scale_flag:
    X_train = preprocessing.scale(X_train_0) 
    X_test = preprocessing.scale(X_test_0) 
else:     
    X_train = X_train_0
    X_test = X_test_0

pca_flag = True
if pca_flag:
    pca = PCA(n_components=7)
    pca.fit(X_train)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)
    
    
# =============================================================================
#  Method
# =============================================================================
    
regr = svm.SVR(gamma=5,C=100)

#regr_1 = MLPRegressor();  regr_2 = MLPRegressor()

regr_1 = RandomForestRegressor(n_estimators=30,random_state=0)  # Best!
regr_2 = RandomForestRegressor(n_estimators=30,random_state=0)  # Best!
regr_1.fit(X_train, y_train[t1])
regr_2.fit(X_train, y_train[t2])

y_PRED = pd.DataFrame({t1:regr_1.predict(pca.transform(X_shell)),
                       t2:regr_2.predict(pca.transform(X_shell))},columns=[t1,t2])
    
y_pred = pd.DataFrame({t1:regr_1.predict(X_test),
                       t2:regr_2.predict(X_test)},columns=[t1,t2])
    
xplot = np.vstack([np.linspace(y_true[t].min()-0.25,y_true[t].max()+0.25,25)\
                   for t in [t1,t2]])
    
# Scatter
plt.figure(figsize=(8,4))
for i,t in enumerate([t1,t2]):
    ax=plt.subplot(1,2,i+1)
    plt.scatter(y_pred[t], y_test[t], alpha=0.1)
    plt.plot(xplot[i],xplot[i],c='k',ls='--',lw=3)
    plt.xlim(xplot[i,0],xplot[i,-1])
plt.tight_layout()

# KDE
plt.figure(figsize=(12,5))
for i,t in enumerate([t1,t2]):
    xx, yy = np.mgrid[xplot[i,0]:xplot[i,-1]:100j, 
                      xplot[i,0]:xplot[i,-1]:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([y_pred[t], y_test[t]])
    kernel = st.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)
    
    ax = plt.subplot(1,2,i+1)
    im = plt.imshow(f.T,cmap='hot',origin='lower',aspect='auto',
                    extent=[xplot[i,0],xplot[i,-1], 
                            xplot[i,0],xplot[i,-1]])
    plt.plot(xplot[i],xplot[i],c='w',ls='--',lw=3)
    cb = plt.colorbar(mappable=im)
    cb.set_label('Density',fontsize="large")
    plt.text(.05,.9,"R$^2=$%.3f"%(r2_score (y_test[t], y_pred[t])),
             color='gold',fontsize=15,transform=ax.transAxes)
    plt.legend(loc=3,fontsize=15)
    plt.title("Prediction for Model Parameter: %s"%t,fontsize='large')
plt.tight_layout()


# RF Tree regression voting
#plt.figure(figsize=(12,2))
#for i in range(5):
#    ax =plt.subplot(1,5,i+1)
#    t1_vote = [tree.predict(X_test)[i] for tree in regr_1.estimators_]
#    t2_vote = [tree.predict(X_test)[i] for tree in regr_2.estimators_]
#    sns.distplot(t1_vote)
#    plt.xlim(xplot[0,0]-0.25,xplot[0,-1]+0.25)
#    ax.axvline(y_test[t1].iloc[i],ls='-',c='r')
#    
#plt.tight_layout()
    

# =============================================================================
# # Monte-Carlo Estimation
# =============================================================================
#N_mc = 1000
#s=0
#y_tot = np.empty((N_mc,24,2))
#for i in range(N_mc):
#    X_prtb = X_test[s:s+24]*(1+np.random.normal(scale=0.05,size=(24,10)))
#    y_tot[i,:,0] = regr_1.predict(X_prtb)
#    y_tot[i,:,1] = regr_2.predict(X_prtb)
#
#
#fig, axes = plt.subplots(nrows=4, ncols=6, 
#                         sharex=True, sharey=True,
#                         figsize=(12,8))
#for i in range(4):
#    for j in range(6):
#        ax = axes[i,j]
##        sns.kdeplot(y_tot[:,i*6+j,0],y_tot[:,i*6+j,1],
##                    n_levels=4,ax=ax)
#        ax.scatter(y_tot[:,i*6+j,0],y_tot[:,i*6+j,1],
#                    marker='.',s=5,alpha=0.1)
#        ax.axvline(y_test[t1].iloc[s+i*6+j],ls='-',c='r')
#        ax.axvline(y_pred[t1].iloc[s+i*6+j],ls='--',c='k')
#        ax.axhline(y_test[t2].iloc[s+i*6+j],ls='-',c='r')
#        ax.axhline(y_pred[t2].iloc[s+i*6+j],ls='--',c='k')
##        ax.text(0.5, 0.85,'T: %.2f'%y_test[s+i*5+j], color='r',
##                horizontalalignment='center',transform=ax.transAxes)
##        ax.text(0.5, 0.75,'P: %.2f'%y_pred[s+i*5+j], color='k',
##                horizontalalignment='center',transform=ax.transAxes)
#        ax.set_title("M: %.1f, t: %.2f\n nH: %d, SFE: %d%%"\
#                     %(tab_test.M_cloud.iloc[s+i*6+j],
#                       tab_test.log_age.iloc[s+i*6+j],
#                       tab_test.nH.iloc[s+i*6+j],
#                       tab_test.SFE_.iloc[s+i*6+j]),
#                       fontsize=10)
#        ax.set_xlim(xplot[0,0],xplot[0,-1])
#        ax.set_ylim(xplot[1,0],xplot[1,-1])
#fig.text(0.5,0.02,'log T',fontsize=15,ha='center')
#fig.text(0.02,0.5,'log M$_{cloud}$',va='center',fontsize=15,rotation='vertical')
#plt.tight_layout(rect=(0.05,0.05,1.,1.))
##plt.savefig("MC_predict_scatter.pdf",dpi=400)

#plt.figure(figsize=(5,4))
#for i in range(5):
#    plt.plot(X_test[s+i,0], X_test[s+i,1],
#             "o",ms=10,label="%d"%i)
#BPT_set()
#plt.legend(loc="best",fontsize=10)


# =============================================================================
#  Model Prediction Likelihood?
# =============================================================================
#likelihood = False
#ind = 1
#mod = tab_shell[tab_shell["#Model_i"]==ind]
#X_mod = np.vstack([mod[line] for line in line_used]).T
#if pca_flag:
#    X_mod = pca.transform(X_mod)  
#
#mod_pred = pd.DataFrame({t1:regr_1.predict(X_mod),
#                         t2:regr_2.predict(X_mod)},columns=[t1,t2])
#                       
#N_mc = 1000
#y_tot = np.empty((N_mc,len(mod),2))
#if likelihood:
#    
#    for i in range(N_mc):
#        X_prtb = X_shell[tab_shell["#Model_i"]==ind]*(1+np.random.normal(scale=0.05,size=(len(mod),10)))
#        if pca_flag:                           
#            y_tot[i,:,0] = regr_1.predict(pca.transform(X_prtb))
#            y_tot[i,:,1] = regr_2.predict(pca.transform(X_prtb))
#        else:
#            y_tot[i,:,0] = regr_1.predict(X_prtb)
#            y_tot[i,:,1] = regr_2.predict(X_prtb)
#            
#    for k in range(len(mod)):
#        fig = plt.figure(figsize=(10,6))
#        gs = mpl.gridspec.GridSpec(2, 2, height_ratios=[1,1], width_ratios=[3,1])
#        
#        #Panel 0
#        ax0 = plt.subplot(gs[:,0])
#        sca = plt.scatter(mod.NII_Ha[:k], mod.OIII_Hb[:k],s=50,
#                    c=np.log10(mod.age_year[:k]),cmap='jet',
#                    zorder=1,label=None)
#        plt.scatter(mod.NII_Ha.iloc[k], mod.OIII_Hb.iloc[k],
#                    s=80, marker="*",edgecolors='k',
#                    c="r",label=None,zorder=3)
#        plt.plot(mod.NII_Ha[:(k+1)],mod.OIII_Hb[:(k+1)],lw=2,zorder=2)
#        BPT_set()
#        plt.title(r"$\rm M_{cl}=10^{%.1f} , n_H=%d , SFE:%.2f$"\
#                 %(mod.M_cloud.unique(),
#                   np.unique(mod.nH),
#                   0.01*np.unique(mod.SFE_)),fontsize=15)
#        cb = plt.colorbar(mappable=sca)
#        cb.set_label('log Age[yr]',fontsize="large")
#        cb.set_clim(mod.log_age.min(), mod.log_age.max())
#        
#        #Panel 1
#        v_p = mod_pred[t1].iloc[k], mod_pred[t2].iloc[k]    # Predicted value
#        v_t = mod[t1].iloc[k], mod[t2].iloc[k]  # True value
#    
#        ax1 = plt.subplot(gs[0,1])
#        t1_vote = [tree.predict(X_mod)[k] for tree in regr_1.estimators_]
#        t2_vote = [tree.predict(X_mod)[k] for tree in regr_2.estimators_]
#        sns.kdeplot(t1_vote,t2_vote,n_levels=5,ax=ax1)
#        ax1.axvline(np.median(t1_vote),ls='-',lw=10,c='g',alpha=0.1)
#        ax1.axhline(np.median(t2_vote),ls='-',lw=10,c='g',alpha=0.1)
#        plt.scatter(t1_vote,t2_vote,s=20,edgecolors="k",color="orange",alpha=0.5)
#    
#        ax1.set_xlim(xplot[0,0],xplot[0,-1])
#        ax1.set_ylim(xplot[1,0],xplot[1,-1])
#        ax1.axvline(v_p[0],ls='--',c='k')
#        ax1.axhline(v_p[1],ls='--',c='k')
#        ax1.axvline(v_t[0],ls='-',c='r')
#        ax1.axhline(v_t[1],ls='-',c='r')
#        ax1.set_xlabel("log T",fontsize=12)
#        ax1.set_ylabel("log M",fontsize=12)
#        ax1.set_title("RF vote",fontsize=12)
#        
#        #Panel 2
#        ax2 = plt.subplot(gs[1,1])
#        sns.kdeplot(y_tot[:,k,0],y_tot[:,k,1], n_levels=5,ax=ax2)
#        ax2.set_xlim(xplot[0,0],xplot[0,-1])
#        ax2.set_ylim(xplot[1,0],xplot[1,-1])
#        ax2.axvline(v_p[0],ls='--',c='k')
#        ax2.axhline(v_p[1],ls='--',c='k')
#        ax2.axvline(v_t[0],ls='-',c='r')
#        ax2.axhline(v_t[1],ls='-',c='r')
#        ax2.set_xlabel("log T",fontsize=12)
#        ax2.set_ylabel("log M",fontsize=12)
#        ax2.set_title("MC perturb",fontsize=12)
#        
#        ax0.text(-2.2,-2.5,"$\Delta T$ = %.2f"%(v_p[0]-v_t[0]),fontsize=12)
#        ax0.text(-2.2,-2.2,"$\Delta M$ = %.2f"%(v_p[1]-v_t[1]),fontsize=12)
#        plt.tight_layout(w_pad=0.2)
#        
#        plt.savefig("Evo_weight/mod%d-stage%d.png"%(ind,k),dpi=200)
#

# =============================================================================
# # Track
# =============================================================================
#plt.figure(figsize=(10,8))
#for ind in np.unique(tab_shell["#Model_i"])[0:1]:
#    mod = tab_shell[tab_shell["#Model_i"]==ind]
#    plt.scatter(mod.NII_Ha, mod.OIII_Hb,s=50,
#                c=np.log10(mod.age_year),cmap='jet',label=None)
#    plt.plot(mod.NII_Ha,mod.OIII_Hb,lw=3,
#             label=r"$\rm M_{cl}=10^5,n_H=500,SFE:%.2f$"\
#             %(0.01*np.unique(mod.SFE_)))
#NII_plot = np.linspace(-2.5,0.0,100)
#NII_plot2 = np.linspace(-2.5,0.45,100)
#plt.plot(NII_plot,BPT_border(NII_plot, 'Kauffmann'),c='k')
#plt.plot(NII_plot2,BPT_border(NII_plot2, 'Kewley'),c='k',ls='--')
#plt.xlim(-2.5,0.4); plt.ylim(-3.,1.5)
#plt.xlabel(r"log([NII]5007/H$\alpha$)",fontsize="large")
#plt.ylabel(r"log([OIII]5007/H$\beta$)",fontsize="large")
#plt.legend(loc="best",fontsize=10)
#cb = plt.colorbar()
#cb.set_label('log Age[yr]',fontsize="large")

# =============================================================================
# # MCMC
# =============================================================================
#import pymc3 as pm
#
#X = np.vstack([tab_shell.OIII_Hb,tab_shell.OII_NII,tab_shell.OIII_OII,
#            tab_shell.NII_Ha,tab_shell.SII_Ha,tab_shell.SII_SII]).T
#
#X_scaled = preprocessing.scale(X) 
#
#X1 = X_scaled[:,0]
#X2 = X_scaled[:,1]
#X3 = X_scaled[:,2]
#X4 = X_scaled[:,3]
#X5 = X_scaled[:,4]
#X6 = X_scaled[:,5]
#
#
#Y = tab_shell[t]
#Y_scale = preprocessing.scale(Y) 
#
#
#basic_model = pm.Model()
#
#with basic_model:
#
#    # Priors for unknown model parameters
#    a0 = pm.Normal('a0', 0., 1.)
#    a1 = pm.Normal('a1', 0., 1.)
#    a2 = pm.Normal('a2', 0., 1.)
#    a3 = pm.Normal('a3', 0., 1.)
#    a4 = pm.Normal('a4', 0., 1.)
#    a5 = pm.Normal('a5', 0., 1.)
#    a6 = pm.Normal('a6', 0., 1.)
#    sigma = pm.HalfNormal('sigma', sd=0.5)
#
#    # Expected value of outcome
#    mu = a0 + a1*X1 + a2*X2 + a3*X3 + a4*X4 + a5*X5 + a6*X6
#
#    # Likelihood (sampling distribution) of observations
#    Y_obs = pm.Normal('Y_obs', mu=mu, sd=sigma, observed=Y_scale)