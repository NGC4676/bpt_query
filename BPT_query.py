#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 16:22:56 2018

@author: Q.Liu
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import scipy.stats as st
from scipy.ndimage import gaussian_filter
from astropy.io import ascii
from astropy.table import Table, vstack, hstack
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

def BPT_plot_dens_models(x, y, bins=80, weights=None,
                         xls=(-2.5,0.5), yls=(-3.5,1.5)):
    if weights is None: 
        age = weights
        d_age = np.append(age[1:],age.iloc[-1]) - age
        tp_last = (np.argwhere(d_age<=0)-1).ravel()  #last time point position
        d_age[d_age<=0]=d_age.iloc[tp_last]
        w = d_age
    else: w = None
    
    H, xbins, ybins = np.histogram2d(x, y, weights=w,
                                     bins=(np.linspace(xls[0]-0.25, xls[1]+0.25, bins), 
                                           np.linspace(yls[0]-0.25, yls[1]+0.25, bins)))
    plt.contourf(np.log(H).T,aspect="auto",origin="lower", cmap='jet',
                 extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]])
    BPT_set(xls,yls)

def BPT_contours(x, y, bins=50, weights=None,
                 xls=(-2.5,0.5), yls=(-3.5,1.5)):
    if weights is None: 
        w = None
    else:
        age = weights
        d_age = np.append(age[1:],age.iloc[-1]) - age
        tp_last = (np.argwhere(d_age<=0)-1).ravel()  #last time point position
        w = d_age.copy()
        w.values[w<=0] = d_age.iloc[tp_last]
    
    H, xbins, ybins = np.histogram2d(x, y, weights=w,
                                     bins=(np.linspace(xls[0], xls[1], bins), 
                                           np.linspace(yls[0], yls[1], bins)))

    XH = np.sort(pd.Series(H[H!=0].ravel()))
    Hsum = XH.sum()
    XH_levels = [np.argmin(abs(np.cumsum(XH)-q*Hsum)) for q in [0.003,0.045,0.318]]
    levels = [XH[k] for k in XH_levels]

    plt.contour(gaussian_filter(H, sigma=1., order=0).T, levels, 
                extent=[xbins.min(),xbins.max(),ybins.min(),ybins.max()],
                linewidths=2, cmap='rainbow',linestyles='solid',alpha=0.3)
    BPT_set(xls,yls)

def read_data(file):    
    
    tab_mod = pd.read_table(file)
    
    tab_mod.loc[:,"M_t"] = np.log10(10**tab_mod.M_cl - 10**(tab_mod.clustermass_log10_g-33))
    
    tab_mod.loc[:,"log_age"] = np.log10(tab_mod.age_yr)
    tab_mod.loc[:,"age_Myr"] = tab_mod.age_yr/1e6
    d_age = np.append(tab_mod.age_Myr[1:],tab_mod.age_Myr.iloc[-1]) - tab_mod.age_Myr
    tp_last = (np.argwhere(d_age<=0)-1).ravel()  #last time point position
    w = d_age.copy()
    w.values[w<=0] = d_age.iloc[tp_last]
    tab_mod.loc[:,"d_age"] = w
    
    Ha = tab_mod["H  1 6562.81A"]         #Ha
    Hb = tab_mod["H  1 4861.33A"]         #Hb
    NII = tab_mod["N  2 6583.45A"]        #NII
    OIII = tab_mod["O  3 5006.84A"]       #OIII
    OII = tab_mod["O  2 3726.03A"] + tab_mod['O  2 3728.81A']    #OII
    SIIa = tab_mod["S  2 6716.44A"]       #SIIa
    SIIb = tab_mod["S  2 6730.82A"]       #SIIb
    
    tab_mod.loc[:,"OIII_Hb"] = np.log10(OIII/Hb)
    tab_mod.loc[:,"NII_Ha"] = np.log10(NII/Ha)
    
    tab_mod.loc[:,"log_OIII5007_Hb"] = np.log10(OIII/Hb)
    tab_mod.loc[:,"log_OIII5007_NII6583"] = np.log10(OIII/NII)
    tab_mod.loc[:,"log_OIII5007_OII3727"] = np.log10(OIII/OII)
    
    tab_mod.loc[:,"log_OII3727_Hb"] = np.log10(OII/Hb)
    tab_mod.loc[:,"log_OII3727_NII6583"] = np.log10(OII/NII)
    
    tab_mod.loc[:,"log_OII3727_OIII5007_Hb"] = np.log10((OIII+OII)/Hb)
    
    tab_mod.loc[:,"log_NII6583_Ha"] = np.log10(NII/Ha)
    
    tab_mod.loc[:,"log_SII6716_6731_Ha"] = np.log10((SIIa+SIIb)/Ha)
    tab_mod.loc[:,"log_SII6716_6731_NII6583"] = np.log10((SIIa+SIIb)/NII)
    tab_mod.loc[:,"log_SII6716_SII6731"] = np.log10(SIIa/SIIb)
    
    return tab_mod

def get_data_err_linear(table_data,line_used,line_used_err):
    table = hstack([table_data[line_used],
                    table_data[line_used_err]]).to_pandas().dropna()
    data_obs = pd.DataFrame(np.vstack([table[l] for l in line_used]).T)
    data_obs_err = pd.DataFrame(np.vstack([table[l+"_err"] for l in line_used]).T)
    data_obs_err_lin = data_obs_err/0.434 
    data_obs_lin = 10**data_obs
    return data_obs_lin, data_obs_err_lin

# =============================================================================
# # Tranining 
# =============================================================================
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score,precision_score,f1_score
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor,MultiOutputClassifier
from sklearn.decomposition import PCA
from sklearn import svm

class BPT_query():
    
    def __init__(self, data,param=["log_age","M_cloud"],use_all=True):
        self.param = param
        self.data = data#.to_pandas()
        y_true = pd.DataFrame({t:self.data[t] for t in param},columns=param)
        self.y_true = y_true
        if use_all:
            self.tab_train, self.tab_test = self.data, self.data
            y_train, y_test = self.y_true, self.y_true
        else:
            tab_train, tab_test, y_train, y_test = train_test_split(self.data, self.y_true, test_size=0.25)
            self.tab_train, self.tab_test = tab_train, tab_test

        self.y_train, self.y_test = y_train, y_test        
        self.y_train_class = self.y_train.astype("str")
        self.y_test_class = self.y_test.astype("str")        
        
        self.xplot = np.vstack([np.linspace(y_true[t].min()-0.25,y_true[t].max()+0.25,25) for t in param])
        
    def set_data(self, line_used, use_pca = True,table_obs = None):
        
        self.line_used=line_used
        self.line_used_err = [l+"_err" for l in line_used]
        
        self.table_obs = table_obs
        self.use_pca = use_pca
        
        X_train = np.vstack([self.tab_train[line] for line in line_used]).T
        X_test = np.vstack([self.tab_test[line] for line in line_used]).T
        
        if use_pca:
            pca = PCA(n_components=max(len(line_used)-2,2))
            pca.fit(X_train)
            self.pca = pca
            
            X_train = pca.transform(X_train)
            X_test = pca.transform(X_test)
            
        self.X_train = X_train
        self.X_test = X_test
        self.X_all = np.vstack([self.X_train,self.X_test])
        self.y_all = np.vstack([self.y_train,self.y_test])
        
    def performance(self):
        # Scatter
        plt.figure(figsize=(3*len(self.param),3))
        for i,t in enumerate(self.param):
            ax=plt.subplot(1,len(self.param),i+1)
            plt.scatter(self.y_pred[t], self.y_test[t], alpha=0.1)
            plt.plot(self.xplot[i],self.xplot[i],c='k',ls='--',lw=3)
            plt.xlim(self.xplot[i,0],self.xplot[i,-1])
        plt.tight_layout(); plt.show()
        
        # KDE
        plt.figure(figsize=(5*len(self.param),4))
        for i,t in enumerate(self.param):
            xx, yy = np.mgrid[self.xplot[i,0]:self.xplot[i,-1]:100j, 
                              self.xplot[i,0]:self.xplot[i,-1]:100j]
            positions = np.vstack([xx.ravel(), yy.ravel()])
            values = np.vstack([self.y_pred[t], self.y_test[t]])
            kernel = st.gaussian_kde(values)
            f = np.reshape(kernel(positions).T, xx.shape)
            
            ax = plt.subplot(1,len(self.param),i+1)
            im = plt.imshow(f.T,cmap='hot',origin='lower',aspect='auto',
                            extent=[self.xplot[i,0],self.xplot[i,-1], 
                                    self.xplot[i,0],self.xplot[i,-1]])
            plt.plot(self.xplot[i],self.xplot[i],c='w',ls='--',lw=3)
            cb = plt.colorbar(mappable=im)
            cb.set_label('Density',fontsize="large")
            plt.text(.05,.9,"R$^2=$%.3f"%(r2_score (self.y_test[t], self.y_pred[t])),
                     color='gold',fontsize=15,transform=ax.transAxes)
            plt.legend(loc=3,fontsize=15)
            plt.title("Prediction for Model Parameter: %s"%t,fontsize='large')
        plt.tight_layout(); plt.show()
        
    def regression(self, method = "RF", **kwargs):
        
        self.method = method
        if method == "RF":
            regr_multi = MultiOutputRegressor(RandomForestRegressor(**kwargs))
            self.n_estimators = regr_multi.estimator.n_estimators
        elif method == "RF-single":    
            regr_multi = RandomForestRegressor(**kwargs)
            self.n_estimators = regr_multi.n_estimators
        elif method == "ExF-single":    
            regr_multi = ExtraTreesRegressor(**kwargs)
            self.n_estimators = regr_multi.n_estimators
        elif method == "SVR":
            regr_multi = MultiOutputRegressor(svm.SVR(**kwargs))
        elif method == "Linear":
            regr_multi = MultiOutputRegressor(LinearRegression(**kwargs))
        
        regr_multi.fit(self.X_train, self.y_train)
        self.regressor = regr_multi
        
        y_pred = regr_multi.predict(self.X_test)
        #y_pred = pd.DataFrame({t:regr.predict(self.X_test) for (regr,t) in zip(regrs_fit,self.param)},columns=self.param)
        self.y_pred = pd.DataFrame({t:col for (t,col) in zip(self.param,y_pred.T)},columns=self.param)
        
        self.r2 = r2_score(self.y_test,self.y_pred,multioutput="raw_values")
        print self.r2
        
        return self.regressor
        
    def predict(self, pos, vote=True): 
        if self.use_pca is True:
            pos_pca = self.pca.transform(pos)
        else:
            pos_pca = pos
        
        
        RF_vote = pd.DataFrame({})
        if self.method == "RF":
            for (t,regr) in zip(self.param, self.regressor.estimators_):
                rf_vote = np.array([tree.predict(pos_pca) for tree in regr.estimators_])
                RF_vote[t] = rf_vote.ravel()
            if vote==True:
                return self.regressor.predict(pos_pca),RF_vote 
            else:
                return self.regressor.predict(pos_pca)
            
        elif self.method == "RF-single":
            rf_vote = np.array([tree.predict(pos_pca) for tree in self.regressor.estimators_])
            RF_vote = pd.DataFrame(np.vstack(rf_vote),columns=self.param)
            if vote==True:
                return self.regressor.predict(pos_pca),RF_vote 
            else:
                return self.regressor.predict(pos_pca)
        
            
        elif self.method == "SVR":
            return self.regressor.predict(pos_pca)
        
        elif self.method == "Linear":
            return self.regressor.predict(pos_pca)
        else: 
            return None
        
    def draw_predict(self,pos,t1,t2):
        
        if self.method=="RF":
            p, p_vote = self.predict(pos,vote=True)
            par_pred = p_vote
            
            plt.figure(figsize=(8,7))
            gs = mpl.gridspec.GridSpec(2, 2, height_ratios=[1, 3],width_ratios=[3, 1])
            ax2 = plt.subplot(gs[2])
            sns.kdeplot(par_pred[t1],par_pred[t2],
                        cmap="Greys",ax=ax2)
            plt.scatter(par_pred[t1],par_pred[t2],
                        color='orange',edgecolor='k',alpha=0.5)
            plt.xlabel(t1,fontsize="large")
            plt.ylabel(t2,fontsize="large")
            
            ax1 = plt.subplot(gs[0])
            sns.distplot(par_pred[t1], bins=10,color='grey')
            ax1.set_xlim(ax2.get_xlim()); ax1.set_xlabel('')
            plt.setp(ax1.get_xticklabels(), visible=False)
            plt.setp(ax1.get_yticklabels(), visible=False)
            
            ax3 = plt.subplot(gs[3])
            sns.distplot(par_pred[t2], bins=10,color='grey',vertical=True)
            ax3.set_ylim(ax2.get_ylim()); ax3.set_ylabel('')
            plt.setp(ax3.get_xticklabels(), visible=False)
            plt.setp(ax3.get_yticklabels(), visible=False)
            
            ax1 = plt.subplot(gs[1])
            BPT_plot_dens_models(self.data.log_NII6583_Ha,
                                 self.data.log_OIII5007_Hb,
                                 weights=self.data.age_year)
            plt.plot(pos[0][0],pos[0][1],c="gold",marker="*",ms=10,mec="k")
            
            plt.subplots_adjust(left=0.12, bottom=0.1, right=0.95, top=0.95, wspace=0.001, hspace=0.001)
            plt.show()
            
            return par_pred
        
        else:
            return None
        
    def predict_MC(self,pos,n_MC=200,vote=True):
        self.n_MC = n_MC
        P1_vote_MC = np.empty((n_MC,self.n_estimators,len(self.param)))
        P1_MC = np.empty((n_MC,len(self.param)))
        for k in range(n_MC):
            noise = np.random.normal(0, 0.05*np.ones_like(pos))
            pos_prtb = pos+noise
            P1, P1_vote = self.predict(pos_prtb,vote=True)
            P1_vote_MC[k] = P1_vote
            P1_MC[k] = P1
        return P1_MC, P1_vote_MC

#    def predict_gp_MC(self,pos,n_MC=200):
#        self.n_MC = n_MC
#                
#        P1_vote_gp_MC = np.empty((n_MC,len(pos),self.n_estimators,len(self.param)))
#        P1_gp_MC = np.empty((n_MC,len(pos),len(self.param)))
#        for k in range(n_MC):
#            noise = np.random.normal(0, 0.05*np.ones_like(pos))
#            pos_prtb = pos+noise
#            P1, P1_vote = self.predict(pos_prtb,vote=True)
#            P1_gp_MC[k] = P1
#            P1_vote_gp_MC[k] = np.reshape(P1_vote.values,[len(pos),self.n_estimators,len(self.param)])
#            
#        return P1_gp_MC, P1_vote_gp_MC
    
    def draw_hist_MC(self,k, P1_MC, P1_vote_MC, ax, b=None):
                
        for i in range(self.n_MC):
            sns.distplot(P1_vote_MC[i,:,k],color='gray',bins=b,
                         hist_kws={"histtype":"step","alpha":0.1},
                         kde=False,ax=ax)
        sns.distplot(np.median(P1_vote_MC[:,:,k],axis=0),color="k",bins=b,
                     hist_kws={"histtype":"step",
                               "linewidth": 3,"alpha":0.9},
                               kde=False,label="Vote MC",ax=ax)
        sns.distplot(P1_MC[:,k], color="k",bins=b,
                     hist_kws={"histtype":"step","linestyle": "--",
                               "linewidth": 3,"alpha":.7},
                               kde=False,label="Best MC",ax=ax)
        plt.legend(loc="best",fontsize=9)
        
    def draw_scatter_MC(self,j,k, P1_MC, P1_vote_MC, ax):
                
        for i in range(self.n_MC):
            ax.scatter(P1_vote_MC[i,:,j],P1_vote_MC[i,:,k],color='gray',
                        s=3,alpha=0.1,zorder=1)
        ax.scatter(np.median(P1_vote_MC[:,:,j],axis=0),
                    np.median(P1_vote_MC[:,:,k],axis=0),
                    color="k",edgecolor='orange',s=20,
                    alpha=0.5,label="Vote MC",zorder=2)
#        sns.kdeplot(np.median(P1_vote_MC[:,:,j],axis=0),
#                    np.median(P1_vote_MC[:,:,k],axis=0),
#                    cmap='Oranges',alpha=0.5,zorder=3,ax=ax)
        ax.scatter(P1_MC[:,j],P1_MC[:,k], color="k",
                   edgecolor='b',marker='^',s=10,
                   alpha=0.5,label="Best MC",zorder=3)
        ax.legend(loc="best",fontsize=8)

    def draw_pred_MC_hist(self,pos,xlabels=None):
                
        plt.figure(figsize=(11,7))
        plt.subplot2grid((4, 3), (0, 0),rowspan=4,colspan=2)
        
#        BPT_contours(self.data.log_NII6583_Ha,self.data.log_OIII5007_Hb,weights=self.data.age_Myr)
        plt.scatter(self.data.log_NII6583_Ha,self.data.log_OIII5007_Hb,
                    c='k',s=5,lw=0,alpha=0.5)
        
        plt.errorbar(self.table_obs.log_NII6583_Ha, self.table_obs.log_OIII5007_Hb,
                     xerr=self.table_obs.log_NII6583_Ha_err,yerr=self.table_obs.log_OIII5007_Hb_err,
                     c="gray",ls="",marker="o",ms=4,mfc="gold",mec="k",lw=.8,alpha=0.8,
                     label=r"SITELLE: symmetric + L$\rm_{H\alpha}>10^{37}$")
        
        plt.scatter(pos[0][0],pos[0][1],c="lime",s=100,edgecolor='gray',alpha=1.,zorder=4)
        plt.scatter(pos[0][0],pos[0][1],marker='x',c="k",s=40,lw=1,alpha=1.,zorder=5)
#        cb = plt.colorbar(sc)
#        cb.set_label(r'$\rm\Delta{\ }Age(Myr)$',fontsize="large")
#        plt.clim(0.,2.)
        BPT_set()
         
        P1_MC, P1_vote_MC = self.predict_MC(pos)

        if xlabels is None: xlabels = self.param
        
        for j,xlab in enumerate(xlabels):
            ax = plt.subplot2grid((4, 3), (j, 2))
            self.draw_hist_MC(j,P1_MC, P1_vote_MC, ax)
            ax.set_xlabel(xlab)
            
        plt.tight_layout()
        
        return None
    
    def draw_pred_MC_sca(self,pos,n_MC=200,labels=None):
                
        plt.figure(figsize=(10,9))
        plt.subplot2grid((4, 4), (0, 2),rowspan=2,colspan=2)
        
#        BPT_contours(self.data.log_NII6583_Ha,self.data.log_OIII5007_Hb,weights=self.data.age_Myr)
        plt.scatter(self.data.log_NII6583_Ha,self.data.log_OIII5007_Hb,
                    c='k',s=5,lw=0,alpha=0.5)
        
        plt.errorbar(self.table_obs.log_NII6583_Ha, self.table_obs.log_OIII5007_Hb,
                     xerr=self.table_obs.log_NII6583_Ha_err,yerr=self.table_obs.log_OIII5007_Hb_err,
                     c="gray",ls="",marker="o",ms=4,mfc="gold",mec="k",lw=.5,alpha=0.7,
                     label=r"SITELLE: symmetric + L$\rm_{H\alpha}>10^{37}$")
      
        plt.scatter(pos[0][0],pos[0][1],c="lime",s=100,edgecolor='gray',alpha=1.,zorder=4)
        plt.scatter(pos[0][0],pos[0][1],marker='x',c="k",s=40,lw=1,alpha=1.,zorder=5)
        BPT_set()
        
        
        P1_MC, P1_vote_MC = self.predict_MC(pos,n_MC=n_MC)

        if labels is None: labels = self.param
        
        for j,lab1 in enumerate(labels[:-1]):
            for k,lab2 in enumerate(labels[j+1:]):
                ax = plt.subplot2grid((4, 4), (j+k+1, j))
                self.draw_scatter_MC(j,j+k+1, P1_MC, P1_vote_MC, ax)
                if j==0: ax.set_ylabel(lab2)
            ax.set_xlabel(lab1)
            
        for j,lab in enumerate(labels):
            ax = plt.subplot2grid((4, 4), (j, j))
            self.draw_hist_MC(j,P1_MC, P1_vote_MC, ax)
            if j==(len(labels)-1): ax.set_xlabel(lab)
            
        plt.tight_layout(h_pad=0.1,w_pad=0.1)
        
        return None

    def pred_MC_grid(self,positions):
        self.positions = positions
        par_guess = np.empty((len(positions),4))
        std_guess = np.empty((len(positions),4))
        up_guess = np.empty((len(positions),4))
        down_guess = np.empty((len(positions),4))
        
        for i,p in enumerate(positions):
            P1_MC,P1_vote_MC = self.predict_MC([p],vote=True)
            P = pd.DataFrame(P1_MC)
            P = pd.DataFrame(np.median(P1_vote_MC,axis=0))
            par_guess[i] = P.mean()
            std_guess[i] = P.std()
            up_guess[i] = P.quantile(0.9)
            down_guess[i] = P.quantile(0.1)
            if np.mod(i+1,25)==0:
                print "%d/%d finished"%(i+1,len(positions))
        
        self.par_guess,self.std_guess,self.up_guess,self.down_guess  = \
            par_guess,std_guess,up_guess,down_guess
            
        return par_guess,std_guess,up_guess,down_guess
    
    def draw_MC_grid_slice(self,xx,yy,n,axis='col',xlabels=None):
        
        shape = (len(yy),len(xx))
        
        plt.figure(figsize=(11,7))
        plt.grid(True)
        ax0 = plt.subplot2grid((4, 5), (0, 0),rowspan=4,colspan=3)
        
#        BPT_contours(self.data.log_NII6583_Ha,self.data.log_OIII5007_Hb,weights=self.data.age_Myr)
        plt.scatter(self.data.log_NII6583_Ha,self.data.log_OIII5007_Hb,
                    c='k',s=5,lw=0,alpha=0.5)
        
        ax0.scatter(self.positions[:,0],self.positions[:,1],
                    color='k',s=15,alpha=0.7)
#        cb = plt.colorbar()
#        cb.set_label('log Age/yr',fontsize="large")
        BPT_set()
        
        if xlabels is None: xlabels = self.param
        
        if axis == 'col':
            plt.axvline(xx[n],color='lime')
            plt.text(xx[n]+0.05,-3.4,"x=%.2f"%xx[n],color="lime")
            print "x = ",xx[n]
            for j,xlab in enumerate(xlabels):
                ax = plt.subplot2grid((4, 5), (j, 3),colspan=2)        
                ax.plot(yy,self.par_guess[:,j].reshape(shape)[:,n],
                        ls='-',lw=2,color='k',alpha=0.8)
                ax.plot(yy,self.up_guess[:,j].reshape(shape)[:,n],
                        ls='--',color='k',alpha=0.8)
                ax.plot(yy,self.down_guess[:,j].reshape(shape)[:,n],
                        ls='--',color='k',alpha=0.8)
                ax.set_ylabel(xlab)
            ax.set_xlabel(r"log([OIII]5007/H$\beta$)",fontsize="large")
        
        elif axis == 'row':
            plt.axhline(yy[n],color='lime')
            plt.text(-2.4,yy[n]+0.05,"x=%.2f"%yy[n],color="lime")
            print "y = ",yy[n]
            for j,xlab in enumerate(xlabels):
                ax = plt.subplot2grid((4, 5), (j, 3),colspan=2)        
                ax.plot(xx,self.par_guess[:,j].reshape(shape)[n],
                        ls='-',lw=2,color='k',alpha=0.8)
                ax.plot(xx,self.up_guess[:,j].reshape(shape)[n],
                        ls='--',color='k',alpha=0.8)
                ax.plot(xx,self.down_guess[:,j].reshape(shape)[n],
                        ls='--',color='k',alpha=0.8)
                ax.set_ylabel(xlab)
            ax.set_xlabel(r"log([NII]6583/H$\alpha$)",fontsize="large")

        else: print " "
        
        plt.tight_layout()
                
        return ax0
    
    def classification(self, method = "RF", **kwargs):
        
        self.method = method
        if method == "RF":        
            clf_multi = MultiOutputClassifier(RandomForestClassifier(**kwargs))
            self.n_estimators = clf_multi.estimator.n_estimators
        elif method == "RF-single":
            clf_multi = RandomForestClassifier(**kwargs)
            self.n_estimators = clf_multi.n_estimators            
        elif method == "SVM":
            clf_multi = MultiOutputClassifier(svm.SVC(**kwargs))
        elif method == "KNN":
            clf_multi = MultiOutputClassifier(KNeighborsClassifier(**kwargs))
        
        clf_multi.fit(self.X_train, self.y_train_class)
        self.classifier = clf_multi
        
        y_pred_class = clf_multi.predict(self.X_test).astype("str")
        if method == "RF-single":
            y_pred_class = np.int_((y_pred_class).astype("float")).astype("str")

        self.y_pred_class = pd.DataFrame({t:col for (t,col) in zip(self.param,y_pred_class.T)},columns=self.param)
        
        self.accuracy = [accuracy_score(self.y_test_class[t], self.y_pred_class[t]) for t in self.param]
        self.precision = [precision_score(self.y_test_class[t], self.y_pred_class[t],average=None) for t in self.param]
    
        print self.accuracy
        print self.precision
        
        self.bins = {t:np.unique(self.data[t]).astype("str") for t in self.param}
        
        return self.classifier
    
    def classify(self, pos, prob=True):  
        pos_pca = self.pca.transform(pos)
        
        RF_vote = pd.DataFrame({})
        if self.method == "RF":
            for (t,clf) in zip(self.param, self.classifier.estimators_):
                rf_vote = np.array([tree.predict(pos_pca) for tree in clf.estimators_])
                RF_vote[t] = rf_vote.ravel()
            if prob==True:
                return self.classifier.predict(pos_pca),RF_vote 
            else:
                return self.classifier.predict(pos_pca)
            
        elif self.method == "RF-single":
            rf_vote = np.array([tree.predict(pos_pca) for tree in self.classifier.estimators_])
            RF_vote = pd.DataFrame(np.vstack(rf_vote),columns=self.param)
            if prob==True:
                return self.classifier.predict(pos_pca),RF_vote 
            else:
                return self.classifier.predict(pos_pca)
            
        elif self.method == "SVM":
            if prob==True:
                return self.classifier.predict(pos_pca),self.classifier.predict_proba(pos_pca)
            else:
                return self.classifier.predict(pos_pca)
        else: 
            return None

    
    def classify_MC(self,pos,n_MC=200,prob=True):
        self.n_MC = n_MC
        if self.method=="SVM":
            P1_vote_MC = {t:np.empty((n_MC,len(self.bins[t]))) for t in self.param}
        elif (self.method == "RF") | (self.method == "RF-single"):  
            P1_vote_MC = np.empty((n_MC,self.n_estimators,len(self.param)))
        else:
            return None
        
        P1_MC = np.empty((n_MC,len(self.param)))
        
        for k in range(n_MC):
            noise = np.random.normal(0, 0.05*np.ones_like(pos))
            pos_prtb = pos + noise
            P1, P1_vote = self.classify(pos_prtb,prob=True)
            if self.method=="SVM":
                for i,t in enumerate(self.param):
                    P1_vote_MC[t][k] = P1_vote[i]
            else:
                P1_vote_MC[k] = P1_vote
            P1_MC[k] = P1
            
        return P1_MC, P1_vote_MC

    def draw_bin_MC(self,k, P1_MC, P1_vote_MC, ax):

        bar_x = self.bins[self.param[k]]
        if (self.method == "RF") | (self.method == "RF-single"):
            for i in range(self.n_MC):
                bar_height = [(P1_vote_MC[i,:,k]==d).sum() for d in range(len(bar_x))]
                plt.bar(bar_x,bar_height,width=0.5,linewidth=1,
                        edgecolor="gray",facecolor="none", alpha=0.1)
                
            bar_height1 = [(np.median(P1_vote_MC[:,:,k],axis=0)==d).sum() for d in range(len(bar_x))]
            plt.bar(bar_x, bar_height1, width=0.5,linewidth=3,
                    edgecolor=["orange"]*3,facecolor="none", alpha=0.7,
                    linestyle="--",label="Vote MC",zorder=3)          
            
        elif self.method == "SVM":
            for i in range(self.n_MC):
                bar_height = P1_vote_MC[self.param[k]][i]*self.n_MC
                plt.bar(bar_x,bar_height,width=0.5,linewidth=1,
                        edgecolor="gray",facecolor="none", alpha=0.1)
                    
            bar_height1 = np.median(P1_vote_MC[self.param[k]],axis=0)*self.n_MC
            plt.bar(bar_x, bar_height1, width=0.5,linewidth=3,
                    edgecolor=["orange"]*3,facecolor="none", alpha=0.7,
                    linestyle="--",label="Prob MC",zorder=3)
        else: 
            return None
            
        bar_height2 = [(P1_MC[:,k]==d).sum() for d in range(len(bar_x))]
        plt.bar(bar_x, bar_height2, width=0.5,linewidth=3,
                edgecolor=["steelblue"]*3,facecolor="none", alpha=0.7,
                label="Best MC",zorder=2)
        plt.legend(loc="best",fontsize=9)
        
    def draw_bubble_MC(self,j,k, P1_MC, P1_vote_MC, ax):
        bubble_x = self.bins[self.param[j]].astype("int")
        bubble_y = self.bins[self.param[k]].astype("int")
        
        if (self.method == "RF") | (self.method == "RF-single"):        
            for b_x in bubble_x:
                for b_y in bubble_y:
                    for i in range(self.n_MC):
                        size = ((P1_vote_MC[i,:,j]==b_x) \
                                & (P1_vote_MC[i,:,k]==b_y)).sum()
                        ax.plot(b_x,b_y,marker='o',ms=size/2.,
                                 mfc='none',mec='gray',mew=1,alpha=0.05)
                
                    ms1 = ((np.median(P1_vote_MC[:,:,j],axis=0)==b_x) \
                           & (np.median(P1_vote_MC[:,:,k],axis=0)==b_y)).sum()
                    ax.plot(b_x,b_y,marker='o',ms=ms1/2.,
                            mec="orange",mfc='none',mew=3,
                            alpha=0.5,label="Vote MC",zorder=2)
                    
        elif self.method == "SVM":
            for b_x in bubble_x:
                for b_y in bubble_y:
                    for i in range(self.n_MC):
                        size = P1_vote_MC[self.param[j]][i][b_x]*\
                                P1_vote_MC[self.param[k]][i][b_y]*self.n_MC
                               
                        ax.plot(b_x,b_y,marker='o',ms=size/2.,
                                         mfc='none',mec='gray',mew=1,alpha=0.05)
                
                    ms1 = np.median(P1_vote_MC[self.param[j]],axis=0)[b_x]*\
                            np.median(P1_vote_MC[self.param[k]],axis=0)[b_y]*self.n_MC
                    ax.plot(b_x,b_y,marker='o',ms=ms1/2.,
                            mec='orange',mfc='none',mew=3,
                            alpha=0.7,label="Vote MC",zorder=3)

        for b_x in bubble_x:
            for b_y in bubble_y:
                ms2 = ((P1_MC[:,j]==b_x) & (P1_MC[:,k]==b_y)).sum()
                ax.plot(b_x,b_y,marker='o',ms=np.log(ms2+1)*10.,
                        mec='steelblue',mfc='none',mew=3,
                        alpha=0.7,label="Best MC",zorder=2) 
        plt.xlim(-.5,b_x+0.5)
        plt.ylim(-.5,b_y+0.5)        
                   
        #ax.legend(loc="best",fontsize=8)
        
    def draw_clf_MC_bin(self,pos,n_MC=200,xlabels=None):
                
        plt.figure(figsize=(11,7))
        plt.subplot2grid((4, 3), (0, 0),rowspan=4,colspan=2)
        
#        BPT_contours(self.data.log_NII6583_Ha,self.data.log_OIII5007_Hb,weights=self.data.age_Myr)
        plt.scatter(self.data.log_NII6583_Ha,self.data.log_OIII5007_Hb,
                    c='k',s=5,lw=0,alpha=0.5)        
        plt.scatter(pos[0][0],pos[0][1],c="lime",s=100,edgecolor='gray',alpha=1.,zorder=4)
        plt.scatter(pos[0][0],pos[0][1],marker='x',c="k",s=40,lw=1,alpha=1.,zorder=5)
        BPT_set()
        
        P1_MC, P1_vote_MC = self.classify_MC(pos,n_MC=n_MC)

        if xlabels is None: xlabels = self.param
        
        for j,xlab in enumerate(xlabels):
            ax = plt.subplot2grid((4, 3), (j, 2))
            self.draw_bin_MC(j,P1_MC, P1_vote_MC, ax)
            ax.set_xlabel(xlab)
            
        plt.tight_layout()
        
        return None
    
    def draw_clf_MC_bub(self,pos,labels=None):
                
        plt.figure(figsize=(10,9))
        plt.subplot2grid((4, 4), (0, 2),rowspan=2,colspan=2)
        
#        BPT_contours(self.data.log_NII6583_Ha,self.data.log_OIII5007_Hb,weights=self.data.age_Myr)
        plt.scatter(self.data.log_NII6583_Ha,self.data.log_OIII5007_Hb,
                    c='k',s=5,lw=0,alpha=0.5)      
        plt.scatter(pos[0][0],pos[0][1],c="lime",s=100,edgecolor='gray',alpha=1.,zorder=4)
        plt.scatter(pos[0][0],pos[0][1],marker='x',c="k",s=40,lw=1,alpha=1.,zorder=5)
        BPT_set()
        
        P1_MC, P1_vote_MC = self.classify_MC(pos)

        if labels is None: labels = self.param
        
        for j,lab1 in enumerate(labels[:-1]):
            for k,lab2 in enumerate(labels[j+1:]):
                ax = plt.subplot2grid((4, 4), (j+k+1, j))
                self.draw_bubble_MC(j,j+k+1, P1_MC, P1_vote_MC, ax)
                if j==0: ax.set_ylabel(lab2)
            ax.set_xlabel(lab1)
            
        for j,lab in enumerate(labels):
            ax = plt.subplot2grid((4, 4), (j, j))
            self.draw_bin_MC(j,P1_MC, P1_vote_MC, ax)
            #ax.set_xlabel(lab)
            
        plt.tight_layout(h_pad=0.1,w_pad=0.1)
        
        return None



    
#####    
def write_Zmock_table(table):
    tab_zmoc=Table()
    for d in [-0.2,-0.1,0.,0.1,0.2]:
        tab_nw=table.copy()
        tab_nw["Z_sol"]+=d
        tab_nw["log_NII6583_Ha"]+=d
        tab_nw["log_OIII5007_NII6583"]-=d
        tab_nw["log_SII6716_6731_Ha"]+=d
        tab_zmoc=vstack([tab_zmoc,tab_nw])
    tab_zmoc.write("Zmock_object_tracks-lines.dat", format='ascii')