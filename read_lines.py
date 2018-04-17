#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 17:04:35 2018

@author: mac
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

NII_plot=np.linspace(-2.,0.0,100)

def BPT_set(xls=(-2.5,0.5), yls=(-3.5,1.5),c="k"):
    NII_plot = np.linspace(-3.,0.0,100)
    NII_plot2 = np.linspace(-3.,0.45,100)
    plt.plot(NII_plot,BPT_border(NII_plot, 'Kauffmann'),c=c)
    plt.plot(NII_plot2,BPT_border(NII_plot2, 'Kewley'),c=c,ls='--')
    plt.xlim(xls); plt.ylim(yls)
    plt.xlabel(r"log([NII]6583/H$\alpha$)",fontsize="large")
    plt.ylabel(r"log([OIII]5007/H$\beta$)",fontsize="large")
    
def BPT_border(log_NII_Ha,linetype = "Kauffmann"):
	if linetype=="Kauffmann": return 1.3 + 0.61 / (log_NII_Ha-0.05)
	elif linetype=="Kewley": return 1.19 + 0.61 / (log_NII_Ha - 0.47)

def BPT_plot_class(NII_Ha, OIII_Hb):
    plt.figure(figsize=(8,6))
    SF_cond = (OIII_Hb<BPT_border(NII_Ha,'Kauffmann')) & (NII_Ha<0.)
    Composite_cond = (OIII_Hb>BPT_border(NII_Ha,'Kauffmann')) & (OIII_Hb<BPT_border(NII_Ha,'Kewley')) & (NII_Ha<0.45)
    plt.scatter(NII_Ha[SF_cond], OIII_Hb[SF_cond],
                    s=10,c="steelblue",alpha=0.5)
    plt.scatter(NII_Ha[Composite_cond], OIII_Hb[Composite_cond],
                    s=10,c="g",alpha=0.5)
    plt.scatter(NII_Ha[~((SF_cond)|(Composite_cond))], OIII_Hb[~((SF_cond)|(Composite_cond))],
                    s=10,c="firebrick",alpha=0.5)
    plt.text(-1.25,-0.6,"%d"%len(NII_Ha[SF_cond]),fontsize=15,color="steelblue")
    plt.text(-0.15,-0.8,"%d"%len(NII_Ha[Composite_cond]),fontsize=15,color="g")
    plt.text(0.1,0.8,"%d"%len(NII_Ha[~((SF_cond)|(Composite_cond))]),fontsize=15,color="firebrick")
    BPT_set()
    
def BPT_plot_dens_models(NII_Ha, OIII_Hb, bins=80, xls=(-2.5,0.5), yls=(-3.5,1.5), weight_flag=True,weights=None):
    plt.figure(figsize=(8,6))

    if weight_flag: 
        age = weights
        d_age = np.append(age[1:],age.iloc[-1]) - age
        d_age[d_age<=0]=500000.0
        w = d_age/500000.0
    else: w = None
    
    H, xbins, ybins = np.histogram2d(NII_Ha, OIII_Hb, weights=w,
    		bins=(np.linspace(-3., 0.25, bins), np.linspace(-3.5, 1.25, bins)))
    C = plt.contourf(np.log(H).T,aspect="auto",origin="lower", cmap='jet',
             extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]])
    
    BPT_set(xls,yls)
    plt.legend(loc="best",fontsize=15)
    cb = plt.colorbar(mappable=C)
    cb.set_label('Density',fontsize="large")
    

def BPT_plot_kde(NII_Ha, OIII_Hb, 
                 xls=(-2.5,0.5), yls=(-3.5,1.5), 
                 bw=None, cmap='hot',classic=True,colorbar=True):
    
    xx, yy = np.mgrid[xls[0]:xls[1]:200j, yls[0]:yls[1]:200j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([NII_Ha, OIII_Hb])
    kernel = st.gaussian_kde(values,bw_method=bw)
    f = np.reshape(kernel(positions).T, xx.shape)
    
    plt.figure(figsize=(8,6))
    im = plt.imshow(f.T,cmap=cmap,origin='lower',aspect='auto',
                    extent=[xls[0],xls[1],yls[0],yls[1]])
    if colorbar:
        cb = plt.colorbar(mappable=im)
        cb.set_label('Density',fontsize=12)
    if classic: BPT_set(xls, yls, c="w")
    plt.tight_layout()
    return im

def BPT_plot_2Dhist(NII_Ha, OIII_Hb, weight,
                    xls=(-2.5,0.5), yls=(-3.5,1.5), 
                    dx=0.05, dy=0.1, cmap='rainbow'):
    
    d_age = np.append(weight[1:],weight.iloc[-1]) - weight
    d_age[d_age<=0]=0.5e6
    w = d_age/0.5e6
    
    H, xbins, ybins = np.histogram2d(NII_Ha, OIII_Hb, weights=w,
                                     bins=(np.arange(xls[0],xls[1]+dx, dx),
                                           np.arange(yls[0],yls[1]+dy, dy)))
    
    plt.figure(figsize=(8,6))
    im = plt.imshow(np.log10(H).T, cmap=cmap, 
                    extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]],
                    origin='lower', aspect='auto',alpha=0.7)
    
    cb = plt.colorbar(mappable=im)
    cb.set_label('log Num',fontsize="large")
    BPT_set(xls, yls, c="k")
    plt.tight_layout()

    
def BPT_plot(NII_Ha, OIII_Hb, col, cmap='rainbow',classic=False):
    s = plt.scatter(NII_Ha, OIII_Hb, c=col,
                cmap=cmap, s=5, alpha=0.5)
    if classic: BPT_set()
    cb = plt.colorbar(mappable=s)
    cb.set_label(col.name,fontsize="large")
    
# =============================================================================
# Read models
# =============================================================================
tab_lines = pd.read_table("object_tracks-lines.dat")

tab_shell = tab_lines[tab_lines.comp=="shell"]

Ha = tab_shell["H  1 6562.81A"]         #Ha
Hb = tab_shell["H  1 4861.33A"]         #Hb
NII = tab_shell["N  2 6583.45A"]        #NII
OIII = tab_shell["O  3 5006.84A"]       #OIII
OII = tab_shell['BLND 3727.00A']     #OII
SIIa = tab_shell["S  2 6716.44A"]    #SIIa
SIIb = tab_shell["S  2 6730.82A"]       #SIIb
SII = tab_shell["BLND 6720.00A"]


tab_shell["log_OIII5007_Hb"] = np.log10(OIII/Hb)
tab_shell["log_OIII5007_NII6583"] = np.log10(OIII/NII)
tab_shell["log_OIII5007_OII3727"] = np.log10(OIII/OII)

tab_shell["log_OII3727_Hb"] = np.log10(OII/Hb)
tab_shell["log_OII3727_NII6583"] = np.log10(OII/NII)

tab_shell["log_OII3727_OIII5007_Hb"] = np.log10((OIII+OII)/Hb)

tab_shell["log_NII6583_Ha"] = np.log10(NII/Ha)

tab_shell["log_SII6716_6731_Ha"] = np.log10((SII)/Ha)
tab_shell["log_SII6716_6731_NII6583"] = np.log10((SII)/NII)
tab_shell["log_SII6716_SII6731"] = np.log10(SIIa/SIIb)

tab_shell["OIII_Hb"] = np.log10(OIII/Hb)
tab_shell["NII_Ha"] = np.log10(NII/Ha)

# =============================================================================
# Draw
# =============================================================================

BPT_plot_kde(tab_shell.NII_Ha, tab_shell.OIII_Hb,bw=0.1)

# =============================================================================
# BPT Evolution Track #
# =============================================================================
#plt.figure(figsize=(10,8))
#for ind in np.unique(tab_shell["#Model_i"])[:3]:
#    mod = tab_shell[tab_shell["#Model_i"]==ind]
#    plt.scatter(mod.NII_Ha, mod.OIII_Hb,s=50,
#                c=np.log10(mod.age_year),cmap='rainbow',label=None)
#    plt.plot(mod.NII_Ha,mod.OIII_Hb,lw=3,
#             label=r"$\rm M_{cl}=10^5,n_H=500,Z=Z_{\odot},SFE:%.2f$"\
#             %(0.01*np.unique(mod.SFE_)))
#
#NII_plot = np.linspace(-2.,0.0,100)
#NII_plot2 = np.linspace(-2.,0.45,100)
#plt.plot(NII_plot,BPT_border(NII_plot, 'Kauffmann'),c='k')
#plt.plot(NII_plot2,BPT_border(NII_plot2, 'Kewley'),c='k',ls='--')
#plt.xlim(-2.,0.4); plt.ylim(-2.5,1.5)
#plt.xlabel(r"log([NII]6583/H$\alpha$)",fontsize="large")
#plt.ylabel(r"log([OIII]5007/H$\beta$)",fontsize="large")
#plt.legend(loc='best',fontsize="large")
#cb = plt.colorbar()
#cb.set_label('log Age[yr]',fontsize="large")

# =============================================================================
# LINE Evolution #
# =============================================================================
#fig, (ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize=(12,5))
#for ind in np.unique(tab_shell["#Model_i"])[:]:
#    mod = tab_shell[tab_shell["#Model_i"]==ind]                         
#    ax1.plot(np.log10(mod.age_year),mod.NII_Ha,lw=3,
#             label=r"$\rm M_{cl}=10^5,n_H=500,Z=Z_{\odot},SFE:%.2f$"\
#             %(0.01*np.unique(mod.SFE_)))
#    ax2.plot(np.log10(mod.age_year),mod.OIII_Hb,lw=3,
#             label=r"$\rm M_{cl}=10^5,n_H=500,Z=Z_{\odot},SFE:%.2f$"\
#             %(0.01*np.unique(mod.SFE_)))
#
#ax1.set_xlabel('log Age[yr]',fontsize="large")
#ax2.set_xlabel('log Age[yr]',fontsize="large")    
#ax1.set_ylabel(r"log([NII]5007/H$\alpha$)",fontsize="large")
#ax2.set_ylabel(r"log([OIII]5007/H$\beta$)",fontsize="large")


### BPT Plot color in param ###
#plt.figure(figsize=(14,10))
#ax1 = plt.subplot(2,2,1)
#BPT_plot(tab_shell["NII_Ha"],tab_shell["OIII_Hb"],tab_shell.M_cloud,'gnuplot')
#ax2 = plt.subplot(2,2,2)
#BPT_plot(tab_shell["NII_Ha"],tab_shell["OIII_Hb"],tab_shell.nH,'copper')
#ax3 = plt.subplot(2,2,3)
#BPT_plot(tab_shell["NII_Ha"],tab_shell["OIII_Hb"],tab_shell.SFE_,'viridis')
#ax4 = plt.subplot(2,2,4)
#BPT_plot(tab_shell["NII_Ha"],tab_shell["OIII_Hb"],np.log10(tab_shell.age_year),'jet')
#plt.tight_layout()


### calculate step ###
#line_used = ["log_NII6583_Ha","log_OIII5007_Hb","log_OIII5007_NII6583","log_OIII5007_OII3727",
#             "log_OII3727_Hb","log_OII3727_NII6583","log_OII3727_OIII5007_Hb", 
#             "log_SII6716_6731_Ha","log_SII6716_6731_NII6583","log_SII6716_SII6731"]
#
#tab_step = pd.DataFrame({})
#for ind in tab_shell["#Model_i"].unique():
#    mod = tab_shell[tab_shell["#Model_i"]==ind]
#    dx = np.append(mod.NII_Ha[1:],0) - mod.NII_Ha
#    dy = np.append(mod.OIII_Hb[1:],0) - mod.OIII_Hb
#    dx.iloc[-1], dy.iloc[-1] = (0, 0)
#    mod_step = mod.iloc[:,:7]
#    mod_step["d_BPTx"] = dx
#    mod_step["d_BPTy"] = dy
#    
#    d_age_yr = np.append(mod.age_year[1:],mod.age_year.iloc[-1]) - mod.age_year
#    d_age_yr[d_age_yr<=0]=500000.0
#    dt = d_age_yr/1e6    # in Myr
#    
#    mod_step["v_BPTx"] = dx/dt
#    mod_step["v_BPTy"] = dy/dt
#    for line in line_used:
#        dl = np.append(mod[line][1:],0) - mod[line]
#        dl.iloc[-1] = 0
#        mod_step["d_"+line] = dl
#    tab_step = pd.concat([tab_step, mod_step])
#
#### Jump points
#jump = tab_step[(abs(tab_step.d_BPTx)>0.1)|(abs(tab_step.d_BPTy)>0.1)]
#tab_jump = tab_shell.iloc[jump.index]
#plt.figure(figsize=(10,8))
#for ind in np.unique(tab_shell["#Model_i"])[:3]:
#    mod = tab_shell[tab_shell["#Model_i"]==ind]
#    plt.scatter(mod.NII_Ha, mod.OIII_Hb,s=50,
#                c=np.log10(mod.age_year),cmap='jet',label=None,alpha=0.5)
#    plt.plot(mod.NII_Ha,mod.OIII_Hb,lw=3,
#             label=r"$\rm M_{cl}=10^5,n_H=500,Z=Z_{\odot},SFE:%.2f$"\
#             %(0.01*np.unique(mod.SFE_)),zorder=2,alpha=0.5)
#
#NII_plot = np.linspace(-2.,0.0,100)
#NII_plot2 = np.linspace(-2.,0.45,100)
#plt.plot(NII_plot,BPT_border(NII_plot, 'Kauffmann'),c='k')
#plt.plot(NII_plot2,BPT_border(NII_plot2, 'Kewley'),c='k',ls='--')
#plt.xlim(-2.,0.4); plt.ylim(-2.5,1.5)
#plt.xlabel(r"log([NII]6583/H$\alpha$)",fontsize="large")
#plt.ylabel(r"log([OIII]5007/H$\beta$)",fontsize="large")
#plt.legend(loc='best',fontsize="large")
#cb = plt.colorbar()
#cb.set_label('log Age[yr]',fontsize="large")
#jump = tab_step[(abs(tab_step.d_BPTx)>0.1)|(abs(tab_step.d_BPTy)>0.1)]
#tab_jump=tab_shell.iloc[jump.index]
#plt.scatter(tab_jump.log_NII6583_Ha,tab_jump.log_OIII5007_Hb,c="k",marker="*",s=100,zorder=1)
#plt.savefig("Evolution Tracks jump.pdf")
#
#
#### BPT Process
#M_c, nH, SFE_ = np.unique((mod.M_cloud,mod.nH,mod.SFE_),axis=1).ravel()
#
#plt.figure(figsize=(10,7))
#plt.subplot(211)
#plt.plot(mod_step.age_year/1e6,mod_step.iloc[:,11:],c="gray",alpha=0.5)
#plt.plot(mod_step.age_year/1e6,mod_step.iloc[:,7],c="r",label="d BPTx")
#plt.plot(mod_step.age_year/1e6,mod_step.iloc[:,8],c="b",label="d BPTy")
#plt.ylabel(r"$\Delta$ (dex)")
#plt.xlabel("Age(Myr)")
#plt.ylim(-1,1.)
#plt.axhline(-0.1,ls="--",c="k")
#plt.axhline(0.1,ls="--",c="k")
#plt.legend(loc="best")
#plt.subplot(212)
#plt.plot(mod_step.age_year/1e6,mod_step.iloc[:,9],c="darkred",label="v BPTx")
#plt.plot(mod_step.age_year/1e6,mod_step.iloc[:,10],c="navy",label="v BPTy")
#plt.ylabel(r"$v$ (dex/Myr)")
#plt.xlabel("Age(Myr)")
#plt.legend(loc="best")
#plt.ylim(-2.5,2.5)
#plt.suptitle(r"$\rm M_{cl}=%d,n_H=%d,Z=Z_{\odot},SFE:%.2f$"%(M_c, nH, SFE_))


# =============================================================================
# FIX parameter
# =============================================================================

#for m_c in np.unique(tab_shell.M_cloud):
#    plt.figure(figsize=(8,7))
#    for SFE in np.unique(tab_shell.SFE_):
#        cond = (tab_shell.M_cloud==m_c) & (tab_shell.SFE_==SFE) #& (tab_shell.nH==100)
#        s = plt.scatter(tab_shell[cond].NII_Ha, 
#                        tab_shell[cond].OIII_Hb,
#                        s=60-tab_shell[cond].nH/10., 
#                        edgecolor=None, label="SFE=%s%%"%SFE, alpha=0.5)
#    plt.legend(loc="best")
#    plt.title("log M$_{cloud}=$%s"%m_c,fontsize="large")
#    BPT_set()
#    plt.tight_layout()
#    plt.savefig("model_interp/M%s-SFE.png"%m_c,dpi=400)
#
#for SFE in np.unique(tab_shell.SFE_):
#    plt.figure(figsize=(8,7))
#    for m_c in np.unique(tab_shell.M_cloud):
#        cond = (tab_shell.SFE_==SFE) & (tab_shell.M_cloud==m_c) #& (tab_shell.nH==100)
#        s = plt.scatter(tab_shell[cond].NII_Ha, 
#                        tab_shell[cond].OIII_Hb,
#                        s=60-tab_shell[cond].nH/10., 
#                        edgecolor=None, label="log M$_{cloud}=$%s"%m_c, alpha=0.5)
#    plt.legend(loc="best")
#    plt.title("SFE$=$%s%%"%SFE,fontsize="large")
#    BPT_set()
#    plt.tight_layout()
#    plt.savefig("model_interp/SFE%s-M.png"%SFE,dpi=400)
#    
#for nH in np.unique(tab_shell.nH):
#    plt.figure(figsize=(8,7))
#    for m_c in np.unique(tab_shell.M_cloud):
#        cond = (tab_shell.nH==nH) & (tab_shell.M_cloud==m_c)
#        s = plt.scatter(tab_shell[cond].NII_Ha, 
#                        tab_shell[cond].OIII_Hb,
#                        s=60-tab_shell[cond].nH/10., 
#                        edgecolor=None, label="log M$_{cloud}=$%s"%m_c, alpha=0.5)
#    plt.legend(loc="best")
#    plt.title("n$_{cloud}=$%s"%nH,fontsize="large")
#    BPT_set()
#    plt.tight_layout()
#    plt.savefig("model_interp/nH%s-M.png"%nH,dpi=400)






