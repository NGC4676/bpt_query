#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 09:33:26 2018

@author: mac
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.stats as st
import seaborn as sns
from astropy.io import fits,ascii
from astropy.table import Table

def BPT_set(xls=(-2.5,0.5), yls=(-3.5,1.5),c="k"):
    NII_plot = np.linspace(-2.5,0.0,100)
    NII_plot2 = np.linspace(-2.5,0.45,100)
    plt.plot(NII_plot,BPT_border(NII_plot, 'Kauffmann'),c=c,ls='--',label="Kauffmann(2003)")
    plt.plot(NII_plot2,BPT_border(NII_plot2, 'Kewley'),c=c,label="Kewley(2001)")
    plt.xlim(xls); plt.ylim(yls)
    plt.xlabel(r"log([NII]$\rm\lambda$6583/H$\alpha$)",fontsize=14)
    plt.ylabel(r"log([OIII]$\rm\lambda$5007/H$\beta$)",fontsize=14)
    

def BPT_border(log_NII_Ha,linetype = "Kauffmann"):
	if linetype=="Kauffmann": return 1.3 + 0.61 / (log_NII_Ha-0.05)
	elif linetype=="Kewley": return 1.19 + 0.61 / (log_NII_Ha - 0.47)
    
def BPT_plot_dens(NII_Ha, OIII_Hb, xls=(-2.5,0.5), yls=(-3.5,1.5),bins=100):
    plt.figure(figsize=(8,6))

    H, xbins, ybins = np.histogram2d(NII_Ha, OIII_Hb,
		bins=(np.linspace(xls[0]-0.5, xls[1], bins), np.linspace(yls[0]-0.5, yls[1], bins)))
    C = plt.contourf(np.log(H).T,aspect="auto",origin="lower", cmap='jet',
             extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]])
    
    BPT_set(xls,yls)
    cb = plt.colorbar(mappable=C)
    cb.set_label('Density',fontsize="large")
    
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
    cb.set_label('Density',fontsize=14)

def BPT_plot_kde(NII_Ha, OIII_Hb, 
                 xls=(-2.5,0.5), yls=(-3.5,1.5), 
                 bw=None, cmap='bone_r',classic=True,colorbar=False):
    
    xx, yy = np.mgrid[xls[0]:xls[1]:200j, yls[0]:yls[1]:200j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([NII_Ha, OIII_Hb])
    kernel = st.gaussian_kde(values,bw_method=bw)
    f = np.reshape(kernel(positions).T, xx.shape)
    
    #plt.figure(figsize=(8,6))
    im = plt.imshow(f.T,cmap=cmap,origin='lower',aspect='auto',
                    extent=[xls[0],xls[1],yls[0],yls[1]])
    if colorbar:
        cb = plt.colorbar(mappable=im)
        cb.set_label('Density',fontsize=12)
    if classic: BPT_set(xls, yls, c="k")
    plt.tight_layout()
    return im

# =============================================================================
# models
# =============================================================================
tab_lines = pd.read_table("object_tracks-lines.dat")

tab_shell = tab_lines[tab_lines.comp=="shell"]

Ha = tab_shell["H  1 6562.81A"]         #Ha
Hb = tab_shell["H  1 4861.33A"]         #Hb
NII = tab_shell["N  2 6583.45A"]        #NII
OIII = tab_shell["O  3 5006.84A"]       #OIII
OIIa = tab_shell['O  2 3726.03A']
OIIb = tab_shell['O  2 3728.81A']
OII = tab_shell['BLND 3727.00A']     #OII
SIIa = tab_shell["S  2 6716.44A"]    #SIIa
SIIb = tab_shell["S  2 6730.82A"]       #SIIb
S_bland = tab_shell["BLND 6720.00A"]


tab_shell["log_OIII5007_Hb"] = np.log10(OIII/Hb)
tab_shell["log_OIII5007_NII6583"] = np.log10(OIII/NII)
tab_shell["log_OIII5007_OII3727"] = np.log10(OIII/OII)

tab_shell["log_OII3727_Hb"] = np.log10(OII/Hb)
tab_shell["log_OII3727_NII6583"] = np.log10(OII/NII)

tab_shell["log_OII3727_OIII5007_Hb"] = np.log10((OIII+OII)/Hb)

tab_shell["log_NII6583_Ha"] = np.log10(NII/Ha)

tab_shell["log_SII6716_6731_Ha"] = np.log10((S_bland)/Ha)
tab_shell["log_SII6716_6731_NII6583"] = np.log10((S_bland)/NII)
tab_shell["log_SII6716_SII6731"] = np.log10(SIIa/SIIb)

tab_shell["OIII_Hb"] = np.log10(OIII/Hb)
tab_shell["NII_Ha"] = np.log10(NII/Ha)

tab_shell["log_age"] = np.log10(tab_shell.age_year)

# =============================================================================
# SITELLE data
# =============================================================================
hdu_SIT = fits.open("SITELLE/NGC628_catalog.fits")[0]

data_SIT = hdu_SIT.data.byteswap().newbyteorder()   # solve a big/little endian issue in fits -> pandas


col_names = ("id","ra","dec","r_kpc","L_Ha","DIG",
             "category","I0","Amp","sig","alpha",
             "R2","size_pc","EBV","EBV_err",
             "log_NII6583_Ha","log_NII6583_Ha_err","log_NII6583_Ha_SNRc",
             "log_SII6716_6731_Ha","log_SII6716_6731_Ha_err","log_SII6716_6731_Ha_SNRc",
             "log_SII6716_6731_NII6583","log_SII6716_6731_NII6583_err","log_SII6716_6731_NII6583_SNRc",
             "log_OIII5007_Hb","log_OIII5007_Hb_err","log_OIII5007_Hb_SNRc",
             "log_OII3727_Hb","log_OII3727_Hb_err","log_OII3727_Hb_SNRc",
             "log_OII3727_OIII5007_Hb","log_OII3727_OIII5007_Hb_err","log_OII3727_OIII5007_Hb_SNRc",
             "log_OIII5007_OII3727","log_OIII5007_OII3727_err","log_OIII5007_OII3727_SNRc",
             "log_OIII5007_NII6583","log_OIII5007_NII6583_err","log_OIII5007_NII6583_SNRc",
             "log_OII3727_NII6583","log_OII3727_NII6583_err","log_OII3727_NII6583_SNRc",
             "SII6716_SII6731","SII6716_SII6731_err","SII6716_SII6731_SNRc")

tab_obs = pd.DataFrame(data_SIT, columns=col_names)
tab_obs["log_SII6716_SII6731"] = np.log10(tab_obs.SII6716_SII6731)
tab_obs["log_SII6716_SII6731_err"] = 0.434*(tab_obs.SII6716_SII6731_err/tab_obs.SII6716_SII6731)


tab_sym = tab_obs[tab_obs.category==1.]     # Symmetric obj

table_M = tab_sym[tab_sym.L_Ha>1e37]        # Massive obj
table_m = tab_sym[tab_sym.L_Ha<1e37]        # Less massive obj

table = table_M.dropna(subset=["log_NII6583_Ha","log_OIII5007_Hb"])

 
#BPT_plot_dens_models(tab_shell["NII_Ha"],tab_shell["OIII_Hb"],bins=80,weight=True)
#plt.errorbar(table.log_NII6583_Ha, table.log_OIII5007_Hb,
#             xerr=table.log_NII6583_Ha_err,yerr=table.log_OIII5007_Hb_err,
#             c="gray",ls="",marker="o",ms=5,mfc="steelblue",mec="k",alpha=0.5,
#             label=r"SITELLE: symmetric+L$_{H\alpha}>10^{37}$")


# =============================================================================
# Line Ratio Fig.26 LN 2018
# =============================================================================
fig = plt.figure(figsize=(11,12))
ax = plt.subplot(321)
BPT_plot_kde(tab_shell["NII_Ha"],tab_shell["OIII_Hb"])
plt.errorbar(table.log_NII6583_Ha, table.log_OIII5007_Hb,
             xerr=table.log_NII6583_Ha_err,yerr=table.log_OIII5007_Hb_err,
             c="gray",ls="",marker="o",ms=5,mfc="steelblue",mec="k",lw=1,alpha=0.4,
             label=r"SITELLE: symmetric + L$\rm_{H\alpha}>10^{37}$")
plt.legend(loc="best",fontsize=10)

ax = plt.subplot(322)
BPT_plot_kde(tab_shell.log_SII6716_6731_Ha,tab_shell.log_OIII5007_Hb,xls=(-1.5,0.5),yls=(-3.5,1.5),classic=False)
plt.errorbar(table.log_SII6716_6731_Ha, table.log_OIII5007_Hb,
             xerr=table.log_SII6716_6731_Ha_err,yerr=table.log_OIII5007_Hb_err,
             c="gray",ls="",marker="o",ms=5,mfc="steelblue",mec="k",lw=1,alpha=0.4,
             label=r"SITELLE: symmetric + L$_{H\alpha}>10^{37}$")
plt.legend(loc="best",fontsize=10)
plt.xlim(-1.5,.5);plt.ylim(-3.5,1.5)
plt.xlabel(r"log([SII]6716,6731/H$\alpha$)",fontsize="large")
plt.ylabel(r"log([OIII]5007/H$\beta$)",fontsize="large")

ax = plt.subplot(323)
BPT_plot_kde(tab_shell.log_OII3727_Hb,tab_shell.log_OIII5007_Hb,xls=(-1.5,2.5),yls=(-2.5,1.5),classic=False)
plt.errorbar(table.log_OII3727_Hb, table.log_OIII5007_Hb,
             xerr=table.log_OII3727_Hb_err,yerr=table.log_OIII5007_Hb_err,
             c="gray",ls="",marker="o",ms=5,mfc="steelblue",mec="k",lw=1,alpha=0.4,
             label=r"SITELLE: symmetric + L$_{H\alpha}>10^{37}$")
plt.legend(loc="best",fontsize=10)
plt.xlim(-1.5,2.5);plt.ylim(-2.5,1.5)
plt.xlabel(r"log([OII]3727/H$\beta$)",fontsize="large")
plt.ylabel(r"log([OIII]5007/H$\beta$)",fontsize="large")

ax = plt.subplot(324)
BPT_plot_kde(tab_shell.log_SII6716_6731_Ha,tab_shell.log_OII3727_Hb,xls=(-1.5,0.5),yls=(-1.5,2.),classic=False)
plt.errorbar(table.log_SII6716_6731_Ha, table.log_OII3727_Hb,
             xerr=table.log_SII6716_6731_Ha_err,yerr=table.log_OII3727_Hb_err,
             c="gray",ls="",marker="o",ms=5,mfc="steelblue",mec="k",lw=1,alpha=0.4,
             label=r"SITELLE: symmetric + L$_{H\alpha}>10^{37}$")
plt.legend(loc="best",fontsize=10)
plt.xlim(-1.5,.5);plt.ylim(-1.5,2.)
plt.xlabel(r"log([SII]6716,6731/H$\alpha$)",fontsize="large")
plt.ylabel(r"log([OII]3727/H$\beta$)",fontsize="large")

ax = plt.subplot(325)
BPT_plot_kde(tab_shell.log_NII6583_Ha,tab_shell.log_OII3727_Hb,xls=(-2.,0.5),yls=(-1.5,2.),classic=False)
plt.errorbar(table.log_NII6583_Ha, table.log_OII3727_Hb,
             xerr=table.log_NII6583_Ha_err,yerr=table.log_OII3727_Hb_err,
             c="gray",ls="",marker="o",ms=5,mfc="steelblue",mec="k",lw=1,alpha=0.4,
             label=r"SITELLE: symmetric + L$_{H\alpha}>10^{37}$")
plt.legend(loc="best",fontsize=10)
plt.xlim(-2.,.5);plt.ylim(-1.5,2.)
plt.xlabel(r"log([NII]6583/H$\alpha$)",fontsize="large")
plt.ylabel(r"log([OII]3727/H$\beta$)",fontsize="large")

plt.tight_layout()

#plt.savefig("SITELLE_HII_regions_on_models_kde_all.pdf")

# =============================================================================
# Other Line ratio
# =============================================================================
fig = plt.figure(figsize=(15,6))

ax = plt.subplot(131)
BPT_plot_kde(tab_shell["NII_Ha"],tab_shell["OIII_Hb"])
plt.errorbar(table_M.log_NII6583_Ha, table_M.log_OIII5007_Hb,
             xerr=table_M.log_NII6583_Ha_err,yerr=table_M.log_OIII5007_Hb_err,
             c="gray",ls="",marker="o",ms=5,mfc="steelblue",mec="k",alpha=0.4,
             label=r"SITELLE: symmetric + L$_{H\alpha}>10^{37}$")
#plt.errorbar(table_m.log_NII6583_Ha, table_m.log_OIII5007_Hb,
#             xerr=table_m.log_NII6583_Ha_err,yerr=table_m.log_OIII5007_Hb_err,
#             c="gray",ls="",marker="o",ms=5,mfc="g",mec="k",alpha=0.4,
#             label=r"SITELLE: symmetric + L$_{H\alpha}<10^{37}$")
plt.legend(loc="best",fontsize=10)

ax = plt.subplot(132)
BPT_plot_kde(tab_shell.log_OIII5007_NII6583,tab_shell.log_OIII5007_Hb,
             xls=(-3.25,2.5),yls=(-3.5,1.5),classic=False)
plt.errorbar(table_M.log_OIII5007_NII6583, table_M.log_OIII5007_Hb,
             xerr=table_M.log_OIII5007_NII6583_err,yerr=table_M.log_OIII5007_Hb_err,
             c="gray",ls="",marker="o",ms=5,mfc="steelblue",mec="k",alpha=0.4,
             label=r"SITELLE: symmetric + L$_{H\alpha}>10^{37}$")
#plt.errorbar(table_m.log_OIII5007_NII6583, table_m.log_OIII5007_Hb,
#             xerr=table_m.log_OIII5007_NII6583_err,yerr=table_m.log_OIII5007_Hb_err,
#             c="gray",ls="",marker="o",ms=5,mfc="g",mec="k",alpha=0.4,
#             label=r"SITELLE: symmetric + L$_{H\alpha}<10^{37}$")
plt.legend(loc="best",fontsize=10)
plt.xlim(-3.25,2.5);plt.ylim(-3.5,1.5)
plt.xlabel(r"log([OIII]5007/[NII]6583)",fontsize=12)
plt.ylabel(r"log([OIII]5007/H$\beta$)",fontsize=12)
plt.tight_layout()

ax = plt.subplot(133)
im = BPT_plot_kde(tab_shell["NII_Ha"],tab_shell.log_OIII5007_NII6583,
                  xls=(-2.5,0.5),yls=(-3.25,2.5),classic=False)
plt.errorbar(table_M.log_NII6583_Ha, table_M.log_OIII5007_NII6583,
             xerr=table_M.log_NII6583_Ha_err,yerr=table_M.log_OIII5007_NII6583_err,
             c="gray",ls="",marker="o",ms=5,mfc="steelblue",mec="k",alpha=0.4,
             label=r"SITELLE: symmetric + L$_{H\alpha}>10^{37}$")
#plt.errorbar(table_m.log_NII6583_Ha, table_m.log_OIII5007_NII6583,
 #            xerr=table_m.log_NII6583_Ha_err,yerr=table_m.log_OIII5007_NII6583_err,
  #           c="gray",ls="",marker="o",ms=5,mfc="g",mec="k",alpha=0.4,
   #          label=r"SITELLE: symmetric + L$_{H\alpha}<10^{37}$")
plt.xlim(-2.5,0.5);plt.ylim(-3.25,2.5)
plt.xlabel(r"log([NII]6583/H$\alpha$)",fontsize=12)
plt.ylabel(r"log([OIII]5007/[NII]6583)",fontsize=12)
plt.legend(loc="best",fontsize=10)

cbar_ax = fig.add_axes([0.2, 0.08, 0.6, 0.04])
cb = fig.colorbar(im, orientation='horizontal',cax=cbar_ax)
cb.set_label('Density',fontsize=12)    

plt.subplots_adjust(bottom=0.22,hspace=0.2,wspace=0.2)

#plt.savefig("Data_on_models_kde.pdf")

