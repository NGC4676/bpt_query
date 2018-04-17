#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 16:32:47 2018

@author: mac
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import seaborn as sns


def BPT_border(log_NII_Ha,linetype = "Kauffmann"):
	if linetype=="Kauffmann": return 1.3 + 0.61 / (log_NII_Ha-0.05)
	elif linetype=="Kewley": return 1.19 + 0.61 / (log_NII_Ha - 0.47)
    
tab_lines = pd.read_table("object_tracks-lines.dat")
#tab_lines = pd.read_table("emp-pop-highres-t-object_tracks-lines.dat")

tab_shell = tab_lines[tab_lines.comp=="shell"]
tab_shell["log_age"] = np.log10(tab_shell.age_year)

Ha = tab_shell["H  1 6562.81A"]         #Ha
Hb = tab_shell["H  1 4861.33A"]         #Hb
NII = tab_shell["N  2 6583.45A"]        #NII
OIII = tab_shell["O  3 5006.84A"]       #OIII
OII = tab_shell["O  2 3726.03A"]        #OII
SIIa = tab_shell["S  2 6716.44A"]       #SIIa
SIIb = tab_shell["S  2 6730.82A"]       #SIIb


tab_shell["OIII_Hb"] = np.log10(OIII/Hb)
tab_shell["OIII_NII"] = np.log10(OIII/NII)
tab_shell["OIII_OII"] = np.log10(OIII/OII)

tab_shell["OII_Hb"] = np.log10(OII/Hb)
tab_shell["OII_NII"] = np.log10(OII/NII)

tab_shell["O_Hb"] = np.log10((OIII+OII)/Hb)

tab_shell["NII_Ha"] = np.log10(NII/Ha)

tab_shell["SII_Ha"] = np.log10((SIIa+SIIb)/Ha)
tab_shell["SII_NII"] = np.log10((SIIa+SIIb)/NII)
tab_shell["SII_SII"] = np.log10(SIIa/SIIb)

# =============================================================================
# SVC
# =============================================================================
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm


X = np.vstack([tab_shell.NII_Ha,tab_shell.OIII_Hb,
               tab_shell.OIII_NII,tab_shell.OII_Hb,tab_shell.OII_NII,
               tab_shell.O_Hb,tab_shell.OIII_OII,
               tab_shell.SII_Ha,tab_shell.NII_Ha, tab_shell.SII_SII]).T
    
t = "M_cloud"

y_true_class = np.around(tab_shell["age_year"]/10**6.,decimals=1).astype("str")
y_true_class = np.around(tab_shell[t], decimals=1).astype("str")


X_train, X_test, y_train, y_test_class = train_test_split(X, y_true_class, test_size=0.33)

# Train
clf = svm.SVC(gamma=4.,C=270,probability=True)

clf.fit(X_train, y_train)
y_pred_class = clf.predict(X_test)

y_true = y_true_class.values.astype("float")
y_test = y_test_class.values.astype("float")
y_pred = y_pred_class.astype("float")

# Scatter
plt.figure()
plt.scatter(y_pred, y_test, alpha=0.1)
xplot = np.linspace(y_true.min(),y_true.max(),25)
plt.plot(xplot,xplot,c='k',ls='--',lw=3)
plt.xlim(y_true.min(),y_true.max())
plt.ylim(y_true.min(),y_true.max())

# KDE
xx, yy = np.mgrid[y_true.min():y_true.max():100j, y_true.min():y_true.max():100j]
positions = np.vstack([xx.ravel(), yy.ravel()])
values = np.vstack([y_pred, y_test])
kernel = st.gaussian_kde(values)
f = np.reshape(kernel(positions).T, xx.shape)

plt.figure(figsize=(8,6))
ax = plt.subplot(111)
im = plt.imshow(f.T,cmap='hot',origin='lower',aspect='auto',
                extent=[y_true.min(),y_true.max(), y_true.min(),y_true.max()])
plt.plot(xplot,xplot,c='w',ls='--',lw=3)
cb = plt.colorbar(mappable=im)
cb.set_label('Density',fontsize="large")
plt.text(.05,.9,"R$^2=$%.3f"%(r2_score (y_test, y_pred)),
         color='gold',fontsize=15,transform=ax.transAxes)
plt.legend(loc=3,fontsize=15)
plt.title("Prediction for Model Parameter: %s"%t,fontsize='large')
plt.tight_layout()

s=50
plt.figure(figsize=(11,2))
for i in range(5):
    ax = plt.subplot(1,5,i+1)
    sns.barplot(np.unique(y_true),clf.predict_proba(X_test)[s+i],
                palette="BuPu",order=np.unique(y_true))    
    plt.axvline(y_test[s+i],ls='--',c='k')
plt.tight_layout()
