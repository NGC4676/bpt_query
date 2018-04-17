#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 16:14:33 2018

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
from sklearn.model_selection import GridSearchCV

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
table_sym = table_obs[table_obs["category"]==1.]     # Symmetric obj
table_M = table_sym[table_sym["L_Ha"]>1e37]        # Massive obj
table_M['log_SII6716_SII6731']= np.log10(table_M['SII6716_SII6731'])
table_M['log_SII6716_SII6731_err']= 0.434*table_M['SII6716_SII6731_err']
table = table_M.to_pandas()

# =============================================================================
# Prediction
# =============================================================================
import BPT_query as B
params = ["log_age","M_cl","SFE","hden0"]

Q = B.BPT_query(data=tab_mod, param=params)
Q.set_data(line_used = ["log_NII6583_Ha","log_OIII5007_Hb"])
Q.regression("RF-single", n_estimators=250,min_samples_leaf=20)


# =============================================================================
# Classification
# =============================================================================
import BPT_query as B
params = ["log_age","M_cl","SFE","hden0"]

Q = B.BPT_query(data=tab_mod, param=params)
Q.set_data(line_used = ["log_NII6583_Ha","log_OIII5007_Hb"])
Q.classification("KNN")

# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = Q.X_train, Q.X_test, Q.y_train, Q.y_test

tuned_parameters = [{'min_samples_split': [2, 5], 
                     'min_samples_leaf': [1, 10, 20, 50]}]

scores = ['r2']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)

    clf = GridSearchCV(B.RandomForestRegressor(n_estimators=100), tuned_parameters, cv=5,
                       scoring='%s' % score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print(clf.best_params_)
    print("Grid scores on development set:")
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
        

# =============================================================================
# Grid Search
# =============================================================================
import BPT_query as B

def binning(x,a1,a2):
    if x<=a1: y=0
    elif x<=a2: y=1
    else: y=2
    return y

tab_mod["M_bin"] = [binning(m,5.75,6.5) for m in tab_mod.M_cl]
tab_mod["SFE_bin"] = [binning(m,3,7) for m in tab_mod.SFE]
tab_mod["nH_bin"] = [binning(m,200.,500.) for m in tab_mod.hden0]

Q = B.BPT_query(data=tab_mod, param=["M_bin","SFE_bin","nH_bin"])
Q.set_data(line_used = line_used0)

X_train, X_test, y_train, y_test = Q.X_train, Q.X_test, Q.y_train_class, Q.y_test_class
        
tuned_parameters = [{'C': [1, 100], 
                     'gamma': [1e-3,5]}]

scores = ['accuracy']

for t in Q.param:
    print("# Tuning hyper-parameters for %s" % score)

    clf = GridSearchCV(B.svm.SVC(), tuned_parameters, cv=5,
                       scoring='accuracy')
    clf.fit(X_train, y_train[t])

    print("Best parameters set found on development set:")
    print(clf.best_params_)
    print("Grid scores on development set:")
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
