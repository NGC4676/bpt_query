#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 10:56:59 2018

@author: mac
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.stats as st
import seaborn as sns
from astropy.io import fits
from astropy.table import Table
from skimage.measure import label, regionprops

hdu_regmap = fits.open("SITELLE/NGC628_ID_regions_map.fits")[0]
reg_map = hdu_regmap.data
props = regionprops(reg_map.astype("int"))

id_prop = [prop.label for prop in props]
e_prop = [prop.eccentricity for prop in props]
s_prop = [prop.solidity for prop in props]
im_prop = [prop.image for prop in props]
conv_im_prop = [prop.convex_image for prop in props]
  
table_reg = pd.DataFrame({"id":id_prop,"e":e_prop,"s":s_prop})

table_reg["im"] = im_prop
table_reg["conv_im"] = conv_im_prop

target = table_reg[(table_reg.e<0.6)&(table_reg.s>0.9)]


#target_rand = target.sample(20)
fig, axes = plt.subplots(nrows=3, ncols=5, 
                         figsize=(10,7))
for i in range(3):
    for j in range(5):
        ax = axes[i,j]
        targ = target.iloc[i*5+j]
        ax.imshow(targ.im+targ.conv_im.astype("int"),
                  origin="lower",aspect="auto",cmap="gnuplot")
        ax.set_title("ID:%d e:%.2f s:%.2f"%(targ.id,targ.e,targ.s),fontsize=12)
plt.suptitle("SITELLE HII Regions: eccentricity<0.6 solidity>0.9",y=0.96,fontsize=15)
plt.subplots_adjust(left=0.08,right=0.92,top=0.9,bottom=0.05,wspace=0.3,hspace=0.35) 
plt.savefig("./HII_regions_morphology.pdf")   
