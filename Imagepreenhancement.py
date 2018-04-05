#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 14:36:25 2018

@author: jingyuwang
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 13:07:52 2018

@author: bickels
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.ndimage as ndi
from scipy.spatial import cKDTree
from skimage import exposure
from skimage.filters import threshold_li#, threshold_otsu
from skimage.feature import blob_log
import numba as nb


#%% FUNCTIONS
def normalize(im):
    im = im.astype(float)
    return (im-im.min())/(im.max()-im.min())
    #return np.int8(((im-im.min())*255)/(im.max()-im.min()))

@nb.jit(nopython=True,nogil=True)
def blur_detect(a, sv_num=3):
    """http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.221.6050&rep=rep1&type=pdf
    blurred Image Region Detection and Classification"""
    s0 = int(len(a)**0.5)
    block = a.reshape((s0,s0))
    u, s, v = np.linalg.svd(block)
    top_sv = np.sum(s[0:sv_num])
    total_sv = np.sum(s)
    return top_sv/total_sv

#%% DEFINE 
print 'load images...',
#TODO: modify path and filenames here
path = r'/Users/jingyuwang/Desktop/ETH 论文/Results/2nd/Day 4/1/65'
gfp = plt.imread(os.path.join(path,r'10x_1_GFP.tif'))
#txr = plt.imread(os.path.join(path,r'day4 10x colony 1_TxRed.tif'))
#dapi = plt.imread(os.path.join(path,r'10x_1_DAPI.tif'))
#trans = plt.imread(os.path.join(path,r'10x_1_TRANS.tif'))

sx,sy = gfp.shape
aspect = sx/float(sy)
dxy = 0.921021E-6 #MicronsPerPixel="0.912021" 
print 'done'

#%% Enhance Channels
print 'normalizing channels...',
gfp = normalize(gfp)
#dapi = normalize(dapi)
#trans = exposure.equalize_adapthist(trans)
print 'done'
##
print 'Enhance images...'
#Histgram Equalization
gfp = exposure.adjust_log(gfp, 2)
print'done'
plt.matshow(gfp)