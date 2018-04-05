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
from skimage.feature import blob_dog
from skimage.feature import blob_doh
import numba as nb
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)
import timeit
from skimage.data import page
from skimage.filters import (threshold_otsu, threshold_niblack,
                             threshold_sauvola)

#%% FUNCTIONS
def normalize(im):
    im = im.astype(float)
    return (im-im.min())/(im.max()-im.min())

@nb.jit(nopython=True,nogil=True)
def blur_detect(a, sv_num=3):
    """http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.221.6050&rep=rep1&type=pdf
    blurred Image Region Detection and Classification"""
    s0 = int(len(a)**0.5)
    block = a.reshape((s0,s0))
    u, s, v = np.linalg.svd(block)
    top_sv = np.sum(s[0:sv_num])
    total_sv = np.sum(s)
    if total_sv is 0:
        return 1
    else:
        return top_sv/total_sv

#%% SET UP SOME COLORMAPS WITH PURE RGB
cdict = mpl.cm.datad['gist_heat'] #load colormap as template

Rdict = cdict.copy()
Rdict['alpha'] = ((0.0, 0.0, 0.0),(0.5, 0.25, 0.25),(1.0, 1.0, 1.0))
Rdict['red'] = ((0.0, 0.0, 0.0),(0.5, 0.25, 0.25),(1.0, 1.0, 1.0))
Rdict['green'] = ((0.0, 0.0, 0.0),(0.5, 0.0, 0.0),(1.0, 0.0, 0.0))
Rdict['blue'] = ((0.0, 0.0, 0.0),(0.5, 0.0, 0.0),(1.0, 0.0, 0.0))
alphared =  mpl.colors.LinearSegmentedColormap('alphared', Rdict)

Gdict = cdict.copy()
Gdict['alpha'] = ((0.0, 0.0, 0.0),(0.5, 0.25, 0.25),(1.0, 1.0, 1.0))
Gdict['red'] = ((0.0, 0.0, 0.0),(0.5, 0.0, 0.0),(1.0, 0.0, 0.0))
Gdict['green'] = ((0.0, 0.0, 0.0),(0.5, 0.25, 0.25),(1.0, 1.0, 1.0))
Gdict['blue'] = ((0.0, 0.0, 0.0),(0.5, 0.0, 0.0),(1.0, 0.0, 0.0))
alphagreen =  mpl.colors.LinearSegmentedColormap('alphagreen', Gdict)

Bdict = cdict.copy()
Bdict['alpha'] = ((0.0, 0.0, 0.0),(0.5, 0.25, 0.25),(1.0, 1.0, 1.0))
Bdict['red'] = ((0.0, 0.0, 0.0),(0.5, 0.0, 0.0),(1.0, 0.0, 0.0))
Bdict['green'] = ((0.0, 0.0, 0.0),(0.5, 0.0, 0.0),(1.0, 0.0, 0.0))
Bdict['blue'] = ((0.0, 0.0, 0.0),(0.5, 0.25, 0.25),(1.0, 1.0, 1.0))
alphablue =  mpl.colors.LinearSegmentedColormap('alphablue', Bdict)

#%% DEFINE 
print 'load images...',
#TODO: modify path and filenames here
path = u'/Users/jingyuwang/Desktop/ETH 论文/Results/2nd/Day 4/1/80-200'
gfp = plt.imread(os.path.join(path,r'10x_2_GFP.tif'))
#txr = plt.imread(os.path.join(path,r'day4 10x colony 1_TxRed.tif'))
dapi = plt.imread(os.path.join(path,r'10x_2_DAPI.tif'))
trans = plt.imread(os.path.join(path,r'10x_2_TRANS.tif'))
 
sx,sy = gfp.shape
aspect = sx/float(sy)
dxy = 0.921021E-6 #MicronsPerPixel="0.912021" 
print 'done'

#%% NORMALIZE/EQUALIZE CHANNELS
print 'normalizing channels...',
gfp = normalize(gfp)
dapi = normalize(dapi)
trans = exposure.equalize_adapthist(trans)
print 'done'
#%% CALCULATE BLUR-MAP
print 'detecting blur...',
start_time = timeit.default_timer()
window =11 #might have to be changed to set appropriate lengthscale of blur
#sv_num = 2
blur_map = ndi.generic_filter(gfp,blur_detect,window)#extra_arguments=(sv_num,))
blur_map = normalize(blur_map) #ranges from 0(clear) to 1(blurry)
elapsed = timeit.default_timer() - start_time
print 'done'
print elapsed

#%% THRESHOLDING TO GET MASK
print 'creating blur mask...',
thres = threshold_li(blur_map)
mask = blur_map > thres
mask=ndi.morphology.binary_fill_holes(mask)
area = np.count_nonzero(mask)*dxy**2
print 'done'

#%%
print 'Enhance images...'
# Estimate the average noise standard deviation across color channels.
sigma_est = estimate_sigma(gfp, multichannel=True, average_sigmas=True)
# Due to clipping in random_noise, the estimate will be a bit smaller than the
# specified sigma.
#remove the noise
gfp = denoise_wavelet(gfp, sigma=1.2*sigma_est)
#Histgram Equalization
#gfp = exposure.equalize_adapthist(gfp, clip_limit=0.03)
#Contrast Stretching (image rescale)
p1, p2 = np.percentile(gfp, (50, 98))
gfp = exposure.rescale_intensity(gfp, in_range=(p1, p2))

print'done'

#%% DETECT BRIGHT SPOTS
print 'detecting blobs...',

blobs = blob_log(gfp,min_sigma=1,max_sigma=10) # have to be adjusted accroding to the enhancement 
y,x,sigma = blobs.T #sigma provides the size of the blob in units of standard deviation
valid = mask[y.astype(int),x.astype(int)]#blobs that are not in blurred region are valid
print 'done'


#%% DEFINE CELLS
bsigma = sigma<4
ncells = len(sigma[bsigma & valid])
density = ncells/area
print 'cell density [#/m-2] = ', density

#%% CALCULATE DISTANCE TO k-NEAREST NEIGHBOR
print 'calculate distances to neighbors...',
#FIXME: This just calculates distances, including distances across blurred regions
if ncells is not 0:
    loc = np.c_[x,y]*dxy
    tree = cKDTree(loc[bsigma & valid])
    dist, ids = tree.query(loc,k=3)
else:
    print 'no cells detected, skipping'
print 'done'

#%% VISUALIZE
nbins=100
plt.hist(dist[:,1]*1e6, bins=nbins, normed=True, histtype='step', label='k=1')
plt.hist(dist[:,2]*1e6, bins=nbins, normed=True, histtype='step', label='k=2')
#plt.xscale('log')
plt.xlabel(r'distance to nearest neighbor $[\mu m]$')
plt.ylabel(r'probability density')
plt.legend(title='neighbor degree')
plt.savefig(os.path.join(path,'histogram.pdf'),filetype='pdf')#change fileformat here
plt.show()
    
extent=(0,sy*dxy*1e6,0,sx*dxy*1e6)
    
plt.imshow(trans,
               cmap='Greys_r', 
               extent=extent,
               interpolation='None',
               label='Trans')
plt.imshow(mask,
               cmap='cool',
               alpha=0.2,
               extent=extent,
               interpolation='None',
               label='In focus area')
plt.imshow(gfp, 
               cmap=alphagreen,
               extent=extent,
               interpolation='None',
               origin='lower',
               label='Gfp')
plt.xlabel(r'$[\mu m]$')
plt.ylabel(r'$[\mu m]$')
plt.title(r'SYTO9 (green) and in focus areas (purple)')
plt.savefig(os.path.join(path,'gfp_and_blur-mask.png'),dpi=300)
plt.show()
    
    
plt.imshow(trans, 
               cmap='Greys_r',
               extent=extent,
               interpolation='None',
               label='Trans')
plt.imshow(gfp, 
               cmap=alphagreen, 
               extent=extent,
               interpolation='None',
               label='Gfp')
plt.imshow(dapi, 
               cmap=alphablue,
               extent=extent,
               interpolation='None')
plt.scatter(x[valid]*dxy*1e6,y[valid]*dxy*1e6,
                s=sigma,
                marker='*',
                color='r', 
                label='detected cells')
plt.legend(loc=3)
plt.xlabel(r'$[\mu m]$')
plt.ylabel(r'$[\mu m]$')
plt.title(r'GFP (green) and DAPI (blue) channels')
plt.savefig(os.path.join(path,'detected_cells.png'),dpi=300)
plt.show()
    
fig,ax = plt.subplots()
plt.imshow(gfp, 
               #cmap='viridis', 
               extent=extent,
               interpolation='None')
#plt.colorbar()
circles = [plt.Circle((xi*dxy*1e6, yi*dxy*1e6), np.sqrt(2)*si) for yi,xi,si in blobs[valid & bsigma]]
p = mpl.collections.PatchCollection(circles,color='r',facecolor=None)
ax.add_collection(p)
plt.xlabel(r'$[\mu m]$')
plt.ylabel(r'$[\mu m]$')
plt.title(r'Detected cells (red)')
plt.savefig(os.path.join(path,'detected_cells2.png'),dpi=300)
plt.show()