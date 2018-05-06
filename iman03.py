#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 13:00:22 2018

@author: bickels, jingyuwang
"""

import sys
import os
import numba as nb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.ndimage as ndi
from scipy.spatial import cKDTree
from skimage import exposure
from skimage.filters import threshold_li  # , threshold_otsu, threshold_niblack, threshold_sauvola
from skimage.feature import blob_log  # , blob_dog, blob_doh
from skimage.restoration import denoise_wavelet, estimate_sigma
import timeit


# %% FUNCTIONS
def normalize(im):
    im = im.astype(float)
    return (im - im.min()) / (im.max() - im.min())


@nb.jit(nopython=True, nogil=True)
def blur_detect(a, sv_num=3):
    """http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.221.6050&rep=rep1&type=pdf
    blurred Image Region Detection and Classification"""
    s0 = int(len(a) ** 0.5)
    block = a.reshape((s0, s0))
    u, s, v = np.linalg.svd(block)
    top_sv = np.sum(s[0:sv_num])
    total_sv = np.sum(s)
    return top_sv / total_sv


# %% SET UP SOME COLORMAPS WITH PURE RGB
cdict = mpl.cm.datad['gist_heat']  # load colormap as template

Rdict = cdict.copy()
Rdict['alpha'] = ((0.0, 0.0, 0.0), (0.5, 0.25, 0.25), (1.0, 1.0, 1.0))
Rdict['red'] = ((0.0, 0.0, 0.0), (0.5, 0.25, 0.25), (1.0, 1.0, 1.0))
Rdict['green'] = ((0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0))
Rdict['blue'] = ((0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0))
alphared = mpl.colors.LinearSegmentedColormap('alphared', Rdict)

Gdict = cdict.copy()
Gdict['alpha'] = ((0.0, 0.0, 0.0), (0.5, 0.25, 0.25), (1.0, 1.0, 1.0))
Gdict['red'] = ((0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0))
Gdict['green'] = ((0.0, 0.0, 0.0), (0.5, 0.25, 0.25), (1.0, 1.0, 1.0))
Gdict['blue'] = ((0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0))
alphagreen = mpl.colors.LinearSegmentedColormap('alphagreen', Gdict)

Bdict = cdict.copy()
Bdict['alpha'] = ((0.0, 0.0, 0.0), (0.5, 0.25, 0.25), (1.0, 1.0, 1.0))
Bdict['red'] = ((0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0))
Bdict['green'] = ((0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0))
Bdict['blue'] = ((0.0, 0.0, 0.0), (0.5, 0.25, 0.25), (1.0, 1.0, 1.0))
alphablue = mpl.colors.LinearSegmentedColormap('alphablue', Bdict)

# %% DEFINE
print
'load images...',
# values passed from main.py
path = sys.argv[1]
f = sys.argv[2]
mag = sys.argv[3]

gfp = plt.imread(os.path.join(path, f + r'_GFP.tif'))
# txr = plt.imread(os.path.join(path,f+r'_TxRed.tif'))
dapi = plt.imread(os.path.join(path, f + r'_DAPI.tif'))
trans = plt.imread(os.path.join(path, f + r'_TRANS.tif'))

sx, sy = gfp.shape
aspect = sx / float(sy)
# TODO: if there are any constants (or settings) specific to magnification, specifiy them here
if mag == '4x':
    dxy = 2.24023e-6  # MicronsPerPixel="2.24023"
    max_sigma = 3
    min_sigma = 1
if mag == '10x':
    dxy = 0.912021e-6  # MicronsPerPixel="0.912021"
    max_sigma = 6
    min_sigma = 1
print
'done'

# %% NORMALIZE/EQUALIZE CHANNELS
print
'normalizing channels...',
gfp = normalize(gfp)
# txr = normalize(txr)
dapi = normalize(dapi)
trans = exposure.equalize_adapthist(trans)
print
'done'
# Estimate the average noise standard deviation across color channels.
sigma_est = estimate_sigma(trans, multichannel=True, average_sigmas=True)
# Due to clipping in random_noise, the estimate will be a bit smaller than the
# specified sigma.
# remove the noise
trans = denoise_wavelet(trans, sigma=1.2 * sigma_est)
# Histgram Equalization
# gfp = exposure.equalize_adapthist(gfp, clip_limit=0.03)
# Contrast Stretching (image rescale)
# %% CALCULATE BLUR-MAP
print
'detecting blur...',
start_time = timeit.default_timer()
window = 11  # might have to be changed to set appropriate lengthscale of blur
blur_map = ndi.generic_filter(gfp, blur_detect, window)
blur_map = normalize(blur_map)  # ranges from 0(clear) to 1(blurry)
elapsed = timeit.default_timer() - start_time
print
'done in', elapsed

# %% THRESHOLDING TO GET MASK
print
'creating blur mask...',
thres = threshold_li(blur_map)
mask = blur_map < thres
mask = ndi.morphology.binary_fill_holes(mask)
area = np.count_nonzero(mask) * dxy ** 2
print
'done'

# %%
print
'Enhance images...'
sigma_est = estimate_sigma(gfp)
# Due to clipping in random_noise, the estimate will be a bit smaller than the
# specified sigma.
# remove the noise
gfp = denoise_wavelet(gfp, sigma=1.2 * sigma_est)
# Histgram Equalization
# gfp = exposure.equalize_adapthist(gfp, clip_limit=0.03)
# Contrast Stretching (image rescale)
p1, p2 = np.percentile(gfp, (50, 98))
gfp = exposure.rescale_intensity(gfp, in_range=(p1, p2))
print
'done'

# %% DETECT BRIGHT SPOTS
print
'detecting blobs...',
thre = threshold_li(gfp)
blobs = blob_log(gfp, min_sigma=min_sigma, max_sigma=max_sigma,
                 threshold=thre)  # have to be adjusted accroding to the enhancement
y, x, sigma = blobs.T  # sigma provides the size of the blob in units of standard deviation

valid = mask[y.astype(int), x.astype(int)]  # blobs that are not in blurred region are valid
print
'done'

# %% DEFINE CELLS
bsigma = sigma < 6
ncells = len(sigma[bsigma & valid])
density = ncells / area
print
'cell density [#/m-2] = ', density

# %% CALCULATE DISTANCE TO k-NEAREST NEIGHBOR
print
'calculate distances to neighbors...',
# FIXME: This just calculates distances, including distances across blurred regions
if ncells is not 0:
    loc = np.c_[x, y] * dxy
    tree = cKDTree(loc[bsigma & valid])
    dist, ids = tree.query(loc, k=3)
else:
    print
    'no cells detected, skipping'
print
'done'

# %% SAVE
print
'saving...',
np.savez(os.path.join(path, f + r'_out.npz'),
         gfp=gfp,
         dapi=dapi,
         #             txr=txr,
         trans=trans,
         mask=mask,
         x=x,
         y=y,
         sigma=sigma,
         ncells=ncells,
         area=area,
         dxy=dxy,
         dist=dist)
print
'done'

# %% VISUALIZE
nbins = 100
plt.hist(dist[:, 1] * 1e6, bins=nbins, normed=True, histtype='step', label='k=1')
plt.hist(dist[:, 2] * 1e6, bins=nbins, normed=True, histtype='step', label='k=2')
# plt.xscale('log')
plt.xlabel(r'distance to nearest neighbor $[\mu m]$')
plt.ylabel(r'probability density')
plt.legend(title='neighbor degree')
plt.savefig(os.path.join(path, f + '_histogram.pdf'), filetype='pdf')  # change fileformat here
plt.show()

extent = (0, sy * dxy * 1e6, 0, sx * dxy * 1e6)

plt.imshow(trans,
           cmap='Greys_r',
           extent=extent,
           interpolation='None',
           origin='lower')
plt.imshow(mask,
           cmap='binary',
           alpha=0.2,
           extent=extent,
           interpolation='None',
           origin='lower')
plt.imshow(gfp,
           cmap=alphagreen,
           extent=extent,
           interpolation='None',
           origin='lower')
plt.xlabel(r'$[\mu m]$')
plt.ylabel(r'$[\mu m]$')
plt.title(r'SYTO9 (green) and out of focus areas (light grey)')
plt.savefig(os.path.join(path, f + '_gfp_and_blur-mask.png'), dpi=300)
plt.show()

plt.imshow(trans,
           cmap='Greys_r',
           extent=extent,
           interpolation='None',
           origin='lower')
plt.imshow(gfp,
           cmap=alphagreen,
           extent=extent,
           interpolation='None',
           origin='lower')
plt.imshow(dapi,
           cmap=alphablue,
           extent=extent,
           interpolation='None',
           origin='lower')
plt.scatter(x[valid] * dxy * 1e6, y[valid] * dxy * 1e6,
            s=sigma,
            marker='*',
            color='r',
            label='detected cells')
plt.legend(loc=3)
plt.xlabel(r'$[\mu m]$')
plt.ylabel(r'$[\mu m]$')
plt.title(r'SYTO9 (green) chalcofluor white (blue)')
plt.savefig(os.path.join(path, f + '_detected_cells.png'), dpi=300)
plt.show()

fig, ax = plt.subplots()
plt.imshow(gfp,
           cmap='viridis',
           extent=extent,
           interpolation='None',
           origin='lower')
# plt.colorbar()
circles = [plt.Circle((xi * dxy * 1e6, yi * dxy * 1e6), si) for yi, xi, si in blobs[valid & bsigma]]
p = mpl.collections.PatchCollection(circles, color='r', facecolor='None')
ax.add_collection(p)
plt.xlabel(r'$[\mu m]$')
plt.ylabel(r'$[\mu m]$')
plt.title(r'SYTO9 (green) detected cells (blue)')
plt.savefig(os.path.join(path, f + '_detected_cells2.png'), dpi=300)
plt.show()

# plt.imshow(gfp,origin='lower')
# plt.plot(x,y,'ro')
# plt.matshow(gfp,origin='lower')
