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
from skimage.filters import threshold_li
from skimage.feature import blob_log
from skimage.restoration import denoise_wavelet, estimate_sigma

Rdict = dict()
Rdict['alpha'] = ((0.0, 0.0, 0.0), (0.5, 0.25, 0.25), (1.0, 1.0, 1.0))
Rdict['red'] = ((0.0, 0.0, 0.0), (0.5, 0.25, 0.25), (1.0, 1.0, 1.0))
Rdict['green'] = ((0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0))
Rdict['blue'] = ((0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0))
alphared = mpl.colors.LinearSegmentedColormap('alphared', Rdict)

Gdict = dict()
Gdict['alpha'] = ((0.0, 0.0, 0.0), (0.5, 0.25, 0.25), (1.0, 1.0, 1.0))
Gdict['red'] = ((0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0))
Gdict['green'] = ((0.0, 0.0, 0.0), (0.5, 0.25, 0.25), (1.0, 1.0, 1.0))
Gdict['blue'] = ((0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0))
alphagreen = mpl.colors.LinearSegmentedColormap('alphagreen', Gdict)

Bdict = dict()
Bdict['alpha'] = ((0.0, 0.0, 0.0), (0.5, 0.25, 0.25), (1.0, 1.0, 1.0))
Bdict['red'] = ((0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0))
Bdict['green'] = ((0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0))
Bdict['blue'] = ((0.0, 0.0, 0.0), (0.5, 0.25, 0.25), (1.0, 1.0, 1.0))
alphablue = mpl.colors.LinearSegmentedColormap('alphablue', Bdict)


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


def normalize_channels(gfp, dapi, trans):
    return normalize(gfp), normalize(dapi), exposure.equalize_adapthist(trans)


def blur_detection(image, window):
    blur_map = ndi.generic_filter(image, blur_detect, window)
    return normalize(blur_map)


def blur_thresholding(blur_map):
    thres = threshold_li(blur_map)
    mask = blur_map < thres
    return ndi.morphology.binary_fill_holes(mask)


def denoise_image(image):
    sigma_est = estimate_sigma(image, multichannel=True, average_sigmas=True)
    return denoise_wavelet(image, sigma=1.2 * sigma_est)


def rescale_image(image, percentile):
    p1, p2 = np.percentile(image, percentile)
    return exposure.rescale_intensity(image, in_range=(p1, p2))


def detect_blobs(image, min_sigma, max_sigma):
    thre = threshold_li(image)
    return blob_log(image, min_sigma=min_sigma, max_sigma=max_sigma, threshold=thre)


def estimate_cell_count(blobs, sigma):
    bsigma = sigma < 6
    return len(sigma[bsigma & blobs])


def calculate_cell_distances(valid_cells, x_cells, y_cells, resolution, sigma):
    if len(valid_cells) is not 0:
        bsigma = sigma < 6
        loc = np.c_[x_cells, y_cells] * resolution
        tree = cKDTree(loc[bsigma & valid_cells])
        dist, _ = tree.query(loc, k=3)
        return dist
    return []


def process_image(path, f, mag, save=False):
    gfp = plt.imread(os.path.join(path, f + r'_GFP.tif'))
    dapi = plt.imread(os.path.join(path, f + r'_DAPI.tif'))
    trans = plt.imread(os.path.join(path, f + r'_TRANS.tif'))

    # TODO: if there are any constants (or settings) specific to magnification, specifiy them here
    if mag == '4x':
        dxy = 2.24023e-6  # MicronsPerPixel="2.24023"
        max_sigma = 3
        min_sigma = 1
    if mag == '10x':
        dxy = 0.912021e-6  # MicronsPerPixel="0.912021"
        max_sigma = 6
        min_sigma = 1

    gfp, dapi, trans = normalize_channels(gfp, dapi, trans)
    blur_map = blur_detection(gfp, window=11)
    mask = blur_thresholding(blur_map)
    area = np.count_nonzero(mask) * dxy ** 2
    gfp = rescale_image(denoise_image(gfp), percentile=(50, 98))
    trans = denoise_image(trans)
    blobs = detect_blobs(gfp, min_sigma, max_sigma)
    y, x, sigma = blobs.T
    valid = mask[y.astype(int), x.astype(int)]
    ncells = estimate_cell_count(valid, sigma)
    dist = calculate_cell_distances(valid, x, y, dxy, sigma)
    if save:
        np.savez(os.path.join(path, f + r'_out.npz'),
                 gfp=gfp,
                 dapi=dapi,
                 trans=trans,
                 mask=mask,
                 x=x,
                 y=y,
                 sigma=sigma,
                 ncells=ncells,
                 area=area,
                 dxy=dxy,
                 dist=dist)
    return (gfp, dapi, trans), mask, blobs, ncells, area, dxy, dist


def visualize_results(channels, mask, blobs, dxy, dist, save=False, path=None, f=None):
    y, x, sigma = blobs.T
    valid_area = mask[y.astype(int), x.astype(int)]
    bsigma = sigma < 6
    gfp, dapi, trans = channels
    sx, sy = gfp.shape

    nbins = 100
    plt.hist(dist[:, 1] * 1e6, bins=nbins, normed=True, histtype='step', label='k=1')
    plt.hist(dist[:, 2] * 1e6, bins=nbins, normed=True, histtype='step', label='k=2')
    plt.xlabel(r'distance to nearest neighbor $[\mu m]$')
    plt.ylabel(r'probability density')
    plt.legend(title='neighbor degree')
    if save:
        plt.savefig(os.path.join(path, f + '_histogram.pdf'), filetype='pdf')  # change fileformat here
    plt.show()

    extent = (0, sy * dxy * 1e6, 0, sx * dxy * 1e6)

    plt.imshow(trans, cmap='Greys_r', extent=extent)
    plt.imshow(mask, cmap='binary', alpha=0.2, extent=extent)
    plt.imshow(gfp, cmap=alphagreen, extent=extent)
    plt.xlabel(r'$[\mu m]$')
    plt.ylabel(r'$[\mu m]$')
    plt.title(r'SYTO9 (green) and out of focus areas (light grey)')
    if save:
        plt.savefig(os.path.join(path, f + '_gfp_and_blur-mask.png'), dpi=300)
    plt.show()

    plt.imshow(trans, cmap='Greys_r', extent=extent)
    plt.imshow(gfp, cmap=alphagreen, extent=extent)
    plt.imshow(dapi, cmap=alphablue, extent=extent)
    plt.scatter(x[valid_area] * dxy * 1e6, y[valid_area] * dxy * 1e6,
                s=sigma,
                marker='*',
                color='r',
                label='detected cells')
    plt.legend(loc=3)
    plt.xlabel(r'$[\mu m]$')
    plt.ylabel(r'$[\mu m]$')
    plt.title(r'SYTO9 (green) chalcofluor white (blue)')
    if save:
        plt.savefig(os.path.join(path, f + '_detected_cells.png'), dpi=300)
    plt.show()

    fig, ax = plt.subplots()
    plt.imshow(gfp, cmap='viridis', extent=extent)
    circles = [plt.Circle((xi * dxy * 1e6, yi * dxy * 1e6), si) for yi, xi, si in blobs[valid_area & bsigma]]
    p = mpl.collections.PatchCollection(circles, color='r', facecolor='None')
    ax.add_collection(p)
    plt.xlabel(r'$[\mu m]$')
    plt.ylabel(r'$[\mu m]$')
    plt.title(r'SYTO9 (green) detected cells (blue)')
    if save:
        plt.savefig(os.path.join(path, f + '_detected_cells2.png'), dpi=300)
    plt.show()


if __name__ == '__main__':
    path, f, mag = sys.argv[1:4]
    channels, mask, blobs, ncells, area, dxy, dist = process_image(path, f, mag, save=True)
    visualize_results(channels, mask, blobs, dxy, dist, save=True, path=path, f=f)
