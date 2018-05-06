import numpy as np
from numba import cfunc, carray
from numba.types import intc, intp, float64, voidptr
from numba.types import CPointer
from scipy.ndimage import generic_filter
from scipy import LowLevelCallable
from skimage.filters import threshold_li
from .image_processing import normalize


def blur_map(image, window=11, sv_num=5):
    if window < 1:
        raise ValueError('Window width must be a positive integer.')
    bm = generic_filter(image, LowLevelCallable(blur_detection.ctypes), window, (sv_num,))
    bm = normalize(bm)
    return bm


def blur_threshold_mask(bmap, threshold=None):
    if threshold is None:
        threshold = threshold_li(bmap)
    elif threshold < 0 or threshold > 1:
        raise ValueError('Threshold must be a real number between 0 and 1.')
    return bmap < threshold


@cfunc(intc(CPointer(float64), intp, CPointer(float64), voidptr), nopython=False, cache=True)
def blur_detection(values_ptr, len_values, result, data):
    print(data)
    sv_num = data[0]
    if sv_num < 1:
        raise ValueError('Must select at least one singular value.')
    side_length = int(len_values ** 0.5)
    block = carray(values_ptr, (side_length, side_length))
    u, s, v = np.linalg.svd(block)
    top_sv = np.sum(s[0:sv_num])
    total_sv = np.sum(s)
    if total_sv == 0:
        result[0] = 1
    else:
        result[0] = top_sv / total_sv
    return 1


def remove_blur(image, window=11):
    bmap = blur_map(image, window)
    bmask = blur_threshold_mask(bmap)
    result = image
    result[~bmask] = 0
    return result, bmap, bmask
