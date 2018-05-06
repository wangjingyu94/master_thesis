import numpy as np
from skimage import img_as_float
from skimage.restoration import estimate_sigma, denoise_wavelet
from skimage.exposure import rescale_intensity, equalize_adapthist
from scipy.ndimage.morphology import white_tophat


def normalize(image):
    return img_as_float(image)


def denoise(image):
    sigma_est = estimate_sigma(image, multichannel=True, average_sigmas=True)
    return denoise_wavelet(image, sigma=sigma_est)


def equalize_histogram(image, saturation_max):
    return equalize_adapthist(image, clip_limit=saturation_max)


def stretch_contrast(image, percentiles):
    p1, p2 = np.percentile(image, percentiles)
    return rescale_intensity(image, in_range=(p1, p2))


def tophat(image, syze):
    return white_tophat(image, size=syze)


def invert(image):
    return 1 - image


def to_uint8(image):
    return (image * 255).astype("uint8")
