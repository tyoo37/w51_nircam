# Simple 1/f noise smoothing method by Dan Coe

import numpy as np
from astropy.stats import sigma_clipped_stats  # , SigmaClip
from astropy.convolution import convolve  # , Gaussian2DKernel, interpolate_replace_nans

def smooth_1overf_correction(data, row_kernel_length=99, column_kernel_length=9):
    # horizontal striping: y-axis row medians
    kernel = np.ones(row_kernel_length) / float(row_kernel_length)
    ymean, ymedian, ystd = sigma_clipped_stats(data, sigma=3, maxiters=10, axis=1)
    ymedian_conv = convolve(ymedian, kernel, boundary='extend')
    row_corrected_data = data + ymedian_conv[:,np.newaxis] - ymedian[:,np.newaxis]
    
    # vertical striping: x-axis column medians
    kernel = np.ones(column_kernel_length) / float(column_kernel_length)
    xmean, xmedian, xstd = sigma_clipped_stats(row_corrected_data, sigma=3, maxiters=10, axis=0)
    xmedian_conv = convolve(xmedian, kernel, boundary='extend')
    corrected_data = row_corrected_data + xmedian_conv[np.newaxis,:] - xmedian[np.newaxis,:]

    return corrected_data