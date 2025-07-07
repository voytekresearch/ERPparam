"""Utilities for working with parameters."""

import numpy as np

###################################################################################################
###################################################################################################


def compute_fwhm(std):
    """Compute the full-width half-max, given the gaussian standard deviation.

    Parameters
    ----------
    std : float
        Gaussian standard deviation.

    Returns
    -------
    float
        Calculated full-width half-max.
    """

    return 2 * np.sqrt(2 * np.log(2)) * std


def compute_gauss_std(fwhm):
    """Compute the gaussian standard deviation, given the full-width half-max.

    Parameters
    ----------
    fwhm : float
        Full-width half-max.

    Returns
    -------
    float
        Calculated standard deviation of a gaussian.
    """

    return fwhm / (2 * np.sqrt(2 * np.log(2)))
