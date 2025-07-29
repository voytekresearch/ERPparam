"""Functions that can be used for model fitting.

NOTES
-----
- ERPparam currently (only) uses the gaussian and skewed gaussian functions.
"""

import numpy as np
from scipy.stats import norm

from ERPparam.core.errors import InconsistentDataError

###################################################################################################
###################################################################################################


def skewed_gaussian_function(xs, *params):
    """Skewed Gaussian fitting function.

    Parameters
    ----------
    xs : 1d array
        Input x-axis values.
    *params : float
        Parameters that define skewed gaussian function:
        * ctr: center of gaussian
        * hgt: height of gaussian
        * wid: width of gaussian
        * skew: skewness of gaussian
    """

    ys = np.zeros_like(xs)

    for ii in range(0, len(params), 4):

        ctr, hgt, wid, skew = params[ii:ii+4]

        ys = ys + skewed_gaussian(xs, ctr, hgt, wid, skew)

    return ys

def skewed_gaussian(xs, *params):
    """Skewed Gaussian PDF function.

    Parameters
    ----------
    xs : 1d array
        Input x-axis values.
    *params : float
        Parameters that define skewed gaussian function:
        * ctr: center of gaussian
        * hgt: height of gaussian
        * wid: width of gaussian
        * skew: skewness of gaussian

    Returns
    -------
    ys : 1d array
        Output values for skewed gaussian function.
    """

    ctr, hgt, wid, skew = params

    # Gaussian distribution
    pdf = gaussian_function(xs, ctr, hgt, wid)

    # Skewed cumulative distribution function
    cdf = norm.cdf(skew * ((xs - ctr) / wid))

    # Skew the gaussian
    ys = pdf * cdf

    # Rescale height
    ys = (ys / np.max(np.abs(ys))) * np.abs(hgt)

    return ys


def gaussian_function(xs, *params):
    """Gaussian fitting function.

    Parameters
    ----------
    xs : 1d array
        Input x-axis values.
    *params : float
        Parameters that define gaussian function.

    Returns
    -------
    ys : 1d array
        Output values for gaussian function.
    """

    ys = np.zeros_like(xs)

    for ii in range(0, len(params), 3):

        ctr, hgt, wid = params[ii:ii+3]

        ys = ys + hgt * np.exp(-(xs-ctr)**2 / (2*wid**2))

    return ys


def get_pe_func(periodic_mode):
    """Select and return specified function for periodic component.

    Parameters
    ----------
    periodic_mode : {'gaussian'}
        Which periodic fitting function to return.

    Returns
    -------
    pe_func : function
        Function for the periodic component.

    Raises
    ------
    ValueError
        If the specified periodic mode label is not understood.

    """

    if periodic_mode == 'gaussian':
        pe_func = gaussian_function
    elif periodic_mode == 'skewed_gaussian':
        pe_func = skewed_gaussian_function
    else:
        raise ValueError("Requested periodic mode not understood.")

    return pe_func
