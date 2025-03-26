"""Functions that can be used for model fitting.

NOTES
-----
- ERPparam currently (only) uses the exponential and gaussian functions.
- Linear & Quadratic functions are from previous versions of ERPparam.
    - They are left available for easy swapping back in, if desired.
"""

import numpy as np

from ERPparam.core.errors import InconsistentDataError

###################################################################################################
###################################################################################################


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

def sigmoid_function(time, amplitude=1, latency=0, slope=1):
    """
    Sigmoid function

    Parameters
    ----------
    time : numpy array
        Input data
    amplitude : float, optional
        Amplitude of the sigmoid function. The default is 1.
    latency : float, optional
        Latency of the sigmoid function. The default is 0.
    slope : float, optional
        Slope of the sigmoid function. The default is 1.

    Returns
    -------
    sigmoid : numpy array
        Sigmoid function output
    
    """

    sigmoid = amplitude / (1 + np.exp(-slope * (time - latency)))

    return sigmoid


def sigmoid_multigauss(time, amplitude=1, latency=0, slope=1, *params,
                       peak_mode='gaussian'):
    """
    Sigmoid function

    Parameters
    ----------
    time : numpy array
        Input data
    amplitude : float, optional
        Amplitude of the sigmoid function. The default is 1.
    latency : float, optional
        Latency of the sigmoid function. The default is 0.
    slope : float, optional
        Slope of the sigmoid function. The default is 1.
    *params : float
        Parameters that define gaussian function.
    peak_mode : {'gaussian'}, optional
        Which kind of component to generate

    Returns
    -------
    sigmoid : numpy array
        Sigmoid function output  
    
    """

    sigmoid = sigmoid_function(time, amplitude, latency, slope)
    peak_function = get_pe_func(peak_mode)
    gauss = peak_function(time, *params)
    sigmulti = sigmoid + gauss

    return sigmulti


def linear_function(xs, *params):
    """Linear fitting function.

    Parameters
    ----------
    xs : 1d array
        Input x-axis values.
    *params : float
        Parameters that define linear function.

    Returns
    -------
    ys : 1d array
        Output values for linear function.
    """

    ys = np.zeros_like(xs)

    offset, slope = params

    ys = ys + offset + (xs*slope)

    return ys


def quadratic_function(xs, *params):
    """Quadratic fitting function.

    Parameters
    ----------
    xs : 1d array
        Input x-axis values.
    *params : float
        Parameters that define quadratic function.

    Returns
    -------
    ys : 1d array
        Output values for quadratic function.
    """

    ys = np.zeros_like(xs)

    offset, slope, curve = params

    ys = ys + offset + (xs*slope) + ((xs**2)*curve)

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
    else:
        raise ValueError("Requested periodic mode not understood.")

    return pe_func


