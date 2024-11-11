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

        ys = ys + hgt * skewed_gaussian(xs, ctr, hgt, wid, skew)

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

    # check if empty
    if len(params) == 0:
        return np.zeros_like(xs)
    
    # compute Gaussian PDF
    if params[3] == 0:
        pdf = gaussian_pdf(xs, *params[:3]) # no skew
    elif params[3] > 0:
        pdf = 2 * gaussian_pdf(xs, *params[:3]) * gaussian_cdf(params[3] * xs, *params[:3])
    elif params[3] < 0:
        pdf = 2 * gaussian_pdf(xs, *params[:3]) * (1 - gaussian_cdf(params[3] * xs, *params[:3]))
    
    # scale to height parameter
    ys = pdf * (params[1] / np.max(np.abs(pdf)))
    if params[1] < 0:
        ys = -ys

    return ys


def gaussian_pdf(xs, *params):
    """Probability density function (PDF) of a Gaussian.

    Parameters
    ----------
    xs : 1d array
        Input x-axis values.
    *params : float
        Parameters that define skewed gaussian function:
        * ctr: center of gaussian
        * hgt: height of gaussian
        * wid: width of gaussian

    Returns
    -------
    ys : 1d array
        Output values for skewed gaussian function.
    """

    gf = gaussian_function(xs, *params) # compute gaussian function
    ys = gf / (np.sqrt(2*np.pi) * params[2]) # normalize to unit area

    return ys


def gaussian_cdf(xs, *params):
    """Cumulative distribution function (CDF) of a Gaussian PDF.

    Parameters
    ----------
    xs : 1d array
        Input x-axis values.
    *params : float
        Parameters that define skewed gaussian function:
        * ctr: center of gaussian
        * hgt: height of gaussian
        * wid: width of gaussian

    Returns
    -------
    ys : 1d array
        Output values for skewed gaussian function.
    """

    gf = gaussian_pdf(xs, *params)
    ys = np.cumsum(gf) / np.sum(gf) 

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


def expo_function(xs, *params):
    """Exponential fitting function, for fitting aperiodic component with a 'knee'.

    NOTE: this function requires linear frequency (not log).

    Parameters
    ----------
    xs : 1d array
        Input x-axis values.
    *params : float
        Parameters (offset, knee, exp) that define Lorentzian function:
        y = 10^offset * (1/(knee + x^exp))

    Returns
    -------
    ys : 1d array
        Output values for exponential function.
    """

    ys = np.zeros_like(xs)

    offset, knee, exp = params

    ys = ys + offset - np.log10(knee + xs**exp)

    return ys


def expo_nk_function(xs, *params):
    """Exponential fitting function, for fitting aperiodic component without a 'knee'.

    NOTE: this function requires linear frequency (not log).

    Parameters
    ----------
    xs : 1d array
        Input x-axis values.
    *params : float
        Parameters (offset, exp) that define Lorentzian function:
        y = 10^off * (1/(x^exp))

    Returns
    -------
    ys : 1d array
        Output values for exponential function, without a knee.
    """

    ys = np.zeros_like(xs)

    offset, exp = params

    ys = ys + offset - np.log10(xs**exp)

    return ys


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


def get_ap_func(aperiodic_mode):
    """Select and return specified function for aperiodic component.

    Parameters
    ----------
    aperiodic_mode : {'fixed', 'knee'}
        Which aperiodic fitting function to return.

    Returns
    -------
    ap_func : function
        Function for the aperiodic component.

    Raises
    ------
    ValueError
        If the specified aperiodic mode label is not understood.
    """

    if aperiodic_mode == 'fixed':
        ap_func = expo_nk_function
    elif aperiodic_mode == 'knee':
        ap_func = expo_function
    else:
        raise ValueError("Requested aperiodic mode not understood.")

    return ap_func


def infer_ap_func(aperiodic_params):
    """Infers which aperiodic function was used, from parameters.

    Parameters
    ----------
    aperiodic_params : list of float
        Parameters that describe the aperiodic component of a power spectrum.

    Returns
    -------
    aperiodic_mode : {'fixed', 'knee'}
        Which kind of aperiodic fitting function the given parameters are consistent with.

    Raises
    ------
    InconsistentDataError
        If the given parameters are inconsistent with any available aperiodic function.
    """

    if len(aperiodic_params) == 2:
        aperiodic_mode = 'fixed'
    elif len(aperiodic_params) == 3:
        aperiodic_mode = 'knee'
    else:
        raise InconsistentDataError("The given aperiodic parameters are "
                                    "inconsistent with available options.")

    return aperiodic_mode
