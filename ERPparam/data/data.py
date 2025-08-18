"""Data objects for ERPparam.

Notes on ERPparam data objects:
- these data objects are NamedTuples, immutable data types with attribute labels
- the namedtuples are wrapped as classes (they are still immutable when doing this)
- wrapping in objects helps to be able to render well formed documentation for them.
- setting `__slots__` as empty voids the dynamic dictionary that usually stores attributes
    - this means no additional attributes can be defined (which is more memory efficient)
"""

from collections import namedtuple

###################################################################################################
###################################################################################################

class ERPparamSettings(namedtuple('ERPparamSettings', ['peak_width_limits', 
                                                       'max_n_peaks',
                                                       'min_peak_height', 
                                                       'peak_threshold',
                                                       'peak_mode', 
                                                       'gauss_overlap_thresh', 
                                                       'maxfev', 
                                                       'amplitude_fraction'])):
    """User defined settings for the fitting algorithm.
    Parameters
    ----------
    peak_width_limits : tuple of (float, float)
        Limits on possible peak width, in Hz, as (lower_bound, upper_bound).
    max_n_peaks : int
        Maximum number of peaks to fit.
    min_peak_height : float
        Absolute threshold for detecting peaks, in units of the input data.
    peak_threshold : float
        Relative threshold for detecting peaks, in units of standard deviation of the input data.
    peak_mode : {'gaussian', 'skewed_gaussian'}
        Mode for fitting the peaks.
    gauss_overlap_thresh : float
        Overlap threshold for Gaussian peaks, as a fraction of the peak width.
    maxfev : int
        Maximum number of function evaluations for the fitting algorithm.
    amplitude_fraction : float, optional, default: 0.5
        Fraction of the peak amplitude to use as a threshold for computing
        the shape parameters of the ERP peak.

    Notes
    -----
    This object is a data object, based on a NamedTuple, with immutable data attributes.
    """
    __slots__ = ()


class ERPparamMetaData(namedtuple('ERPparamMetaData', ['time_range', 'fs', 'baseline', 'time_res'])):
    """Metadata information about a power spectrum.

    Parameters
    ----------
    time_range : list of [float, float]
        Time range of the signal, as [start_time, end_time].
    fs : float
        Sampling frequency of the signal.

    Notes
    -----
    This object is a data object, based on a NamedTuple, with immutable data attributes.
    """
    __slots__ = ()


class ERPparamResults(namedtuple('ERPparamResults', ['gaussian_params',
                                                     'shape_params',
                                                     'peak_indices',
                                                     'r_squared', 
                                                     'error',
                                                     'adj_r_squared'])):
    """Model results from parameterizing a power spectrum.

    Parameters
    ----------
    gaussian_params : 2d array
        Parameters that define the gaussian fit( s).
        Each row is a gaussian, as [mean, height, standard deviation].
    shape_params : 2d array
        ERP shape parameters 
        Each row is a waveform, as [FWHM, rise-time, decay-time, rise-decay symmetry,
        sharpness, rising sharpeness, decaying sharpeness, CF, PW, BW].
    peak_indices : 1d array, 
        Indices of the peaks in the input data.
    r_squared : float
        R-squared of the fit between the full model fit and the input data.
    error : float
        Error of the full model fit.
    adj_r_squared : float
        Adjusted R-squared of the fit between the full model fit and the input data.

    Notes
    -----
    This object is a data object, based on a NamedTuple, with immutable data attributes.
    """
    __slots__ = ()


class SimParams(namedtuple('SimParams', ['peak_params', 'nlv'])):
    """Parameters that define a simulated ERP.

    Parameters
    ----------
    peak_params : list or list of lists
        Parameters that define the peaks component.
    nlv : float
        Noise level added to simulated spectrum.

    Notes
    -----
    This object is a data object, based on a NamedTuple, with immutable data attributes.
    """
    __slots__ = ()
