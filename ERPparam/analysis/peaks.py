"""Functions to analyze and investigate ERPparam results - periodic components."""

import numpy as np

from ERPparam.core.items import PEAK_INDS, GAUS_INDS
from ERPparam.data.data import ERPparamResults

###################################################################################################
###################################################################################################

def get_window_peak_ep(fm, time_range, select_highest=True, threshold=None, thresh_param='PW',
                     attribute='shape_params', extract_param=False, dict_format = False):
    """Extract peaks from a time range of interest from a ERPparam object.

    Parameters
    ----------
    fm : ERPparam
        Object to extract peak data from.
    time_range : tuple of (float, float)
        Time range of interest.
        Defined as: (lower_time, upper_time).
    select_highest : bool, optional, default: True
        Whether to return single peak (if True) or all peaks within the range found (if False).
        If True, returns the highest amplitude peak within the search range.
    threshold : float, optional
        A minimum threshold value to apply.
    thresh_param : {'PW', 'BW'}
        Which parameter to threshold on. 'PW' is power and 'BW' is bandwidth.
    attribute : {'shape_params', 'gaussian_params'}
        Which attribute of peak data to extract data from.
    extract_param : False or {'MN', 'HT', 'SD', 'SK} for gaussian_params, or {'CT', 'PW', 'BW', 'SQ', 'FWHM', 
                    'rise_time', 'decay_time', 'symmetry','sharpness', 'sharpness_rise', 'sharpness_decay'} 
                    for shape_params, optional, Default False
        Which attribute of peak data to return.
    dict_format : bool, Default False
        Whether or not to format results as a dictionary with keys corresponding to 
        the parameter label and values corresponding to the parameter.
    

    Returns
    -------
    1d or 2d array, dict, or None
        Peak data. Each row is a peak, as [MN, HT, SD, SK] if attribute == "gaussian_params" and extract_param is False,
        and [CT, PW, BW, SQ, FWHM, rise_time, decay_time, symmetry,sharpness, sharpness_rise, sharpness_decay] 
        if attribute == "shape_params" and extract_param is False. 
        
        Return parameters in a dictionary as {parameter label : peak data} if dict_format is True.
        
        Returns None if the ERPparam model doesn't have valid parameters, or if there are 
        not peaks in the requested time range or matching the given criteria. 

    Examples
    --------
    Select a peak from an already fit ERPparam object 'fm', selecting the highest peak:

    >>> p3 = get_window_peak_fm(fm, [0.25, 0.6], select_highest=True)  # doctest:+SKIP

    Select beta peaks from a ERPparam object 'fm', extracting all peaks in the range:

    >>> erps = get_window_peak_fm(fm, [0.0, 1.0], select_highest=False)  # doctest:+SKIP
    """

    if attribute not in ['shape_params','gaussian_params']:
        msg = "Parameter values '{0}' not understood.".format(attribute)
        raise ValueError(msg)
    
    params = getattr(fm, attribute + '_')
    inds, thresh_param = infer_desired_params(params, thresh_param, verbose=True)

    if extract_param:
        assert extract_param in list(inds.keys())

    # Return nan array if empty input
    if params.size > 0:
        # Find indices of peaks in the specified range, and check the number found
        peak_inds = (params[:, 0] >= time_range[0]) & (params[:, 0] <= time_range[1])
        n_peaks = sum(peak_inds)

        # If there are no peaks within the specified range, return nan
        #   Note: this also catches and returns if the original input was empty
        if n_peaks > 0:
            window_peaks = params[peak_inds, :]

            # Apply a minimum threshold, if one was provided
            if (len(window_peaks) > 0) and threshold:
                window_peaks = threshold_peaks(window_peaks, threshold, inds, thresh_param)

            # If results > 1 and select_highest, then we return the highest peak
            #    Call a sub-function to select highest power peak in window
            if (len(window_peaks) > 1) and select_highest:
                window_peaks = get_highest_peak(window_peaks)

            if (len(window_peaks) > 0):
                if window_peaks.ndim == 1:
                    window_peaks = window_peaks[np.newaxis,:]

                if extract_param: 
                    window_peaks = window_peaks[:, inds[extract_param]]
                if dict_format:
                    if extract_param:
                        window_peaks = {extract_param : window_peaks}
                    elif not extract_param:
                        construct = {pl:window_peaks[:,inds[pl]] for pl in inds.keys()}   
                        window_peaks = construct
                return window_peaks
    
    if (params.size == 0) or (n_peaks < 1) or (window_peaks.shape[0] == 0):
        return None


def get_window_peak_eg(fg, time_range, threshold=None, thresh_param='PW', 
                     attribute='shape_params', extract_param=False, dict_format = False):
    """Extract peaks from a window of interest from a ERPparamGroup object.

    Parameters
    ----------
    fg : ERPparamGroup
        Object to extract peak data from.
    time_range : tuple of (float, float)
        Time range of interest.
        Defined as: (lower_frequency_bound, upper_frequency_bound).
    threshold : float, optional
        A minimum threshold value to apply. 
        If the peak doesn't meet the threshold, it is recorded as an array of NaNs
    thresh_param : {'PW', 'BW'}
        Which parameter to threshold on. 'PW' is power and 'BW' is bandwidth.
    attribute : {'shape_params', 'gaussian_params'}
        Which attribute of peak data to extract data from.
    extract_param : False or {'MN', 'HT', 'SD','SK', 'CT', 'PW', 'BW', 'SQ', 
                    'FWHM', 'rise_time', 'decay_time', 'symmetry','sharpness', 
                    'sharpness_rise', 'sharpness_decay'}, optional, Default False
        Which attribute of peak data to return.
    dict_format : bool, Default False
        Whether or not to format results as a dictionary with keys corresponding to 
        the parameter label and values corresponding to the parameter.

    Returns
    -------
    List of length N ERPs, or None
        Peak data. Each entry is result of applying get_window_peak_fm() on a single ERP
        Results can be formatted as [MN, HT, SD, SK] if attribute == "gaussian_params" and extract_param is False,
        or [CT, PW, BW, SQ, FWHM, rise_time, decay_time, symmetry,sharpness, sharpness_rise, sharpness_decay] 
        if attribute == "shape_params". The list entry for a peak will be None if the ERPparam model doesn't have 
        valid parameters, or if there are not peaks in the requested time range or matching the given criteria. 

        Returns None if the ERPparamGroup does not have any peaks fit.

    """

    n_fits = len(fg.group_results) 

    if n_fits > 0:

        # Extracts an array per ERPparam fit, and extracts window pe(aks from it
        window_peaks = []
        for ind in range(n_fits):
            each_param = get_window_peak_ep(fg.get_ERPparam(ind, regenerate=False),
                                            time_range=time_range, select_highest=True,
                                            threshold=threshold,
                                            thresh_param=thresh_param,
                                            attribute=attribute,
                                            extract_param=extract_param,
                                            dict_format=dict_format)
            if not each_param is None:
                window_peaks.append(each_param)
            else:
                window_peaks.append(None)

        return window_peaks
    else:
        return None

def get_window_peak_arr(peak_params, time_range, select_highest=True, threshold=None, thresh_param='PW'):
    """Extract peaks within a given time range of interest.

    Parameters
    ----------
    peak_params : 2d array
        Peak parameters, with shape of [n_peaks, 3].
    time_range : tuple of (float, float)
        Time range of interest.
        Defined as: (start_time, end_time).
    select_highest : bool, optional, default: True
        Whether to return single peak (if True) or all peaks within the range found (if False).
        If True, returns the highest peak within the search range.
    threshold : float, optional
        A minimum threshold value to apply.
    thresh_param : {'PW', 'BW'}
        Which parameter to threshold on. 'PW' is power and 'BW' is bandwidth.

    Returns
    -------
    window_peaks : 1d or 2d array
        Peak data. Each row is a peak, as [MN, HT, SD, SK] if gaussian_params and [CT, PW, BW, SQ, FWHM, rise_time, decay_time, symmetry,sharpness, sharpness_rise, sharpness_decay] if shape params.
    """
    len_params_arr = peak_params.shape[1]

    if peak_params.size != 0:
        inds, thresh_param = infer_desired_params(peak_params, thresh_param, verbose=True)

        # Find indices of peaks in the specified range, and check the number found
        peak_inds = (peak_params[:, 0] >= time_range[0]) & (peak_params[:, 0] <= time_range[1])
        n_peaks = sum(peak_inds)

        window_peaks = peak_params[peak_inds, :]

        # Apply a minimum threshold, if one was provided
        if threshold:
            window_peaks = threshold_peaks(window_peaks, threshold, inds, thresh_param)

        # If results > 1 and select_highest, then we return the highest peak
        #    Call a sub-function to select highest power peak in window
        if window_peaks.size > 1 and select_highest:
            window_peaks = get_highest_peak(window_peaks)

        if window_peaks.shape[0] != 0:
            # Squeeze so that if there is only 1 result, return single peak in flat array
            return window_peaks
    
    # Return nan array if empty input
    if (peak_params.size == 0) or (n_peaks==0) or (window_peaks.shape[0] == 0):
        return np.array([np.nan]*len_params_arr)

def get_window_peak_group_arr(fg_results, time_range, threshold=None, 
                            select_highest = True,
                            attribute = 'shape_params',
                            thresh_param='PW',
                            rmv_nans=False):
    """Extract peaks within a given window of interest, from peaks from a group fit.

    Parameters
    ----------
    fg_results : list of ERPparamGroup Results 
        List of ERPparamGroup Results objects, one for each ERP input to the Group object.
        Generated by ERPparamGroup.get_results()
    time_range : tuple of (float, float)
        Time range of interest.
        Defined as: (start_time, end_time).
    threshold : float, optional
        A minimum threshold value to apply.
    select_highest : bool, optional, default: True
        Whether to return single peak (if True) or all peaks within the range found (if False).
        If True, returns the highest peak within the search range.
    attribute : {'shape_params', 'gaussian_params'}
        Which attribute of peak data to extract data from.
    thresh_param : {'PW', 'BW'}
        Which parameter to threshold on. 'PW' is power and 'BW' is bandwidth.
    rmv_nans : bool, default : False
        Whether or not to remove rows where there were no peaks detected at all, or peaks didn't fit the search criteria

    Returns
    -------
    window_peaks : 2d array
        Peak data. Each row is a peak found in one of the signals input to the ERPparamGroup object.
        Each row represents an individual model from the input array of all peaks, which will be one per
        ERP signal if select_highest = True and a fit is found in the given time range.

    Notes
    -----
    - Each row reflects an individual model fit, in order, filled with nan if no peak was present or if no peaks fit the search criteria.
    - First dimension is >= the number of signals input into the ERPparamGroup object,
        with at least one entry per signal (full of NaNs if no matching peaks were found), 
        or more if more than one matching peak is detected. If rmv_nans = True then the first dimension can be less
        than the number of ERPs the Group was fit on.
    """
    if (not isinstance(fg_results, list)) or (not isinstance(fg_results[0], ERPparamResults)):
        raise TypeError('Input to fg_results should be an ERPparamResults object')
    n_fits = len(fg_results) # how many signals were input to the Group object

    # Extracts an array per model fit, and extracts window peaks from it
    window_peaks = []
    for sig_fit in range(n_fits):
        peak_params = getattr(fg_results[sig_fit],attribute) # for each signal, get the shape or gauss params of the fit
        this_window_arr = get_window_peak_arr(peak_params, window=time_range, \
                                          select_highest=select_highest, \
                                            threshold=threshold, thresh_param=thresh_param)
        window_peaks.append(this_window_arr)

    # stack peaks across fits
    window_peaks = np.vstack(window_peaks)
    if rmv_nans:
        # check for NaNs, if there are rows where there's a whole row of missing params
        window_peaks = window_peaks[np.sum(np.isnan(window_peaks), axis=1) <= 1] 
    return window_peaks

def get_highest_peak(peak_params):
    """Extract the highest power peak.

    Parameters
    ----------
    peak_params : 2d array
        Peak parameters, with shape of [n_peaks, 3].

    Returns
    -------
    1d array
        Peak data. The row is a peak, as [MN, HT, SD, SK].

    Notes
    -----
    This function returns the singular highest power peak from the input set, and as
    such is defined to work on periodic parameters from a single model fit.
    """
    high_ind = np.argmax(peak_params[:, 1])

    return peak_params[high_ind, :]

def threshold_peaks(peak_params, threshold, inds, param='PW'):
    """Extract peaks that are above a given threshold value.

    Parameters
    ----------
    peak_params : 2d array
        Peak parameters, with shape of [n_peaks, 3] or [n_peaks, 4].
    threshold : float
        A minimum threshold value to apply.
    inds : dict
        Dictionary of attributes : indices for gaussian or shape parameters
    param : {'PW', 'BW', 'MN', or 'SD'}
        Which parameter to threshold on. 'PW' is power and 'BW' is bandwidth.

    Returns
    -------
    thresholded_peaks : 2d array
        Peak parameters, with shape of [n_peaks, :].

    Notes
    -----
    This function can be applied to periodic parameters from an individual model,
    or a set or parameters from a group.
    """
    # Apply a mask for the requested threshold
    thresh_mask = peak_params[:, inds[param]] > threshold
    thresholded_peaks = peak_params[thresh_mask]

    return thresholded_peaks

def infer_desired_params(peak_params, thresh_param, verbose=True):

    len_params_arr = peak_params.shape[-1]
    ## infer whether we're dealing with shape or gaussian params
    inferred = thresh_param
    if len_params_arr == len(PEAK_INDS.keys()):
        inds = PEAK_INDS
        params_label = 'shape_params'
        if thresh_param == 'HT':
            inferred = 'PW'
        elif thresh_param == 'SD':
            inferred = 'BW'
    elif len_params_arr == len(GAUS_INDS.keys()):
        inds = GAUS_INDS
        params_label = 'gaussian_params'
        if thresh_param == 'PW':
            inferred = "HT"
        elif thresh_param == 'BW':
            inferred = 'SD'
    else:
        msg = "The type of parameters input cannot be inferred. The parameter array is shape {0}, which doesn't match expected gaussian or shape parameters".format(peak_params.shape)
        raise ValueError(msg)

    if (inferred != thresh_param) and verbose:
        print(f"Inferring that the intended thresh_param is {inferred}, and the input parameters are {[params_label]}")

    return inds, inferred

