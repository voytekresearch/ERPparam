"""Conversion functions for organizing model results into alternate representations."""

import numpy as np

from ERPparam import Bands
from ERPparam.core.info import (get_peak_indices, get_shape_indices, 
                                get_gauss_indices)
from ERPparam.core.modutils import safe_import, check_dependency
from ERPparam.analysis.periodic import get_band_peak

pd = safe_import('pandas')

###################################################################################################
###################################################################################################

def model_to_dict(fit_results, peak_org):
    """Convert model fit results to a dictionary.

    Parameters
    ----------
    fit_results : ERPparamResults
        Results of a model fit.
    peak_org : int or Bands
        How to organize peaks.
        If int, extracts the first n peaks.
        If Bands, extracts peaks based on band definitions.

    Returns
    -------
    dict
        Model results organized into a dictionary.
    """

    fr_dict = {}

    # concatenate peak and shape parameters
    peak_params = fit_results.peak_params
    shape_params = fit_results.shape_params
    gaussian_params = fit_results.gaussian_params
    peaks = np.hstack((peak_params, shape_params, gaussian_params))

    # get indices for peak and shape parameters
    peak_indices = get_peak_indices()
    shape_indices = get_shape_indices()
    gauss_indices = get_gauss_indices()
    for key in shape_indices.keys():
        shape_indices[key] += len(peak_indices)
    for key in gauss_indices.keys():
        gauss_indices[key] += (len(peak_indices) + len(shape_indices))
    indices = {**peak_indices, **shape_indices, **gauss_indices}

    if isinstance(peak_org, int):
        if len(peaks) < peak_org:
            nans = [np.array([np.nan] * 14) for _ in range(peak_org-len(peaks))]
            peaks = np.vstack((peaks, nans))

        for ind, peak in enumerate(peaks[:peak_org, :]):
            for pe_label, pe_param in zip(indices, peak):
                fr_dict[pe_label.lower() + '_' + str(ind)] = pe_param

    elif isinstance(peak_org, Bands):
        for band, f_range in peak_org:
            for label, param in zip(indices, get_band_peak(peaks, f_range)):
                fr_dict[band + '_' + label.lower()] = param

        # goodness-of-fit metrics
        fr_dict['error'] = fit_results.error
        fr_dict['r_squared'] = fit_results.r_squared

    return fr_dict

@check_dependency(pd, 'pandas')
def model_to_dataframe(fit_results, peak_org):
    """Convert model fit results to a dataframe.

    Parameters
    ----------
    fit_results : ERPparamResults
        Results of a model fit.
    peak_org : int or Bands
        How to organize peaks.
        If int, extracts the first n peaks.
        If Bands, extracts peaks based on band definitions.

    Returns
    -------
    pd.Series
        Model results organized into a dataframe.
    """

    return pd.Series(model_to_dict(fit_results, peak_org))


@check_dependency(pd, 'pandas')
def group_to_dataframe(fit_results, peak_org):
    """Convert a group of model fit results into a dataframe.

    Parameters
    ----------
    fit_results : list of ERPparamResults
        List of ERPparamResults objects.
    peak_org : int or Bands
        How to organize peaks.
        If int, extracts the first n peaks.
        If Bands, extracts peaks based on band definitions.

    Returns
    -------
    pd.DataFrame
        Model results organized into a dataframe.
    """

    return pd.DataFrame([model_to_dataframe(f_res, peak_org) for f_res in fit_results])
