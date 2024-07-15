"""Conversion functions for organizing model results into alternate representations."""

import numpy as np

from ERPparam import Bands
from ERPparam.core.info import get_peak_indices
from ERPparam.core.modutils import safe_import, check_dependency
from ERPparam.analysis.periodic import get_band_peak

pd = safe_import('pandas')

###################################################################################################
###################################################################################################

def model_to_dict(fit_results, peak_subset=None):
    """Convert model fit results to a dictionary.

    Parameters
    ----------
    fit_results : ERPparamResults
        Results of a model fit.
    peak_subset : int or None
        Extracts the first n peaks.
        If None, extracts all peaks

    Returns
    -------
    dict
        Model results organized into a dictionary.
    """

    fr_dict = {}

    # peaks parameters
    peaks = fit_results.peak_params

    if peak_subset == None:
        # if the user didn't specify the first N peaks to extract, set the desired N to be equal to the number of peaks
        peak_subset = peaks.shape[0]

    if len(peaks) < peak_subset:
        nans = [np.array([np.nan] * 3) for ind in range(peak_subset-len(peaks))]
        peaks = np.vstack((peaks, nans))

    for ind, peak in enumerate(peaks[:peak_subset, :]):
        for pe_label, pe_param in zip(get_peak_indices(), peak):
            fr_dict[pe_label.lower() + '_' + str(ind)] = pe_param

    # goodness-of-fit metrics
    fr_dict['error'] = fit_results.error
    fr_dict['r_squared'] = fit_results.r_squared

    return fr_dict

@check_dependency(pd, 'pandas')
def model_to_dataframe(fit_results, peak_subset=None):
    """Convert model fit results to a dictionary.

    Parameters
    ----------
    fit_results : ERPparamResults
        Results of a model fit.
    peak_subset : int or None
        Extracts the first n peaks.
        If None, extracts all peaks

    Returns
    -------
    pd.Series
        Model results organized into a dataframe.
    """

    return pd.Series(model_to_dict(fit_results, peak_subset))


@check_dependency(pd, 'pandas')
def group_to_dataframe(fit_results, peak_subset=None):
    """Convert model fit results to a dictionary.

    Parameters
    ----------
    fit_results : ERPparamResults
        Results of a model fit.
    peak_subset : int or None
        Extracts the first n peaks.
        If None, extracts all peaks

    Returns
    -------
    pd.DataFrame
        Model results organized into a dataframe.
    """

    return pd.DataFrame([model_to_dataframe(f_res, peak_subset) for f_res in fit_results])
