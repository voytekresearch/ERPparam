"""Internal functions to manage info related to ERPparam objects."""

###################################################################################################
###################################################################################################

def get_description():
    """Get dictionary specifying ERPparam attributes, and what kind of data they store.

    Returns
    -------
    attributes : dict
        Mapping of ERPparam object attributes, and what kind of data they are.

    Notes
    -----
    This function organizes public ERPparam object attributes into:

    - results : parameters for and measures of the model
    - settings : model settings
    - data : input data
    - meta_data : meta data of the inputs
    - arrays : data stored in arrays
    - model_components : component pieces of the model
    - descriptors : descriptors of the object status and model results
    """

    attributes = {'results' : ['gaussian_params_', 'shape_params_', 'peak_indices_', 'r_squared_', 'error_', 'adj_r_squared_'],
                  'settings' : ['peak_width_limits', 'max_n_peaks', 'min_peak_height', 
                                'peak_threshold', 'peak_mode', 'gauss_overlap_thresh', 
                                'maxfev','amplitude_fraction'], 
                  'hidden_settings' : ['_bw_std_edge', '_cf_bound', 
                                       '_error_metric', '_max_n_iters', 
                                       '_debug', '_check_times', '_check_data'],
                  'data' : ['signal', 'time', 'baseline_signal', 'uncropped_time', 'uncropped_signal'],
                  'meta_data' : ['time_range', 'fs', 'baseline', 'time_res'],
                  'arrays' : ['time', 'signal', 'gaussian_params_', 
                              'shape_params_', 'baseline_signal', 
                              'uncropped_time', 'uncropped_signal'],
                  'model_components' : ['_peak_fit'],
                  'descriptors' : ['has_data', 'has_model', 'n_peaks_']
                  }

    return attributes


def get_gauss_indices():
    """Get a mapping from column labels to indices for peak parameters.

    Returns
    -------
    indices : dict
        Mapping of the column labels and indices for the peak parameters.
    """

    indices = {
        'MN' : 0,
        'HT' : 1,
        'SD' : 2,
        'SK' : 3,
    }

    return indices


def get_shape_indices():
    """Get a mapping from column labels to indices for rise-decay
     symmetry parameters.

    Returns
    -------
    indices : dict
        Mapping of the column labels and indices for the peak parameters.
    """

    indices = {
        'latency' : 0,
        'amplitude' : 1,
        'width' : 2,
        'skew' : 3,
        'fwhm' : 4,
        'rise_time' : 5,
        'decay_time' : 6,
        'symmetry' : 7,
        'sharpness' : 8,
        'sharpness_rise' : 9,
        'sharpness_decay' : 10
    }

    return indices


def get_indices(param_type='shape_params'):
    """Get a mapping from column labels to indices for all parameters.

    Parameters
    ----------

    Returns
    -------
    indices : dict
        Mapping of the column labels and indices for all parameters.
    """

    # Get the indices for the specified parameter type
    if param_type == 'gaussian_params':
        indices = get_gauss_indices()
    elif param_type=='shape_params':
        indices = get_shape_indices()
    else:
        raise ValueError('Unknown parameter type: %s' % param_type)

    return indices


def get_info(ERPparam_obj, aspect):
    """Get a selection of information from a ERPparam derived object.

    Parameters
    ----------
    ERPparam_obj : ERPparam or ERPparamGroup
        Object to get attributes from.
    aspect : {'settings', 'meta_data', 'results'}
        Which set of attributes to compare the objects across.

    Returns
    -------
    dict
        The set of specified info from the ERPparam derived object.
    """

    return {key : getattr(ERPparam_obj, key) for key in get_description()[aspect]}
