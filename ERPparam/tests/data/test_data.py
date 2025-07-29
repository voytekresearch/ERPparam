"""Tests for the ERPparam.data.data.

For testing the data objects, the testing approach is to check that the object
has the expected fields, given what is defined in the object description.
"""

from ERPparam.core.items import OBJ_DESC

from ERPparam.data.data import *

###################################################################################################
###################################################################################################

def test_ERPparam_settings():

    settings = ERPparamSettings([1, 8], 8, 0.25, 2, 'gaussian', 0.75, 500, 0.5) #['peak_width_limits', 'max_n_peaks','min_peak_height', 'peak_threshold',
                                                                            #'peak_mode', 'gauss_overlap_thresh', 'maxfev', 
                                                                            # 'amplitude_fraction']
    assert settings

    for field in OBJ_DESC['settings']:
        assert getattr(settings, field)

def test_ERPparam_meta_data():

    meta_data = ERPparamMetaData([1, 50], 0.5) #['time_range', 'fs']
    assert meta_data

    for field in OBJ_DESC['meta_data']:
        assert getattr(meta_data, field)

def test_ERPparam_results():
    # [ 'r_squared', 'error', 'gaussian_params','shape_params', 'peak_indices']
    results = ERPparamResults(0.95, 
                              0.05,
                              [10, 0.5, 1], 
                              [0.05, 0.05, 0.025, 0.5, 0.97,  0.97, 0.98, 10, 0.5, 1], 
                              [0]) 
    assert results

    results_fields = OBJ_DESC['results']
    for field in results_fields:
        assert getattr(results, field.strip('_'))

def test_sim_params():

    sim_params = SimParams([10, 0.5, 1], 0.05) # ['peak_params', 'nlv']
    assert sim_params

    for field in [ 'peak_params', 'nlv']:
        assert getattr(sim_params, field)
