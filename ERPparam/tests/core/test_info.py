"""Tests for ERPparam.core.info."""

from ERPparam.core.info import *

###################################################################################################
###################################################################################################

def test_get_description(tfm):

    desc = get_description()
    objs = dir(tfm)

    # Test that everything in dict is a valid component of the ERPparam object
    for ke, va in desc.items():
        for it in va:
            assert it in objs

def test_get_gauss_indices():

    indices = get_gauss_indices()

    # Check it returns a valid object & that values are correct
    assert indices
    for ind, val in enumerate(['MN', 'HT', 'SD','SK']):
        assert indices[val] == ind

def test_get_shape_indices():

    indices = get_shape_indices()

    # Check it returns a valid object & that values are correct
    assert indices
    for ind, val in enumerate(['CT', 'PW', 'BW','SQ', 'FWHM', 'rise_time', 'decay_time', 'symmetry', 'sharpness', 'sharpness_rise', 'sharpness_decay']):
        assert indices[val] == ind

def test_get_info(tfm, tfg):

    for f_obj in [tfm, tfg]:
        assert get_info(f_obj, 'settings')
        assert get_info(f_obj, 'meta_data')
        assert get_info(f_obj, 'results')
