"""Test functions for ERPparam.analysis.periodic."""

import numpy as np

from ERPparam.analysis.periodic import *
from ERPparam import ERPparam, ERPparamGroup
from ERPparam.sim import simulate_erp

###################################################################################################
###################################################################################################

def test_get_band_peak_ep():

    time_range = (-0.2, 1)
    nlv = 0.0
    fs = 1000

    # simulate ERP with a single peak
    erp_latency = [ 0.5]
    erp_amplitude = [1.0]
    erp_width = [ 0.1]
    erp_params = np.ravel(np.column_stack([erp_latency, erp_amplitude, erp_width]))
    time, erp = simulate_erp(time_range, erp_params, nlv=nlv, fs=fs)

    # Apply ERPparam
    tfm = ERPparam(verbose=False, max_n_peaks=1)
    tfm.fit(time, erp, time_range=[0, 1.0])

    # test whether any output is given
    assert np.all(get_band_peak_ep(tfm, (0,1)))
    # test whether None is returned if there's no peak in our time range
    assert get_band_peak_ep(tfm, (0,0.5)) is None
    # test whether None is returned if there's no peak in our threshold
    assert get_band_peak_ep(tfm, (0,1), threshold = 1.1) is None
    # test whether we output a dictionary when desired
    assert type(get_band_peak_ep(tfm, (0,1), dict_format=True)) == dict
    # test whether we correctly extract the center time
    assert np.isclose(get_band_peak_ep(tfm, (0,1), extract_param='CT')[0], 0.5)

    # simulate ERP
    erp_latency = [ 0.25, 0.75]
    erp_amplitude = [1.0, 0.50]
    erp_width = [ 0.1, 0.1]
    erp_params = np.ravel(np.column_stack([erp_latency, erp_amplitude, erp_width]))
    time, erp = simulate_erp(time_range, erp_params, nlv=nlv, fs=fs)

    # Apply ERPparam
    tfm = ERPparam(verbose=False, max_n_peaks=2)
    tfm.fit(time, erp, time_range=[0, 1.0])

    # test whether we find that there's one peak in our time range
    assert np.all(get_band_peak_ep(tfm, (0,0.5)))
    assert (get_band_peak_ep(tfm, (0,0.5))).shape[0] == 1
    # test whether we find that there's two peaks in our time range
    assert (get_band_peak_ep(tfm, (0,1), select_highest=False)).shape[0] == 2
    # test whether we find that there's one peak over our threshold
    assert (get_band_peak_ep(tfm, (0,1), threshold = 0.9)).shape[0] == 1
    # test whether we correctly extract the center time of the second peak
    assert np.isclose(get_band_peak_ep(tfm, (0.5,1), extract_param='CT')[0], 0.75)
    # test whether we correctly extract the center time of the second peak from gauss params
    assert np.isclose(get_band_peak_ep(tfm, (0.5,1), attribute='gaussian_params')[0][0], 0.75)

# def test_get_band_peak_fg(tfg):

#     assert np.all(get_band_peak_fg(tfg, (8, 12)))

# def test_get_band_peak_group():

#     data = np.array([[10, 1, 1.8, 0], [13, 1, 2, 2], [14, 2, 4, 2]])

#     out1 = get_band_peak_group(data, [8, 12], 3)
#     assert out1.shape == (3, 3)
#     assert np.array_equal(out1[0, :], [10, 1, 1.8])

#     out2 = get_band_peak_group(data, [12, 16], 3)
#     assert out2.shape == (3, 3)
#     assert np.array_equal(out2[2, :], [14, 2, 4])

# def test_get_band_peak():

#     data = np.array([[10, 1, 1.8], [14, 2, 4]])

#     # Test single result
#     assert np.array_equal(get_band_peak(data, [10, 12]), [10, 1, 1.8])

#     # Test no results - returns nan
#     assert np.all(np.isnan(get_band_peak(data, [4, 8])))

#     # Test multiple results - return all
#     assert np.array_equal(get_band_peak(data, [10, 15], select_highest=False),
#                           np.array([[10, 1, 1.8], [14, 2, 4]]))

#     # Test multiple results - return one
#     assert np.array_equal(get_band_peak(data, [10, 15], select_highest=True),
#                           np.array([14, 2, 4]))

#     # Test applying a threshold
#     assert np.array_equal(get_band_peak(data, [10, 15], threshold=1.5, select_highest=False),
#                           np.array([14, 2, 4]))

# def test_get_highest_peak():

#     data = np.array([[10, 1, 1.8], [14, 2, 4], [12, 3, 2]])

#     assert np.array_equal(get_highest_peak(data), [12, 3, 2])

# def test_threshold_peaks():

#     # Check it works, with a standard height threshold
#     data = np.array([[10, 1, 1.8], [14, 2, 4], [12, 3, 2.5]])
#     assert np.array_equal(threshold_peaks(data, 2.5), np.array([[12, 3, 2.5]]))

#     # Check it works using a bandwidth threshold
#     data = np.array([[10, 1, 1.8], [14, 2, 4], [12, 3, 2.5]])
#     assert np.array_equal(threshold_peaks(data, 2, param='BW'),
#                           np.array([[14, 2, 4], [12, 3, 2.5]]))

#     # Check it works with an [n_peaks, 4] array, as from FOOOFGroup
#     data = np.array([[10, 1, 1.8, 0], [13, 1, 2, 2], [14, 2, 4, 2]])
#     assert np.array_equal(threshold_peaks(data, 1.5), np.array([[14, 2, 4, 2]]))

def test_empty_inputs():

    data = np.empty(shape=[0, 3])

    assert np.all(get_band_peak(data, [8, 12]))
    assert np.all(get_highest_peak(data))
    assert np.all(threshold_peaks(data, 1))

    data = np.empty(shape=[0, 4])

    assert np.all(get_band_peak_group(data, [8, 12], 0))
