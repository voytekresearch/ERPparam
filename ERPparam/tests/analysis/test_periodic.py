"""Test functions for ERPparam.analysis.periodic."""

import numpy as np

from ERPparam.analysis.periodic import *
from ERPparam import ERPparam, ERPparamGroup
from ERPparam.sim import simulate_erp
from ERPparam.data.data import ERPparamResults
from ERPparam.core.items import PEAK_INDS, GAUS_INDS

from pytest import raises

###################################################################################################
###################################################################################################

def test_get_window_peak_ep():

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
    assert np.all(get_window_peak_ep(tfm, (0,1)))
    # test whether None is returned if there's no peak in our time range
    assert get_window_peak_ep(tfm, (0,0.5)) is None
    # test whether None is returned if there's no peak in our threshold
    assert get_window_peak_ep(tfm, (0,1), threshold = 1.1) is None
    # test whether we output a dictionary when desired
    assert type(get_window_peak_ep(tfm, (0,1), dict_format=True)) == dict
    # test whether we correctly extract the center time
    assert np.isclose(get_window_peak_ep(tfm, (0,1), extract_param='CT')[0], 0.5)

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
    assert np.all(get_window_peak_ep(tfm, (0,0.5)))
    assert (get_window_peak_ep(tfm, (0,0.5))).shape[0] == 1
    # test whether we find that there's two peaks in our time range
    assert (get_window_peak_ep(tfm, (0,1), select_highest=False)).shape[0] == 2
    # test whether we find that there's one peak over our threshold
    assert (get_window_peak_ep(tfm, (0,1), threshold = 0.9)).shape[0] == 1
    # test whether we correctly extract the center time of the second peak
    assert np.isclose(get_window_peak_ep(tfm, (0.5,1), extract_param='CT')[0], 0.75)
    # test whether we correctly extract the center time of the second peak from gauss params
    assert np.isclose(get_window_peak_ep(tfm, (0.5,1), attribute='gaussian_params')[0][0], 0.75)

def test_get_window_peak_group_arr():
    time_range = (-0.2, 1)
    nlv = 0.0
    fs = 1000

    erp_latency = [ 0.25, 0.75]
    erp_amplitude = [1.0, 0.5]
    erp_width = [ 0.1, 0.1]
    erp_params = np.ravel(np.column_stack([erp_latency, erp_amplitude, erp_width]))
    time, erp = simulate_erp(time_range, erp_params, nlv=nlv, fs=fs)

    erp_latency = [ 0.15, 0.6]
    erp_amplitude = [1.2, 0.5]
    erp_width = [ 0.1, 0.1]
    erp_params2 = np.ravel(np.column_stack([erp_latency, erp_amplitude, erp_width]))
    _, erp2 = simulate_erp(time_range, erp_params2, nlv=nlv, fs=fs)

    erp_latency = [ 0.4]
    erp_amplitude = [0.75]
    erp_width = [ 0.15]
    erp_params3 = np.ravel(np.column_stack([erp_latency, erp_amplitude, erp_width]))
    _, erp3 = simulate_erp(time_range, erp_params3, nlv=nlv, fs=fs)

    erps = np.vstack([erp,erp2,erp3])
    eg = ERPparamGroup(verbose=False, max_n_peaks=3)
    eg.fit(time=time, signals=erps, time_range=[0,1])

    # test whether we get out the same number of peaks that our group has
    out = get_window_peak_group_arr(eg.get_results(), (0.0,1), select_highest=False, threshold=None, attribute='gaussian_params')
    assert out.shape[0] == 5
    # test whether we get out one peak per signals since select_highest is True
    out = get_window_peak_group_arr(eg.get_results(), (0.0,1), select_highest=True, threshold=None, attribute='gaussian_params', rmv_nans=True)
    assert out.shape[0] == 3
    # test whether our amplitude filter works
    out = get_window_peak_group_arr(eg.get_results(), (0.0,1), select_highest=False, threshold=0.8, attribute='gaussian_params', rmv_nans=True)
    assert out.shape[0] == 2
    # test whether our bandwidth filter works
    out = get_window_peak_group_arr(eg.get_results(), (0.0,1), select_highest=False, threshold=0.11, thresh_param='BW', attribute='gaussian_params', rmv_nans=True)
    assert out.shape[0] == 1

def test_get_highest_peak():

    data = np.array([[10, 1, 1.8], [14, 2, 4], [12, 3, 2]])

    assert np.array_equal(get_highest_peak(data), [12, 3, 2])

def test_threshold_peaks():

    # Check it works, with a standard height threshold
    data = np.array([[10, 1, 1.8], [14, 2, 4], [12, 3, 2.5]])
    assert np.array_equal(threshold_peaks(data, 2.5, PEAK_INDS), np.array([[12, 3, 2.5]]))

    # Check it works using a bandwidth threshold
    data = np.array([[10, 1, 1.8], [14, 2, 4], [12, 3, 2.5]])
    assert np.array_equal(threshold_peaks(data, 2, PEAK_INDS, param='BW'),
                          np.array([[14, 2, 4], [12, 3, 2.5]]))

def test_empty_inputs():

    data = np.empty(shape=[0, 4])

    assert np.sum(np.isnan(get_window_peak_arr(data, [8, 12]))) == 4

    with raises(TypeError):
        get_window_peak_group_arr(data, [1,2]) 
