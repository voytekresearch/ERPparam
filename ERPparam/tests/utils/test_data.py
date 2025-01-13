"""Test functions for ERPparam.utils.data."""

import numpy as np

from ERPparam.sim.gen import simulate_erp, simulate_erps

from ERPparam.utils.data import *

###################################################################################################
###################################################################################################

def test_trim_signal():

    f_in = np.array([0., 1., 2., 3., 4., 5.])
    p_in = np.array([1., 2., 3., 4., 5., 6.])

    f_out, p_out = trim_signal(f_in, p_in, [2., 4.])

    assert np.array_equal(f_out, np.array([2., 3., 4.]))
    assert np.array_equal(p_out, np.array([3., 4., 5.]))

def test_interpolate_signal(tfm):

    times = tfm.time
    signal = tfm.signal

    # Test with single buffer exclusion zone
    exclude = [0.05, 0.15]
    times_out, signal_out = interpolate_signal(times, signal, exclude)

    assert np.array_equal(times, times_out)
    assert np.all(signal)
    assert signal.shape == signal_out.shape
    mask = np.logical_and(times >= exclude[0], times <= exclude[1])
    assert signal[mask].sum() > signal_out[mask].sum()

    # Test with multiple buffer exclusion zones
    exclude = [[0.05, 0.15], [0.4, 0.6]]

    times_out, signal_out = interpolate_signal(times, signal, exclude)
    assert np.array_equal(times, times_out)
    assert np.all(signal)
    assert signal.shape == signal_out.shape

    for f_range in exclude:
        mask = np.logical_and(times >= f_range[0], times <= f_range[1])
        assert signal[mask].sum() > signal_out[mask].sum()

def test_subsample_signal(tfg):

    n_sim = len(tfg)
    signals = tfg.signals

    # Test with int input
    n_select = 2
    out = subsample_signal(signals, n_select)
    assert isinstance(out, np.ndarray)
    assert out.shape == (n_select, signals.shape[1])

    # Test with foat input
    prop_select = 0.75
    out = subsample_signal(signals, prop_select)
    assert isinstance(out, np.ndarray)
    assert out.shape == (int(prop_select * n_sim), signals.shape[1])

    # Test returning indices
    out, inds = subsample_signal(signals, n_select, return_inds=True)
    assert len(set(inds)) == n_select
    for ind, spectrum in zip(inds, out):
        assert np.array_equal(spectrum, signals[ind, :])

    out, inds = subsample_signal(signals, prop_select, return_inds=True)
    assert len(set(inds)) == int(prop_select * n_sim)
    for ind, spectrum in zip(inds, out):
        assert np.array_equal(spectrum, signals[ind, :])
