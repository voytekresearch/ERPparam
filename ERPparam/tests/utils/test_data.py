"""Test functions for ERPparam.utils.data."""

import numpy as np

from ERPparam.sim.gen import simulate_erp, simulate_erps

from ERPparam.utils.data import *

###################################################################################################
###################################################################################################

def test_trim_spectrum():

    f_in = np.array([0., 1., 2., 3., 4., 5.])
    p_in = np.array([1., 2., 3., 4., 5., 6.])

    f_out, p_out = trim_spectrum(f_in, p_in, [2., 4.])

    assert np.array_equal(f_out, np.array([2., 3., 4.]))
    assert np.array_equal(p_out, np.array([3., 4., 5.]))

def test_interpolate_spectrum():

    # Test with single buffer exclusion zone
    times, signals = simulate_erp(\
        [1, 75], [1, 1], [[10, 0.5, 1.0], [60, 2, 0.1]])

    exclude = [58, 62]

    times_out, signals_out = interpolate_spectrum(times, signals, exclude)

    assert np.array_equal(times, times_out)
    assert np.all(signals)
    assert signals.shape == signals_out.shape
    mask = np.logical_and(times >= exclude[0], times <= exclude[1])
    assert signals[mask].sum() > signals_out[mask].sum()

    # Test with multiple buffer exclusion zones
    times, signals = simulate_erp(\
        [1, 150], [1, 100, 1], [[10, 0.5, 1.0], [60, 1, 0.1], [120, 0.5, 0.1]])

    exclude = [[58, 62], [118, 122]]

    times_out, signals_out = interpolate_spectrum(times, signals, exclude)
    assert np.array_equal(times, times_out)
    assert np.all(signals)
    assert signals.shape == signals_out.shape

    for f_range in exclude:
        mask = np.logical_and(times >= f_range[0], times <= f_range[1])
        assert signals[mask].sum() > signals_out[mask].sum()

def test_subsample_spectra():

    # Simulate spectra, each with unique osc peak (for checking)
    n_sim = 10
    oscs = [[10 + ind, 0.25, 0.5] for ind in range(n_sim)]
    times, signals = simulate_erps(\
        n_sim, [1, 50], [1, 1], oscs)

    # Test with int input
    n_select = 2
    out = subsample_spectra(signals, n_select)
    assert isinstance(out, np.ndarray)
    assert out.shape == (n_select, signals.shape[1])

    # Test with foat input
    prop_select = 0.75
    out = subsample_spectra(signals, prop_select)
    assert isinstance(out, np.ndarray)
    assert out.shape == (int(prop_select * n_sim), signals.shape[1])

    # Test returning indices
    out, inds = subsample_spectra(signals, n_select, return_inds=True)
    assert len(set(inds)) == n_select
    for ind, spectrum in zip(inds, out):
        assert np.array_equal(spectrum, signals[ind, :])

    out, inds = subsample_spectra(signals, prop_select, return_inds=True)
    assert len(set(inds)) == int(prop_select * n_sim)
    for ind, spectrum in zip(inds, out):
        assert np.array_equal(spectrum, signals[ind, :])
