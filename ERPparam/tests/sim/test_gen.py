"""Test functions for ERPparam.sim.gen"""

import numpy as np
from numpy import array_equal

from ERPparam.tests.tutils import default_group_params

from ERPparam.sim.gen import *

###################################################################################################
###################################################################################################

def test_gen_times():

    t_range = [0, 10]
    fs = 1000

    times = gen_time_vector(t_range, fs)

    assert times.min() == t_range[0]
    assert times.max() == t_range[1]

    assert len(times) == int(t_range[1]*fs)+1

def test_gen_erp():

    time_range = (-0.5, 2)
    erp_latency = [0.1, 0.2]
    erp_amplitude = [2, -1.5]
    erp_width = [0.03, 0.05]
    erp_params = np.ravel(np.column_stack([erp_latency, erp_amplitude, erp_width]))
    nlv = 0.0 # no noise

    xs, ys = simulate_erp(time_range, erp_params, nlv)

    assert np.all(xs)
    assert np.all(ys)
    assert len(xs) == len(ys)

def test_gen_erp_return_params():

    time_range = (-0.5, 2)
    erp_latency = [0.1, 0.2]
    erp_amplitude = [2, -1.5]
    erp_width = [0.03, 0.05]
    erp_params = np.ravel(np.column_stack([erp_latency, erp_amplitude, erp_width]))
    nlv = 0.0 # no noise

    xs, ys, sp = simulate_erp(time_range, erp_params, nlv, return_params=True)

    # Test returning parameters
    assert array_equal(np.hstack(sp.peak_params), erp_params)
    assert sp.nlv == nlv

def test_gen_group_erps():

    n_sigs = 3

    xs, ys = simulate_erps(n_sigs, *default_group_params())

    assert np.all(xs)
    assert np.all(ys)
    assert ys.ndim == 2
    assert ys.shape[0] == n_sigs

def test_gen_group_return_params():

    n_sigs = 3

    pes = [10, 0.5, 0.03]
    nlv = 0.01

    xs, ys, sim_params = simulate_erps(n_sigs, [1, 50], pes, nlv,
                                                 return_params=True)

    assert n_sigs == ys.shape[0] == len(sim_params)
    sp = sim_params[0]
    assert array_equal(sp.peak_params, [pes])
    assert sp.nlv == nlv

def test_gen_peaks_skewed():

    xs = gen_time_vector([0, 1], 1000)

    # positive skew
    pe_params = [0.5, 1, 0.1, 4]
    pe_vals = sim_erp(xs, pe_params, peak_mode='skewed_gaussian')
    assert np.all(np.invert(np.isnan(pe_vals)))
    assert np.isclose(xs[np.argmax(pe_vals)], 0.5, 1)
    print(xs[np.argmax(pe_vals)])

    # zero skew
    pe_params = [0.5, 1, 0.1, 0]
    pe_vals = sim_erp(xs, pe_params, peak_mode='skewed_gaussian')
    assert np.all(np.invert(np.isnan(pe_vals)))
    assert np.isclose(xs[np.argmax(pe_vals)], 0.5, 1)

    # negative skew
    pe_params = [0.5, 1, 0.1, -4]
    pe_vals = sim_erp(xs, pe_params, peak_mode='skewed_gaussian')
    assert np.all(np.invert(np.isnan(pe_vals)))
    assert np.isclose(xs[np.argmax(pe_vals)], 0.5, 1)

def test_gen_peaks():

    xs = gen_time_vector([0, 50], 1000)
    pe_params = [10, 2, 0.01]

    pe_vals = sim_erp(xs, pe_params)

    assert np.all(np.invert(np.isnan(pe_vals)))
    assert xs[np.argmax(pe_vals)] == 10

def test_gen_noise():

    t_range = [0,10]
    pe_params = [10, 2, 0.01]
    fs = 1000

    nlv = 0.1
    noise = sim_noise(t_range, pe_params, fs, nlv)
    assert np.all(np.invert(np.isnan(noise)))
    assert np.isclose(np.std(noise), (nlv*pe_params[1]), 0.25)

    nlv = 0.5
    noise = sim_noise(t_range, pe_params, fs, nlv)
    assert np.all(np.invert(np.isnan(noise)))
    assert np.isclose(np.std(noise), (nlv*pe_params[1]), 0.25)

def test_gen_signal_values():

    xs = gen_time_vector([0, 50], 1000)
    pe_params = [10, 2, 0.01]
    nlv = 0.1
    fs =1000

    ys, time = gen_signal(xs, [0, 50], pe_params,fs, nlv)

    assert np.all(ys)

