"""Tests for ERPparam.core.funcs."""

from pytest import raises

import numpy as np
from scipy.stats import norm, linregress

from ERPparam.core.errors import InconsistentDataError

from ERPparam.core.funcs import *

###################################################################################################
###################################################################################################

def test_gaussian_function():

    ctr, hgt, wid = 50, 5, 10

    xs = np.arange(1, 100)
    ys = gaussian_function(xs, ctr, hgt, wid)

    assert np.all(ys)

    # Check distribution matches generated gaussian from scipy
    #  Generated gaussian is normalized for this comparison, height tested separately
    assert max(ys) == hgt
    assert np.allclose([i/sum(ys) for i in ys], norm.pdf(xs, ctr, wid))

def test_linear_function():

    off, sl = 10, 2

    xs = np.arange(1, 100)
    ys = linear_function(xs, off, sl)

    assert np.all(ys)

    sl_meas, off_meas, _, _, _ = linregress(xs, ys)

    assert np.isclose(off_meas, off)
    assert np.isclose(sl_meas, sl)

def test_quadratic_function():

    off, sl, curve = 10, 3, 2

    xs = np.arange(1, 100)
    ys = quadratic_function(xs, off, sl, curve)

    assert np.all(ys)

    curve_meas, sl_meas, off_meas = np.polyfit(xs, ys, 2)

    assert np.isclose(off_meas, off)
    assert np.isclose(sl_meas, sl)
    assert np.isclose(curve_meas, curve)

def test_get_pe_func():

    pe_ga_func = get_pe_func('gaussian')
    assert callable(pe_ga_func)

    with raises(ValueError):
        get_pe_func('bad')
