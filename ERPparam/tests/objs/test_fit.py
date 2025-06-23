"""Tests for ERPparam.objs.fit, including the ERPparam object and it's methods.

NOTES
-----
The tests here are not strong tests for accuracy.
They serve rather as 'smoke tests', for if anything fails completely.
"""

import numpy as np
from pytest import raises

from ERPparam import ERPparam
from ERPparam.core.items import OBJ_DESC
from ERPparam.core.errors import FitError
from ERPparam.core.utils import group_three, group_four
from ERPparam.core.modutils import safe_import
from ERPparam.core.errors import DataError, NoDataError, InconsistentDataError
from ERPparam.sim import gen_time_vector, simulate_erp
from ERPparam.data import ERPparamSettings, ERPparamMetaData, ERPparamResults

pd = safe_import('pandas')

from ERPparam.tests.settings import TEST_DATA_PATH
from ERPparam.tests.tutils import get_tfm, plot_test, default_params

from ERPparam.objs.fit import *

###################################################################################################
###################################################################################################

def test_ERPparam():
    """Check ERPparam object initializes properly."""

    assert ERPparam(verbose=False)

def test_ERPparam_has_data(tfm):
    """Test the has_data property attribute, with and without model fits."""

    assert tfm.has_data

    ntfm = ERPparam()
    assert not ntfm.has_data

def test_ERPparam_has_model(tfm):
    """Test the has_model property attribute, with and without model fits."""
    
    assert tfm.has_model

    ntfm = ERPparam()
    assert not ntfm.has_model

def test_ERPparam_n_peaks(tfm):
    """Test the n_peaks property attribute."""

    assert tfm.n_peaks_

def test_ERPparam_fit():
    """Test ERPparam fit."""

    time_range, erp_params, nlv = default_params()
    xs, ys = simulate_erp(time_range, erp_params, nlv)

    tfm = ERPparam(verbose=False, max_n_peaks=4)
    tfm.fit(xs, ys, time_range=[0, time_range[1]])

    # Check model results - gaussian parameters
    for ii, gauss in enumerate(group_three(erp_params)):
        assert np.allclose(gauss, tfm.peak_params_[ii, :3], 
                           [2.0, 0.5, 1.0])
        assert np.isnan(tfm.peak_params_[ii, 3])

def test_ERPparam_fit_noise():
    """Test ERPparam fit on noisy data, to make sure nothing breaks."""

    time_range, erp_params, _ = default_params()
    nlv = 0.3

    xs, ys = simulate_erp(time_range, erp_params, nlv)

    tfm = ERPparam(max_n_peaks=2, verbose=False)
    tfm.fit(xs, ys, time_range=[0, time_range[1]])

    # No accuracy checking here - just checking that it ran
    assert tfm.has_model

def test_ERPparam_fit_skew():
    """Test ERPparam fit on skewed data with peak_mode='skewed_gaussian'."""

    time_range, erp_params_d, nlv = default_params()
    for skew in [-2, 0, 2]:
        erp_params = np.concatenate([erp_params_d[:3], [skew]])
        xs, ys = simulate_erp(time_range, erp_params, nlv, 
                            peak_mode='skewed_gaussian')

        tfm = ERPparam(verbose=False, max_n_peaks=2, peak_mode='skewed_gaussian')
        tfm.fit(xs, ys, time_range=[0, time_range[1]])

        # Check model results - gaussian parameters
        for ii, gauss in enumerate(group_four(erp_params)):
            assert np.allclose(gauss, tfm.peak_params_[ii], 
                            [2.0, 1.0, 1.0, 2.0])

def test_ERPparam_fit_measures():
    """Test goodness of fit & error metrics, post model fitting."""

    tfm = ERPparam(verbose=False)

    # Hack fake data with known properties: total error magnitude 2
    tfm.signal = np.array([1, 2, 3, 4, 5])
    tfm._peak_fit = np.array([1, 2, 5, 4, 5])

    # Check default goodness of fit and error measures
    tfm._calc_r_squared()
    assert np.isclose(tfm.r_squared_, 0.75757575)
    tfm._calc_error()
    assert np.isclose(tfm.error_, 0.4)

    # Check with alternative error fit approach
    tfm._calc_error(metric='MSE')
    assert np.isclose(tfm.error_, 0.8)
    tfm._calc_error(metric='RMSE')
    assert np.isclose(tfm.error_, np.sqrt(0.8))
    with raises(ValueError):
        tfm._calc_error(metric='BAD')

def test_ERPparam_checks():
    """Test various checks, errors and edge cases in ERPparam.
    This tests all the input checking done in `_prepare_data`.
    """
    xs, ys = simulate_erp(*default_params())

    tfm = ERPparam(verbose=False)

    ## Check checks & errors done in `_prepare_data`

    # Check wrong data type error
    with raises(DataError):
        tfm.fit(list(xs), list(ys))

    # Check dimension error
    with raises(DataError):
        tfm.fit(xs, np.reshape(ys, [1, len(ys)]))

    # Check shape mismatch error
    with raises(InconsistentDataError):
        tfm.fit(xs[:-1], ys)

    # Check complex inputs error
    with raises(DataError):
        tfm.fit(xs, ys.astype('complex'))

    # Check time_range cropping 
    tfm.fit(xs, ys, [1, 2])

    # Check error for `check_times` - for if there is non-evenly spaced time values
    with raises(DataError):
        tfm.fit(np.array([1, 2, 4]), np.array([1, 2, 3]))

    ## Check errors & errors done in `fit`

    # Check fit, and string report model error (no data / model fit)
    tfm = ERPparam(verbose=False)
    with raises(NoDataError):
        tfm.fit()

def test_ERPparam_load():
    """Test load into ERPparam. Note: loads files from test_core_io."""

    # Test loading just results
    tfm = ERPparam(verbose=False)
    file_name_res = 'test_ERPparam_res'
    tfm.load(file_name_res, TEST_DATA_PATH, regenerate=False)
    # Check that result attributes get filled
    for result in OBJ_DESC['results']:
        assert not np.all(np.isnan(getattr(tfm, result)))
    # Test that settings and data are None
    #   Except for aperiodic mode, which can be inferred from the data
    for setting in OBJ_DESC['settings']:
        if setting != 'aperiodic_mode':
            assert getattr(tfm, setting) is None
    assert getattr(tfm, 'signal') is None

    # Test loading just settings
    tfm = ERPparam(verbose=False)
    file_name_set = 'test_ERPparam_set'
    tfm.load(file_name_set, TEST_DATA_PATH)
    for setting in OBJ_DESC['settings']:
        assert getattr(tfm, setting) is not None
    # Test that results and data are None
    for result in OBJ_DESC['results']:
        assert np.all(np.isnan(getattr(tfm, result)))
    assert tfm.signal is None

    # Test loading just data
    tfm = ERPparam(verbose=False)
    file_name_dat = 'test_ERPparam_dat'
    tfm.load(file_name_dat, TEST_DATA_PATH, regenerate=False)
    assert tfm.signal is not None
    # Test that settings and results are None
    for setting in OBJ_DESC['settings']:
        assert getattr(tfm, setting) is None
    for result in OBJ_DESC['results']:
        assert np.all(np.isnan(getattr(tfm, result)))

    # Test loading all elements
    tfm = ERPparam(verbose=False)
    file_name_all = 'test_ERPparam_all'
    tfm.load(file_name_all, TEST_DATA_PATH)
    for result in OBJ_DESC['results']:
        assert not np.all(np.isnan(getattr(tfm, result)))
    for setting in OBJ_DESC['settings']:
        assert getattr(tfm, setting) is not None
    for data in OBJ_DESC['data']:
        assert getattr(tfm, data) is not None
    for meta_dat in OBJ_DESC['meta_data']:
        assert getattr(tfm, meta_dat) is not None

def test_add_data():
    """Tests method to add data to ERPparam objects."""

    # This test uses it's own ERPparam object, to not add stuff to the global one
    tfm = get_tfm()

    # Test data for adding
    time, signal = np.array([1, 2, 3]), np.array([10, 10, 10])

    # Test adding data
    tfm.add_data(time, signal)
    assert tfm.has_data
    assert np.all(tfm.time == time)
    assert np.all(tfm.signal == signal)

    # Test that prior data does not get cleared, when requesting not to clear
    tfm._reset_data_results(True, True, True)
    tfm.add_results(ERPparamResults(peak_params=np.asarray([[ 0.102     ,  1.7758259 ,  0.05875653],
       [ 0.198     , -1.48916601,  0.09412952]]), 
       r_squared=0.9973886245863048, error=0.01234299919871499, 
       gaussian_params=np.asarray([[ 0.10157924,  1.8103013 ,  0.02937827],
       [ 0.1980535 , -1.39908145,  0.04706476]]), 
       shape_params=np.asarray([[0.064     , 0.04      , 0.024     , 0.625     , 0.97706828,
        0.97134   , 0.98279656],
       [0.103     , 0.041     , 0.062     , 0.39805825, 0.9560461 ,
        0.96498029, 0.9471119 ]]), 
        peak_indices=np.asarray([[562, 602, 626],
       [657, 698, 760]])))
    tfm.add_data(time, signal, clear_results=False)
    assert tfm.has_data
    assert tfm.has_model

    # Test that prior data does get cleared, when requesting to clear
    tfm._reset_data_results(True, True, True)
    tfm.add_data(time, signal, clear_results=True)
    assert tfm.has_data
    assert not tfm.has_model

def test_add_settings():
    """Tests method to add settings to ERPparam objects."""

    # This test uses it's own ERPparam object, to not add stuff to the global one
    tfm = get_tfm()

    # Test adding settings
    ERPparam_settings = ERPparamSettings([1, 4], 6, 0, 2, "gaussian", 0.75, 500)
    tfm.add_settings(ERPparam_settings)
    for setting in OBJ_DESC['settings']:
        assert getattr(tfm, setting) == getattr(ERPparam_settings, setting)

def test_add_meta_data():
    """Tests method to add meta data to ERPparam objects."""

    # This test uses it's own ERPparam object, to not add stuff to the global one
    tfm = get_tfm()

    # Test adding meta data
    ERPparam_meta_data = ERPparamMetaData([3, 40], 0.5)
    tfm.add_meta_data(ERPparam_meta_data)
    for meta_dat in OBJ_DESC['meta_data']:
        assert getattr(tfm, meta_dat) == getattr(ERPparam_meta_data, meta_dat)

def test_add_results():
    """Tests method to add results to ERPparam objects."""

    # This test uses it's own ERPparam object, to not add stuff to the global one
    tfm = get_tfm()

    # Test adding results
    ERPparam_results = ERPparamResults(peak_params=np.asarray([[ 0.1     ,  1.7 ,  0.053],
                                                                 [ 0.1     , -1.4,  0.09]]), 
       r_squared=0.99, error=0.01, 
       gaussian_params=np.asarray([[ 0.10,  1.8 ,  0.02],
                                    [ 0.19 , -1.39,  0.047]]), 
       shape_params=np.asarray([[0.064 , 0.04 , 0.024, 0.625 , 0.97, 0.97   , 0.98],
                                  [0.103 , 0.041 , 0.062 , 0.39, 0.951 , 0.964, 0.947 ]]), 
        peak_indices=np.asarray([[562, 602, 626],  [657, 698, 760]]))
    tfm.add_results(ERPparam_results)
    assert tfm.has_model
    for setting in OBJ_DESC['results']:
        assert np.sum(getattr(tfm, setting) != getattr(ERPparam_results, setting.strip('_'))) == 0

def test_obj_gets(tfm):
    """Tests methods that return ERPparam data objects.

    Checks: get_settings, get_meta_data, get_results
    """

    settings = tfm.get_settings()
    assert isinstance(settings, ERPparamSettings)
    meta_data = tfm.get_meta_data()
    assert isinstance(meta_data, ERPparamMetaData)
    results = tfm.get_results()
    assert isinstance(results, ERPparamResults)

def test_get_params(tfm):
    """Test the get_params method."""

    for dname in ['peak_params', 'peak','shape','shape_params',
                  'error', 'r_squared', 'gaussian_params', 'gaussian']:
        assert np.any(tfm.get_params(dname))

        if dname == 'peak_params' or dname == 'peak':
            for dtype in ['CT', 'PW', 'BW']:
                assert np.any(tfm.get_params(dname, dtype))

        if dname == 'gaussian_params' or dname == 'gaussian':
            for dtype in ['MN','HT','SD']:
                assert np.any(tfm.get_params(dname, dtype))

        if dname == 'shape_params' or dname == 'shape':
            for dtype in ['FWHM', 'rise_time', 'decay_time', 'symmetry',
            'sharpness', 'sharpness_rise', 'sharpness_decay']:
                assert np.any(tfm.get_params(dname, dtype))

def test_copy():
    """Test copy ERPparam method."""

    tfm = ERPparam(verbose=False)
    ntfm = tfm.copy()

    assert tfm != ntfm

def test_ERPparam_prints(tfm):
    """Test methods that print (alias and pass through methods).

    Checks: print_settings, print_results, print_report_issue.
    """

    tfm.print_settings()
    tfm.print_results()
    tfm.print_report_issue()

@plot_test
def test_ERPparam_plot(tfm, skip_if_no_mpl):
    """Check the alias to plot ERPparam."""

    tfm.plot()

def test_ERPparam_resets():
    """Check that all relevant data is cleared in the reset method."""

    # Note: uses it's own tfm, to not clear the global one
    tfm = get_tfm()

    tfm._reset_data_results(True, True, True)
    tfm._reset_internal_settings()

    for data in ['data', 'model_components']:
        for field in OBJ_DESC[data]:
            assert getattr(tfm, field) is None
    for field in OBJ_DESC['results']:
        assert np.all(np.isnan(getattr(tfm, field)))
    assert tfm.time is None and tfm._peak_fit is None

def test_ERPparam_report(skip_if_no_mpl):
    """Check that running the top level model method runs."""

    tfm = ERPparam(verbose=False)
    tfm.report(*simulate_erp(*default_params()))

    assert tfm

def test_ERPparam_fit_failure():
    """Test ERPparam fit failures."""

    ## Induce a runtime error, and check it runs through
    tfm = ERPparam(verbose=False)
    tfm.maxfev = 5

    tfm.fit(*simulate_erp(*default_params()))

    # Check after failing out of fit, all results are reset
    for result in OBJ_DESC['results']:
        assert np.all(np.isnan(getattr(tfm, result)))

    ## Monkey patch to check errors in general
    #  This mimics the main fit-failure, without requiring bad data / waiting for it to fail.
    tfm = ERPparam(verbose=False)
    def raise_runtime_error(*args, **kwargs):
        raise FitError('Test-MonkeyPatch')
    tfm._fit_peaks = raise_runtime_error

    # Run a ERPparam fit - this should raise an error, but continue in try/except
    tfm.fit(*simulate_erp(*default_params()))

    # Check after failing out of fit, all results are reset
    for result in OBJ_DESC['results']:
        assert np.all(np.isnan(getattr(tfm, result)))

def test_ERPparam_debug():
    """Test ERPparam in debug mode, including with fit failures."""

    tfm = ERPparam(verbose=False)
    tfm.maxfev = 5

    tfm.set_debug_mode(True)
    assert tfm._debug is True

    with raises(FitError):
        tfm.fit(*simulate_erp(*default_params()))

def test_ERPparam_check_data():
    """Test ERPparam in with check data mode turned off, including with NaN data."""

    tfm = ERPparam(verbose=False)

    tfm.set_check_data_mode(False)
    assert tfm._check_data is False

    # Add data, with check data turned off
    #   In check data mode, adding data with NaN should run
    freqs = gen_time_vector([3, 50], 0.5)
    powers = np.ones_like(freqs) * np.nan
    tfm.add_data(freqs, powers)
    assert tfm.has_data

    # Model fitting should execute, but return a null model fit, given the NaNs, without failing
    tfm.fit()
    assert not tfm.has_model

def test_ERPparam_to_df(tfm, tbands, skip_if_no_pandas):

    df1 = tfm.to_df(2)
    assert isinstance(df1, pd.Series)
    df2 = tfm.to_df(tbands)
    assert isinstance(df2, pd.Series)
