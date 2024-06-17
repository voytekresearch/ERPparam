"""Tests for the ERPparam.objs.group, including the ERPparamGroup object and it's methods.

NOTES
-----
The tests here are not strong tests for accuracy.
They serve rather as 'smoke tests', for if anything fails completely.
"""

import os

import numpy as np
from numpy.testing import assert_equal
from pytest import raises

from ERPparam.core.items import OBJ_DESC
from ERPparam.core.modutils import safe_import
from ERPparam.core.errors import DataError, NoDataError, InconsistentDataError
from ERPparam.data import ERPparamResults
from ERPparam.sim import simulate_erps
from ERPparam.sim.params import param_sampler

pd = safe_import('pandas')

from ERPparam.tests.settings import TEST_DATA_PATH, TEST_REPORTS_PATH
from ERPparam.tests.tutils import default_group_params, plot_test

from ERPparam.objs.group import *

###################################################################################################
###################################################################################################

def test_fg():
    """Check ERPparamGroup object initializes properly."""

    # Note: doesn't assert fg itself, as it return false when group_results are empty
    #  This is due to the __len__ used in ERPparamGroup
    fg = ERPparamGroup(verbose=False)
    assert isinstance(fg, ERPparamGroup)

def test_fg_iter(tfg):
    """Check iterating through ERPparamGroup."""

    for res in tfg:
        assert res

def test_fg_getitem(tfg):
    """Check indexing, from custom __getitem__, in ERPparamGroup."""

    assert tfg[0]

def test_fg_has_data(tfg):
    """Test the has_data property attribute, with and without data."""

    assert tfg.has_model

    ntfg = ERPparamGroup()
    assert not ntfg.has_data

    

def test_fg_has_model(tfg):
    """Test the has_model property attribute, with and without model fits."""

    assert tfg.has_model

    ntfg = ERPparamGroup()
    assert not ntfg.has_model

    with raises(NoDataError):
        ntfg.fit()

def test_ERPparam_n_peaks(tfg):
    """Test the n_peaks property attribute."""

    assert tfg.n_peaks_

def test_n_null(tfg):
    """Test the n_null_ property attribute."""

    # Since there should have been no failed fits, this should return 0
    assert tfg.n_null_ == 0

def test_null_inds(tfg):
    """Test the null_inds_ property attribute."""

    # Since there should be no failed fits, this should return an empty list
    assert tfg.null_inds_ == []

def test_fg_fit_nk():
    """Test ERPparamGroup fit, no noise. Intialize empty Group obj, then feed data into fit func. """

    n_signals = 2
    xs, ys = simulate_erps(n_signals, *default_group_params(), nlvs=0)

    tfg = ERPparamGroup(verbose=False)
    tfg.fit(xs, ys)
    out = tfg.get_results()

    assert out
    assert len(out) == n_signals
    assert isinstance(out[0], ERPparamResults)
    assert np.all(out[1].peak_params)

def test_fg_fit_nk_noise():
    """Test ERPparamGroup fit, on noisy data, to make sure nothing breaks."""

    n_signals = 5
    xs, ys = simulate_erps(n_signals, *default_group_params(), nlvs=0.10)

    tfg = ERPparamGroup(max_n_peaks=8, verbose=False)
    tfg.fit(xs, ys)

    # No accuracy checking here - just checking that it ran
    assert tfg.has_model

def test_fg_fit_progress(tfg):
    """Test running ERPparamGroup fitting, with a progress bar."""

    tfg.fit(progress='tqdm')

def test_fg_fail():
    """Test ERPparamGroup fit, in a way that some fits will fail.
    Also checks that model failures don't cause errors.
    """

    # Create some noisy spectra that will be hard to fit    
    time_range = (-0.5, 2)
    erp_latency = [0.1, 0.5]
    erp_amplitude = [2, -1.5]
    erp_width = [0.03, 0.05]
    erp_params =  param_sampler( [np.ravel(np.column_stack([erp_latency, erp_amplitude, erp_width])),
                                np.ravel(np.column_stack([erp_latency[1], erp_amplitude[1], erp_width[1]])),
                                np.ravel(np.column_stack([erp_latency[0], erp_amplitude[0], erp_width[0]]))
                                ] )
    nlv = 0.3

    xs, ys = simulate_erps(3, time_range, erp_params, nlv)

    # Use a fg with the max iterations set so low that it will fail to converge
    ntfg = ERPparamGroup(max_n_peaks=3, peak_threshold=1.5)
    ntfg._maxfev = 5

    # Fit models, where some will fail, to see if it completes cleanly
    ntfg.fit(xs, ys)

    # Check that results are all
    for res in ntfg.get_results():
        assert res

    # Test that get_params works with failed model fits
    outs1 = ntfg.get_params('peak_params')
    outs2 = ntfg.get_params('shape_params', 'sharpness')
    outs3 = ntfg.get_params('gaussian_params')
    outs4 = ntfg.get_params('peak_params', 0)
    outs5 = ntfg.get_params('gaussian_params', 2)

    # Test shortcut labels
    outs6 = ntfg.get_params('gaussian')
    outs6 = ntfg.get_params('peak', 'CT')

    # Test the property attributes related to null model fits
    #   This checks that they do the right thing when there are null fits (failed fits)
    assert ntfg.n_null_ > 0
    assert ntfg.null_inds_

def test_fg_drop():
    """Test function to drop results from ERPparamGroup."""

    n_signals = 3
    xs, ys = simulate_erps(n_signals, *default_group_params())

    tfg = ERPparamGroup(verbose=False, peak_threshold=1.5, max_n_peaks=3)

    # Test dropping one ind
    tfg.fit(xs, ys)
    tfg.drop(0)

    dropped_fres = tfg.group_results[0]
    for field in dropped_fres._fields:
        assert np.all(np.isnan(getattr(dropped_fres, field)))

    # Test dropping multiple inds
    tfg.fit(xs, ys)
    drop_inds = [0, 2]
    tfg.drop(drop_inds)

    for drop_ind in drop_inds:
        dropped_fres = tfg.group_results[drop_ind]
        for field in dropped_fres._fields:
            assert np.all(np.isnan(getattr(dropped_fres, field)))

    # Test that a ERPparamGroup that has had inds dropped still works with `get_params`
    cfs = tfg.get_params('peak_params', 1)
    exps = tfg.get_params('gaussian_params', 'MN')
    assert np.all(np.isnan([exps[i,0] for i in range(exps.shape[0]) if exps[i,1] in drop_inds ]))
    #assert np.all(np.invert(np.isnan(np.delete(exps, drop_inds))))

def test_fg_fit_par():
    """Test ERPparamGroup fit, running in parallel."""

    n_signals = 2
    xs, ys = simulate_erps(n_signals, *default_group_params())

    tfg = ERPparamGroup(verbose=False)
    tfg.fit(xs, ys, n_jobs=2)
    out = tfg.get_results()

    assert out
    assert len(out) == n_signals
    assert isinstance(out[0], ERPparamResults)
    assert np.all(out[1].gaussian_params)

def test_fg_print(tfg):
    """Check print method (alias)."""

    tfg.print_results()
    assert True

def test_save_model_report(tfg):

    file_name = 'test_group_model_report'
    tfg.save_model_report(0, file_name, TEST_REPORTS_PATH)

    assert os.path.exists(os.path.join(TEST_REPORTS_PATH, file_name + '.pdf'))

def test_get_results(tfg):
    """Check get results method."""

    assert tfg.get_results()

def test_get_params(tfg):
    # """Check get_params method."""

    # for dname in ['aperiodic_params', 'peak_params', 'error', 'r_squared', 'gaussian_params']:
    #     assert np.any(tfg.get_params(dname))

    #     if dname == 'aperiodic_params':
    #         for dtype in ['offset', 'exponent']:
    #             assert np.any(tfg.get_params(dname, dtype))

    #     if dname == 'peak_params':
    #         for dtype in ['CF', 'PW', 'BW']:
    #             assert np.any(tfg.get_params(dname, dtype))

    """Test the get_params method."""

    for dname in ['peak_params', 'peak','shape','shape_params',
                  'error', 'r_squared', 'gaussian_params', 'gaussian']:
        assert np.any(tfg.get_params(dname))

        if dname == 'peak_params' or dname == 'peak':
            for dtype in ['CT', 'PW', 'BW']:
                assert np.any(tfg.get_params(dname, dtype))

        if dname == 'gaussian_params' or dname == 'gaussian':
            for dtype in ['MN','HT','SD']:
                assert np.any(tfg.get_params(dname, dtype))

        if dname == 'shape_params' or dname == 'shape':
            for dtype in ['FWHM', 'rise_time', 'decay_time', 'symmetry',
            'sharpness', 'sharpness_rise', 'sharpness_decay']:
                assert np.any(tfg.get_params(dname, dtype))

@plot_test
def test_fg_plot(tfg, skip_if_no_mpl):
    """Check alias method for plot."""

    tfg.plot()

def test_fg_load():
    """Test load into ERPparamGroup. Note: loads files from test_core_io."""

    file_name_res = 'test_ERPparamgroup_res'
    file_name_set = 'test_ERPparamgroup_set'
    file_name_dat = 'test_ERPparamgroup_dat'

    # Test loading just results
    tfg = ERPparamGroup(verbose=False)
    tfg.load(file_name_res, TEST_DATA_PATH)
    assert len(tfg.group_results) > 0
    # Test that settings and data are None
    #   Except for aperiodic mode, which can be inferred from the data
    for setting in OBJ_DESC['settings']:
        if setting != 'aperiodic_mode':
            assert getattr(tfg, setting) is None
    assert tfg.power_spectra is None

    # Test loading just settings
    tfg = ERPparamGroup(verbose=False)
    tfg.load(file_name_set, TEST_DATA_PATH)
    for setting in OBJ_DESC['settings']:
        assert getattr(tfg, setting) is not None
    # Test that results and data are None
    for result in OBJ_DESC['results']:
        assert np.all(np.isnan(getattr(tfg, result)))
    assert tfg.power_spectra is None

    # Test loading just data
    tfg = ERPparamGroup(verbose=False)
    tfg.load(file_name_dat, TEST_DATA_PATH)
    assert tfg.power_spectra is not None
    # Test that settings and results are None
    for setting in OBJ_DESC['settings']:
        assert getattr(tfg, setting) is None
    for result in OBJ_DESC['results']:
        assert np.all(np.isnan(getattr(tfg, result)))

    # Test loading all elements
    tfg = ERPparamGroup(verbose=False)
    file_name_all = 'test_ERPparamgroup_all'
    tfg.load(file_name_all, TEST_DATA_PATH)
    assert len(tfg.group_results) > 0
    for setting in OBJ_DESC['settings']:
        assert getattr(tfg, setting) is not None
    assert tfg.power_spectra is not None
    for meta_dat in OBJ_DESC['meta_data']:
        assert getattr(tfg, meta_dat) is not None

def test_fg_report(skip_if_no_mpl):
    """Check that running the top level model method runs."""

    n_signals = 2
    xs, ys = simulate_erps(n_signals, *default_group_params())

    tfg = ERPparamGroup(verbose=False)
    tfg.report(xs, ys)

    assert tfg

def test_fg_get_ERPparam(tfg):
    """Check return of an individual model fit to a ERPparam object from ERPparamGroup."""

    # Check without regenerating
    tfm0 = tfg.get_ERPparam(0, False)
    assert tfm0
    # Check that settings are copied over properly
    for setting in OBJ_DESC['settings']:
        assert getattr(tfg, setting) == getattr(tfm0, setting)

    # Check with regenerating
    tfm1 = tfg.get_ERPparam(1, True)
    assert tfm1
    # Check that regenerated model is created
    for result in OBJ_DESC['results']:
        assert np.all(getattr(tfm1, result))

    # Test when object has no data (clear a copy of tfg)
    new_tfg = tfg.copy()
    new_tfg._reset_data_results(False, True, True, True)
    tfm2 = new_tfg.get_ERPparam(0, True)
    assert tfm2
    # Check that data info is copied over properly
    for meta_dat in OBJ_DESC['meta_data']:
        assert getattr(tfm2, meta_dat)

def test_fg_get_group(tfg):
    """Check the return of a sub-sampled ERPparamGroup."""

    # Check with list index
    inds1 = [1, 2]
    nfg1 = tfg.get_group(inds1)
    assert isinstance(nfg1, ERPparamGroup)

    # Check with range index
    inds2 = range(0, 2)
    nfg2 = tfg.get_group(inds2)
    assert isinstance(nfg2, ERPparamGroup)

    # Check that settings are copied over properly
    for setting in OBJ_DESC['settings']:
        assert getattr(tfg, setting) == getattr(nfg1, setting)
        assert getattr(tfg, setting) == getattr(nfg2, setting)

    # Check that data info is copied over properly
    for meta_dat in OBJ_DESC['meta_data']:
        assert getattr(nfg1, meta_dat)
        assert getattr(nfg2, meta_dat)

    # Check that the correct data is extracted
    assert_equal(tfg.power_spectra[inds1, :], nfg1.power_spectra)
    assert_equal(tfg.power_spectra[inds2, :], nfg2.power_spectra)

    # Check that the correct results are extracted
    assert [tfg.group_results[ind] for ind in inds1] == nfg1.group_results
    assert [tfg.group_results[ind] for ind in inds2] == nfg2.group_results

def test_fg_to_df(tfg, tbands, skip_if_no_pandas):

    df1 = tfg.to_df(2)
    assert isinstance(df1, pd.DataFrame)
    df2 = tfg.to_df(tbands)
    assert isinstance(df2, pd.DataFrame)
