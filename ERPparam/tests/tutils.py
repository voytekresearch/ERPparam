"""Utilities for testing ERPparam."""

from functools import wraps

import numpy as np

from ERPparam.bands import Bands
from ERPparam.data import ERPparamResults
from ERPparam.objs import ERPparam, ERPparamGroup
from ERPparam.core.modutils import safe_import
from ERPparam.sim.params import param_sampler
from ERPparam.sim.gen import simulate_erp, simulate_erps

plt = safe_import('.pyplot', 'matplotlib')

###################################################################################################
###################################################################################################

def get_tfm():
    """Get a ERPparam object, with a fit power spectrum, for testing."""

    fs = 1000
    xs, ys = simulate_erp(*default_params(), fs=fs)

    tfm = ERPparam(verbose=False, max_n_peaks=4)
    tfm.fit(xs, ys)

    return tfm


def get_tfg():
    """Get a ERPparamGroup object, with some fit power spectra, for testing."""

    n_signals = 3
    xs, ys = simulate_erps(n_signals, *default_group_params())

    tfg = ERPparamGroup(verbose=False, max_n_peaks=4)
    tfg.fit(xs, ys)

    return tfg


def get_tbands():
    """Get a bands object, for testing."""

    return Bands({'p1' : (0, 0.2), 'p3' : (0.2, 0.4)})


def get_twindow():
    """Get a time window list, for testing."""

    return [0, 10]


def get_tresults():
    """Get a ERPparamResults objet, for testing."""

    return ERPparamResults(shape_params=np.array([[0.08, 0.05, 0.02, 0.6, 0.97, 0.97, 0.98, 0.1, 2, 0.06], 
                                                  [0.08, 0.05, 0.02, 0.6, 0.97, 0.97, 0.98, 0.2, -1.5, 0.1]]), 
                           # [duration, rise-time, decay-time, rise-decay symmetry, FWHM, rising sharpness, decaying sharpness, CT, PW, BW]
                           r_squared=0.97, error=0.01, adj_r_squared=0.95,
                           gaussian_params=np.array([[0.1, 2, 0.03], [0.2, -1.5, 0.05]]), # [mean, height, standard deviation]
                           peak_indices=np.array([[1,2,3],[4,5,6]])
                           )

def default_params():
    time_range = (-0.5, 2)
    erp_latency = [0.1, 0.2]
    erp_amplitude = [2, -1.5]
    erp_width = [0.03, 0.05]
    erp_params = np.ravel(np.column_stack([erp_latency, erp_amplitude, erp_width]))
    nlv = 0.1

    return time_range, erp_params, nlv


def default_group_params():
    """Create default parameters for generating a test group of power spectra."""

    time_range = [-0.5, 2]
    erp_latency = [0.1, 0.2]
    erp_amplitude = [2, -1.5]
    erp_width = [0.03, 0.05]
    erp_params = param_sampler( [np.ravel(np.column_stack([erp_latency, erp_amplitude, erp_width])),
                                np.ravel(np.column_stack([erp_latency[:2], erp_amplitude[:2], erp_width[:2]])),
                                np.ravel(np.column_stack([erp_latency[0], erp_amplitude[0], erp_width[0]]))
                                ] )
    nlvs = 0.1

    return time_range, erp_params, nlvs


def plot_test(func):
    """Decorator for simple testing of plotting functions.

    Notes
    -----
    This decorator closes all plots prior to the test.
    After running the test function, it checks an axis was created with data.
    It therefore performs a minimal test - asserting the plots exists, with no accuracy checking.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):

        plt.close('all')

        func(*args, **kwargs)

        ax = plt.gca()
        assert ax.has_data()

    return wrapper
