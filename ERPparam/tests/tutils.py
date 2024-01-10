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

    freq_range = [3, 50]
    ap_params = [50, 2]
    gaussian_params = [10, 0.5, 2, 20, 0.3, 4]

    xs, ys = simulate_erp(freq_range, ap_params, gaussian_params)

    tfm = ERPparam(verbose=False)
    tfm.fit(xs, ys)

    return tfm

def get_tfg():
    """Get a ERPparamGroup object, with some fit power spectra, for testing."""

    n_signals = 3
    xs, ys = simulate_erps(n_signals, *default_group_params())

    tfg = ERPparamGroup(verbose=False)
    tfg.fit(xs, ys)

    return tfg

def get_tbands():
    """Get a bands object, for testing."""

    return Bands({'theta' : (4, 8), 'alpha' : (8, 12), 'beta' : (13, 30)})

def get_tresults():
    """Get a ERPparamResults objet, for testing."""

    return ERPparamResults(aperiodic_params=np.array([1.0, 1.00]),
                        peak_params=np.array([[10.0, 1.25, 2.0], [20.0, 1.0, 3.0]]),
                        r_squared=0.97, error=0.01,
                        gaussian_params=np.array([[10.0, 1.25, 1.0], [20.0, 1.0, 1.5]]))

def default_group_params():
    """Create default parameters for generating a test group of power spectra."""

    time_range = [-0.5, 2]
    erp_latency = [0.1, 0.2, 0.5]
    erp_amplitude = [2, -1.5, 0.75]
    erp_width = [0.03, 0.05, 0.1]
    erp_params = param_sampler( [np.ravel(np.column_stack([erp_latency, erp_amplitude, erp_width])),
                                 np.ravel(np.column_stack([erp_latency[:2], erp_amplitude[:2], erp_width[:2]])),
                                 np.ravel(np.column_stack([erp_latency[0], erp_amplitude[0], erp_width[0]]))
                                 ] )
                        #param_sampler([[erp_latency[0], erp_amplitude[0], erp_width[0]],
                        #        [erp_latency[1], erp_amplitude[1], erp_width[1]],
                         #       [erp_latency[2], erp_amplitude[2], erp_width[2]]])#

    return time_range, erp_params

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
