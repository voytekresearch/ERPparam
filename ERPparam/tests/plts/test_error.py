"""Tests for ERPparam.plts.error."""

import numpy as np

from ERPparam.tests.tutils import plot_test
from ERPparam.tests.settings import TEST_PLOTS_PATH

from ERPparam.plts.error import *

###################################################################################################
###################################################################################################

@plot_test
def test_plot_signals_error(skip_if_no_mpl):

    fs = np.arange(3, 41, 1)
    errs = np.ones(len(fs))

    plot_signals_error(fs, errs, save_fig=True, file_path=TEST_PLOTS_PATH,
                        file_name='test_plot_signal_error.png')
