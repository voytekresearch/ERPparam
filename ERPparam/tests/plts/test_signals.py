"""Tests for ERPparam.plts.signals."""

from pytest import raises

import numpy as np

from ERPparam.tests.tutils import plot_test
from ERPparam.tests.settings import TEST_PLOTS_PATH

from ERPparam.plts.signals import *

###################################################################################################
###################################################################################################

@plot_test
def test_plot_signals(tfm, tfg, skip_if_no_mpl):

    # Test with 1d inputs - 1d freq array and list of 1d signal
    plot_signals(tfm.time, tfm.signal,
                 save_fig=True, file_path=TEST_PLOTS_PATH, 
                 file_name='test_plot_signals_1d.png')

    # Test with 1d inputs - 1d freq array and list of 1d signal
    plot_signals(tfg.time, [tfg.signals[0, :], tfg.signals[1, :]],
                 save_fig=True, file_path=TEST_PLOTS_PATH, 
                 file_name='test_plot_signals_list_1d.png')

    # Test with multiple freq inputs - list of 1d freq array and list of 1d signal
    plot_signals([tfg.time, tfg.time], [tfg.signals[0, :], tfg.signals[1, :]],
                 save_fig=True, file_path=TEST_PLOTS_PATH,
                 file_name='test_plot_signals_lists_1d.png')

    # Test with 2d array inputs
    plot_signals(np.vstack([tfg.time, tfg.time]),
                 np.vstack([tfg.signals[0, :], tfg.signals[1, :]]),
                 save_fig=True, file_path=TEST_PLOTS_PATH, 
                 file_name='test_plot_signals_2d.png')

    # Test with labels
    plot_signals(tfg.time, [tfg.signals[0, :], tfg.signals[1, :]], 
                 labels=['A', 'B'], save_fig=True, file_path=TEST_PLOTS_PATH, 
                 file_name='test_plot_signals_labels.png')

@plot_test
def test_plot_signals_shading(tfm, tfg, skip_if_no_mpl):

    plot_signals_shading(tfm.time, tfm.signal, shades=[8, 12], 
                         add_center=True, save_fig=True, 
                         file_path=TEST_PLOTS_PATH,
                         file_name='test_plot_signal_shading1.png')

    plot_signals_shading(tfg.time, [tfg.signals[0, :], tfg.signals[1, :]],
                         shades=[8, 12], add_center=True, save_fig=True, 
                         file_path=TEST_PLOTS_PATH,
                         file_name='test_plot_signals_shading2.png')

    # Test with **kwargs that pass into plot_signals
    plot_signals_shading(tfg.time, [tfg.signals[0, :], tfg.signals[1, :]],
                         shades=[8, 12], add_center=True,
                         labels=['A', 'B'], save_fig=True, 
                         file_path=TEST_PLOTS_PATH,
                         file_name='test_plot_signals_shading_kwargs.png')

@plot_test
def test_plot_signals_yshade(skip_if_no_mpl, tfg):

    time = tfg.time
    signals = tfg.signals

    # Invalid 1d array, without shade
    with raises(ValueError):
        plot_signals_yshade(time, signals[0])

    # Plot with 2d array
    plot_signals_yshade(time, signals, shade='std',
                        save_fig=True, file_path=TEST_PLOTS_PATH,
                        file_name='test_plot_signals_yshade1.png')

    # Plot shade with given 1d array
    plot_signals_yshade(time, np.mean(signals, axis=0),
                        shade=np.std(signals, axis=0),
                        save_fig=True, file_path=TEST_PLOTS_PATH,
                        file_name='test_plot_signals_yshade2.png')

    # Plot shade with different average and shade approaches
    plot_signals_yshade(time, signals, shade='sem', average='median',
                        save_fig=True, file_path=TEST_PLOTS_PATH,
                        file_name='test_plot_signals_yshade3.png')

    # Plot shade with custom average and shade callables
    def _average_callable(signals): return np.mean(signals, axis=0)
    def _shade_callable(signals): return np.std(signals, axis=0)

    plot_signals_yshade(time, signals, shade=_shade_callable,  
                        average=_average_callable, save_fig=True, 
                        file_path=TEST_PLOTS_PATH,
                        file_name='test_plot_signals_yshade4.png')
