"""signal ERP plotting functions.

Notes
-----
This file contains functions for plotting signals, that take in data directly.
"""

from inspect import isfunction
from itertools import repeat, cycle

import numpy as np
from scipy.stats import sem

from ERPparam.core.modutils import safe_import, check_dependency
from ERPparam.plts.settings import PLT_FIGSIZES
from ERPparam.plts.style import style_ERP_plot, style_plot
from ERPparam.plts.utils import check_ax, add_shades, savefig, check_plot_kwargs

plt = safe_import('.pyplot', 'matplotlib')

###################################################################################################
###################################################################################################

@savefig
@style_plot
@check_dependency(plt, 'matplotlib')
def plot_signals(time, signals, colors=None, labels=None, ax=None, **plot_kwargs):
    """Plot one or multiple ERP signals.

    Parameters
    ----------
    time : 1d or 2d array or list of 1d array
        Time values, to be plotted on the x-axis.
    signals : 1d or 2d array or list of 1d array
        ERP signal values, to be plotted on the y-axis.
    colors : list of str, optional, default: None
        Line colors of the spectra.
    labels : list of str, optional, default: None
        Legend labels for the spectra.
    ax : matplotlib.Axes, optional
        Figure axes upon which to plot.
    **plot_kwargs
        Keyword arguments to pass into the ``style_plot``.
    """

    ax = check_ax(ax, plot_kwargs.pop('figsize', PLT_FIGSIZES['signal']))

    # Create the plot
    plot_kwargs = check_plot_kwargs(plot_kwargs, {'linewidth' : 2.0})

    # Make inputs iterable if need to be passed multiple times to plot each ERP
    plt_signals = np.reshape(signals, (1, -1)) if np.ndim(signals) == 1 else \
        signals
    plt_time = repeat(time) if isinstance(time, np.ndarray) and time.ndim == 1 else time

    # Set labels
    labels = plot_kwargs.pop('label') if 'label' in plot_kwargs.keys() and labels is None else labels
    labels = repeat(labels) if not isinstance(labels, list) else cycle(labels)
    colors = repeat(colors) if not isinstance(colors, list) else cycle(colors)

    # Plot
    for time, signals, color, label in zip(plt_time, plt_signals, colors, labels):

        # Set plot data, and collect color, if absent
        if color:
            plot_kwargs['color'] = color

        ax.plot(time, signals, label=label, **plot_kwargs)

    style_ERP_plot(ax)


@savefig
@check_dependency(plt, 'matplotlib')
def plot_signals_shading(time, signals, shades, shade_colors='r',
                         add_center=False, ax=None, **plot_kwargs):
    """Plot one or multiple singals with a shaded time region (or regions).

    Parameters
    ----------
    time : 1d or 2d array or list of 1d array
        Time values, to be plotted on the x-axis.
    signals : 1d or 2d array or list of 1d array
        ERP signal values, to be plotted on the y-axis.
    shades : list of [float, float] or list of list of [float, float]
        Shaded region(s) to add to plot, defined as [lower_bound, upper_bound].
    shade_colors : str or list of string
        Color(s) to plot shades.
    add_center : bool, optional, default: False
        Whether to add a line at the center point of the shaded regions.
    ax : matplotlib.Axes, optional
        Figure axes upon which to plot.
    **plot_kwargs
        Keyword arguments to pass into :func:`~.plot_signals`.

    Notes
    -----
    Parameters for `plot_signals` can also be passed into this function as keyword arguments.

    This includes `labels`. See `plot_signals` for usage details.
    """

    ax = check_ax(ax, plot_kwargs.pop('figsize', PLT_FIGSIZES['signal']))

    plot_signals(time, signals, ax=ax, **plot_kwargs)

    add_shades(ax, shades, shade_colors, add_center)

    style_ERP_plot(ax)


@savefig
@style_plot
@check_dependency(plt, 'matplotlib')
def plot_signals_yshade(time, signals, shade='std', average='mean', scale=1,
                        color=None, label=None,
                        ax=None, **plot_kwargs):
    """Plot standard deviation or error as a shaded region around the mean ERP.

    Parameters
    ----------
    time : 1d array
        Time values, to be plotted on the x-axis.
    signals : 1d or 2d array
        signal values, to be plotted on the y-axis. ``shade`` must be provided if 1d.
    shade : 'std', 'sem', 1d array or callable, optional, default: 'std'
        Approach for shading above/below the mean ERP.
    average : 'mean', 'median' or callable, optional, default: 'mean'
        Averaging approach for the average ERP to plot. Only used if signals is 2d.
    scale : int, optional, default: 1
        Factor to multiply the plotted shade by.
    color : str, optional, default: None
        Line color of the ERP.
    label : str, optional, default: None
        Legend label for the ERP.
    ax : matplotlib.Axes, optional
        Figure axes upon which to plot.
    **plot_kwargs
        Keyword arguments to be passed to `plot_signals` or to the plot call.
    """

    if (isinstance(shade, str) or isfunction(shade)) and signals.ndim != 2:
        raise ValueError('ERPs must be 2d if shade is not given.')

    ax = check_ax(ax, plot_kwargs.pop('figsize', PLT_FIGSIZES['signal']))

    # Set plot data & labels
    plt_time = time
    plt_signals = signals

    # Organize mean ERP to plot
    avg_funcs = {'mean' : np.mean, 'median' : np.median}

    if isinstance(average, str) and plt_signals.ndim == 2:
        avg_signals = avg_funcs[average](plt_signals, axis=0)
    elif isfunction(average) and plt_signals.ndim == 2:
        avg_signals = average(plt_signals)
    else:
        avg_signals = plt_signals

    # Plot average signal ERP
    ax.plot(plt_time, avg_signals, linewidth=2.0, color=color, label=label)

    # Organize shading to plot
    shade_funcs = {'std' : np.std, 'sem' : sem}

    if isinstance(shade, str):
        shade_vals = scale * shade_funcs[shade](plt_signals, axis=0)
    elif isfunction(shade):
        shade_vals = scale * shade(plt_signals)
    else:
        shade_vals = scale * shade

    upper_shade = avg_signals + shade_vals
    lower_shade = avg_signals - shade_vals

    # Plot +/- yshading around ERP
    alpha = plot_kwargs.pop('alpha', 0.25)
    ax.fill_between(plt_time, lower_shade, upper_shade,
                    alpha=alpha, color=color, **plot_kwargs)

    style_ERP_plot(ax)
