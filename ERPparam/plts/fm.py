"""Plots for the ERPparam object.

Notes
-----
This file contains plotting functions that take as input a ERPparam object.
"""

import numpy as np

from ERPparam.core.utils import nearest_ind
from ERPparam.core.modutils import safe_import, check_dependency
from ERPparam.sim.gen import sim_erp
from ERPparam.utils.data import trim_signal
from ERPparam.utils.params import compute_fwhm
from ERPparam.plts.signals import plot_signals
from ERPparam.plts.settings import PLT_FIGSIZES, PLT_COLORS
from ERPparam.plts.utils import check_ax, check_plot_kwargs, savefig
from ERPparam.plts.style import style_ERP_plot, style_plot

plt = safe_import('.pyplot', 'matplotlib')

###################################################################################################
###################################################################################################

@savefig
@style_plot
@check_dependency(plt, 'matplotlib')
def plot_fm(fm, plot_peaks=None, add_legend=True, ax=None, data_kwargs=None, 
            model_kwargs=None, peak_kwargs=None, **plot_kwargs):
    """Plot the signals and model fit results from a ERPparam object.

    Parameters
    ----------
    fm : ERPparam
        Object containing ERP signals and (optionally) results from fitting.
    plot_peaks : None or {'shade', 'dot', 'outline', 'line'}, optional
        What kind of approach to take to plot peaks. If None, peaks are not specifically plotted.
        Can also be a combination of approaches, separated by '-', for example: 'shade-line'.
    add_legend : boolean, optional, default: False
        Whether to add a legend describing the plot components.
    ax : matplotlib.Axes, optional
        Figure axes upon which to plot.
    data_kwargs, model_kwargs, peak_kwargs : None or dict, optional
        Keyword arguments to pass into the plot call for each plot element.
    **plot_kwargs
        Keyword arguments to pass into the ``style_plot``.

    """

    ax = check_ax(ax, plot_kwargs.pop('figsize', PLT_FIGSIZES['signal']))

    # Plot the data, if available
    if fm.has_data:
        data_defaults = {'color' : PLT_COLORS['data'], 'linewidth' : 2.0,
                         'label' : 'Original Signal' if add_legend else None}
        data_kwargs = check_plot_kwargs(data_kwargs, data_defaults)
        plot_signals(fm.time, fm.signal, ax=ax, **data_kwargs)

    # Add the full model fit, and components (if requested)
    if fm.has_model:
        model_defaults = {'color' : PLT_COLORS['model'], 'linewidth' : 3.0, 'alpha' : 0.5,
                          'label' : 'Full Model Fit' if add_legend else None}
        model_kwargs = check_plot_kwargs(model_kwargs, model_defaults)
        plot_signals(fm.time, fm._peak_fit, ax=ax, **model_kwargs)

        # Plot the periodic components of the model fit
        if plot_peaks:
            _add_peaks(fm, 'dot', ax, peak_kwargs)

    # Apply default style to plot
    style_ERP_plot(ax)


def _add_peaks(fm, approach, ax, peak_kwargs):
    """Add peaks to a model plot.

    Parameters
    ----------
    fm : ERPparam
        ERPparam object containing results from fitting.
    approach : {'shade', 'dot', 'outline', 'outline', 'line'}
        What kind of approach to take to plot peaks.
        Can also be a combination of approaches, separated by '-' (for example 'shade-line').
    ax : matplotlib.Axes
        Figure axes upon which to plot.
    peak_kwargs : None or dict
        Keyword arguments to pass into the plot call.
        This can be a flat dictionary, with plot keyword arguments,
        or a dictionary of dictionaries, with keys as labels indicating an `approach`,
        and values which contain a dictionary of plot keywords for that approach.

    Notes
    -----
    This is a pass through function, that takes a specification of one
    or multiple add peak approaches to use, and calls the relevant function(s).
    """

    # Input for kwargs could be None, so check if dict and typecast if not
    peak_kwargs = peak_kwargs if isinstance(peak_kwargs, dict) else {}

    # Split up approaches, in case multiple are specified, and apply each
    for cur_approach in approach.split('-'):

        try:

            # This unpacks kwargs, if it's embedded dictionaries for each approach
            plot_kwargs = peak_kwargs.get(cur_approach, peak_kwargs)

            # Pass through to the peak plotting function
            ADD_PEAK_FUNCS[cur_approach](fm, ax, **plot_kwargs)

        except KeyError:
            raise ValueError("Plot peak type not understood.")


def _add_peaks_shade(fm, ax, **plot_kwargs):
    """Add a shading in of all peaks.

    Parameters
    ----------
    fm : ERPparam
        ERPparam object containing results from fitting.
    ax : matplotlib.Axes
        Figure axes upon which to plot.
    **plot_kwargs
        Keyword arguments to pass into the ``fill_between``.
    """

    defaults = {'color' : PLT_COLORS['periodic'], 'alpha' : 0.25}
    plot_kwargs = check_plot_kwargs(plot_kwargs, defaults)

    for peak in fm.gaussian_params_:

        peak_line = sim_erp(fm.time, peak)

        ax.fill_between(fm.time, peak_line, fm.signal, **plot_kwargs)


def _add_peaks_dot(fm,  ax, **plot_kwargs):
    """Add a short line, from aperiodic to peak, with a dot at the top.

    Parameters
    ----------
    fm : ERPparam
        ERPparam object containing results from fitting.
    ax : matplotlib.Axes
        Figure axes upon which to plot.
    **plot_kwargs
        Keyword arguments to pass into the plot call.
    """

    defaults = {'color' : PLT_COLORS['periodic'], 'alpha' : 0.6, 'lw' : 2.5, 'ms' : 6}
    plot_kwargs = check_plot_kwargs(plot_kwargs, defaults)

    for peak in fm.shape_params_:

        ap_point = np.interp(peak[7], fm.time, fm.signal)
        freq_point = peak[7]

        # Add the line from the aperiodic fit up the tip of the peak
        ax.plot([freq_point, freq_point], [ap_point, ap_point + peak[1]], **plot_kwargs)

        # Add an extra dot at the tip of the peak
        ax.plot(freq_point, ap_point + peak[1], marker='o', **plot_kwargs)


def _add_peaks_outline(fm, ax, **plot_kwargs):
    """Add an outline of each peak.

    Parameters
    ----------
    fm : ERPparam
        ERPparam object containing results from fitting.
    ax : matplotlib.Axes
        Figure axes upon which to plot.
    **plot_kwargs
        Keyword arguments to pass into the plot call.
    """

    defaults = {'color' : PLT_COLORS['periodic'], 'alpha' : 0.7, 'lw' : 1.5}
    plot_kwargs = check_plot_kwargs(plot_kwargs, defaults)

    for peak in fm.gaussian_params_:

        # Define the frequency range around each peak to plot - peak bandwidth +/- 3
        peak_range = [peak[0] - peak[2]*3, peak[0] + peak[2]*3]

        # Generate a peak reconstruction for each peak, and trim to desired range
        peak_line = sim_erp(fm.time, peak)
        peak_time, peak_line = trim_signal(fm.time, peak_line, peak_range)

        # Plot the peak outline
        ax.plot(peak_time, peak_line, **plot_kwargs)


def _add_peaks_line(fm, ax, **plot_kwargs):
    """Add a long line, from the top of the plot, down through the peak, with an arrow at the top.

    Parameters
    ----------
    fm : ERPparam
        ERPparam object containing results from fitting.
    ax : matplotlib.Axes
        Figure axes upon which to plot.
    **plot_kwargs
        Keyword arguments to pass into the plot call.
    """

    defaults = {'color' : PLT_COLORS['periodic'], 'alpha' : 0.7, 'lw' : 1.4, 'ms' : 10}
    plot_kwargs = check_plot_kwargs(plot_kwargs, defaults)

    ylims = ax.get_ylim()

    for peak in fm.shape_params_:

        freq_point = peak[7]
        ax.plot([freq_point, freq_point], ylims, '-', **plot_kwargs)
        ax.plot(freq_point, ylims[1], 'v', **plot_kwargs)


def _add_peaks_width(fm, ax, **plot_kwargs):
    """Add a line across the width of peaks.

    Parameters
    ----------
    fm : ERPparam
        ERPparam object containing results from fitting.
    ax : matplotlib.Axes
        Figure axes upon which to plot.
    **plot_kwargs
        Keyword arguments to pass into the plot call.

    Notes
    -----
    This line represents the bandwidth (width or gaussian standard deviation) of
    the peak, though what is literally plotted is the full-width half-max.
    """

    defaults = {'color' : PLT_COLORS['periodic'], 'alpha' : 0.6, 'lw' : 2.5, 'ms' : 6}
    plot_kwargs = check_plot_kwargs(plot_kwargs, defaults)

    for peak in fm.gaussian_params_:

        peak_top = fm.power_spectrum[nearest_ind(fm.time, peak[0])]
        bw_time = [peak[0] - 0.5 * compute_fwhm(peak[2]),
                    peak[0] + 0.5 * compute_fwhm(peak[2])]

        ax.plot(bw_time, [peak_top-(0.5*peak[1]), peak_top-(0.5*peak[1])], **plot_kwargs)


# Collect all the possible `add_peak_*` functions together
ADD_PEAK_FUNCS = {
    'shade' : _add_peaks_shade,
    'dot' : _add_peaks_dot,
    'outline' : _add_peaks_outline,
    'line' : _add_peaks_line,
    'width' : _add_peaks_width
}
