"""Plots for periodic fits and parameters."""

from itertools import cycle

import numpy as np

from ERPparam.sim.gen import gen_time_vector
from ERPparam.core.funcs import gaussian_function
from ERPparam.core.modutils import safe_import, check_dependency
from ERPparam.plts.settings import PLT_FIGSIZES
from ERPparam.plts.style import style_param_plot, style_plot
from ERPparam.plts.utils import check_ax, recursive_plot, savefig, check_plot_kwargs

plt = safe_import('.pyplot', 'matplotlib')

###################################################################################################
###################################################################################################

@savefig
@style_plot
@check_dependency(plt, 'matplotlib')
def plot_peak_params(peaks, time_range=None, colors=None, labels=None, ax=None, **plot_kwargs):
    """Plot peak parameters as dots representing center time, power and bandwidth.

    Parameters
    ----------
    peaks : 2d array or list of 2d array
        Peak data. Each row is a peak, as [CF, PW, BW].
    time_range : list of [float, float] , optional
        The time range to plot the peak parameters across, as [f_min, f_max].
    colors : str or list of str, optional
        Color(s) to plot data.
    labels : list of str, optional
        Label(s) for plotted data, to be added in a legend.
    ax : matplotlib.Axes, optional
        Figure axes upon which to plot.
    **plot_kwargs
        Keyword arguments to pass into the ``style_plot``.
    """

    ax = check_ax(ax, plot_kwargs.pop('figsize', PLT_FIGSIZES['params']))

    # If there is a list, use recurse function to loop across arrays of data and plot them
    if isinstance(peaks, list):
        recursive_plot(peaks, plot_peak_params, ax, colors=colors, labels=labels)

    # Otherwise, plot the array of data
    else:

        # Unpack data: CF as x; PW as y; BW as size
        xs, ys = peaks[:, 0], peaks[:, 1]
        sizes = peaks[:, 2] * plot_kwargs.pop('s', 150)

        # Create the plot
        plot_kwargs = check_plot_kwargs(plot_kwargs, {'alpha' : 0.7})
        ax.scatter(xs, ys, sizes, c=colors, label=labels, **plot_kwargs)

    # Add axis labels
    ax.set_xlabel('Center Time')
    ax.set_ylabel('Amplitude')

    # Set plot limits
    if time_range:
        ax.set_xlim(time_range)
    ax.set_ylim([0, ax.get_ylim()[1]])

    style_param_plot(ax)


@savefig
@style_plot
def plot_peak_fits(peaks, time_range=None, colors=None, labels=None, ax=None, **plot_kwargs):
    """Plot reconstructions of model peak fits.

    Parameters
    ----------
    peaks : 2d array
        Peak data. Each row is a peak, as [CF, PW, BW].
    time_range : list of [float, float] , optional
        The time range to plot the peak fits across, as [f_min, f_max].
        If not provided, defaults to +/- 4 around given peak center timeuencies.
    colors : str or list of str, optional
        Color(s) to plot data.
    labels : list of str, optional
        Label(s) for plotted data, to be added in a legend.
    ax : matplotlib.Axes, optional
        Figure axes upon which to plot.
    **plot_kwargs
        Keyword arguments to pass into the plot call.
    """

    ax = check_ax(ax, plot_kwargs.pop('figsize', PLT_FIGSIZES['params']))

    if isinstance(peaks, list):

        if not colors:
            colors = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

        recursive_plot(peaks, plot_function=plot_peak_fits, ax=ax,
                       time_range=tuple(time_range) if time_range else time_range,
                       colors=colors, labels=labels, **plot_kwargs)

    else:

        if not time_range:

            # Extract all the CF values, excluding any NaNs
            cfs = peaks[~np.isnan(peaks[:, 0]), 0]

            # Define the time range as +/- buffer around the data range
            #   This also doesn't let the plot range drop below 0
            f_buffer = 4
            time_range = [cfs.min() - f_buffer if cfs.min() - f_buffer > 0 else 0,
                          cfs.max() + f_buffer]

        # Create the time axis, which will be the plot x-axis
        times = gen_time_vector(time_range, 0.1)

        colors = colors[0] if isinstance(colors, list) else colors

        avg_vals = np.zeros(shape=[len(times)])

        for peak_params in peaks:

            # Create & plot the peak model from parameters
            peak_vals = gaussian_function(times, *peak_params)
            ax.plot(times, peak_vals, color=colors, alpha=0.35, linewidth=1.25)

            # Collect a running average average peaks
            avg_vals = np.nansum(np.vstack([avg_vals, peak_vals]), axis=0)

        # Plot the average across all components
        avg = avg_vals / peaks.shape[0]
        avg_color = 'black' if not colors else colors
        ax.plot(times, avg, color=avg_color, linewidth=3.75, label=labels)

    # Add axis labels
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude')

    # Set plot limits
    ax.set_xlim(time_range)
    ax.set_ylim([0, ax.get_ylim()[1]])

    # Apply plot style
    style_param_plot(ax)
