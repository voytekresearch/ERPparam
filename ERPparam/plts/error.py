"""Plots for visualizing model error."""

import numpy as np

from ERPparam.core.modutils import safe_import, check_dependency
from ERPparam.plts.signals import plot_signals
from ERPparam.plts.settings import PLT_FIGSIZES
from ERPparam.plts.style import style_ERP_plot, style_plot
from ERPparam.plts.utils import check_ax, savefig

plt = safe_import('.pyplot', 'matplotlib')

###################################################################################################
###################################################################################################

@savefig
@style_plot
@check_dependency(plt, 'matplotlib')
def plot_signals_error(time, error, shade=None, ax=None, **plot_kwargs):
    """Plot timepoint by timepoint error values.

    Parameters
    ----------
    time : 1d array
        Time values, to be plotted on the x-axis.
    error : 1d array
        Calculated error values or mean error values across Time, to plot on the y-axis.
    shade : 1d array, optional
        Values to shade in around the plotted error.
        This could be, for example, the standard deviation of the errors.
    ax : matplotlib.Axes, optional
        Figure axes upon which to plot.
    **plot_kwargs
        Keyword arguments to pass into the ``style_plot``.
    """

    ax = check_ax(ax, plot_kwargs.pop('figsize', PLT_FIGSIZES['signal']))

    plt_time = time

    plot_signals(plt_time, error, ax=ax, linewidth=3)

    if np.any(shade):
        ax.fill_between(plt_time, error-shade, error+shade, alpha=0.25)

    ymin, ymax = ax.get_ylim()
    if ymin < 0:
        ax.set_ylim([0, ymax])
    ax.set_xlim(plt_time.min(), plt_time.max())

    style_ERP_plot(ax)
    ax.set_ylabel('Absolute Error')
