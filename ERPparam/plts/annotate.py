"""Plots for annotating ERP fittings and models."""

import numpy as np

from ERPparam.core.utils import nearest_ind
from ERPparam.core.errors import NoModelError
from ERPparam.core.funcs import gaussian_function
from ERPparam.core.modutils import safe_import, check_dependency

#from ERPparam.sim.gen import gen_aperiodic
from ERPparam.analysis.periodic import get_band_peak_fm
from ERPparam.utils.params import compute_fwhm

from ERPparam.plts.signals import plot_signals
from ERPparam.plts.utils import check_ax, savefig
from ERPparam.plts.settings import PLT_FIGSIZES, PLT_COLORS
from ERPparam.plts.style import style_ERP_plot

plt = safe_import('.pyplot', 'matplotlib')
mpatches = safe_import('.patches', 'matplotlib')

###################################################################################################
###################################################################################################

@savefig
@check_dependency(plt, 'matplotlib')
def plot_annotated_peak_search(fm):
    """Plot a series of plots illustrating the peak search from a flattened ERP.

    Parameters
    ----------
    fm : ERPparam
        ERPparam object, with model fit, data and settings available.
    """

    # Get signal with all peaks
    flatspec = fm.signal.copy()

    # Calculate ylims of the plot that are scaled to the range of the data
    ylims = [min(flatspec) - 0.1 * np.abs(min(flatspec)), max(flatspec) + 0.1 * max(flatspec)]

    # Sort parameters by peak height
    gaussian_params = fm.gaussian_params_[fm.gaussian_params_[:, 1].argsort()][::-1]

    # Loop through the iterative search for each peak
    for ind in range(fm.n_peaks_ + 1):

        # This forces the creation of a new plotting axes per iteration
        ax = check_ax(None, PLT_FIGSIZES['signal'])

        plot_signals(fm.time, flatspec, ax=ax, linewidth=2.5,
                     label='Flattened Signal', color=PLT_COLORS['data'])
        plot_signals(fm.time, [fm.peak_threshold * np.std(fm.signal)]*len(fm.time), ax=ax,
                     label='Relative Threshold', color='orange', linewidth=2.5, linestyle='dashed')
        plot_signals(fm.time, -1*(np.asarray([fm.peak_threshold * np.std(fm.signal)]*len(fm.time))), ax=ax,
                      color='orange', linewidth=2.5, linestyle='dashed')
        plot_signals(fm.time, [fm.min_peak_height]*len(fm.time), ax=ax,
                     label='Absolute Threshold', color='red', linewidth=2.5, linestyle='dashed')
        plot_signals(fm.time, -1*(np.asarray([fm.min_peak_height]*len(fm.time))), ax=ax,
                      color='red', linewidth=2.5, linestyle='dashed')

        maxi = np.argmax(flatspec)
        mini = np.argmin(flatspec)
        ax.plot(fm.time[maxi], flatspec[maxi], '.',
                color=PLT_COLORS['periodic'], alpha=0.75, markersize=30)
        ax.plot(fm.time[mini], flatspec[mini], '.',
                color=PLT_COLORS['periodic'], alpha=0.75, markersize=30)


        ax.set_ylim(ylims)
        ax.set_title('Iteration #' + str(ind+1), fontsize=16)

        if ind < fm.n_peaks_:

            gauss = gaussian_function(fm.time, *gaussian_params[ind, :])
            plot_signals(fm.time, gauss, ax=ax, label='Gaussian Fit',
                         color=PLT_COLORS['periodic'], linestyle=':', linewidth=3.0)

            flatspec = flatspec - gauss

        style_ERP_plot(ax)


@savefig
@check_dependency(plt, 'matplotlib')
def plot_annotated_model(fm, annotate_peaks=True, ax=None):
    """Plot an annotated ERP and model, from a ERPparam object.

    Parameters
    ----------
    fm : ERPparam
        ERPparam object, with model fit, data and settings available.
    annotate_peaks : boolean, optional, default: True
        Whether to annotate the periodic components of the model fit.
    ax : matplotlib.Axes, optional
        Figure axes upon which to plot.

    Raises
    ------
    NoModelError
        If there are no model results available to plot.
    """

    # Check that model is available
    if not fm.has_model:
        raise NoModelError("No model is available to plot, can not proceed.")

    # Settings
    fontsize = 15
    lw1 = 4.0
    lw2 = 3.0
    ms1 = 12

    # Create the baseline figure
    ax = check_ax(ax, PLT_FIGSIZES['signal'])
    fm.plot(ax) 
    #         plot_peaks='dot-shade-width', ax=ax,
    #         data_kwargs={'lw' : lw1, 'alpha' : 0.6},
    #         aperiodic_kwargs={'lw' : lw1, 'zorder' : 10},
    #         model_kwargs={'lw' : lw1, 'alpha' : 0.5},
    #         peak_kwargs={'dot' : {'color' : PLT_COLORS['periodic'], 'ms' : ms1, 'lw' : lw2},
    #                      'shade' : {'color' : PLT_COLORS['periodic']},
    #                      'width' : {'color' : PLT_COLORS['periodic'], 'alpha' : 0.75, 'lw' : lw2}}

    # Get time for plotting, and convert to log if needed
    time = fm.time 

    ## Buffers: for spacing things out on the plot (scaled by plot values)
    x_buff1 = max(time) * 0.1
    x_buff2 = max(time) * 0.25
    y_buff1 = 0.15 * np.ptp(ax.get_ylim())
    shrink = 0.1

    # There is a bug in annotations for some perpendicular lines, so add small offset
    #   See: https://github.com/matplotlib/matplotlib/issues/12820. Fixed in 3.2.1.
    bug_buff = 0.000001

    if annotate_peaks and fm.n_peaks_:

        # Extract largest peak, to annotate, grabbing gaussian params
        gauss = get_band_peak_fm(fm, fm.time_range, attribute='gaussian_params')

        peak_ctr, peak_hgt, peak_wid = gauss
        bw_time = [peak_ctr - 0.5 * compute_fwhm(peak_wid),
                    peak_ctr + 0.5 * compute_fwhm(peak_wid)]

        peak_top = fm.signal[nearest_ind(time, peak_ctr)]

        # Annotate Peak CF
        ax.annotate('Center  Time',
                    xy=(peak_ctr, peak_top),
                    xytext=(peak_ctr, peak_top+np.abs(0.6*peak_hgt)),
                    verticalalignment='center',
                    horizontalalignment='center',
                    arrowprops=dict(facecolor=PLT_COLORS['periodic'], shrink=shrink),
                    color=PLT_COLORS['periodic'], fontsize=fontsize)

        # Annotate Peak PW
        ax.annotate('Height',
                    xy=(peak_ctr, peak_top),
                    xytext=(peak_ctr+x_buff1, peak_top),
                    verticalalignment='center',
                    arrowprops=dict(facecolor=PLT_COLORS['periodic'], shrink=shrink),
                    color=PLT_COLORS['periodic'], fontsize=fontsize)

        # Annotate Peak BW
        bw_buff = (peak_ctr - bw_time[0])/2
        ax.annotate('Bandwidth',
                    xy=(peak_ctr-bw_buff+bug_buff, peak_top-(0.5*peak_hgt)),
                    xytext=(peak_ctr-bw_buff, peak_top-(1.5*peak_hgt)),
                    verticalalignment='center',
                    horizontalalignment='right',
                    arrowprops=dict(facecolor=PLT_COLORS['periodic'], shrink=shrink),
                    color=PLT_COLORS['periodic'], fontsize=fontsize, zorder=20)

    
    # Apply style to plot & tune grid styling
    style_ERP_plot(ax)
    ax.grid(True, alpha=0.5)

    # Add labels to plot in the legend
    da_patch = mpatches.Patch(color=PLT_COLORS['data'], label='Original Data')
    pe_patch = mpatches.Patch(color=PLT_COLORS['periodic'], label='Peak Parameters')
    mo_patch = mpatches.Patch(color=PLT_COLORS['model'], label='Full Model')

    handles = [da_patch, 
               pe_patch if annotate_peaks else None, mo_patch]
    handles = [el for el in handles if el is not None]

    ax.legend(handles=handles, handlelength=1, fontsize='x-large')
