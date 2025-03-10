"""Plots for the ERPparam object.

Notes
-----
This file contains plotting functions that take as input a ERPparam object.
"""

import numpy as np

from ERPparam.core.modutils import safe_import, check_dependency
from ERPparam.plts.settings import PLT_FIGSIZES, PLT_COLORS
from ERPparam.plts.style import style_ERP_plot

plt = safe_import('.pyplot', 'matplotlib')

###################################################################################################
###################################################################################################

@check_dependency(plt, 'matplotlib')
def plot_ERPparam(model, ax=None, y_label=None):
    """Plot ERP and model fit results."""

    # create figure
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=PLT_FIGSIZES['signal'])

    # plot signal
    ax.plot(model.time, model.signal, alpha=0.75, label='ERP', 
            color=PLT_COLORS['data'], linewidth=3)
    
    # plot full model fit
    ax.plot(model.time, model._full_fit, linestyle='--', 
            color=PLT_COLORS['model'], label='Model fit')

    # plot peak and half-mag points    
    if model._peak_fit is not None:
        ax.scatter(model.time[model.peak_indices_[:,1]], model.signal[model.peak_indices_[:,1]], color='r', label='Peak fit')
        half_mag_indices = np.concatenate((model.peak_indices_[:,0], model.peak_indices_[:,2]))
        ax.scatter(model.time[half_mag_indices], model.signal[half_mag_indices], color='b', label='Half-mag fit')

    # plot Gaussian and sigmoid fit, if available
    if model.fit_offset:
        # plot gaussian fit
        ax.plot(model.time, model._peak_fit, linestyle='--', 
                color=PLT_COLORS['peak'], label='Gaussian fit')
        
        # plot sigmoid fit
        ax.plot(model.time, model._sigmoid_fit, linestyle='--', 
                color=PLT_COLORS['offset'], label='Offset fit')
        
    # label
    if y_label is not None:
        ax.set(xlabel="Time (s)", ylabel=y_label)
    else:
        ax.set(xlabel="Time (s)", ylabel="Amplitude")
    ax.legend()

    # style
    style_ERP_plot(ax)
