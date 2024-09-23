"""Utilities for working with data and models."""

from itertools import repeat

import numpy as np

###################################################################################################
###################################################################################################

def trim_spectrum(time, signal, t_range):
    """Extract a time range from signals.

    Parameters
    ----------
    time : 1d array
        Time array for the ERP
    signal : 1d or 2d array
        ERP signal values.
    t_range: list of [float, float]
        Time range to restrict to, as [lowest_time, highest_time].

    Returns
    -------
    time_ext : 1d array
        Extracted time values for the ERP signal.
    signal_ext : 1d or 2d array
        Extracted signal values.

    Notes
    -----
    This function extracts time ranges >= t_low and <= t_high.
    It does not round to below or above t_low and t_high, respectively.


    Examples
    --------
    Using a simulated signal, extract a time range:

    >>> from ERPparam.sim import simulate_erps
    >>> time, signals = simulate_erps([1, 50], [1, 1], [10, 0.5, 1.0])
    >>> time, signals = trim_spectrum(time, signals, [3, 30])
    """

    # Create mask to index only requested times
    f_mask = np.logical_and(time >= t_range[0], time <= t_range[1])

    # Restrict time & signals to requested range
    #   The if/else is to cover both 1d or 2d arrays
    time_ext = time[f_mask]
    signal_ext = signal[f_mask] if signal.ndim == 1 \
        else signal[:, f_mask]

    return time_ext, signal_ext


def interpolate_spectrum(time, signals, interp_range, buffer=3):
    """Interpolate a time region in a ERP signal.

    Parameters
    ----------
    time : 1d array
        time values for the ERP signal.
    signals : 1d array
        Amplitude values for the ERP signal.
    interp_range : list of float or list of list of float
        time range to interpolate, as [lowest_time, highest_time].
        If a list of lists, applies each as it's own interpolation range.
    buffer : int or list of int
        The number of samples to use on either side of the interpolation
        range, that are then averaged and used to calculate the interpolation.

    Returns
    -------
    time : 1d array
        time values for the ERP signal.
    signals : 1d array
        Amplitude values, with interpolation, for the ERP signal.

    Notes
    -----
    This function takes in, and returns, linearly spaced values.

    The interpolation range is taken as the range from >= interp_range_low and
    <= interp_range_high. It does not round to below or above interp_range_low
    and interp_range_high, respectively.

    To be more robust to noise, this approach takes a number of samples on either
    side of the interpolation range (the number of which is controlled by `buffer`)
    and averages these points to linearly interpolate between them.
    Setting `buffer=1` is equivalent to a linear interpolation between
    the points adjacent to the interpolation range.

    Examples
    --------
    Using a simulated spectrum, interpolate away a line noise peak:

    >>> from ERPparam.sim import simulate_erps
    >>> time, signals = simulate_erps([1, 50], [1, 1], [10, 0.5, 1.0])
    >>> time, signals = interpolate_spectrum(time, signals, [3, 30])
    """

    # If given a list of interpolation zones, recurse to apply each one
    if isinstance(interp_range[0], list):
        buffer = repeat(buffer) if isinstance(buffer, int) else buffer
        for interp_zone, cur_buffer in zip(interp_range, buffer):
            time, signals = interpolate_spectrum(time, signals, interp_zone, cur_buffer)

    # Assuming list of two floats, interpolate a single time range
    else:

        # Take a copy of the array, to not change original array
        signals = np.copy(signals)

        # Get the set of time values that need to be interpolated
        interp_mask = np.logical_and(time >= interp_range[0], time <= interp_range[1])
        interp_time = time[interp_mask]

        # Get the indices of the interpolation range
        ii1, ii2 = np.flatnonzero(interp_mask)[[0, -1]]

        # Extract the requested range of data to use around interpolated range
        xs1 = (time[ii1-buffer:ii1])
        xs2 = (time[ii2:ii2+buffer])
        ys1 = (signals[ii1-buffer:ii1])
        ys2 = (signals[ii2:ii2+buffer])

        # Linearly interpolate, in log-log space, between averages of the extracted points
        vals = np.interp((interp_time),
                         [np.median(xs1), np.median(xs2)],
                         [np.median(ys1), np.median(ys2)])
        signals[interp_mask] = vals

    return time, signals


def subsample_signals(signals, selection, return_inds=False):
    """Subsample a group of power signals.

    Parameters
    ----------
    signals : 2d array
        A group of signals to subsample from.
    selection : int or float
        The number of signals to subsample.
        If int, is the number to select, if float, is a proportion based on input size.
    return_inds : bool, optional, default: False
        Whether to return the list of indices that were selected.

    Returns
    -------
    subsample : 2d array
        A subsampled selection of power signals.
    inds : list of int
        A list of which indices where subsampled.
        Only returned if `return_inds` is True.

    Examples
    --------
    Using a group of simulated signals, subsample a specific number:

    >>> from ERPparam.sim import gen_group_signal
    >>> time, signals = gen_group_signal(10, [1, 50], [1, 1], [10, 0.5, 1.0])
    >>> subsample = subsample_signals(signals, 5)

    Using a group of simulated signals, subsample a proportion:

    >>> from ERPparam.sim import gen_group_signal
    >>> time, signals = gen_group_signal(10, [1, 50], [1, 1], [10, 0.5, 1.0])
    >>> subsample = subsample_signals(signals, 0.25)
    """

    n_signals = signals.shape[0]

    if isinstance(selection, float):
        n_sample = int(n_signals * selection)
    else:
        n_sample = selection

    inds = np.random.choice(n_signals, n_sample, replace=False)
    subsample = signals[inds, :]

    if return_inds:
        return subsample, inds
    else:
        return subsample
