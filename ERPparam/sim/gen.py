"""Functions for generating model components and simulated ERPs."""

import numpy as np
from neurodsp.sim import sim_powerlaw

from ERPparam.core.utils import check_iter
from ERPparam.core.funcs import  get_pe_func

from ERPparam.sim.params import collect_sim_params

###################################################################################################
###################################################################################################

def gen_time_vector(time_range, fs):
    """Generate a frequency vector.

    Parameters
    ----------
    time_range : list of [float, float]
        Time range to create time vector across, as [start_time, end_time], inclusive.
    fs : float
        Sampling frequency for desired time vector.

    Returns
    -------
    time : 1d array
        Time values.

    Examples
    --------
    Generate a vector of time values from -1 to 1:

    >>> time = gen_time_vector([-1, 1], fs=2000)
    """

    # The end value has something added to it, to make sure the last value is included
    #   It adds a fraction to not accidentally include points beyond range
    #   due to rounding / or uneven division of the fs into range to simulate
    time = np.arange(time_range[0], time_range[1] + (0.5 / fs), (1/fs))

    return time


def simulate_erp(time_range,  params, nlv=0.005, fs=1000, 
                  peak_mode='gaussian', return_params=False):
    """Generate a simulated ERP.

    Parameters
    ----------
    time_range : list of [float, float]
        Time range to create time vector across, as [start_time, end_time], inclusive.
    params : list of float or generator
        Parameters for the ERP components.
        Length of n_components * 3.
    nlvs : float or list of float or generator, optional, default: 0.005
        Noise level to add to generated ERP.
    fs : float
        Sampling frequency for desired time vector.
    peak_mode : {'gaussian'}, optional
        Which kind of component to generate.
    return_params : bool, optional, default: False
        Whether to return the parameters for the simulated ERP.

    Returns
    -------
    time : 1d array
        Time vector.
    signals : 2d array
        Matrix of signals [n_signals, n_times-1].
    sim_params : list of SimParams
        Definitions of parameters used for each signal. Has length of n_signals.
        Only returned if `return_params` is True.
    """

    time = gen_time_vector(time_range, fs)[:-1]
    signal, time = gen_signal(time, time_range, params, fs, nlv,
                                peak_mode=peak_mode)

    if return_params:
        sim_params = collect_sim_params(params, nlv, peak_mode=peak_mode)
        return time, signal, sim_params
    else:
        return time, signal



def simulate_erps(n_signals, time_range, params, nlvs=0.005, fs=1000, 
                  peak_mode='gaussian', return_params=False):
    """Generate a group of simulated ERPs.

    Parameters
    ----------
    n_signals : int
        The number of ERPs to generate.
    time_range : list of [float, float]
        Time range to create time vector across, as [start_time, end_time], inclusive.
    params : list of float or generator
        Parameters for the ERP components.
        Length of n_peaks * 3.
    nlvs : float or list of float or generator, optional, default: 0.005
        Noise level to add to generated ERPs.
    fs : float
        Sampling frequency for desired time vector.
    peak_mode : {'gaussian'}, optional
        Which kind of component to generate.
    return_params : bool, optional, default: False
        Whether to return the parameters for the simulated ERPs.

    Returns
    -------
    time : 1d array
        Time vector.
    signals : 2d array
        Matrix of signals [n_signals, n_times-1].
    sim_params : list of SimParams
        Definitions of parameters used for each signal. Has length of n_signals.
        Only returned if `return_params` is True.
    """

    # Initialize things
    time = gen_time_vector(time_range, fs)[:-1]
    signals = np.zeros([n_signals, (len(time))])
    sim_params = [None] * n_signals

    # Check if inputs are generators, if not, make them into repeat generators
    pe_params = check_iter(params, n_signals)
    nlvs = check_iter(nlvs, n_signals)

    # Simulate ERPs
    for ind, pe, nlv in zip(range(n_signals), pe_params, nlvs):
        signals[ind, :], time = gen_signal(time, time_range, pe, fs, nlv,
                                            peak_mode=peak_mode)
        sim_params[ind] = collect_sim_params(pe, nlv, peak_mode=peak_mode)

    if return_params:
        return time, signals, sim_params
    else:
        return time, signals


def sim_erp(time, params, peak_mode='gaussian'):
    """Generate signal values.

    Parameters
    ----------
    time : 1d array
        Time vector to create peak values for.
    params : list of float
        Parameters to create the component.
    peak_mode : {'gaussian'}, optional
        Which kind of component to generate.

    Returns
    -------
    peak_vals : 1d array
        Peak values.
    """

    pe_func = get_pe_func(peak_mode)

    pe_vals = pe_func(time, *params)

    return pe_vals


def sim_noise(time_range, params, fs, nlv):
    """Generate noise values for a simulated ERP.

    Parameters
    ----------
    time_range : list of [float, float]
        Times to create noise values.
    params : list of float
        Parameters of ERP components to scale noise to
    fs : float
        Sampling frequency for desired time vector.
    nlv : float
        Noise level to generate.

    Returns
    -------
    noise_vals : 1d vector
        Noise values.

    Notes
    -----
    This approach generates noise as randomly distributed white noise.
    The 'level' of noise is controlled as the scale of the normal distribution.
    """
    noise_vals = sim_powerlaw(time_range[1]-time_range[0], fs) * (nlv*params[1])

    return noise_vals


def gen_signal(time, time_range, params, fs, nlv, peak_mode='gaussian'):
    """Generate simulated ERP.

    Parameters
    ----------
    time : 1d array
        Time vector to create values for.
    params : list of float
        Parameters to create the ERP components.
    fs : float
        Sampling frequency for desired time vector.
    nlv : float
        Noise level to add to generated signal.

    Returns
    -------
    signal : 1d vector
        simulated signal.

    """

    pe_vals = sim_erp(time, params, peak_mode=peak_mode)
    if nlv != 0:
        noise = sim_noise(time_range, params, fs, nlv)
    else:
        noise = np.zeros(len(pe_vals))

    if len(noise) == (len(pe_vals)-1):
        pe_vals = pe_vals[:-1]
        time = time[:-1]
    signal = pe_vals + noise 

    return signal, time
