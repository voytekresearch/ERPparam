"""
Correction functions for ERPparam.
- correct_overlapping_peaks: Correct the indices of overlapping peaks
- _refine_peak_index: update peak index to signal extremum between half-maximums
- _find_overlapping_peaks: helper func to identify overlapping peaks
- _find_troughs: helper func to identify troughs between overlapping peaks
- _find_stumpy_peaks: identify peaks that don't have sufficient rise/decay

"""

# imports
import numpy as np


def correct_overlapping_peaks(signal, peak_indices, gaussian_params, min_rise_decay_height):
    """ 
    Correct the indices of overlapping peaks fit with ERPparam. If the
    start of a peak overlaps with the previous peak, the start of the peak is
    set to the trough between the peaks. If the end of a peak overlaps with the
    next peak, the end of the peak is set to the trough between the peaks.
    """

    # find overlapping peaks and the troughs between them
    overlap_start, overlap_end = _find_overlapping_peaks(peak_indices)
    idx_trough = _find_troughs(signal, peak_indices, overlap_start, overlap_end)
    
    # create temporary array that we can modify
    peak_indices_temp = peak_indices.copy()
    # update peak indices to the troughs
    for i_peak in range(len(peak_indices_temp)):
        if overlap_start[i_peak]:
            peak_indices_temp[i_peak][0] = idx_trough[i_peak]
        if overlap_end[i_peak]:
            peak_indices_temp[i_peak][2] = idx_trough[i_peak+1]

    # find the new max of the signals
    peak_indices_temp = _refine_peak_index(signal, peak_indices_temp)

    # detect signals that don't have sufficient height between the half-maximum points and the peak
    peak_indices_drop = _find_stumpy_peaks(signal, peak_indices_temp, min_rise_decay_height)

    if (peak_indices_drop is not None) and (peak_indices_drop.size > 0):
        # if we've detected peaks to drop due to stumpiness, then we want to drop them and re-run this function as they never existed
        peak_indices_dropped = np.delete(peak_indices.copy(), peak_indices_drop, axis=0)
        gaussian_params_dropped = np.delete(gaussian_params.copy(), peak_indices_drop, axis=0)
        peak_indices, gaussian_params = correct_overlapping_peaks(
            signal, peak_indices_dropped, gaussian_params_dropped, min_rise_decay_height)
    else:
        # otherwise we assign our peak_indices to the modified array
        peak_indices = peak_indices_temp

    return peak_indices, gaussian_params


def _refine_peak_index(signal, peak_indices):
    """
    Find signal extrema between corrected half-maximum points
    """
    if np.size(peak_indices) == 0:
        return peak_indices
    else:
        refined_indices = []
        for start, peak, end in peak_indices:
            if np.isnan(start) or np.isnan(peak) or np.isnan(end):
                refined_indices.append([np.nan, np.nan, np.nan])
                continue

            # Find local maxima/minima between the half-maximum points
            local_signal = signal[int(start):int(end)]
            local_max = np.argmax(np.abs(local_signal))

            # Refine the peak index
            refined_peak = start + local_max
            refined_indices.append([start, refined_peak, end])

        return np.array(refined_indices)


def _find_stumpy_peaks(signal, peak_indices, min_rise_decay_height):
    """
    Drop peak indices in which the distance between the left or right
    half-maximum point and the signal peak is not a sufficiently large portion 
    of the total amplitude
    """
    if np.size(peak_indices) == 0:
        return  np.array([])
    else:
        short_peak_idx = []
        for i_peak in range(len(peak_indices)):
            start, peak, end = peak_indices[i_peak]
            if np.isnan(start):
                continue
            sig_height = signal[int(peak)]
            left_height = signal[int(start)]
            right_height = signal[int(end)]
            
            # get the signal height between the left rise point as a percent of total amplitude
            amp_ratio_rise = ((sig_height - left_height) / sig_height) 
            amp_ratio_decay = ((sig_height - right_height) / sig_height)
            
            # check that these portions are not less than the designated threshold
            if ((amp_ratio_rise <= min_rise_decay_height) or (amp_ratio_decay <= min_rise_decay_height)):
                short_peak_idx.append(i_peak)
            
        return np.array(short_peak_idx)


def _find_overlapping_peaks(peak_indices):
    """
    Find indices of overlapping peaks in a list of peak indices.

    Parameters
    ----------
    peak_indices : list of tuples
        List of tuples, where each tuple contains the start, peak, and end 
        indices of a peak.

    Returns
    -------
    overlap_start : 1d array
        Boolean array indicating if the start of a peak overlaps with the 
        previous peak.
    overlap_end : 1d array
        Boolean array indicating if the end of a peak overlaps with the next 
        peak.
    """

    # chech if start of peak is before the previous peak
    overlap_start = np.zeros(len(peak_indices), dtype=bool)
    for i_peak in range(1, len(peak_indices)):
        if peak_indices[i_peak][0] < peak_indices[i_peak-1][1]:
            overlap_start[i_peak] = True

    # check if end of peak is after the next peak
    overlap_end = np.zeros(len(peak_indices), dtype=bool)
    for i_peak in range(len(peak_indices)-1):
        if peak_indices[i_peak][2] > peak_indices[i_peak+1][1]:
            overlap_end[i_peak] = True


    return overlap_start, overlap_end


def _find_troughs(signal, peak_indices, overlap_start, overlap_end):
    """
    Find the troughs between overlapping peaks in a signal.

    Parameters
    ----------
    signal : 1d array
        Signal containing the peaks.
    peak_indices : list of tuples
        List of tuples, where each tuple contains the start, peak, and end
        indices of a peak.
    overlap_start : 1d array
        Boolean array indicating which peaks have an starting index that overlaps
        with the previous peak.
    overlap_end : 1d array
        Boolean array indicating which peaks have an ending index that overlaps
        with the following peak.

    Returns
    -------
    idx_trough : 1d array
        Array of indices of the troughs between the overlapping peaks.
    
    """

    # initialize
    idx_trough = np.zeros_like(overlap_start) * np.nan
    for i_overlap in range(len(overlap_start)):
        if overlap_start[i_overlap]:
            overlap = signal[int(peak_indices[i_overlap-1][1]) : \
                             int(peak_indices[i_overlap][1])]
            idx_trough[i_overlap] = np.argmin(np.abs(overlap)) + \
                peak_indices[i_overlap-1][1]
        
    for i_overlap in range(len(overlap_start)-1):      
        if overlap_end[i_overlap] and not overlap_start[i_overlap+1]:
            overlap = signal[int(peak_indices[i_overlap][1]) : \
                             int(peak_indices[i_overlap+1][1])]
            idx_trough[i_overlap] = np.argmin(np.abs(overlap)) + \
                peak_indices[i_overlap][1]
            
    return idx_trough
