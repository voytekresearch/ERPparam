"""
Correction functions for ERPparam.
- correct_overlapping_peaks: Correct the indices of overlapping peaks.
- _find_overlapping_peaks: helper func to identify overlapping peaks
- _find_troughs: helper func to identify troughs between overlapping peaks

"""

# imports
import numpy as np


def correct_overlapping_peaks(signal, peak_indices):
    """ 
    Correct the indices of overlapping peaks fit with ERPparam. If the
    start of a peak overlaps with the previous peak, the start of the peak is
    set to the trough between the peaks. If the end of a peak overlaps with the
    next peak, the end of the peak is set to the trough between the peaks.
    """

    # find overlapping peaks and the troughs between them
    overlap_start, overlap_end = _find_overlapping_peaks(peak_indices)
    idx_trough = _find_troughs(signal, peak_indices, overlap_start, overlap_end)
    
    # update peak indices
    for i_peak in range(len(peak_indices)):
        if overlap_start[i_peak]:
            peak_indices[i_peak][0] = idx_trough[i_peak]
        if overlap_end[i_peak]:
            peak_indices[i_peak][2] = idx_trough[i_peak+1]
            
    return peak_indices


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
