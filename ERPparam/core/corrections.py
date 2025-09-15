"""
Correction functions for ERPparam.
- correct_overlapping_peaks: Correct the indices of overlapping peaks.
- _find_overlapping_peaks: helper func to identify overlapping peaks
- _find_troughs: helper func to identify troughs between overlapping peaks

"""

# imports
import numpy as np
from ERPparam.core.funcs import gaussian_function, skewed_gaussian_function

def correct_overlapping_peaks(signal, time, peak_indices, gaussian_params):
    """ 
    Correct the indices of overlapping peaks fit with ERPparam. If the
    start of a peak overlaps with the previous peak, the start of the peak is
    set to the trough between the peaks. If the end of a peak overlaps with the
    next peak, the end of the peak is set to the trough between the peaks.
    """
    # find and remove duplicate peaks
    peak_indices, gaussian_params = _find_identical_peaks_and_remove(peak_indices, gaussian_params, signal, time)

    # find overlapping peaks and the troughs between them
    overlap_start, overlap_end = _find_overlapping_peaks(peak_indices)
    idx_trough = _find_troughs(signal, peak_indices, overlap_start, overlap_end)
    
    # update peak indices
    for i_peak in range(len(peak_indices)):
        if overlap_start[i_peak]:
            peak_indices[i_peak][0] = idx_trough[i_peak]
        if overlap_end[i_peak]:
            peak_indices[i_peak][2] = idx_trough[i_peak+1]
            
    return peak_indices, gaussian_params

def _find_identical_peaks_and_remove(peak_indices, gaussian_params, signal, times):
    # get a reduced version of the peak_indices array, where only unique rows are shown
    peak_indices_unique, counts = np.unique(peak_indices, axis=0, return_counts=True)
    # detect which of these unique rows has more than one count, if any
    duplicate_counts = np.where((counts>1))[0]
    # if there are any duplicates, there could be multiple sets

    if len(duplicate_counts) > 0:
        discard = []
        # so we have to loop through each set of duplicates
        for dc in duplicate_counts:
            # and find the array of peak_index values which was duplicated
            duplicate_peak_indices_arr = peak_indices_unique[dc, :]
            # and figure out which indices in the original peak_indices array that they correspond to (should be at least two)
            duplicate_peak_indices_idx_bool = np.all((peak_indices==duplicate_peak_indices_arr) , axis=1)
            duplicate_peak_idx_in_original_array = np.where(duplicate_peak_indices_idx_bool)[0]
            # loop through each candidate peak and extract the gaussians, and get the r-squared
            get_rsqs = []
            for duplicate_gauss_idx in duplicate_peak_idx_in_original_array:
                # get gaussian param of each idx, where this gaussian led to the same peak_index as some other
                dup_gauss = gaussian_params[duplicate_gauss_idx, :]
                # check if the last param (skew) is a nan, and get rid of it if so (and use correct gauss func)
                if np.isnan( dup_gauss[-1] ): 
                    # if the last element (skew param) is nan, then we want the gauss func to be the normal one and we drop the skew param
                    gauss_fit = gaussian_function(times, *dup_gauss[:-1])
                else:
                    # otherwise we generate the skewed signal
                    gauss_fit = skewed_gaussian_function(times, *dup_gauss)
                # compute the r-sq of this signal
                r_fit = _calc_r_squared(signal, gauss_fit)
                get_rsqs.append(r_fit)
            # get the index of the best fit gaussian, so that we know which one to keep and which to toss out
            get_rsqs = np.asarray(get_rsqs)
            worse_fits = np.where((get_rsqs != np.max(get_rsqs)))[0]
            inds_to_toss = duplicate_peak_idx_in_original_array[worse_fits]
            for i in inds_to_toss:
                discard.append(i)
        peak_indices = np.delete(peak_indices, np.asarray(discard), axis=0)  
        gaussian_params = np.delete(gaussian_params, np.asarray(discard), axis=0)  
                
    return peak_indices, gaussian_params

def _calc_r_squared(signal, gaussian_fit):
    """Calculate the r-squared goodness of fit of the model compared to the 
    original data."""

    r_val = np.corrcoef(signal, gaussian_fit)
    return (r_val[0][1] ** 2)

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
