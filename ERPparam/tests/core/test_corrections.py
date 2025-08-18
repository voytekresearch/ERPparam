"""Tests for ERPparam.core.corrections."""

import numpy as np

from ERPparam import ERPparam
from ERPparam.sim import simulate_erp
from ERPparam.core.corrections import ( correct_overlapping_peaks, 
                                       _find_overlapping_peaks, _find_troughs)

###################################################################################################
###################################################################################################

def test_correct_overlapping_peaks():
    # simulate ERP with overlapping peaks
    time_range = [-0.25, 1.]
    erp_params = np.asarray([0.1, 0.5, 0.03, 0.2, 0.4, 0.05])
    time, erp = simulate_erp(time_range, erp_params, nlv=0)

    # fit peaks with positive polarity
    model = ERPparam(max_n_peaks=2, min_peak_height=0.1)
    model.fit(time, erp)
    assert np.isclose(model.peak_indices_[0, 2], model.peak_indices_[1, 0]) # overlap set to trough

    # fit peaks with negative polarity
    model = ERPparam(max_n_peaks=2, min_peak_height=0.1)
    model.fit(time, -erp)
    assert np.isclose(model.peak_indices_[0, 2], model.peak_indices_[1, 0]) # overlap set to trough

    # test with no peaks fit
    gaussian_params = np.ones([0,4])*np.nan
    peak_indices = np.empty((len(gaussian_params), 3))
    peak_indices_corr = correct_overlapping_peaks(erp, peak_indices)
    assert peak_indices_corr.shape == (0, 3)

def test_find_overlapping_peaks():
    # 2 overlapping peaks
    peak_indices = np.asarray([[316., 350., 493.], [308., 450., 509.]])
    overlap_start, overlap_end = _find_overlapping_peaks(peak_indices)
    assert np.all(overlap_start == [False, True])
    assert np.all(overlap_end == [True, False])

    # more overlapping peaks
    peak_indices = np.asarray([[316., 350., 493.], 
                               [308., 450., 509.], 
                               [400., 500., 609.]])
    overlap_start, overlap_end = _find_overlapping_peaks(peak_indices)
    assert np.all(overlap_start == [False, True, True])
    assert np.all(overlap_end == [True, True, False])

    # no overlap
    peak_indices = np.asarray([[316., 350., 409.], [409., 450., 509.]])
    overlap_start, overlap_end = _find_overlapping_peaks(peak_indices)
    assert np.all(overlap_start == [False, False])
    assert np.all(overlap_end == [False, False])

def test_find_troughs():
    # simulate ERP with overlapping peaks
    time_range = [-0.25, 1.]
    erp_params = np.asarray([0.1, 0.5, 0.03, 0.2, 0.4, 0.05])
    _, erp = simulate_erp(time_range, erp_params, nlv=0)

    # define (uncorrected) peak indices and overlap (based on prior fitting)
    peak_indices = np.asarray([[316., 350., 493.], [308., 450., 509.]])
    overlap_start = np.array([False, True], dtype=bool)
    overlap_end = np.array([True, False], dtype=bool)

    # test trough finding for overlapping peaks
    idx_trough = _find_troughs(erp, peak_indices, overlap_start, overlap_end)
    assert np.isnan(idx_trough[0]) # no overlap for start of peak 0
    assert idx_trough[1] > peak_indices[0, 1] and idx_trough[1] < peak_indices[1, 1] # trough between peaks

    # test again with no overlap
    idx_trough = _find_troughs(erp, peak_indices, [False, False], [False, False])
    assert np.isnan(idx_trough).all()
