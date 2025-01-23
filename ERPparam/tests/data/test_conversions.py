"""Tests for the ERPparam.data.conversions."""

from copy import deepcopy

import numpy as np

from ERPparam.core.modutils import safe_import
pd = safe_import('pandas')

from ERPparam.data.conversions import *

###################################################################################################
###################################################################################################

def test_model_to_dict(tresults, twindow):

    out = model_to_dict(tresults, peak_org=1)
    assert isinstance(out, dict)
    assert 'ct_0' in out
    assert out['ct_0'] == tresults.peak_params[0, 0]
    assert not 'ct_1' in out

    out = model_to_dict(tresults, peak_org=2)
    assert 'ct_0' in out
    assert 'ct_1' in out
    assert out['ct_1'] == tresults.peak_params[1, 0]

    out = model_to_dict(tresults, peak_org=3)
    assert 'ct_2' in out
    assert np.isnan(out['ct_2'])

    out = model_to_dict(tresults, peak_org=twindow)
    assert 'ct' in out
    assert out['ct'] == tresults.peak_params[0, 0]


def test_model_to_dataframe(tresults, skip_if_no_pandas):

    for peak_org in [1, 2, 3]:
        out = model_to_dataframe(tresults, peak_org=peak_org)
        assert isinstance(out, pd.Series)


def test_group_to_dataframe(tresults, skip_if_no_pandas):

    fit_results = [deepcopy(tresults), deepcopy(tresults), deepcopy(tresults)]

    for peak_org in [1, 2, 3]:
        out = group_to_dataframe(fit_results, peak_org=peak_org)
        assert isinstance(out, pd.DataFrame)
