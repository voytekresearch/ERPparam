"""Test functions for ERPparam.utils.params."""

import numpy as np

from ERPparam.utils.params import *

###################################################################################################
###################################################################################################


def test_compute_fwhm():

    assert compute_fwhm(1.5)

def test_compute_gauss_std():

    assert compute_gauss_std(1.0)
