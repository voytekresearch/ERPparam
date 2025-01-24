"""Test functions for ERPparam.utils.io."""

import numpy as np

from ERPparam.core.items import OBJ_DESC
from ERPparam.objs import ERPparam, ERPparamGroup

from ERPparam.tests.settings import TEST_DATA_PATH

from ERPparam.utils.io import *

###################################################################################################
###################################################################################################

def test_load_ERPparam():

    file_name = 'test_ERPparam_all'

    tfm = load_ERPparam(file_name, TEST_DATA_PATH)

    assert isinstance(tfm, ERPparam)

    # Check that all elements get loaded
    for result in OBJ_DESC['results']:
        assert not np.all(np.isnan(getattr(tfm, result)))
    for setting in OBJ_DESC['settings']:
        assert getattr(tfm, setting) is not None
    for data in OBJ_DESC['data']:
        assert getattr(tfm, data) is not None
    for meta_dat in OBJ_DESC['meta_data']:
        assert getattr(tfm, meta_dat) is not None

def test_load_ERPparamgroup():

    file_name = 'test_ERPparamgroup_all'
    tfg = load_ERPparamGroup(file_name, TEST_DATA_PATH)

    assert isinstance(tfg, ERPparamGroup)

    # Check that all elements get loaded
    assert len(tfg.group_results) > 0
    for setting in OBJ_DESC['settings']:
        assert getattr(tfg, setting) is not None
    assert tfg.signals is not None
    for meta_dat in OBJ_DESC['meta_data']:
        assert getattr(tfg, meta_dat) is not None
