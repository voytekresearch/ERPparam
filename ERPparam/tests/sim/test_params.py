"""Test functions for ERPparam.sim.params."""

from pytest import raises

from numpy import array_equal

from ERPparam.core.errors import InconsistentDataError

from ERPparam.sim.params import *

###################################################################################################
###################################################################################################

def test_collect_sim_params():

    pe = [10, 1, 1]
    nlv = 0.05

    sp = collect_sim_params(pe, nlv)

    assert array_equal(sp.peak_params, [pe])
    assert sp.nlv == nlv

    # Check it organizes peaks into embedded lists (or equivalent)
    pe = [10, 1, 1, 20, 1, 1]
    sp = collect_sim_params(pe, nlv)
    assert array_equal(sp.peak_params, [[10, 1, 1], [20, 1, 1]])

    # check for peak_mode argument
    pe = [10, 1, 1]
    sp = collect_sim_params(pe, nlv, peak_mode='gaussian')
    assert array_equal(sp.peak_params, [pe])
    assert sp.nlv == nlv

    pe = [10, 1, 1, 0]
    sp = collect_sim_params(pe, nlv, peak_mode='skewed_gaussian')
    assert array_equal(sp.peak_params, [pe])
    assert sp.nlv == nlv   


def test_stepper():

    stepper = Stepper(8, 12, 1)
    assert stepper
    assert len(stepper) == 4
    for val in stepper:
        assert True

def test_stepper_errors():
    """Test the checks in Stepper._check_values()"""

    with raises(ValueError):
        Stepper(-1, 0, 0)

    with raises(ValueError):
        Stepper(10, 8, 1)

    with raises(ValueError):
        Stepper(8, 12, 5)

def test_param_iter():

    # Test oscillations
    step = Stepper(8, 12, .1)
    osc = [step, .5, .5]
    iter_1 = param_iter(osc)

    for ind, val in enumerate(iter_1):
        assert val == [8 + (.1*ind), .5, .5]

    # Test aperiodic
    step = Stepper(.25, 3, .25)
    ap_params = [0, step]
    iter_1 = param_iter(ap_params)

    for ind, val in enumerate(iter_1):
        assert val == [0, .25 + (.25*ind)]

    # Test n oscillations
    step = Stepper(8, 12, .1)
    oscs = [step, .5, .5, 10, .25, 1]
    iter_1 = param_iter(oscs)

    for ind, val in enumerate(iter_1):
        assert val == [8 + (.1*ind), .5, .5, 10, .25, 1]

    # Test list of lists
    step = Stepper(8, 12, .1)
    osc_1 = [1, 2, 3]
    osc_2 = [4, 5, 6]
    osc_3 = [7, 8, step]
    oscs = [osc_1, osc_2, osc_3]
    iter_2 = param_iter(oscs)

    for ind, val in enumerate(iter_2):
        assert val == [1, 2, 3, 4, 5, 6, 7, 8, 8 + (.1*ind)]

    # Test multiple stepper error
    step = Stepper(8, 12, .1)
    with raises(ValueError):
        for params in param_iter([[step, step, step]]):
            continue

def test_param_sampler():

    pos = [1, 2, 3, 4]

    gen = param_sampler(pos)
    for ind, el in zip(range(3), gen): assert el in pos

    # Test can take prob inputs, and size mismatch error
    gen = param_sampler(pos, probs=[0.85, 0.05, 0.05, 0.05])
    for ind, el in zip(range(3), gen):
        assert el in pos

    gen = param_sampler(pos, probs=[0.5])
    with raises(ValueError):
        next(gen)

def test_param_jitter():

    params = [1, 1]

    # Check that jitter does get applied when it should
    #   Note: the zip range is used because otherwise this is an infinite loop
    jitterer = param_jitter(params, [0.5, 0.5])
    for ind, jits in zip(range(3), jitterer):
        for p1, j1 in zip(jits, params):
            assert p1 != j1

    # Check that jitter does not get applied when it should not
    jitterer = param_jitter(params, [0, 0.5])
    for ind, jits in zip(range(3), jitterer):
        assert jits[0] == params[1]
