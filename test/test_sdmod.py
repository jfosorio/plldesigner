from __future__ import division, print_function, absolute_import
import numpy as np
from plldesigner.sdmod import gen_mash, SDModulator

def test_gen_mash():
    from numpy.testing import assert_almost_equal
    import numpy.random as rnd
    # Test the floating number is well reproduced
    # for the mash, order 1, the signal is fix because other wise
    # it can fail quite often
    floatnum = 0.309 * np.ones(100000)
    # order one
    sequence, period = gen_mash(1, 19, (floatnum * 2 ** 19).astype(int))
    assert_almost_equal(sequence.mean(), floatnum.mean(), 4)
    assert period == None
    sequence, period = gen_mash(1, 19, 0.25 * np.ones(1000) * 2 ** 19)
    assert period == 4
    floatnum = rnd.rand() * np.ones(100000)
    # order two
    sequence, period = gen_mash(2, 19, (floatnum * 2 ** 19).astype(int))
    assert_almost_equal(sequence.mean(), floatnum.mean(), 4)
    assert period == None
    sequence, period = gen_mash(2, 19, 0.25 * np.ones(1000) * 2 ** 19)
    assert period == 8

    # order three
    sequence, period = gen_mash(3, 19, (floatnum * 2 ** 19).astype(int))
    assert_almost_equal(sequence.mean(), floatnum.mean(), 4)
    assert period == None
    sequence, period = gen_mash(3, 19, 0.25 * np.ones(1000) * 2 ** 19)
    assert_almost_equal(sequence.mean(), 0.25, 2)
    assert period == 8
    sequence, period = gen_mash(3, 19, 0.25 * np.ones(1000) * 2 ** 19, init=(1,0,0))
    assert_almost_equal(sequence.mean(), 0.25, 2)
    assert period == None

    # order four
    sequence, period = gen_mash(4, 19, (floatnum * 2 ** 19).astype(int))
    assert_almost_equal(sequence.mean(), floatnum.mean(), 4)
    assert period == None
    sequence, period = gen_mash(4, 19, 0.25 * np.ones(1000) * 2 ** 19)
    assert period == 16

    # Using the class the test is done as:
    sd_mash = SDModulator('mash', 3, 19,
                          (np.array([0.323232] * 100000) * 2 ** 19).astype(int))
    assert_almost_equal(sd_mash.seq.mean(), 0.323232, 4)