from __future__ import division, print_function, absolute_import
from plldesigner.pll import AnalogPLL
from plldesigner.pnoise import Pnoise
import scipy.constants as k
import numpy as np

def test_loopcalc():
    from numpy.testing import  assert_almost_equal
    myAnalogPLL=AnalogPLL(4, 521.8e+06, Navg=55.22, prescaler=2, plltype=2)
    myAnalogPLL.loopcalc(1e6, 60.0, -107.8, 1e6, 0.7, 300)
    # Assert filter values for a specific design
    assert_almost_equal(myAnalogPLL.filter_vals['C1'], 4.2842e-10, 3)
    assert_almost_equal(myAnalogPLL.filter_vals['C2'], 3.31384e-11, 3)
    assert_almost_equal(myAnalogPLL.filter_vals['C3'], 3.31384e-12, 3)
    assert_almost_equal(myAnalogPLL.filter_vals['Icp'], 516.691e-6, 3)
    assert_almost_equal(myAnalogPLL.filter_vals['R1'], 1.3864303e3, 3)
    assert_almost_equal(myAnalogPLL.filter_vals['R2'], 1.28688908e3, 3)
    
def test_add_noise_sources():
    from numpy.testing import  assert_almost_equal
    import matplotlib.pyplot as plt
    myAnalogPLL=AnalogPLL(4, 521.8e+06, Navg=10, prescaler=2, plltype=2)
    myAnalogPLL.loopcalc(1e6, 60.0, -130.0, 1e6, 0.1, 300)
    myAnalogPLL.lti()
    pinput = [Pnoise([1e3, 1e6, 1e9],[-140, -140, -140], label='input', fc=48e6)]
    poutput = [Pnoise([1e3, 1e6, 1e9],[-70, -120, -180], label='VCO', fc=480e6)]

    fm = np.logspace(4,8,100)
    pn_total, pn_in_colored, pn_out_colored = \
        myAnalogPLL.add_noise_sources(fm, pn_inputs=pinput, pn_outputs=poutput)
    assert_almost_equal(pn_total.interp1d(1e8), -160, 2)
    assert_almost_equal(pn_total.interp1d(1e4), -120, 2)

def test_lti():
    from numpy.testing import  assert_almost_equal
    import matplotlib.pyplot as plt

    # Create a PLL and obtain the lti system
    for order in [3, 4]:
        myAnalogPLL=AnalogPLL(order, 521.8e+06, Navg=10, prescaler=2, plltype=2)
        myAnalogPLL.loopcalc(1e6, 60.0, -107.8, 1e6, 0.7, 300)
        G, T, H = myAnalogPLL.lti()

        # The open loop transfer function G
        f = np.linspace(0.95e6, 1.05e6, num=100)
        w, mag, phase = G.bode(w=2*k.pi*f)
        ix = np.max(np.where(mag>=0))
        assert_almost_equal(f[ix]/1e6, 1, 1)

