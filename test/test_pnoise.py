"""
Testing
"""
from __future__ import division, print_function, absolute_import
import numpy as np
from numpy import log10
from numpy.testing import assert_almost_equal
import scipy.signal as sig
from plldesigner.pnoise import Pnoise
import scipy.interpolate as intp


def test_at_fc():
    pnobj = Pnoise([1e4, 1e6, 1e8],[-80,-100,-120], fc=2e9)
    pnob2 = pnobj.at_fc(20e9)
    assert np.all(pnob2.ldbc == [-60,-80,-100])


def test_fm_fc_scaling():
    pnobj = Pnoise([1e4, 1e6, 1e8],[-80,-100,-120], fc=2e9)
    pnobj.fc = 20e9
    assert np.all(pnobj.ldbc == [-60,-80,-100])
    pnobj.fm = [1e5, 1e6, 1e7]
    assert np.all(pnobj.ldbc == [-70,-80,-90])

def test_generate_samples():
    pnobj = Pnoise.with_points_slopes([1e5, 1e6, 1e9],
                                      [-80,-100,-120],
                                      [-30,-20,0])
    npoints = 2 ** 16
    fs = 500e6
    phi_t = pnobj.generate_samples(npoints, fs)
    f, pxx = sig.welch(phi_t, fs, window='blackman', nperseg=2**8)
    ldbc_noise = 10 * np.log10(pxx / 2)
    pnobj.fm = f[1:]
    error = np.max(np.abs((pnobj.ldbc - ldbc_noise[1:]) / pnobj.ldbc * 100))
    assert error < 2.5

def test_fc_settler():
    fm = np.array([1e3, 1e5, 1e7])
    ldbc = 10 * np.log10(1 / (fm * fm))
    lorentzian = Pnoise(fm, ldbc, fc=1e9, label='Lorentzian')
    lorentzian.fc = 10e9
    assert_almost_equal(lorentzian.ldbc, ldbc + 20*log10(10e9/1e9))

def test_interp1d_class():
    fm = np.array([1e3, 1e5, 1e7])
    lorentzian = Pnoise(fm, 10 * np.log10(1 / (fm * fm)),
                        label='Lorentzian')
    val = lorentzian.interp1d(1e6)
    assert_almost_equal(val, -120, 4)



def test__init__():
    # Test __init__
    fm = np.logspace(3, 9, 100)
    lorentzian = Pnoise(fm, 10 * np.log10(1 / (fm * fm)),
                        label='Lorentzian')
    white = Pnoise(fm, -120 * np.ones(fm.shape), label='white')
    added = white + lorentzian
    added.label = "addition"
    assert_almost_equal(added.ldbc[0], -60, 4)
    assert_almost_equal(added.ldbc[-1], -120, 4)
    ix, = np.where(fm > 1e6)
    assert_almost_equal(added.ldbc[ix[0]], -117.2822, 4)


def test_private_functions():
    # test the new
    fi = np.array([1e4, 1e9])
    ldbc_fi = np.array([-40, -150])
    slopes = np.array([-30, -20])
    fm = np.logspace(3, 9, 20)
    ldbc_model = Pnoise._pnoise_point_slopes(fi, ldbc_fi, slopes, fm)
    func = intp.interp1d(log10(fm), ldbc_model, kind='linear')
    ldbc_0 = func(log10(fi[0]))
    ldbc_1 = func(log10(fi[1]))
    assert_almost_equal(ldbc_0, ldbc_fi[0], 0)
    assert_almost_equal(ldbc_1, ldbc_fi[1], 0)


def test_with_points_slopes():
    from copy import copy
    # test the new
    fi = np.array([1e4, 1e9])
    ldbc_fi = np.array([-40, -150])
    slopes = np.array([-30, -20])
    pnoise_model = Pnoise.with_points_slopes(fi, ldbc_fi, slopes)
    fm = np.logspace(3, 9, 20)
    pnoise_extrapolated = copy(pnoise_model)
    pnoise_extrapolated.fm = fm


def test_integration():
    fm = np.logspace(4,8,1000)
    lorentzian  = Pnoise(fm,10*np.log10(1/(fm*fm)),label='Lorentzian')
    white = Pnoise(fm,-120*np.ones(fm.shape),label='white')
    added = lorentzian+white
    iadded_gardner = added.integrate()
    iadded_trapz = added.integrate(method='trapz')
    f_int = lambda fm : 2.0e-12*fm-2.0/fm
    i_f_int = np.sqrt(f_int(fm[-1])-f_int(fm[0]))
    assert_almost_equal(iadded_trapz, i_f_int, 5)
    assert_almost_equal(iadded_trapz, i_f_int, 5)


def test_add_points():
    obj = Pnoise([1e3, 1e6, 1e9], [-100, -120, -140])
    obj.add_points([1e2, 1e3, 1e10], [-80, -70, -160])
    assert np.all(obj.fm == [1e2, 1e3, 1e6, 1e9, 1e10])
    assert np.all(obj.ldbc == [-80, -70, -120, -140, -160])
    # Check deep the interpolation
    obj.fm = [1e2, 1e6, 1.1e5, 7e7]
    obj.fm = [1e3, 1e6, 1.1e5,7e7, 1e10]

