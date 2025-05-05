# -*- coding: utf-8 -*-
"""
A clase to design and analyses PLL's using the phase domain model
"""

from __future__ import (absolute_import, division, print_function)

import numpy as np
from numpy import sqrt, log10, tan
import scipy.constants as k
import scipy.signal.ltisys as lti
from .pnoise import Pnoise


class AnalogPLL(object):
    def __init__(self, order, Kvco, Navg=1.0, prescaler=1, plltype=2,
                 filter_vals={}):
        """
        Create a AnalogPLL with different parameter

        Parameters
        ----------
        order : int
            The order of the pll
        Kvco : float
            Oscillator gain in Hz/V
        Navg : float
            Division ration average
        prescaler : int
            The prescaler value not implemented yet

        Return
        ------
        filter_val      Is a dictionary with all the filter components

        """
        
        self.order, self.Kvco, self.Navg = order, Kvco, Navg
        self.prescaler, self.plltype = prescaler, plltype
        self.filter_vals = filter_vals

    def loopcalc(self, fc, pm, Lvco, Lvco_fr, DL, Temp=300.13):
        """
        Calculates the filter of a pll of type2 and 2nd Order

        Parameters
        ----------
        fc : float
            The cut off frequency in Hertz
        pm : float
            The phase margin in degrees
        Lvco : float
            VCO phase noise in dBc/Hz at frequency Lvco_fr
        Lvco_fr : float
            Frequency where the  Lvco is specified
        DL : float
            Ratio of the noise of the Filter to the the noise of
            the VCO due to R1
        Kvco : float
            Oscillator gain in Hz/V
        Navg : float
            Division ration average
        Temp : float
            temperature in K

        Return
        ------
        filter_val      Is a dictionary with all the filter components

        """
        self.fc, self.pm, self.Lvco = fc, pm, Lvco
        self.Lvco_fr, self.DL, self.Temp = Lvco_fr, DL, Temp
        Kvco = self.Kvco
        # phisical constants
        valid_orders_types = ((3, 2), (4, 2))
        if (self.order, self.plltype) not in valid_orders_types:
            print('Order and Type not implemented')
            raise
        if self.order == 3 and self.plltype == 2:
            b = (tan(pm * k.pi / 180 / 2 + k.pi /4)) ** 2
            wc = 2 * k.pi * fc
            wp = wc * sqrt(b)
            wz = wc / sqrt(b)
            tz = 1 / wz
            tp = 1 / wp
            phi_fm = sqrt(2 * 10 ** (Lvco/10))
            R1 = (10 ** (DL / 10) - 1) / ((b - 1) / b) / \
                (4 * k.k * Temp * Kvco ** 2) * phi_fm ** 2 * Lvco_fr ** 2
            C1 = tz / R1
            C2 = tz * tp / R1 / (tz - tp)
            Icp = (2 * k.pi * self.Navg * fc * b) / (R1 * Kvco * (b - 1))
            self.filter_vals = {'C1' : C1, 'C2' : C2, 'R1' : R1, 'Icp' : Icp}
        if self.order == 4 and self.plltype == 2:
            b = (tan(pm * k.pi / 180 / 2 + k.pi / 4)) ** 2
            wc = 2 * k.pi * fc
            wp = wc * sqrt(b)
            wz = wc / sqrt(b)
            tz = 1 / wz
            tp = 1 / wp
            phi_fm = sqrt(2 * 10 ** (Lvco / 10))
            R1 = (10 ** (DL / 10) - 1) / ((b - 1) / b)/(4 * k.k * Temp * Kvco ** 2)\
                * phi_fm ** 2 * Lvco_fr ** 2
            C1 = tz / R1
            C2 = tz * tp / R1 / (tz - tp)
            Icp = (2 * k.pi * self.Navg * fc * b) / (R1 * Kvco * (b-1))
            tp2 = tp / 10
            C3 = C2 / 10 # C2 should not be that big
            R2 = tp2 / C3
            self.filter_vals = {'C1': C1, 'C2':C2, 'C3':C3, 'R1':R1, 'R2':R2,
                                'Icp': Icp}
    def lti(self):
        """
        Calculates the lti models for the PLL

        Return
        ------
        G : lti
            The open loop transfer function
        T : lti
            The error transfer function
        H : lti
            the input to output transfer function

        """
        Navg, Kvco, fvals = (self.Navg, self.Kvco, self.filter_vals)
        if self.order == 3 and self.plltype == 2:
            C1, C2, R1, Icp = (fvals['C1'], fvals['C2'], fvals['R1'],
                               fvals['Icp'])
            tp = R1 * C1 * C2 / (C1 + C2)
            tz = R1 * C1
            Kf = 1 / (C1 + C2)
            Kpfd = Icp / 2 / k.pi
            K = Kf * Kpfd / Navg * 2 * k.pi * Kvco
            G = lti.lti([K*tz, K], [tp, 1, 0, 0])
            T = lti.lti([tp, 1, 0, 0], [tp, 1, K*tz, K])
            H = lti.lti([Navg*K*tz, Navg*K], [tp, 1, K*tz, K])
        if self.order == 4 and self.plltype == 2:
            Navg, Kvco, fvals = (self.Navg, self.Kvco, self.filter_vals)
            C1, C2, C3, R1, R2, Icp = (fvals['C1'], fvals['C2'], fvals['C3'],
                fvals['R1'], fvals['R2'],fvals['Icp'])
            tz = R1 * C1
            Kf = 1 / (C1 + C2 + C3)
            a2 = C1 * C2 * C3 * R1 * R2 * Kf
            a1 = (C1 * C2 * R1 + C1 * C3 * R1 + C1 * C3 * R2 + C2 * C3 * R2) * Kf
            a0 = 1
            Kpfd = Icp / 2 / k.pi
            K = Kf * Kpfd / Navg * 2 * k.pi * Kvco
            G = lti.lti([K*tz, K], [a2, a1, a0, 0, 0])
            T = lti.lti([a2, a1, a0, 0, 0], [a2, a1, a0, K*tz, K])
            H = lti.lti( [Navg*K*tz, Navg*K],  [a2, a1, a0, K*tz, K])
        if self.order not in(3, 4):
            print('order not implemented, the noise can not be calculated')
            raise
        self.G, self.T, self.H = G, T, H
        return G, T, H

    def calcTF(self, fm):
        unused, Hfm = self.H.freqresp(w=2 * k.pi * fm)
        unused, Gfm = self.G.freqresp(w=2 * k.pi * fm)
        unused, Tfm = self.T.freqresp(w=2 * k.pi * fm)
        return Hfm, Gfm, Tfm

    def calcfcpm(self,fm):
        Hfm, Gfm, Tfm = self.calcTF(fm)


    def filter_vn2(self, fm, Temp=300.13):
        fvals = self.filter_vals
        s = 2*k.pi*fm*1j
        if self.order==3 and self.plltype == 2:
            C1, C2, R1 = (fvals['C1'], fvals['C2'], fvals['R1'])
            HvnR1 = C1 / ((C1 * C2 * R1) * s + C1 + C2)
            vn2 = 4 * k.k * Temp * R1 * abs(HvnR1) ** 2
        if self.order==4  and self.plltype == 2:
            C1, C2, C3, R1, R2 = (fvals['C1'], fvals['C2'], fvals['C3'],
                              fvals['R1'], fvals['R2'])
            HvnR1 = C1 / (C1 * C2 * C3 * R1 * R2 * s ** 2 +
                  (C1 * C2 * R1 + C1 * C3 * R1 + C1 * C3 * R2 + C2 * C3 * R2) * s + C1 + C2 + C3)
            vn2R1 = 4 * k.k * Temp * R1 * np.abs(HvnR1) ** 2
            HvnR2 = ((C1 * C2 * R1) * s + C1 + C2) / (C1 * C2 * C3 * R1 * R2 * s ** 2 +
                    (C1 * C2 * R1 + C1 * C3 * R1 + C1 * C3 * R2 + C2 * C3 * R2) * s + C1 + C2 + C3)
            vn2R2 = 4 * k.k * Temp * R2 * np.abs(HvnR2) ** 2
            vn2 = vn2R1 + vn2R2
        return vn2

    def pnoise_calc(Lin_ref, Lout_ref, fm, Mult=1, Div=1):
        phi2_in_ref = 2 * 10 ** (Lin_ref / 10)
        Hfm = repmat(Hfm, size(phi2_in_ref, 1), 1)
        phi2_in_ref *= abs(Hfm) ** 2
        Lin_ref = 10 * log10(phi2_in_ref / 2) + 20 * log10(Mult) + 20 * log10(1 / Div)
        # Filter the noise of  the output referred sources
        phi2_out_ref = 2*10 ** (Lout_ref / 10)
        Tfm = repmat(Tfm, size(phi2_out_ref, 1), 1)
        phi2_out_ref = phi2_out_ref * abs(Tfm) ** 2
        Lout_ref = 10 * log10(phi2_out_ref / 2) + 20 * log10(Mult) + 20 * log10(1 / Div)
        # sum column wise the numbers and add them afterwards
        phi2_tot = (sum(phi2_in_ref, 1) + sum(phi2_out_ref, 1)) * Mult**2/Div**2
        # division
        Ltot = 10 * log10(phi2_tot / 2)
        pnLtot  = Pnoise(fm, Ltot)
        phi_int = pnLtot.integrate()
        return phi_int

    def filter_repr(self):
        str_val  = 'Filter report\n'
        str_val += '=============\n'
        str_val += 'Input parameters:\n'
        str_val += 'fc = {:2.3f} (MHz), pm = {:2.1f} (degrees)\n'.format(self.fc / 1e6,self.pm)
        str_val += 'Lvco {:2.3f} (dBc/Hz) @ {:2.1f} (MHz), DL = {:2.1f}\n'.format(self.Lvco, self.Lvco_fr/1e6, self.DL)
        str_val += 'Ideal values:\n'
        str_val += 'Icp = {:2.3f} (uA), R1 = {:2.3f} (Kohms), R2 = {:2.3f} (Kohms)\n'.format(
            self.filter_vals['Icp'] / 1e-6, self.filter_vals['R1'] / 1e3,
            self.filter_vals['R2'] / 1e3)
        str_val += 'C1 = {:2.3f} (pf), C2 = {:2.3f} (pf), C3 = {:2.3f} (pf)\n'.format(
        self.filter_vals['C1'] / 1e-12, self.filter_vals['C2'] / 1e-12,
        self.filter_vals['C3'] / 1e-12)
        return str_val

    def add_noise_sources(self,fm, pn_inputs=[], pn_outputs=[]):
        # Calculated the transfer functions
        Hfm, Gfm, Tfm = self.calcTF(fm)
        # All the noise sources use the same offset
        for pn_elem in pn_inputs+pn_outputs:
            pn_elem.fm = fm
        # Calcualte the noise of the filter
        phi2_filter = self.filter_vn2(fm) * (self.Kvco / fm) ** 2
        pn_filter = Pnoise(fm,phi2_filter,label='filter',units='rad**/Hz')
        pn_filter_colored = pn_filter * (np.abs(Tfm) ** 2)
        # Calculate the noise of the other sources
        pn_out_colored = [pn_elm * (np.abs(Tfm) ** 2) for pn_elm in pn_outputs]
        pn_out_colored.append(pn_filter_colored)
        pn_in_colored  = [pn_elm * (np.abs(Hfm) ** 2) for pn_elm in pn_inputs]

        pn_total = pn_out_colored[0]
        for pn_elem in pn_out_colored[1:]:
            pn_total += pn_elem
        for pn_elem in pn_in_colored:
            pn_total += pn_elem
        return pn_total, pn_in_colored, pn_out_colored

    def sim(self, t, U = []):
        """
        Simulate the PLL with a input

        Parameters
        ----------
        t : array_like
            Vector with the time
        U: array_like
            Input to the block with the same lenght of t, if a empty is given

        Return
        ------
        out : array like response to U
        error : error
        """
        if not len(U):
            U = np.ones_like(t)
        t, signal_out, X = lti.lsim(self.H, U=U, T=t)
        error_out = signal_out - U * self.Navg
        return signal_out, error_out



class AnalogPLLDict(AnalogPLL):
    def __init__(self, _dict):
        self.__dict__.update(_dict)
        self.loopcalc(_dict['fc'], _dict['pm'], _dict['Lvco'],
            _dict['Lvco_fr'], _dict['DL'], Temp=_dict['Temp'])
