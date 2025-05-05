"""
Implementation of a sigma delta modulator and the functions related
To do generalize the algorithm
"""

import numpy as np
from numpy import (zeros, arange, log10, sin, pi)
import matplotlib.pyplot as plt
from .pnoise import Pnoise


class SDModulator(object):
    """ Creates a Sigma-Delta modulator object

    Parameters
    ----------
    modtype : string
        type of SDModulator implemented types are:
        ('mash')
    *arg  : Positional arguments for the different types of SDMOD
    **argk :
    """

    def __init__(self, modtype, *arg, **argk):
        """ """
        self.modtype = modtype
        func = {'mash': gen_mash}
        self.seq, self.period = func[modtype](*arg, **argk)

    def plotseq(self, *arg, **argk):
        xval = np.arange(len(self.seq))
        plt.step(xval, self.seq, *arg, **argk)

    def LdB_theoretical(self, fm, fref, n=1.0):
        func = {'mash': L_mash_dB}
        return func[self.modtype](fm, fref, n)


def gen_mash(order, n, k, init=()):
    """ Generates a mash type $\Sigma-\Delta$ sequence

    Parameters
    ----------
    order : int
        order of the $\Sigma-\Delta$ modulator.
    n : int
        Number of bits of the modulator.
    k : int
        Value being converted
    init: tuple
        init is a tuple that initialize the register of the
        mash sd, at with length equal to the order.

    Returns
    -------
    sd : array
        sigma delta modulator sequence
    period : int
        Period of the sequence

    Commentaries
    ------------
    This implementation is not really effective from the computational
    point of view but is representative of the architecture of the
    system.
    """

    # Assertions
    assert len(init) == order or len(init) == 0, (
        'Initial condition length must be equal to the order of the modulator')
    assert 1 <= order <= 4, 'This orders have not been implemented'

    # Modulator of order 1
    _maxvalue = 2 ** n - 1
    L = len(k)
    
    if len(init) == 0:
        init = [0]*order
    sd = [0]*L

    # Modulator of order 1        
    if order == 1:
        # initialize the registers
        overflow0 = [0]*L
        state0 = [0]*L
        # initialize the state
        state0[0] = init[0]
        for j in range(1, L):
            state0[j] = state0[j - 1] + k[j - 1]
            if state0[j] > _maxvalue:
                overflow0[j] = 1
                state0[j] -= _maxvalue + 1
            sd[j] = overflow0[j]
        cycles, = np.where(np.asarray(state0) == init[0])
        if len(cycles) > 1:
            period = cycles[1] - cycles[0]
        else:
            period = None


    # Modulator of order 2
    elif order == 2:
        # initialize the registers
        state0, state1 = [0]*L,[0]*L
        overflow0, overflow1 = [0]*L, [0]*L
        
        # initialize the state
        state0[0], state1[0] = init

        # Implement the SDM
        for j in range(1, L):
            state0[j] = state0[j - 1] + k[j - 1]
            if state0[j] > _maxvalue:
                overflow0[j] = 1
                state0[j] -= _maxvalue + 1
            state1[j] = state1[j - 1] + state0[j]
            if state1[j] > _maxvalue:
                overflow1[j] = 1
                state1[j] -= _maxvalue + 1
            sd[j] = overflow0[j] + overflow1[j] - overflow1[j-1]
        state = np.vstack((state0, state1))
        cycles, = np.where((state[0,:]==init[0]) & (state[1,:]==init[1]))
        if len(cycles) > 1:
            period = cycles[1] - cycles[0]
        else:
            period = None

    # Modulator of order 3
    elif order == 3:
        # initialize the registers
        state0, state1, state2 = [0]*L,[0]*L, [0]*L
        overflow0, overflow1, overflow2 = [0]*L, [0]*L, [0]*L
        
        #initaitlize the state
        state0[0], state1[0], state2[0] = init

        # Implement the SDM
        for j in range(1, L):
            state0[j] = state0[j - 1] + k[j - 1]
            if state0[j] > _maxvalue:
                overflow0[j] = 1
                state0[j] -= _maxvalue + 1

            state1[j] = state1[j - 1] + state0[j]
            if state1[j] > _maxvalue:
                overflow1[j] = 1
                state1[j] -= _maxvalue + 1

            state2[j] = state2[j - 1] + state1[j]
            if state2[j] > _maxvalue:
                overflow2[j] = 1
                state2[j] -= _maxvalue + 1
            sd[j] = overflow0[j]
            sd[j] += overflow1[j] - overflow1[j-1]
            sd[j] += overflow2[j] - 2 * overflow2[j-1]
            sd[j] += overflow2[j-2]
            
        state = np.vstack((state0, state1, state2))
        cycles, = np.where((state[0,:]==init[0]) & (state[1,:]==init[1]) &
                            (state[2,:]==init[2]))
        if len(cycles) > 1:
            period = cycles[1] - cycles[0]
        else:
            period = None

    elif order == 4:
        # initialize the registers
        state0, state1, state2, state3 = [0]*L,[0]*L, [0]*L, [0]*L
        overflow0, overflow1 = [0]*L, [0]*L
        overflow2, overflow3 = [0]*L, [0]*L
        
        #initaitlize the state
        state0[0], state1[0], state2[0], state3[0] = init
        
        # Implement the SDM
        for j in range(1, L):
            state0[j] = state0[j - 1] + k[j - 1]
            if state0[j] > _maxvalue:
                overflow0[j] = 1
                state0[j] -= _maxvalue + 1
            state1[j] = state1[j - 1] + state0[j]
            if state1[j] > _maxvalue:
                overflow1[j] = 1
                state1[j] -= _maxvalue + 1
            state2[j] = state2[j - 1] + state1[j]
            if state2[j] > _maxvalue:
                overflow2[j] = 1
                state2[j] -= _maxvalue + 1
            state3[j] = state3[j - 1] + state2[j]
            if state3[j] > _maxvalue:
                overflow3[j] = 1
                state3[j] -= _maxvalue + 1
                
            sd[j] = overflow0[j]
            sd[j] += overflow1[j] - overflow1[j-1]
            sd[j] += overflow2[j] - 2 * overflow2[j-1]
            sd[j] += overflow2[j-2]
            sd[j] += overflow3[j] - 3 * overflow3[j-1]
            sd[j] += 3 * overflow3[j-2]
            sd[j] -= overflow3[j-3]

            
        state = np.vstack((state0, state1, state2, state3))
        cycles, = np.where((state[0,:]==init[0]) & (state[1,:]==init[1]) &
                            (state[2,:]==init[2]) & (state[3,:]==init[3]))
        if len(cycles) > 1:
            period = cycles[1] - cycles[0]
        else:
            period = None

    return np.asarray(sd), period


def L_mash_dB(m, fref, n=1.0):
    """ Phase noise theoretical value of noise produced by a mash111 SDM

    This procedure calculates the noise at the output of the SD modulator

    Parameters
    ----------
    m : int
        The order of the SD modulator
    fref : float
        Reference frequency that is used to compare the output of the SD
        modulator
    n: float
        It is the average division ratio.

    return
    ------
    ldbc : Pnoise
        Return a function object of the noise
    """
    func_ldbc = lambda fm : 10 * log10((2 * pi) ** 2 / (12 * fref) *
        (2 * sin(pi * fm / fref)) ** (2 * (m - 1)) / n ** 2)
    ldbc = Pnoise.with_function(func_ldbc, fc=fref, label='sdm')
    return ldbc
