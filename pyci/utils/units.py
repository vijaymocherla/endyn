#!/usr/bin/env python
#
#
"""units.py
"""

import numpy as np
import scipy.constants as const

const.value('vacuum electric permittivity')
const.value('atomic unit of permittivity')
const.value('atomic mass unit-inverse meter relationship')
const.value('atomic unit of time')
const.value('atomic unit of electric field')

c = const.value('speed of light in vacuum')

fs_to_au = 41.341374575751

def freq_to_wavelength(freq):
    wavelength = 2*np.pi*c/freq * 1/(1e15*fs_to_au)
    return wavelength

def wavelength_to_freq(wavelength):
    freq = 2*np.pi*c/wavelength * 1/(1e15*fs_to_au)
    return freq
