#!/usr/bin/env python
#
#
"""units.py
"""

import numpy as np
import scipy.constants as const


c = const.value('speed of light in vacuum')

fs_to_au = 41.341374575751

def fwhm(w0, n):
    sigma_fs = n * freq_to_wavelength(w0)/c *1e15
    sigma = sigma_fs * fs_to_au
    return sigma

def watt_per_cm2_to_au(I):
        eps0 = const.value('vacuum electric permittivity')
        c = const.value('speed of light in vacuum')
        I = I*1e4 # I in W/cm-2 * 1e4 
        E0 = np.sqrt(I/(0.5*eps0*c)) /const.value('atomic unit of electric field')
        # print('E0 : {E0:10.16f}'.format(E0=E0))
        return E0

def freq_to_wavelength(freq):
    wavelength = 2*np.pi*c/freq * 1/(1e15*fs_to_au)
    return wavelength

def wavelength_to_freq(wavelength):
    freq = 2*np.pi*c/wavelength * 1/(1e15*fs_to_au)
    return freq

# const.value('atomic unit of time')
# const.value('vacuum electric permittivity')
# const.value('atomic unit of permittivity')
# const.value('atomic mass unit-inverse meter relationship')
# const.value('atomic unit of electric field')
