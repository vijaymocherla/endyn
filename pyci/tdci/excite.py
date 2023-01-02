#!/usr/bin/env python
#
# Author : Sai Vijay Mocherla <vijaysai.mocherla@gmail.com>
#
"""excite.py

"""

import numpy as np
from pyci.utils.units import fwhm

def sin2_pulse(t, params):
    E0, w0, ti, n, phase = params
    sigma = fwhm(w0, n)
    E = lambda t : E0 * np.sin(np.pi/(2*sigma)*t)**2 * np.sin(w0*t + phase) 
    if t >= ti and t <= 2*sigma:
        sin2_pulse = E(t)
    else: 
        sin2_pulse = 0
    return sin2_pulse

def gaussian_pulse(t, params):
    E0, w0, ti, n, phase = params
    sigma = fwhm(w0, n)
    E = lambda t : E0 * np.exp(-4*np.log(2)*((t-sigma)/sigma)**2) * np.sin(w0*t + phase)
    if t >= ti and t <= 2*sigma:
        gaussian_pulse = E(t)
    else: 
        gaussian_pulse = 0
    return gaussian_pulse