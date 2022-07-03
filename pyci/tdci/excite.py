#!/usr/bin/env python
#
# Author : Sai Vijay Mocherla <vijaysai.mocherla@gmail.com>
#
"""excite.py

"""

import numpy as np
 
def sin2_pulse(t, params):
    E0, w0, ti, tf, sigma, phase = params
    E = lambda t : E0 * np.sin(np.pi/(2*sigma)*t)**2 * np.sin(w0*t + phase) 
    if t >= ti and t <= tf:
        sin2_pulse = E(t)
    else: 
        sin2_pulse = 0
    return sin2_pulse

def gaussian_pulse(t, params):
    E0, w0, ti, tf, sigma, phase = params
    tc = (ti+tf)/2
    E = lambda t : E0 * np.exp(-(t - tc)**2/(2*sigma**2)) * np.sin(w0*t + phase)
    if t >= ti and t <= tf:
        gaussian_pulse = E(t)
    else: 
        gaussian_pulse = 0
    return gaussian_pulse
