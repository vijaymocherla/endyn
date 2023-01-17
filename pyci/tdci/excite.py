#!/usr/bin/env python
#
# Author : Sai Vijay Mocherla <vijaysai.mocherla@gmail.com>
#
"""excite.py

"""

import numpy as np
from pyci.utils.units import fwhm

def sinesqr_pulse(t, params):
    E0, w0, tp, n, phase = params
    sigma = fwhm(w0, n)
    if t >= (tp-sigma) and t < tp: 
        pulse =  E0 * np.sin(np.pi/2*((t-tp)/sigma + 1))**2 * np.sin(w0*(t-tp) + phase) 
    elif t > tp and t <= (tp+sigma):
        pulse =  E0 * np.sin(np.pi/2*((t-tp)/sigma + 1))**2 * np.sin(w0*(t-tp) + phase) 
    else: 
        pulse = 0.0
    return pulse

def gaussian_pulse(t, params):
    E0, w0, tp, n, phase = params
    sigma = fwhm(w0, n)
    if t >= (tp-sigma) and t < tp: 
        pulse =  E0 * np.exp(-4*np.sqrt(np.log(2))*((t-tp)/sigma)**2) * np.sin(w0*(t-tp) + phase) 
    elif t > tp and t <= (tp+sigma):
        pulse =  E0 * np.exp(-4*np.sqrt(np.log(2))*((t-tp)/sigma)**2) * np.sin(w0*(t-tp) + phase) 
    else: 
        pulse = 0.0
    return pulse

def trapezoidal_pulse(t, params):
    E0, w0, tp, n, phase = params 
    sigma = fwhm(w0, n)
    tm = (2*fwhm(w0, 3))
    slope = 1.0/tm
    if t > (tp-sigma) and t <= (tp-sigma)+tm:
        pulse = E0 * np.sin(w0*(t-tp) + phase) * slope* (t-(tp-sigma))
    elif t > (tp-sigma) + 0.2*sigma and t < (tp+sigma) - tm:
        pulse = E0 * np.sin(w0*(t-tp) + phase) * 1.0
    elif t > (tp+sigma)-tm and t < (tp+sigma):
        pulse = E0 * np.sin(w0*(t-tp) + phase) * slope*((tp+sigma)-t)        
    else:
        pulse = 0.0
    return pulse