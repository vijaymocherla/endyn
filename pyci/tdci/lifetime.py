#!/usr/bin/env python
# 
# Author : Sai Vijay Mocherla <vijaysai.mocherla@gmail.com>
# 
""" heuristic.py 
A python module for calculating lifetime for many-body states above a certain
ionization threshold in configuration interaction calculations using the 
heuristic lifetime model proposed by Klinkusch et al as presented in [1].

Ref:
[1] Klinkusch, S., Saalfrank, P., & Klamroth, T.
J. Chem. Phys. 131, 114304 (2009); https://doi.org/10.1063/1.3218847
[2] Woźniak, A. P., Przybytek, M., Lewenstein, M., et al.
J. Chem. Phys. 156, 174106 (2022); https://doi.org/10.1063/5.0087384
"""

import numpy as np


class heuristic:
    def __init__(self, eigvals, eigvecs, csfs, mo_eps):
        self.eigvals = eigvals
        self.eigvecs = eigvecs
        self.csfs = csfs
        self.mo_eps = mo_eps
    
    def gamma(self, x, d1):
        gamma = 0
        if x!=0 and self.mo_eps[x] >= 0.0 :
            gamma = np.sqrt(2*self.mo_eps[x])/d1
        return gamma

    def lifetime(self, idx, d1): 
        eigvec = self.eigvecs[idx]
        lifetime = np.sum([(eigvec[idx]**2 * self.gamma(csf[2], d1) 
                          + eigvec[idx]**2 * self.gamma(csf[3], d1))
                            for idx, csf in enumerate(self.csfs)])    
        return lifetime

    def cmplx_energies(self, params):
        E0, w0, IP = params
        Ecut = IP + 3.17*(E0/(4*w0))**2
        d1, d2 = E0/w0**2, 0.1
        threshold = self.eigvals[0] + IP
        cmplx_energies = []
        for idx, Ei in enumerate(self.eigvals):
            if Ei >= threshold:   
                if Ei > Ecut:
                    cmplx_energies.append(Ei - 0.5j*self.lifetime(idx, d2))
                else:    
                    cmplx_energies.append(Ei - 0.5j*self.lifetime(idx, d1))
            else:
                cmplx_energies.append(Ei)
        return cmplx_energies

