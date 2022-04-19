#!/usr/bin/env python
#
# Author : Sai Vijay Mocherla <vijaysai.mocherla@gmail.com>
#
""" A Module for generating configuration state functions(CSFs)
"""

from pyci.configint.bitstrings import bitDet, SlaterCondon

class CSF(object):
    """A class construct for configuration state functions
    """
    def __init__(self, csf_dict, spin=0):
        self.Dets = csf_dict.keys()
        self.coeff = csf_dict.values()
        self.Ns = 2*spin + 1

def gen_singlet_singles(refDet: type(bitDet()), occ_list, vir_list, csf_list=[], exc_list=[]):
    """Generates single excited spin-singlet CSFs
    """
    D_ar = [(a,r) for a in occ_list for r in vir_list]
    # ^{1}\Psi_{a}^{r}
    for exc in D_ar:
        a,r = exc
        det1 = refDet.copy()
        det1.remove_alpha(a), det1.add_alpha(r)
        det2 = refDet.copy()
        det2.remove_beta(a), det2.add_beta(r)
        csf_list.append(CSF({det1: 0.7071067811865475, 
                             det2: 0.7071067811865475}))
        exc_list.append(exc)
    return csf_list, exc_list


def gen_singlet_doubles(refDet: type(bitDet()), occ_list, vir_list, csf_list=[], exc_list=[]):
    """Generates double excited spin-singlet CSFs
    """
    D_aarr = [(a,a,r,r) for a in occ_list for r in vir_list]
    D_abrr = [(a,b,r,r) for a in occ_list for b in occ_list for r in vir_list if a!=b]
    D_aars = [(a,a,r,s) for a in occ_list for r in vir_list for s in vir_list if r!=s]
    D_abrs = [(a,b,r,s) for a in occ_list for b in occ_list 
                        for r in vir_list for s in vir_list if a!=b and r!=s]
    # ^{1}\Psi_{a,a}^{r,r}
    for exc in D_aarr:
        a,ax,r,rx = exc
        det1 = refDet.copy()
        det1.remove_alpha(a), det1.add_alpha(r)
        det1.remove_beta(a), det1.add_beta(r)
        csf_list.append(CSF({det1 : 1.0}))
        exc_list.append(exc)
    # ^{1}\Psi_{a,a}^{r,s}
    for exc in D_aars:
        a,ax,r,s = exc
        det1 = refDet.copy()
        det1.remove_alpha(a), det1.add_alpha(r)
        det1.remove_beta(a), det1.add_beta(s)
        det2 = refDet.copy()
        det2.remove_alpha(a), det1.add_alpha(s)
        det2.remove_beta(a), det1.add_beta(r)
        csf_list.append(CSF({det1: 0.7071067811865475, 
                            det2: 0.7071067811865475}))
        exc_list.append(exc)
    # ^{1}\Psi_{a,b}^{r,r}
    for exc in D_abrr:
        a,b,r,rx = exc
        det1 = refDet.copy()
        det1.remove_alpha(a), det1.add_alpha(r)
        det1.remove_beta(b), det1.add_beta(r)
        det2 = refDet.copy()
        det2.remove_alpha(b), det1.add_alpha(r)
        det2.remove_beta(a), det1.add_beta(r)
        csf_list.append(CSF({det1: 0.7071067811865475, 
                            det2: 0.7071067811865475}))
        exc_list.append(exc)
    # ^{A}\Psi_{a,b}^{r,s}
    for exc in D_abrs:
        a,b,r,s = exc
        det1 = refDet.copy()
        det1.remove_alpha(a), det1.add_alpha(r)
        det1.remove_alpha(b), det1.add_alpha(s)
        det2 = refDet.copy()
        det2.remove_beta(a), det1.add_beta(r)
        det2.remove_beta(b), det1.add_beta(s)
        det3 = refDet.copy()
        det3.remove_alpha(a), det1.add_alpha(r)
        det3.remove_beta(b), det1.add_beta(s)
        det4 = refDet.copy()
        det4.remove_alpha(b), det1.add_alpha(s)
        det4.remove_beta(a), det1.add_beta(r)
        det5 = refDet.copy()
        det5.remove_alpha(a), det1.add_alpha(s)
        det5.remove_beta(b), det1.add_beta(r)
        det6 = refDet.copy()
        det6.remove_alpha(b), det1.add_alpha(s)
        det6.remove_beta(a), det1.add_beta(r)
        csf_list.append(CSF({det1: +0.5773502691896258, 
                            det2: +0.5773502691896258,
                            det3: +0.2886751345948129, 
                            det4: +0.2886751345948129,
                            det5: -0.2886751345948129, 
                            det6: -0.2886751345948129}))
        exc_list.append(exc)
    # ^{B}\Psi_{a,b}^{r,s}
    for exc in D_abrs:
        a,b,r,s = exc
        det1 = refDet.copy()
        det1.remove_alpha(a), det1.add_alpha(r)
        det1.remove_beta(b), det1.add_beta(s)
        det2 = refDet.copy()
        det2.remove_alpha(a), det1.add_alpha(s)
        det2.remove_beta(b), det1.add_beta(r)
        det3 = refDet.copy()
        det3.remove_alpha(b), det1.add_alpha(r)
        det3.remove_beta(a), det1.add_beta(s)
        det4 = refDet.copy()
        det4.remove_alpha(b), det1.add_alpha(s)
        det4.remove_beta(a), det1.add_beta(r)
        csf_list.append(CSF({det1: 0.50, 
                            det2: 0.50,
                            det3: 0.50, 
                            det4: 0.50}))
        exc_list.append(exc)
    return csf_list, exc_list
