#!/usr/bin/env python
#
# Author : Sai Vijay Mocherla <vijaysai.mocherla@gmail.com>
#
""" CIS(D) Spin MO version
"""
from itertools import product
from functools import partial
from opt_einsum import contract 
from pyci.utils.multproc import pool_jobs
import numpy as np


def comp_delta(mo_eps, nel, nso):
    """Computes the tensor delta_{ab}^{rs} with orbital energy differences
    """
    occ_list = range(nel)
    vir_list = range(nel, nso)
    so_mo_eps = np.kron(mo_eps, np.array([1.0, 1.0]))
    delta = np.full(([nso]*4), 1.0)
    iterlist = list(product(vir_list, vir_list, occ_list, occ_list))
    for conf in iterlist:
        a, b, i, j = conf
        delta[conf] = so_mo_eps[a] + so_mo_eps[b] - so_mo_eps[i] - so_mo_eps[j]
    delta = delta[nel:, nel:, :nel, :nel]  # reshaping delta to vvoo
    return delta

def comp_e0mp2(w_tensor, g_tensor, nel):
    """Computes MP2 correction for ground-state energy
    """
    E0_mp2 = 0.25*contract('abij,ijab',
                        w_tensor, g_tensor[:nel, :nel, nel:, nel:],
                        optimize=True)
    return E0_mp2

def comp_utensor(g_tensor, nel, Ck):
    """Computes U_{ab}^{rs} tensor for each SA-CIS state
    """
    utensor = (contract('abcj,ic->abij', g_tensor[nel:, nel:, nel:, :nel], Ci, optimize=True)
             - contract('abci,jc->abij', g_tensor[nel:, nel:, nel:, :nel], Ci, optimize=True)
             + contract('kaij,kb->abij', g_tensor[:nel, nel:, :nel, :nel], Ci, optimize=True)
             - contract('kbij,ka->abij', g_tensor[:nel, nel:, :nel, :nel], Ci, optimize=True))
    return utensor

def comp_varray(w_tensor, g_tensor, nel, Ck):
    """Computes V_{a}^{r} array for each SA-CIS state
    """
    term1 = contract('jkbc,ib,cajk->ia', g_tensor[:nel, :nel, nel:, nel:], Ci, w_tensor, optimize=True)
    term2 = contract('jkbc,ja,cbik->ia', g_tensor[:nel, :nel, nel:, nel:], Ci, w_tensor, optimize=True)
    term3 = contract('jkbc,jb,acik->ia', g_tensor[:nel, :nel, nel:, nel:], Ci, w_tensor, optimize=True)
    varray = 0.5*(term1 + term2 + (2*term3))
    return varray

def get_idx(occ_idx, vir_idx, nocc, nvir):
    """Get basis index of single det in cis eigenvectors
    """
    idx = (occ_idx*nvir) + (vir_idx-nocc)
    return idx

def spin_block_ci(Ck, iterlist, nocc, nvir):
    """Spin blocks Ci into spin blocked form
    """
    new_Ci = np.zeros((nocc, nvir))
    for state in iterlist: # check later to see the loop can removed by .reshape
        i, a = state
        new_Ci[i, (a-nocc)] = Ci[get_idx(i, a)]
    spin_Ci = np.kron(new_Ci, np.array([[1.0, 0.0], [0.0, 1.0]]))  # spin blocking the cis vector
    return spin_Ci

def comp_dcorr(ecis, ccis, delta, w_tensor, g_tensor, nel, k):
    Ek = ecis[k]
    Ck = ccis[k]
    spin_Ck = spin_block_ci(1/np.sqrt(2) * Ck)
    Uk = comp_utensor(g_tensor, nel, spin_Ck)
    Vk = comp_varray(w_tensor, g_tensor, nel, spin_Ck)
    Ek_d_corr = -0.25*contract('ijab,abij', 
                                Uk.transpose(2,3,0,1), 
                                Uk/(delta-Ek), 
                                optimize=True)
    Ek_t_corr = np.sum(spin_Ck*Vk)
    Ek_corr = Ek_d_corr + Ek_t_corr
    # print('%3.4f \t %3.4f \t %3.4f \t %3.4f' % (Ek_d_corr, Ek_t_corr, Ek_corr, Ek+Ek_corr))
    return Ek_corr

def kernel(ecis, ccis, mo_eps, mo_so_erints, orbinfo, nevals, ncore=4):
    nocc, nmo = orbinfo
    nso = int(2*nmo)
    nel = int(nocc/2)
    delta = comp_delta(mo_eps, nel, nso)
    g_tensor = mo_so_erints - mo_so_erints.transpose(0, 1, 3, 2)
    w_tensor = - g_tensor[nel:, nel:, :nel, :nel]/delta  # g_vvoo / delta[vvoo]
    pfunc_comp_dcorr = partial(comp_dcorr, ecis, ccis, delta, 
                                w_tensor, g_tensor, nel)
    argslist = list(range(nevals))
    E_corr = pool_jobs(pfunc_comp_dcorr, argslist, ncore)
    E_cis_d = ecis[:nevals] + E_corr
    return E_cis_d
