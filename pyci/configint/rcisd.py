#!/usr/bin/env python
#
# Author : Sai Vijay Mocherla <vijaysai.mocherla@gmail.com>
#
""" rcisd.py
A python program with implementation of CISD in terms of CSF basis.

References:
[1] Supplimentary Material of "Effects of electronic correlation on the 
    high harmonic generation in helium: a time-dependent 
    configuration interaction singles vs time-dependent 
    full configuration interaction study." 
    Woźniak, A. P., Przybytek, M., Lewenstein, M., & Moszyński, R. (2022). 
    J.Chem.Phys., 156(17), 174106.
"""
import numpy as np

from functools import partial
from multiprocessing import Pool

# input : eps, Ca, ao_oeints, ao_erints
# methods : 
#   1. Transform AO to MO basis
#   2. Options: singles, doubles, full_cis and active space.
#   3. Calculate operators in CSF basis using explicit formulaes from ref[1]
#       a. calculate OEPROPs(dipoles, charges etc)
#       b. Hamiltonian 
#       c. parallelise routines   
#   4. Caculate 1-RDM from a state in CSF basis
#
# singles = True
# doubles = True
# full_cis = True
# orbinfo = (16, 45, 45)
# active_space = (5,37)


def generate_csfs(orbinfo, active_space, options):
    singles, full_cis, doubles = options
    nel, nbf, nmo = orbinfo
    nocc, nvir = int(nel/2), int((nmo-nel)/2)
    act_occ, act_vir = active_space
    occ_list = range(nocc-act_occ, nocc)
    vir_list = range(nocc, nocc+act_vir)
    csfs = [(0,0,0,0)]
    num_csfs = [1,0,0,0,0,0,0]
    if singles:
        if full_cis:
            D_ia = [(i,0,a,0) for i in range(0,nocc) for a in range(nocc,nmo)]
        else:    
            D_ia = [(i,0,a,0) for i in occ_list for a in vir_list]
        csfs += D_ia
        num_csfs[1] = len(D_ia)
    if doubles:
        D_iiaa = [(i,i,a,a) for i in occ_list for a in vir_list]
        csfs += D_iiaa
        num_csfs[2] = len(D_iiaa)
        D_iiab = [(i,i,a,b) for i in occ_list for a in vir_list for b in vir_list if a!=b]
        csfs += D_iiab
        num_csfs[3] = len(D_iiab)
        D_ijaa = [(i,j,a,a) for i in occ_list for j in occ_list for a in vir_list if i!=j]
        csfs += D_ijaa
        num_csfs[4] = len(D_ijaa)
        D_ijab = [(i,j,a,b) for i in occ_list for j in occ_list 
                          for a in vir_list for b in vir_list if i!=j and a!=b]
        D_ijab_A = D_ijab_B = D_ijab
        csfs += D_ijab_A
        num_csfs[5] = len(D_ijab_A)
        csfs += D_ijab_B
        num_csfs[6] = len(D_ijab_B)
    return csfs, num_csfs

H = np.zeros((N,N))
def comp_hrow_hf(mo_eps, mo_eris, csfs, num_csfs, options, params):
    singles, full_cis, doubles = options
    nocc, nvirt, N, E0 = params
    row = np.zeros(N)
    i,j,a,b = csfs[P]
    row[0] = E0
    Q = 1
    if doubles:
        # then do doubles
        n_iiaa = num_csfs[2]
        n_iiab = num_csfs[3]
        n_ijaa = num_csfs[4]
        n_ijab_A = num_csfs[5]
        n_ijab_B = num_csfs[6]
        for right_ex in csfs[Q:Q+n_iiaa]:
            k,l,c,d = right_ex
            row[Q] = mo_eris[k,l,k,l]
            Q += 1
        for right_ex in csfs[Q:Q+n_iiab]:
            k,l,c,d = right_ex
            row[Q] = np.sqrt(2)*mo_eris[c,k,d,k]
            Q += 1 
        for right_ex in csfs[Q:Q+n_ijaa]:
            k,l,c,d = right_ex
            row[Q] = np.sqrt(2)*mo_eris[c,k,c,l]
            Q += 1
        for right_ex in csfs[Q:Q+n_ijab_A]:
            # A 
            k,l,c,d = right_ex
            row[Q] = np.sqrt(3)*(mo_eris[c,k,d,l] - mo_eris[c,l,d,k])
            Q += 1
        for  right_ex in csfs[Q:Q+n_ijab_B]:
            # B 
            k,l,c,d = right_ex
            row[Q] = mo_eris[c,k,d,l] + mo_eris[c,l,d,k]
            Q += 1
    if Q != N:
        raise Exception("ERROR: posval not equal nCSFs")
    return row

# calculate rows for singles csf
def comp_hrow_ia(mo_eps, mo_eris, csfs, num_csfs, options, params, P):
    singles, full_cis, doubles = options
    nocc, nvirt, N, E0 = params
    row = np.zeros(N)
    i,j,a,b = csfs[P]
    row[0] = 0.0
    Q = 1
    n_ia = num_csfs[1]    
    for right_ex in csfs[Q:n_ia]:
        k,l,c,d = right_ex
        row[Q] = ((i==k)*(a==c)*(E0 - mo_eps[i]+ mo_eps[a])
                            +2*mo_eris[a,i,c,k] - mo_eris[c,a,k,i]) 
        Q += 1
    if doubles:
        # then do doubles
        n_iiaa = num_csfs[2]
        n_iiab = num_csfs[3]
        n_ijaa = num_csfs[4]
        n_ijab_A = num_csfs[5]
        n_ijab_B = num_csfs[6]
        for right_ex in csfs[Q:Q+n_iiaa]:
            k,l,c,d = right_ex
            row[Q] = np.sqrt(2) * ((i==k)*mo_eris[c,a,c,i]
                                        - (a==c)*mo_eris[k,a,k,i])
            Q += 1
        for right_ex in csfs[Q:Q+n_iiab]:
            k,l,c,d = right_ex
            row[Q] =  ((i==k)*(mo_eris[d,a,c,i] + mo_eris[c,a,d,i])
                            - (a==c)*mo_eris[k,d,k,i]
                            - (a==d)*mo_eris[k,c,k,i])
            Q += 1 
        for right_ex in csfs[Q:Q+n_ijaa]:
            k,l,c,d = right_ex
            row[Q] = ((i==k)*mo_eris[c,a,c,l]
                        + (i==l)*mo_eris[c,a,c,k]
                        + (a==c)*(mo_eris[a,l,k,i] + mo_eris[a,k,l,i]))
            Q += 1
        for right_ex in csfs[Q:Q+n_ijab_A]:
            # A 
            k,l,c,d = right_ex
            row[Q] = np.sqrt(1.5) * ((i==k)*(mo_eris[a,c,d,l] - mo_eris[a,d,c,l])
                                -(i==l)*(mo_eris[a,c,d,k] - mo_eris[a,d,c,k])
                                +(a==c)*(mo_eris[d,k,l,i] - mo_eris[d,l,k,i])
                                -(a==d)*(mo_eris[c,k,l,i] - mo_eris[c,l,k,i]))
            Q += 1
        for  right_ex in csfs[Q:Q+n_ijab_B]:
            # B 
            k,l,c,d = right_ex
            row[Q] = np.sqrt(0.5) * ((i==k)*(mo_eris[a,c,d,l]- mo_eris[a,d,c,l])
                                +(i==l)*(mo_eris[a,c,d,k] - mo_eris[a,d,c,k])
                                -(a==c)*(mo_eris[d,k,l,i] - mo_eris[d,l,k,i])
                                -(a==d)*(mo_eris[c,k,l,i] - mo_eris[c,l,k,i]))
            Q += 1
    if Q != N:
        raise Exception("ERROR: posval not equal nCSFs")
    return row
# calculate rows for doubles csf
def comp_hrow_iiaa(mo_eps, mo_eris, csfs, num_csfs, options, params, P):
    singles, full_cis, doubles = options
    nocc, nvirt, N, E0 = params
    row = np.zeros(N)
    i,j,a,b = csfs[P]
    row[0] = mo_eris[a,i,a,i]
    Q = 1
    if singles:
        n_ia = num_csfs[1]    
        for right_ex in csfs[Q:n_ia]:
            k,l,c,d = right_ex
            row[Q] = np.sqrt(2) * ((k==i)*mo_eris[a,c,a,k]
                                        - (c==a)*mo_eris[i,c,i,k])
            Q += 1
    n_iiaa = num_csfs[2]
    n_iiab = num_csfs[3]
    n_ijaa = num_csfs[4]
    n_ijab_A = num_csfs[5]
    n_ijab_B = num_csfs[6]
    for right_ex in csfs[Q:Q+n_iiaa]:
        k,l,c,d = right_ex
        row[Q] = ((i==k)*(a==c) *(E0- 2*mo_eps[i] + 2*mo_eps[a] - 4*mo_eris[a,a,i,i] +2*mo_eris[a,i,a,i]) 
                    + (i==k)*mo_eris[c,a,c,a]
                    - (a==c)*mo_eris[k,i,k,i])
        Q += 1
    for right_ex in csfs[Q:Q+n_iiab]:
        k,l,c,d = right_ex
        row[Q] = np.sqrt(2)*(((i==k)*(a==c)*(mo_eris[a,i,d,i]-mo_eris[a,d,i,i]))
                    + (i==k)*(a==d)*(mo_eris[a,i,c,i] - 2*mo_eris[a,c,i,i])
                    + (i==k)*mo_eris[a,d,a,c])
        Q += 1 
    for right_ex in csfs[Q:Q+n_ijaa]:
        k,l,c,d = right_ex
        row[Q] = np.sqrt(2)*(((i==k)*(a==c)*(mo_eris[a,i,a,l]- 2*mo_eris[a,a,l,i]))
                                + (i==l)*(a==c)*(mo_eris[a,i,a,k] - 2*mo_eris[a,a,k,i])
                                + (a==c)*mo_eris[k,i,l,i])
        Q += 1
    for right_ex in csfs[Q:Q+n_ijab_A]:
        # A 
        k,l,c,d = right_ex
        row[Q] = np.sqrt(3)*((i==k)*(a==c)*mo_eris[a,i,d,l]
                            - (i==k)*(a==d)*mo_eris[a,i,c,l]
                            - (i==l)*(a==c)*mo_eris[a,i,d,k]
                            + (i==l)*(a==d)*mo_eris[a,i,c,k])
        Q += 1
    for  right_ex in csfs[Q:Q+n_ijab_B]:
        # B 
        k,l,c,d = right_ex
        row[Q] = ((i==k)*(a==c)*(mo_eris[a,i,d,l] - 2*mo_eris[a,d,l,i])
                            + (i==k)*(a==d)*(mo_eris[a,i,c,l] - 2*mo_eris[a,c,l,i])
                            + (i==l)*(a==c)*(mo_eris[a,i,d,k] - 2*mo_eris[a,d,k,i])
                            + (i==l)*(a==d)*(mo_eris[a,i,c,k] - 2*mo_eris[a,c,k,i]))
        Q += 1
    if Q != N:
        raise Exception("ERROR: posval not equal nCSFs")
    return row

def comp_hrow_iiab(mo_eps, mo_eris, csfs, num_csfs, options, params, P):
    singles, full_cis, doubles = options
    nocc, nvirt, N, E0 = params
    row = np.zeros(N)
    i,j,a,b = csfs[P]
    row[0] = mo_eris[a,i,a,i]
    Q = 1
    if singles:
        n_ia = num_csfs[1]    
        for right_ex in csfs[Q:n_ia]:
            k,l,c,d = right_ex
            row[Q] = np.sqrt(2) * ((k==i)*mo_eris[a,c,a,k]
                                        - (c==a)*mo_eris[i,c,i,k])
            Q += 1
    n_iiaa = num_csfs[2]
    n_iiab = num_csfs[3]
    n_ijaa = num_csfs[4]
    n_ijab_A = num_csfs[5]
    n_ijab_B = num_csfs[6]
    for right_ex in csfs[Q:Q+n_iiaa]:
        k,l,c,d = right_ex
        row[Q] =  
        Q += 1
    for right_ex in csfs[Q:Q+n_iiab]:
        k,l,c,d = right_ex
        row[Q] = ((i==k)*(a==c)*(b==d)*(E0 - 2*mo_eps[i] + mo_eps[a] + mo_eps[b])
                    + (i==k)*(a==c)*(mo_eris[b,i,d,i] - 2*mo_eris[b,d,i,i])
                    + (i==k)*(a==d)*(mo_eris[b,i,c,i] - 2*mo_eris[b,c,i,i])
                    + (i==k)*(b==c)*(mo_eris[a,i,d,i] - 2*mo_eris[a,d,i,i])
                    + (i==k)*(b==d)*(mo_eris[a,i,c,i] - 2*mo_eris[a,c,i,i])
                    + (i==k)*(mo_eris[a,c,b,d] + mo_eris[a,d,b,c])
                    + (a==c)*(b==d)*(mo_eris[k,i,k,i]))
        Q += 1 
    for right_ex in csfs[Q:Q+n_ijaa]:
        k,l,c,d = right_ex
        row[Q] = ((i==k)*(a==c)*(mo_eris[a,l,b,i] - 2*mo_eris[a,b,l,i])
                   +(i==k)*(b==c)*(mo_eris[b,l,a,i] - 2*mo_eris[b,a,l,i])
                   +(i==l)*(a==c)*(mo_eris[a,k,b,i] - 2*mo_eris[a,b,k,i])
                   +(i==l)*(b==c)*(mo_eris[b,k,a,i] - 2*mo_eris[b,a,k,i]))
        Q += 1
    for right_ex in csfs[Q:Q+n_ijab_A]:
        # A 
        k,l,c,d = right_ex
        row[Q] = np.sqrt(1.5)*((i==k)*(a==c)*mo_eris[b,i,d,l] 
                               - (i==k)*(a==d)*mo_eris[b,i,c,l]
                               + (i==k)*(b==c)*mo_eris[a,i,d,l]
                               - (i==k)*(b==d)*mo_eris[a,i,c,l]
                               - (i==k)*(a==c)*mo_eris[b,i,d,k]
                               + (i==l)*(a==d)*mo_eris[b,i,c,k]
                               - (i==l)*(b==c)*mo_eris[a,i,d,k]
                               + (i==l)*(b==d)*mo_eris[a,i,c,k])

        Q += 1
    for  right_ex in csfs[Q:Q+n_ijab_B]:
        # B 
        k,l,c,d = right_ex
        row[Q] = np.sqrt(0.5)*((i==k)*(a==c)*(mo_eris[b,i,d,l]- 2*mo_eris[b,d,l,i])
                                +(i==k)*(a==d)*(mo_eris[b,i,c,l]- 2*mo_eris[b,c,l,i])
                                +(i==k)*(b==c)*(mo_eris[a,i,d,l]- 2*mo_eris[a,d,l,i])
                                +(i==k)*(b==d)*(mo_eris[a,i,c,l]- 2*mo_eris[a,c,l,i])
                                +(i==l)*(a==c)*(mo_eris[b,i,d,k]- 2*mo_eris[b,d,k,i])
                                +(i==l)*(a==d)*(mo_eris[b,i,c,k]- 2*mo_eris[b,c,k,i])
                                +(i==l)*(b==c)*(mo_eris[a,i,d,k]- 2*mo_eris[a,d,k,i])
                                +(i==l)*(b==d)*(mo_eris[a,i,c,k]- 2*mo_eris[a,c,k,i])
                                +(a==c)*(b==d)*2*mo_eris[k,i,l,i])
        Q += 1
    if Q != N:
        raise Exception("ERROR: posval not equal nCSFs")
    return row
# im here
def comp_hrow_ijaa(mo_eps, mo_eris, csfs, num_csfs, options, params, P):
    singles, full_cis, doubles = options
    nocc, nvirt, N, E0 = params
    row = np.zeros(N)
    i,j,a,b = csfs[P]
    row[0] = mo_eris[a,i,a,i]
    Q = 1
    if singles:
        n_ia = num_csfs[1]    
        for right_ex in csfs[Q:n_ia]:
            k,l,c,d = right_ex
            row[Q] =((k==i)*(mo_eris[b,c,a,k] + mo_eris[a,c,b,k])
                            - (c==a)*mo_eris[i,b,i,k]
                            - (c==b)*mo_eris[i,a,i,k])
            Q += 1
    n_iiaa = num_csfs[2]
    n_iiab = num_csfs[3]
    n_ijaa = num_csfs[4]
    n_ijab_A = num_csfs[5]
    n_ijab_B = num_csfs[6]
    for right_ex in csfs[Q:Q+n_iiaa]:
        k,l,c,d = right_ex
        row[Q] = 
        Q += 1
    for right_ex in csfs[Q:Q+n_iiab]:
        k,l,c,d = right_ex
        row[Q] =
        Q += 1 
    for right_ex in csfs[Q:Q+n_ijaa]:
        k,l,c,d = right_ex
        row[Q] =
        Q += 1
    for right_ex in csfs[Q:Q+n_ijab_A]:
        # A 
        k,l,c,d = right_ex
        row[Q] =
        Q += 1
    for  right_ex in csfs[Q:Q+n_ijab_B]:
        # B 
        k,l,c,d = right_ex
        row[Q] =
        Q += 1
    if Q != N:
        raise Exception("ERROR: posval not equal nCSFs")
    return row

def comp_hrow_ijab_A(mo_eps, mo_eris, csfs, num_csfs, options, params, P):
    singles, full_cis, doubles = options
    nocc, nvirt, N, E0 = params
    row = np.zeros(N)
    i,j,a,b = csfs[P]
    row[0] = mo_eris[a,i,a,i]
    Q = 1
    if singles:
        n_ia = num_csfs[1]    
        for right_ex in csfs[Q:n_ia]:
            k,l,c,d = right_ex
            row[Q] = np.sqrt(2) * ((k==i)*mo_eris[a,c,a,k]
                                        - (c==a)*mo_eris[i,c,i,k])
            Q += 1
    n_iiaa = num_csfs[2]
    n_iiab = num_csfs[3]
    n_ijaa = num_csfs[4]
    n_ijab_A = num_csfs[5]
    n_ijab_B = num_csfs[6]
    for right_ex in csfs[Q:Q+n_iiaa]:
        k,l,c,d = right_ex
        row[Q] = 
        Q += 1
    for right_ex in csfs[Q:Q+n_iiab]:
        k,l,c,d = right_ex
        row[Q] =
        Q += 1 
    for right_ex in csfs[Q:Q+n_ijaa]:
        k,l,c,d = right_ex
        row[Q] =
        Q += 1
    for right_ex in csfs[Q:Q+n_ijab_A]:
        # A 
        k,l,c,d = right_ex
        row[Q] =
        Q += 1
    for  right_ex in csfs[Q:Q+n_ijab_B]:
        # B 
        k,l,c,d = right_ex
        row[Q] =
        Q += 1
    if Q != N:
        raise Exception("ERROR: posval not equal nCSFs")
    return row

def comp_hrow_ijab_B(mo_eps, mo_eris, csfs, num_csfs, options, params, P):
    singles, full_cis, doubles = options
    nocc, nvirt, N, E0 = params
    row = np.zeros(N)
    i,j,a,b = csfs[P]
    row[0] = mo_eris[a,i,a,i]
    Q = 1
    if singles:
        n_ia = num_csfs[1]    
        for right_ex in csfs[Q:n_ia]:
            k,l,c,d = right_ex
            row[Q] = np.sqrt(2) * ((k==i)*mo_eris[a,c,a,k]
                                        - (c==a)*mo_eris[i,c,i,k])
            Q += 1
    n_iiaa = num_csfs[2]
    n_iiab = num_csfs[3]
    n_ijaa = num_csfs[4]
    n_ijab_A = num_csfs[5]
    n_ijab_B = num_csfs[6]
    for right_ex in csfs[Q:Q+n_iiaa]:
        k,l,c,d = right_ex
        row[Q] = 
        Q += 1
    for right_ex in csfs[Q:Q+n_iiab]:
        k,l,c,d = right_ex
        row[Q] =
        Q += 1 
    for right_ex in csfs[Q:Q+n_ijaa]:
        k,l,c,d = right_ex
        row[Q] =
        Q += 1
    for right_ex in csfs[Q:Q+n_ijab_A]:
        # A 
        k,l,c,d = right_ex
        row[Q] =
        Q += 1
    for  right_ex in csfs[Q:Q+n_ijab_B]:
        # B 
        k,l,c,d = right_ex
        row[Q] =
        Q += 1
    if Q != N:
        raise Exception("ERROR: posval not equal nCSFs")
    return row


def comp_hcisd():
    hrow_hf = comp_hrow_hf(mo_eps, mo_eris, csfs, num_csfs, options)
    hrow_ia = partial(comp_hrow_ia, mo_eps, mo_eris, csfs, num_csfs, options, params)
    hrow_iiaa = partial(comp_hrow_iiaa, mo_eps, mo_eris, csfs, num_csfs, options, params)
    hrow_iiab = partial(comp_hrow_iiab, mo_eps, mo_eris, csfs, num_csfs, options, params)
    hrow_ijaa = partial(comp_hrow_ijaa, mo_eps, mo_eris, csfs, num_csfs, options, params)
    hrow_ijab_A = partial(comp_hrow_ijab_A, mo_eps, mo_eris, csfs, num_csfs, options, params)
    hrow_ijab_B = partial(comp_hrow_ijab_B, mo_eps, mo_eris, csfs, num_csfs, options, params)
    hf_row = []
    single_rows = []
    double_rows = []
    if singles:
        # compute singles
    if doubles:
        # compute doubles
    return hcisd
