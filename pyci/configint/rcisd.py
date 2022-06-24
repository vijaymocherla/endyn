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

# input : eps, Ca, ao_oeints, ao_eris
# methods : 
#   1. Transform AO to MO basis
#   2. Options: singles, doubles, full_cis and active space.
#   3. Calculate operators in CSF basis using explicit formulaes from ref[1]
#       a. calculate OEPROPs(dipoles, charges etc)
#       b. Hamiltonian 
#       c. parallelise routines   
#   4. Caculate 1-RDM from a state in CSF basis
#

def generate_csfs(orbinfo, active_space, options):
    singles, full_cis, doubles, doubles_options = options
    nocc, nmo = orbinfo
    (doubles_iiaa, doubles_iiab, doubles_ijaa, 
    doubles_ijab_A, doubles_ijab_B) = doubles_options
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
        csfs.extend(D_ia)
        num_csfs[1] = len(D_ia)
    if doubles:
        if doubles_iiaa:
            D_iiaa = [(i,i,a,a) for i in occ_list for a in vir_list]
            csfs.extend(D_iiaa)
            num_csfs[2] = len(D_iiaa)
        if  doubles_iiab:       
            D_iiab = [(i,i,a,b) for i in occ_list for a in vir_list for b in vir_list if a>b]
            csfs.extend(D_iiab)
            num_csfs[3] = len(D_iiab)
        if doubles_ijaa:
            D_ijaa = [(i,j,a,a) for i in occ_list for j in occ_list for a in vir_list if i>j]
            csfs.extend(D_ijaa)
            num_csfs[4] = len(D_ijaa)
        D_ijab = [(i,j,a,b) for i in occ_list for j in occ_list 
                            for a in vir_list for b in vir_list if i>j and a>b]
        if doubles_ijab_A:
            D_ijab_A = D_ijab        
            csfs.extend(D_ijab_A)
            num_csfs[5] = len(D_ijab_A)
        if doubles_ijab_B:        
            D_ijab_B = D_ijab
            csfs.extend(D_ijab_B)
            num_csfs[6] = len(D_ijab_B)
    return csfs, num_csfs

def multproc_comp_rows(pfunc, Plist, ncore=4, run_checks=False):
    n = len(Plist)
    with Pool(processes=ncore) as pool:
        async_object = pool.map_async(pfunc, Plist)
        data = async_object.get()
        pool.close()
        pool.join() 
    rows = [data[i][0] for i in range(n)]
    log = [data[i][1] for i in range(n)]
    if run_checks:
        print('Checking if all process finished succesfully...\n')
        if sum(log) != n:
            for idx, P in enumerate(Plist):
                if log[idx] != 1:
                    print('Something went wrong while computing row %i' % (P))
        else:
            print("All rows were computed successfully! \n")
    return rows, log

def comp_hrow_hf(mo_eps, mo_eris, scf_energy, csfs, num_csfs, options):
    singles, full_cis, doubles, doubles_options = options
    (doubles_iiaa, doubles_iiab, doubles_ijaa, 
    doubles_ijab_A, doubles_ijab_B) = doubles_options
    N = sum(num_csfs)
    E0 = scf_energy
    row = np.zeros(N)
    i,j,a,b = csfs[0]
    try:
        row[0] = E0
        Q = 1
        if singles:
            n_ia = num_csfs[1]
            Q += n_ia
        if doubles:
            # then do doubles
            n_iiaa = num_csfs[2]
            n_iiab = num_csfs[3]
            n_ijaa = num_csfs[4]
            n_ijab_A = num_csfs[5]
            n_ijab_B = num_csfs[6]
            if doubles_iiaa:
                for right_ex in csfs[Q:Q+n_iiaa]:
                    k,l,c,d = right_ex
                    row[Q] = mo_eris[k,l,k,l]
                    Q += 1
            if doubles_iiab:
                for right_ex in csfs[Q:Q+n_iiab]:
                    k,l,c,d = right_ex
                    row[Q] = np.sqrt(2)*mo_eris[c,k,d,k]
                    Q += 1 
            if doubles_ijaa:            
                for right_ex in csfs[Q:Q+n_ijaa]:
                    k,l,c,d = right_ex
                    row[Q] = np.sqrt(2)*mo_eris[c,k,c,l]
                    Q += 1
            if doubles_ijab_A:
                for right_ex in csfs[Q:Q+n_ijab_A]:
                    # A 
                    k,l,c,d = right_ex
                    row[Q] = np.sqrt(3)*(mo_eris[c,k,d,l] - mo_eris[c,l,d,k])
                    Q += 1
            if doubles_ijab_B:
                for  right_ex in csfs[Q:Q+n_ijab_B]:
                    # B 
                    k,l,c,d = right_ex
                    row[Q] = mo_eris[c,k,d,l] + mo_eris[c,l,d,k]
                    Q += 1
        return row, 1
    except :
        raise Exception("Something went wrong while computing row %i"%(0))
        return row, 0

# calculate rows for singles csf
def comp_hrow_ia(mo_eps, mo_eris, scf_energy, csfs, num_csfs, options, P):
    singles, full_cis, doubles, doubles_options = options
    (doubles_iiaa, doubles_iiab, doubles_ijaa, 
    doubles_ijab_A, doubles_ijab_B) = doubles_options
    N = sum(num_csfs)
    E0 = scf_energy
    row = np.zeros(N)
    i,j,a,b = csfs[P]
    try:    
        row[0] = 0.0
        Q = 1
        n_ia = num_csfs[1]    
        for right_ex in csfs[Q:Q+n_ia]:
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
            if doubles_iiaa:
                for right_ex in csfs[Q:Q+n_iiaa]:
                    k,l,c,d = right_ex
                    row[Q] = np.sqrt(2) * ((i==k)*mo_eris[c,a,c,i]
                                                - (a==c)*mo_eris[k,a,k,i])
                    Q += 1
            if doubles_iiab:
                for right_ex in csfs[Q:Q+n_iiab]:
                    k,l,c,d = right_ex
                    row[Q] =  ((i==k)*(mo_eris[d,a,c,i] + mo_eris[c,a,d,i])
                                    - (a==c)*mo_eris[k,d,k,i]
                                    - (a==d)*mo_eris[k,c,k,i])
                    Q += 1
            if doubles_ijaa:
                for right_ex in csfs[Q:Q+n_ijaa]:
                    k,l,c,d = right_ex
                    row[Q] = ((i==k)*mo_eris[c,a,c,l]
                                + (i==l)*mo_eris[c,a,c,k]
                                + (a==c)*(mo_eris[a,l,k,i] + mo_eris[a,k,l,i]))
                    Q += 1
            if doubles_ijab_A:        
                for right_ex in csfs[Q:Q+n_ijab_A]:
                    # A 
                    k,l,c,d = right_ex
                    row[Q] = np.sqrt(1.5) * ((i==k)*(mo_eris[a,c,d,l] - mo_eris[a,d,c,l])
                                        -(i==l)*(mo_eris[a,c,d,k] - mo_eris[a,d,c,k])
                                        +(a==c)*(mo_eris[d,k,l,i] - mo_eris[d,l,k,i])
                                        -(a==d)*(mo_eris[c,k,l,i] - mo_eris[c,l,k,i]))
                    Q += 1
            if doubles_ijab_B:
                for  right_ex in csfs[Q:Q+n_ijab_B]:
                    # B 
                    k,l,c,d = right_ex
                    row[Q] = np.sqrt(0.5) * ((i==k)*(mo_eris[a,c,d,l]- mo_eris[a,d,c,l])
                                        +(i==l)*(mo_eris[a,c,d,k] - mo_eris[a,d,c,k])
                                        -(a==c)*(mo_eris[d,k,l,i] - mo_eris[d,l,k,i])
                                        -(a==d)*(mo_eris[c,k,l,i] - mo_eris[c,l,k,i]))
                    Q += 1
        return row, 1
    except:
        raise Exception("Something went wrong while computing row %i" % (P))
        return row, 0

# calculate rows for doubles csf
def comp_hrow_iiaa(mo_eps, mo_eris, scf_energy, csfs, num_csfs, options, P):
    singles, full_cis, doubles, doubles_options = options
    (doubles_iiaa, doubles_iiab, doubles_ijaa, 
    doubles_ijab_A, doubles_ijab_B) = doubles_options
    N = sum(num_csfs)
    E0 = scf_energy
    row = np.zeros(N)
    i,j,a,b = csfs[P]
    try:
        row[0] = mo_eris[a,i,a,i]
        Q = 1
        if singles:
            n_ia = num_csfs[1]    
            for right_ex in csfs[Q:Q+n_ia]:
                k,l,c,d = right_ex
                row[Q] = np.sqrt(2) * ((k==i)*mo_eris[a,c,a,k]
                                    - (c==a)*mo_eris[i,c,i,k])
                Q += 1
        n_iiaa = num_csfs[2]
        n_iiab = num_csfs[3]
        n_ijaa = num_csfs[4]
        n_ijab_A = num_csfs[5]
        n_ijab_B = num_csfs[6]
        if doubles_iiaa:
            for right_ex in csfs[Q:Q+n_iiaa]:
                k,l,c,d = right_ex
                row[Q] = ((i==k)*(a==c) *(E0 - 2*mo_eps[i] + 2*mo_eps[a] 
                                        - 4*mo_eris[a,a,i,i] + 2*mo_eris[a,i,a,i]) 
                            + (i==k)*mo_eris[c,a,c,a]
                            + (a==c)*mo_eris[k,i,k,i])
                Q += 1
        if doubles_iiab:
            for right_ex in csfs[Q:Q+n_iiab]:
                k,l,c,d = right_ex
                row[Q] = np.sqrt(2)*( (i==k)*(a==c)*(mo_eris[a,i,d,i] - 2*mo_eris[a,d,i,i])
                                    + (i==k)*(a==d)*(mo_eris[a,i,c,i] - 2*mo_eris[a,c,i,i])
                                    + (i==k)*mo_eris[a,d,a,c])
                Q += 1
        if doubles_ijaa: 
            for right_ex in csfs[Q:Q+n_ijaa]:
                k,l,c,d = right_ex
                row[Q] = np.sqrt(2)*( (i==k)*(a==c)*(mo_eris[a,i,a,l] - 2*mo_eris[a,a,l,i])
                                    + (i==l)*(a==c)*(mo_eris[a,i,a,k] - 2*mo_eris[a,a,k,i])
                                    + (a==c)*mo_eris[k,i,l,i])
                Q += 1
        if doubles_ijab_A:
            for right_ex in csfs[Q:Q+n_ijab_A]:
                # A 
                k,l,c,d = right_ex
                row[Q] = np.sqrt(3)*( (i==k)*(a==c)*mo_eris[a,i,d,l]
                                    - (i==k)*(a==d)*mo_eris[a,i,c,l]
                                    - (i==l)*(a==c)*mo_eris[a,i,d,k]
                                    + (i==l)*(a==d)*mo_eris[a,i,c,k])
                Q += 1
        if doubles_ijab_B:
            for  right_ex in csfs[Q:Q+n_ijab_B]:
                # B 
                k,l,c,d = right_ex
                row[Q] = ((i==k)*(a==c)*(mo_eris[a,i,d,l] - 2*mo_eris[a,d,l,i])
                        + (i==k)*(a==d)*(mo_eris[a,i,c,l] - 2*mo_eris[a,c,l,i])
                        + (i==l)*(a==c)*(mo_eris[a,i,d,k] - 2*mo_eris[a,d,k,i])
                        + (i==l)*(a==d)*(mo_eris[a,i,c,k] - 2*mo_eris[a,c,k,i]))
                Q += 1
        return row, 1
    except:
        raise Exception("Something went wrong while computing row %i" % (P))
        return row, 0

def comp_hrow_iiab(mo_eps, mo_eris, scf_energy, csfs, num_csfs, options, P):
    singles, full_cis, doubles, doubles_options = options
    (doubles_iiaa, doubles_iiab, doubles_ijaa, 
    doubles_ijab_A, doubles_ijab_B) = doubles_options
    N = sum(num_csfs)
    E0 = scf_energy
    row = np.zeros(N)
    i,j,a,b = csfs[P]
    try :
        row[0] = np.sqrt(2)*mo_eris[a,i,b,i]
        Q = 1
        if singles:
            n_ia = num_csfs[1]    
            for right_ex in csfs[Q:Q+n_ia]:
                k,l,c,d = right_ex
                row[Q] = ((k==i)*(mo_eris[b,c,a,k] + mo_eris[a,c,b,k])
                                - (c==a)*mo_eris[i,b,i,k]
                                - (c==b)*mo_eris[i,a,i,k])
                Q += 1
        n_iiaa = num_csfs[2]
        n_iiab = num_csfs[3]
        n_ijaa = num_csfs[4]
        n_ijab_A = num_csfs[5]
        n_ijab_B = num_csfs[6]
        if doubles_iiaa:
            for right_ex in csfs[Q:Q+n_iiaa]:
                k,l,c,d = right_ex
                row[Q] = np.sqrt(2)*(((k==i)*(c==a)*(mo_eris[c,k,b,k] - 2*mo_eris[c,b,k,k]))
                            + (k==i)*(c==b)*(mo_eris[c,k,a,k] - 2*mo_eris[c,a,k,k])
                            + (k==i)*mo_eris[c,b,c,a])  
                Q += 1
        if doubles_iiab:                
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
        if doubles_ijaa: 
            for right_ex in csfs[Q:Q+n_ijaa]:
                k,l,c,d = right_ex
                row[Q] =   ((i==k)*(a==c)*(mo_eris[a,l,b,i] - 2*mo_eris[a,b,l,i])
                        +(i==k)*(b==c)*(mo_eris[b,l,a,i] - 2*mo_eris[b,a,l,i])
                        +(i==l)*(a==c)*(mo_eris[a,k,b,i] - 2*mo_eris[a,b,k,i])
                        +(i==l)*(b==c)*(mo_eris[b,k,a,i] - 2*mo_eris[b,a,k,i]))
                Q += 1
        if doubles_ijab_A:
            for right_ex in csfs[Q:Q+n_ijab_A]:
                # A 
                k,l,c,d = right_ex
                row[Q] = np.sqrt(1.5)*  ((i==k)*(a==c)*mo_eris[b,i,d,l] 
                                    - (i==k)*(a==d)*mo_eris[b,i,c,l]
                                    + (i==k)*(b==c)*mo_eris[a,i,d,l]
                                    - (i==k)*(b==d)*mo_eris[a,i,c,l]
                                    - (i==l)*(a==c)*mo_eris[b,i,d,k]
                                    + (i==l)*(a==d)*mo_eris[b,i,c,k]
                                    - (i==l)*(b==c)*mo_eris[a,i,d,k]
                                    + (i==l)*(b==d)*mo_eris[a,i,c,k])
                Q += 1
        if doubles_ijab_B:
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
        return row, 1
    except:
        raise Exception("Something went wrong while computing row %i" % (P))
        return row, 0

def comp_hrow_ijaa(mo_eps, mo_eris, scf_energy, csfs, num_csfs, options, P):
    singles, full_cis, doubles, doubles_options = options
    (doubles_iiaa, doubles_iiab, doubles_ijaa, 
    doubles_ijab_A, doubles_ijab_B) = doubles_options
    N = sum(num_csfs)
    E0 = scf_energy
    row = np.zeros(N)
    i,j,a,b = csfs[P]
    try:
        row[0] = np.sqrt(2)*mo_eris[a,i,a,j]
        Q = 1
        if singles:
            n_ia = num_csfs[1]    
            for right_ex in csfs[Q:Q+n_ia]:
                k,l,c,d = right_ex
                row[Q] =((k==i)*mo_eris[a,c,a,j]
                            + (k==j)*mo_eris[a,c,a,i]
                            + (c==a)*(mo_eris[c,j,i,k] + mo_eris[c,i,j,k]))
                Q += 1
        n_iiaa = num_csfs[2]
        n_iiab = num_csfs[3]
        n_ijaa = num_csfs[4]
        n_ijab_A = num_csfs[5]
        n_ijab_B = num_csfs[6]
        if doubles_iiaa:
            for right_ex in csfs[Q:Q+n_iiaa]:
                k,l,c,d = right_ex
                row[Q] =  np.sqrt(2)*(((k==i)*(c==a)*(mo_eris[c,k,c,j]- 2*mo_eris[c,c,j,k]))
                                        + (k==j)*(c==a)*(mo_eris[c,k,c,i] - 2*mo_eris[c,c,i,k])
                                        + (c==a)*mo_eris[i,k,j,k])
                Q += 1
        if doubles_iiab:
            for right_ex in csfs[Q:Q+n_iiab]:
                k,l,c,d = right_ex
                row[Q] =   ((k==i)*(c==a)*(mo_eris[c,j,d,k] - 2*mo_eris[c,d,j,k])
                        +(k==i)*(d==a)*(mo_eris[d,j,c,k] - 2*mo_eris[d,c,j,k])
                        +(k==j)*(c==a)*(mo_eris[c,i,d,k] - 2*mo_eris[c,d,i,k])
                        +(k==j)*(d==a)*(mo_eris[d,i,c,k] - 2*mo_eris[d,c,i,k]))
                Q += 1
        if doubles_ijaa: 
            for right_ex in csfs[Q:Q+n_ijaa]:
                k,l,c,d = right_ex
                row[Q] = ((i==k)*(j==l)*(a==c)*(E0-mo_eps[i]-mo_eps[j]+2*mo_eps[a])
                        + (i==k)*(a==c)*(mo_eris[a,l,a,j] -2*mo_eris[a,a,l,j])
                        + (i==l)*(a==c)*(mo_eris[a,k,a,j] -2*mo_eris[a,a,k,j])
                        + (j==k)*(a==c)*(mo_eris[a,l,a,i] -2*mo_eris[a,a,l,i])
                        + (j==l)*(a==c)*(mo_eris[a,k,a,i] -2*mo_eris[a,a,k,i])
                        + (a==c)*(mo_eris[k,i,l,j] + mo_eris[l,i,k,j])
                        + (i==k)*(j==l)*(mo_eris[c,a,c,a]))
                Q += 1
        if doubles_ijab_A:
            for right_ex in csfs[Q:Q+n_ijab_A]:
                # A 
                k,l,c,d = right_ex
                row[Q] = np.sqrt(1.5)*((i==k)*(a==c)*(mo_eris[a,j,d,l])
                                    - (i==k)*(a==d)*(mo_eris[a,j,c,l])
                                    - (i==l)*(a==c)*(mo_eris[a,j,d,k])
                                    + (i==l)*(a==d)*(mo_eris[a,j,c,k])
                                    + (j==k)*(a==c)*(mo_eris[a,i,d,l])
                                    - (j==k)*(a==d)*(mo_eris[a,i,c,l])
                                    - (j==l)*(a==c)*(mo_eris[a,i,d,k])
                                    + (j==l)*(a==d)*(mo_eris[a,i,c,k]))
                Q += 1
        if doubles_ijab_B:
            for  right_ex in csfs[Q:Q+n_ijab_B]:
                # B 
                k,l,c,d = right_ex
                row[Q] = np.sqrt(0.5)*((i==k)*(a==c)*(mo_eris[a,j,d,l] -2*mo_eris[a,d,j,l])
                                    + (i==k)*(a==d)*(mo_eris[a,j,c,l] -2*mo_eris[a,c,j,l])
                                    + (i==l)*(a==c)*(mo_eris[a,j,d,k] -2*mo_eris[a,d,j,k])
                                    + (i==l)*(a==d)*(mo_eris[a,j,c,k] -2*mo_eris[a,c,j,k])
                                    + (j==k)*(a==c)*(mo_eris[a,i,d,l] -2*mo_eris[a,d,i,l])
                                    + (j==k)*(a==d)*(mo_eris[a,i,c,l] -2*mo_eris[a,c,i,l])
                                    + (j==l)*(a==c)*(mo_eris[a,i,d,k] -2*mo_eris[a,d,i,k])
                                    + (j==l)*(a==d)*(mo_eris[a,i,c,k] -2*mo_eris[a,c,i,k])
                                    + (i==k)*(j==l)*2*mo_eris[c,a,d,a])
                Q += 1
        return row, 1
    except:
        raise Exception("Something went wrong while computing row %i" % (P))
        return row, 0

def comp_hrow_ijab_A(mo_eps, mo_eris, scf_energy, csfs, num_csfs, options, P):
    singles, full_cis, doubles, doubles_options = options
    (doubles_iiaa, doubles_iiab, doubles_ijaa, 
    doubles_ijab_A, doubles_ijab_B) = doubles_options
    N = sum(num_csfs)
    E0 = scf_energy
    row = np.zeros(N)
    i,j,a,b = csfs[P]
    try :
        row[0] = np.sqrt(3)*(mo_eris[a,i,b,j] - mo_eris[a,j,b,i])
        Q = 1
        if singles:
            n_ia = num_csfs[1]    
            for right_ex in csfs[Q:Q+n_ia]:
                k,l,c,d = right_ex
                row[Q] = np.sqrt(1.5) * ((k==i)*(mo_eris[c,a,b,j] - mo_eris[c,b,a,j])
                                - (k==j)*(mo_eris[c,a,b,i] - mo_eris[c,b,a,i])
                                + (c==a)*(mo_eris[b,i,j,k] - mo_eris[b,j,i,k])
                                - (c==b)*(mo_eris[a,i,j,k] - mo_eris[a,j,i,k]))
                Q += 1
        n_iiaa = num_csfs[2]
        n_iiab = num_csfs[3]
        n_ijaa = num_csfs[4]
        n_ijab_A = num_csfs[5]
        n_ijab_B = num_csfs[6]
        if doubles_iiaa:
            for right_ex in csfs[Q:Q+n_iiaa]:
                k,l,c,d = right_ex
                row[Q] = np.sqrt(3)*( (k==i)*(c==a)*mo_eris[c,k,b,j]
                                    - (k==i)*(c==b)*mo_eris[c,k,a,j]
                                    - (k==j)*(c==a)*mo_eris[c,k,b,i]
                                    + (k==j)*(c==b)*mo_eris[c,k,a,i])
                Q += 1
        if doubles_iiab:
            for right_ex in csfs[Q:Q+n_iiab]:
                k,l,c,d = right_ex
                row[Q] = np.sqrt(1.5) * ((k==i)*(c==a)*mo_eris[d,k,b,j] 
                                    - (k==i)*(c==b)*mo_eris[d,k,a,j]
                                    + (k==i)*(d==a)*mo_eris[c,k,b,j]
                                    - (k==i)*(d==b)*mo_eris[c,k,a,j]
                                    - (k==j)*(c==a)*mo_eris[d,k,b,i]
                                    + (k==j)*(c==b)*mo_eris[d,k,a,i]
                                    - (k==j)*(d==a)*mo_eris[c,k,b,i]
                                    + (k==j)*(d==b)*mo_eris[c,k,a,i])
                Q += 1
        if doubles_ijaa: 
            for right_ex in csfs[Q:Q+n_ijaa]:
                k,l,c,d = right_ex
                row[Q] = np.sqrt(1.5)*((k==i)*(c==a)*(mo_eris[c,l,b,j])
                                    - (k==i)*(c==b)*(mo_eris[c,l,a,j])
                                    - (k==j)*(c==a)*(mo_eris[c,l,b,i])
                                    + (k==j)*(c==b)*(mo_eris[c,l,a,i])
                                    + (l==i)*(c==a)*(mo_eris[c,k,b,j])
                                    - (l==i)*(c==b)*(mo_eris[c,k,a,j])
                                    - (l==j)*(c==a)*(mo_eris[c,k,b,i])
                                    + (l==j)*(c==b)*(mo_eris[c,k,a,i]))
                
                Q += 1
        if doubles_ijab_A:
            for right_ex in csfs[Q:Q+n_ijab_A]:
                # A 
                k,l,c,d = right_ex
                row[Q] = ((i==k)*(j==l)*(a==c)*(b==d)*(E0-mo_eps[i]-mo_eps[j]+mo_eps[a]+mo_eps[b])
                        +(i==k)*(a==c)*(1.5*mo_eris[b,j,d,l] - mo_eris[b,d,l,j])
                        -(i==k)*(a==d)*(1.5*mo_eris[b,j,c,l] - mo_eris[b,c,l,j])
                        -(i==k)*(b==c)*(1.5*mo_eris[a,j,d,l] - mo_eris[a,d,l,j])
                        +(i==k)*(b==d)*(1.5*mo_eris[a,j,c,l] - mo_eris[a,c,l,j])
                        -(i==l)*(a==c)*(1.5*mo_eris[b,j,d,k] - mo_eris[b,d,k,j])
                        +(i==l)*(a==d)*(1.5*mo_eris[b,j,c,k] - mo_eris[b,c,k,j])
                        +(i==l)*(b==c)*(1.5*mo_eris[a,j,d,k] - mo_eris[a,d,k,j])
                        -(i==l)*(b==d)*(1.5*mo_eris[a,j,c,k] - mo_eris[a,c,k,j])
                        -(j==k)*(a==c)*(1.5*mo_eris[b,i,d,l] - mo_eris[b,d,l,i])
                        +(j==k)*(a==d)*(1.5*mo_eris[b,i,c,l] - mo_eris[b,c,l,i])
                        +(j==k)*(b==c)*(1.5*mo_eris[a,i,d,l] - mo_eris[a,d,l,i])
                        -(j==k)*(b==d)*(1.5*mo_eris[a,i,c,l] - mo_eris[a,c,l,i])
                        +(j==l)*(a==c)*(1.5*mo_eris[b,i,d,k] - mo_eris[b,d,k,i])
                        -(j==l)*(a==d)*(1.5*mo_eris[b,i,c,k] - mo_eris[b,c,k,i])
                        -(j==l)*(b==c)*(1.5*mo_eris[a,i,d,k] - mo_eris[a,d,k,i])
                        +(j==l)*(b==d)*(1.5*mo_eris[a,i,c,k] - mo_eris[a,c,k,i])
                        +(i==k)*(j==l)*(mo_eris[a,c,d,b]- mo_eris[a,d,c,b])
                        +(a==c)*(b==d)*(mo_eris[i,k,l,j]- mo_eris[i,l,k,j]))
                Q += 1
        if doubles_ijab_B:
            for  right_ex in csfs[Q:Q+n_ijab_B]:
                # B 
                k,l,c,d = right_ex
                row[Q] = np.sqrt(0.75)* ((i==k)*(a==c)*(mo_eris[b,j,d,l])
                                        +(i==k)*(a==d)*(mo_eris[b,j,c,l])
                                        -(i==k)*(b==c)*(mo_eris[a,j,d,l])
                                        -(i==k)*(b==d)*(mo_eris[a,j,c,l])
                                        +(i==l)*(a==c)*(mo_eris[b,j,d,k])
                                        +(i==l)*(a==d)*(mo_eris[b,j,c,k])
                                        -(i==l)*(b==c)*(mo_eris[a,j,d,k])
                                        -(i==l)*(b==d)*(mo_eris[a,j,c,k])
                                        -(j==k)*(a==c)*(mo_eris[b,i,d,l])
                                        -(j==k)*(a==d)*(mo_eris[b,i,c,l])
                                        +(j==k)*(b==c)*(mo_eris[a,i,d,l])
                                        +(j==k)*(b==d)*(mo_eris[a,i,c,l])
                                        -(j==l)*(a==c)*(mo_eris[b,i,d,k])
                                        -(j==l)*(a==d)*(mo_eris[b,i,c,k])
                                        +(j==l)*(b==c)*(mo_eris[a,i,d,k])
                                        +(j==l)*(b==d)*(mo_eris[a,i,c,k]))
                Q += 1
        return row, 1
    except:
        raise Exception("Something went wrong while computing row %i" % (P))
        return row, 0

def comp_hrow_ijab_B(mo_eps, mo_eris, scf_energy, csfs, num_csfs, options, P):
    singles, full_cis, doubles, doubles_options = options
    (doubles_iiaa, doubles_iiab, doubles_ijaa, 
    doubles_ijab_A, doubles_ijab_B) = doubles_options
    N = sum(num_csfs)
    E0 = scf_energy
    row = np.zeros(N)
    i,j,a,b = csfs[P]
    try:
        row[0] = mo_eris[a,i,b,j] + mo_eris[a,j,b,i]
        Q = 1
        if singles:
            n_ia = num_csfs[1]    
            for right_ex in csfs[Q:Q+n_ia]:
                k,l,c,d = right_ex
                row[Q] = np.sqrt(0.5) * ((k==i)*(mo_eris[c,a,b,j]- mo_eris[c,b,a,j])
                                    + (k==j)*(mo_eris[c,a,b,i] - mo_eris[c,b,a,i])
                                    - (c==a)*(mo_eris[b,i,j,k] - mo_eris[b,j,i,k])
                                    - (c==b)*(mo_eris[a,i,j,k] - mo_eris[a,j,i,k]))
                Q += 1
        n_iiaa = num_csfs[2]
        n_iiab = num_csfs[3]
        n_ijaa = num_csfs[4]
        n_ijab_A = num_csfs[5]
        n_ijab_B = num_csfs[6]
        if doubles_iiaa:
            for right_ex in csfs[Q:Q+n_iiaa]:
                k,l,c,d = right_ex
                row[Q] = ((k==i)*(c==a)*(mo_eris[c,k,b,j] - 2*mo_eris[c,b,j,k])
                        + (k==i)*(c==b)*(mo_eris[c,k,a,j] - 2*mo_eris[c,a,j,k])
                        + (k==j)*(c==a)*(mo_eris[c,k,b,i] - 2*mo_eris[c,b,i,k])
                        + (k==j)*(c==b)*(mo_eris[c,k,a,i] - 2*mo_eris[c,a,i,k]))

                Q += 1
        if doubles_iiab:
            for right_ex in csfs[Q:Q+n_iiab]:
                k,l,c,d = right_ex
                row[Q] = np.sqrt(0.5)*((k==i)*(c==a)*(mo_eris[d,k,b,j]- 2*mo_eris[d,b,j,k])
                                    + (k==i)*(c==b)*(mo_eris[d,k,a,j]- 2*mo_eris[d,a,j,k])
                                    + (k==i)*(d==a)*(mo_eris[c,k,b,j]- 2*mo_eris[c,b,j,k])
                                    + (k==i)*(d==b)*(mo_eris[c,k,a,j]- 2*mo_eris[c,a,j,k])
                                    + (k==j)*(c==a)*(mo_eris[d,k,b,i]- 2*mo_eris[d,b,i,k])
                                    + (k==j)*(c==b)*(mo_eris[d,k,a,i]- 2*mo_eris[d,a,i,k])
                                    + (k==j)*(d==a)*(mo_eris[c,k,b,i]- 2*mo_eris[c,b,i,k])
                                    + (k==j)*(d==b)*(mo_eris[c,k,a,i]- 2*mo_eris[c,a,i,k])
                                    + (c==a)*(d==b)*2*mo_eris[i,k,j,k])
                Q += 1
        if doubles_ijaa: 
            for right_ex in csfs[Q:Q+n_ijaa]:
                k,l,c,d = right_ex
                row[Q] = np.sqrt(0.5)*((k==i)*(c==a)*(mo_eris[c,l,b,j] -2*mo_eris[c,b,l,j])
                                    + (k==i)*(c==b)*(mo_eris[c,l,a,j] -2*mo_eris[c,a,l,j])
                                    + (k==j)*(c==a)*(mo_eris[c,l,b,i] -2*mo_eris[c,b,l,i])
                                    + (k==j)*(c==b)*(mo_eris[c,l,a,i] -2*mo_eris[c,a,l,i])
                                    + (l==i)*(c==a)*(mo_eris[c,k,b,j] -2*mo_eris[c,b,k,j])
                                    + (l==i)*(c==b)*(mo_eris[c,k,a,j] -2*mo_eris[c,a,k,j])
                                    + (l==j)*(c==a)*(mo_eris[c,k,b,i] -2*mo_eris[c,b,k,i])
                                    + (l==j)*(c==b)*(mo_eris[c,k,a,i] -2*mo_eris[c,a,k,i])
                                    + (k==i)*(l==j)*2*mo_eris[a,c,b,c])
                Q += 1
        if doubles_ijab_A:
            for right_ex in csfs[Q:Q+n_ijab_A]:
                # A 
                k,l,c,d = right_ex
                row[Q] = np.sqrt(0.75)* ((k==i)*(c==a)*(mo_eris[d,l,b,j])
                                        +(k==i)*(c==b)*(mo_eris[d,l,a,j])
                                        -(k==i)*(d==a)*(mo_eris[c,l,b,j])
                                        -(k==i)*(d==b)*(mo_eris[c,l,a,j])
                                        +(k==j)*(c==a)*(mo_eris[d,l,b,i])
                                        +(k==j)*(c==b)*(mo_eris[d,l,a,i])
                                        -(k==j)*(d==a)*(mo_eris[c,l,b,i])
                                        -(k==j)*(d==b)*(mo_eris[c,l,a,i])
                                        -(l==i)*(c==a)*(mo_eris[d,k,b,j])
                                        -(l==i)*(c==b)*(mo_eris[d,k,a,j])
                                        +(l==i)*(d==a)*(mo_eris[c,k,b,j])
                                        +(l==i)*(d==b)*(mo_eris[c,k,a,j])
                                        -(l==j)*(c==a)*(mo_eris[d,k,b,i])
                                        -(l==j)*(c==b)*(mo_eris[d,k,a,i])
                                        +(l==j)*(d==a)*(mo_eris[c,k,b,i])
                                        +(l==j)*(d==b)*(mo_eris[c,k,a,i]))
                Q += 1
        if doubles_ijab_B:
            for  right_ex in csfs[Q:Q+n_ijab_B]:
                # B 
                k,l,c,d = right_ex
                row[Q] = ((i==k)*(j==l)*(a==c)*(b==d)*(E0-mo_eps[i]-mo_eps[j]+mo_eps[a]+mo_eps[b])
                        +(i==k)*(a==c)*(0.5*mo_eris[b,j,d,l] - mo_eris[b,d,l,j])
                        +(i==k)*(a==d)*(0.5*mo_eris[b,j,c,l] - mo_eris[b,c,l,j])
                        +(i==k)*(b==c)*(0.5*mo_eris[a,j,d,l] - mo_eris[a,d,l,j])
                        +(i==k)*(b==d)*(0.5*mo_eris[a,j,c,l] - mo_eris[a,c,l,j])
                        +(i==l)*(a==c)*(0.5*mo_eris[b,j,d,k] - mo_eris[b,d,j,k])
                        +(i==l)*(a==d)*(0.5*mo_eris[b,j,c,k] - mo_eris[b,c,k,j])
                        +(i==l)*(b==c)*(0.5*mo_eris[a,j,d,k] - mo_eris[a,d,k,j])
                        +(i==l)*(b==d)*(0.5*mo_eris[a,j,c,k] - mo_eris[a,c,k,j])
                        +(j==k)*(a==c)*(0.5*mo_eris[b,i,d,l] - mo_eris[b,d,l,i])
                        +(j==k)*(a==d)*(0.5*mo_eris[b,i,c,l] - mo_eris[b,c,l,i])
                        +(j==k)*(b==c)*(0.5*mo_eris[a,i,d,l] - mo_eris[a,d,l,i])
                        +(j==k)*(b==d)*(0.5*mo_eris[a,i,c,l] - mo_eris[a,c,l,i])
                        +(j==l)*(a==c)*(0.5*mo_eris[b,i,d,k] - mo_eris[b,d,k,i])
                        +(j==l)*(a==d)*(0.5*mo_eris[b,i,c,k] - mo_eris[b,c,k,i])
                        +(j==l)*(b==c)*(0.5*mo_eris[a,i,d,k] - mo_eris[a,d,k,i])
                        +(j==l)*(b==d)*(0.5*mo_eris[a,i,c,k] - mo_eris[a,c,k,i])
                        +(i==k)*(j==l)*(mo_eris[a,c,d,b] + mo_eris[a,d,c,b])
                        +(a==c)*(b==d)*(mo_eris[i,k,j,l] + mo_eris[i,l,k,j]))
                Q += 1
        return row, 1
    except:
        raise Exception("Something went wrong while computing row %i" % (P))
        return row, 0

def comp_hcisd(mo_eps, mo_eris, scf_energy, orbinfo, active_space, options, ncore=4):
    singles, full_cis, doubles, doubles_options = options
    (doubles_iiaa, doubles_iiab, doubles_ijaa, 
    doubles_ijab_A, doubles_ijab_B) = doubles_options
    csfs, num_csfs = generate_csfs(orbinfo, active_space, options)
    N = sum(num_csfs)
    hcisd = []
    P = 0
    row_hf, log_hf = comp_hrow_hf(mo_eps, mo_eris, scf_energy, csfs, num_csfs, options)
    hcisd += [row_hf]
    P += 1
    if singles:
        n_ia = num_csfs[1]
        pfunc_hrow_ia = partial(comp_hrow_ia, mo_eps, mo_eris, scf_energy, csfs, num_csfs, options)
        Plist_ia = list(range(P,P+n_ia))
        rows_ia, log_ia = multproc_comp_rows(pfunc_hrow_ia, Plist_ia, ncore=ncore)
        hcisd += rows_ia
        P += n_ia
    if doubles:
        if doubles_iiaa:
            n_iiaa = num_csfs[2]
            pfunc_hrow_iiaa = partial(comp_hrow_iiaa, mo_eps, mo_eris, scf_energy, csfs, num_csfs, options)
            Plist_iiaa = list(range(P,P+n_iiaa))
            rows_iiaa, log_iiaa = multproc_comp_rows(pfunc_hrow_iiaa, Plist_iiaa, ncore=ncore)
            hcisd += rows_iiaa
            P += n_iiaa
        if doubles_iiab:        
            n_iiab = num_csfs[3]
            pfunc_hrow_iiab = partial(comp_hrow_iiab, mo_eps, mo_eris, scf_energy, csfs, num_csfs, options)
            Plist_iiab = list(range(P,P+n_iiab))
            rows_iiab, log_iiab = multproc_comp_rows(pfunc_hrow_iiab, Plist_iiab, ncore=ncore)
            hcisd += rows_iiab
            P += n_iiab
        if doubles_ijaa:
            n_ijaa = num_csfs[4]
            pfunc_hrow_ijaa = partial(comp_hrow_ijaa, mo_eps, mo_eris, scf_energy, csfs, num_csfs, options)
            Plist_ijaa = list(range(P,P+n_ijaa))
            rows_ijaa, log_ijaa = multproc_comp_rows(pfunc_hrow_ijaa, Plist_ijaa, ncore=ncore)
            hcisd += rows_ijaa
            P += n_ijaa
        if doubles_ijab_A:
            n_ijab_A = num_csfs[5]
            pfunc_hrow_ijab_A = partial(comp_hrow_ijab_A, mo_eps, mo_eris, scf_energy, csfs, num_csfs, options)
            Plist_ijab_A = list(range(P,P+n_ijab_A))
            rows_ijab_A, log_ijab_A = multproc_comp_rows(pfunc_hrow_ijab_A, Plist_ijab_A, ncore=ncore)
            hcisd += rows_ijab_A
            P += n_ijab_A
        if doubles_ijab_B:        
            n_ijab_B = num_csfs[6]
            pfunc_hrow_ijab_B = partial(comp_hrow_ijab_B, mo_eps, mo_eris, scf_energy, csfs, num_csfs, options)
            Plist_ijab_B = list(range(P,P+n_ijab_B))
            rows_ijab_B, log_ijab_B = multproc_comp_rows(pfunc_hrow_ijab_B, Plist_ijab_B, ncore=ncore)
            hcisd += rows_ijab_B
            P += n_ijab_B
    if P != N:
        raise Exception("ERROR: posval not equal nCSFs")
    hcisd = np.array(hcisd)
    return hcisd

def comp_oeprop_hf(mo_oeprop, csfs, num_csfs, options):
    singles, full_cis, doubles, doubles_options = options
    (doubles_iiaa, doubles_iiab, doubles_ijaa,
    doubles_ijab_A, doubles_ijab_B) = doubles_options
    N = sum(num_csfs)
    row = np.zeros(N)
    i,j,a,b = csfs[0]
    try:
        row[0.0] = 0.0
        Q = 1
        if singles:
            n_ia = num_csfs[1]
            for right_ex in csfs[Q:Q+n_ia]:
                k,l,c,d = right_ex
                row[Q] = 0.0
                Q += 1
        n_iiaa = num_csfs[2]
        n_iiab = num_csfs[3]
        n_ijaa = num_csfs[4]
        n_ijab_A = num_csfs[5]
        n_ijab_B = num_csfs[6]
        if doubles_iiaa:
            for right_ex in csfs[Q:Q+n_iiaa]:
                k,l,c,d = right_ex
                row[Q] = 0.0
                Q += 1
        if doubles_iiab:
            for right_ex in csfs[Q:Q+n_iiab]:
                k,l,c,d = right_ex
                row[Q] = 0.0
                Q +=1
        if doubles_ijaa:
            for right_ex in csfs[Q:Q+n_ijaa]:
                k,l,c,d = right_ex
                row[Q] = 0.0
                Q += 1
        if doubles_ijab_A:
            for right_es in csfs[Q:Q+n_ijab_A]:
                k,l,c,d = right_ex
                row[Q] = 0.0
                Q += 1
        if doubles_ijab_B:
            for right_ex in csfs[Q:Q+n_ijab_B]:
                row[Q] = 0.0
                Q += 1
        return row, 1           
    except:
        raise Exception("Something went wronh while computing row %i"%(P))
    return row, 0

def comp_oeprop_ia(mo_oeprop, csfs, num_csfs, options, P):
    singles, full_cis, doubles, doubles_options = options
    (doubles_iiaa, doubles_iiab, doubles_ijaa,
    doubles_ijab_A, doubles_ijab_B) = doubles_options
    N = sum(num_csfs)
    row = np.zeros(N)
    i,j,a,b = csfs[P]
    try:
        row[0.0] = 0.0
        Q = 1
        if singles:
            n_ia = num_csfs[1]
            for right_ex in csfs[Q:Q+n_ia]:
                k,l,c,d = right_ex
                row[Q] = 0.0
                Q += 1
        n_iiaa = num_csfs[2]
        n_iiab = num_csfs[3]
        n_ijaa = num_csfs[4]
        n_ijab_A = num_csfs[5]
        n_ijab_B = num_csfs[6]
        if doubles_iiaa:
            for right_ex in csfs[Q:Q+n_iiaa]:
                k,l,c,d = right_ex
                row[Q] = 0.0
                Q += 1
        if doubles_iiab:
            for right_ex in csfs[Q:Q+n_iiab]:
                k,l,c,d = right_ex
                row[Q] = 0.0
                Q +=1
        if doubles_ijaa:
            for right_ex in csfs[Q:Q+n_ijaa]:
                k,l,c,d = right_ex
                row[Q] = 0.0
                Q += 1
        if doubles_ijab_A:
            for right_es in csfs[Q:Q+n_ijab_A]:
                k,l,c,d = right_ex
                row[Q] = 0.0
                Q += 1
        if doubles_ijab_B:
            for right_ex in csfs[Q:Q+n_ijab_B]:
                row[Q] = 0.0
                Q += 1
        return row, 1           
    except:
        raise Exception("Something went wronh while computing row %i"%(P))
    return row, 0

def comp_oeprop_iiaa(mo_oeprop, csfs, num_csfs, options, P):
    singles, full_cis, doubles, doubles_options = options
    (doubles_iiaa, doubles_iiab, doubles_ijaa,
    doubles_ijab_A, doubles_ijab_B) = doubles_options
    N = sum(num_csfs)
    row = np.zeros(N)
    i,j,a,b = csfs[P]
    try:
        row[0.0] = 0.0
        Q = 1
        if singles:
            n_ia = num_csfs[1]
            for right_ex in csfs[Q:Q+n_ia]:
                k,l,c,d = right_ex
                row[Q] = 0.0
                Q += 1
        n_iiaa = num_csfs[2]
        n_iiab = num_csfs[3]
        n_ijaa = num_csfs[4]
        n_ijab_A = num_csfs[5]
        n_ijab_B = num_csfs[6]
        if doubles_iiaa:
            for right_ex in csfs[Q:Q+n_iiaa]:
                k,l,c,d = right_ex
                row[Q] = 0.0
                Q += 1
        if doubles_iiab:
            for right_ex in csfs[Q:Q+n_iiab]:
                k,l,c,d = right_ex
                row[Q] = 0.0
                Q +=1
        if doubles_ijaa:
            for right_ex in csfs[Q:Q+n_ijaa]:
                k,l,c,d = right_ex
                row[Q] = 0.0
                Q += 1
        if doubles_ijab_A:
            for right_es in csfs[Q:Q+n_ijab_A]:
                k,l,c,d = right_ex
                row[Q] = 0.0
                Q += 1
        if doubles_ijab_B:
            for right_ex in csfs[Q:Q+n_ijab_B]:
                row[Q] = 0.0
                Q += 1
        return row, 1           
    except:
        raise Exception("Something went wronh while computing row %i"%(P))
    return row, 0

def comp_oeprop_iiab(mo_oeprop, csfs, num_csfs, options, P):
    singles, full_cis, doubles, doubles_options = options
    (doubles_iiaa, doubles_iiab, doubles_ijaa,
    doubles_ijab_A, doubles_ijab_B) = doubles_options
    N = sum(num_csfs)
    row = np.zeros(N)
    i,j,a,b = csfs[P]
    try:
        row[0.0] = 0.0
        Q = 1
        if singles:
            n_ia = num_csfs[1]
            for right_ex in csfs[Q:Q+n_ia]:
                k,l,c,d = right_ex
                row[Q] = 0.0
                Q += 1
        n_iiaa = num_csfs[2]
        n_iiab = num_csfs[3]
        n_ijaa = num_csfs[4]
        n_ijab_A = num_csfs[5]
        n_ijab_B = num_csfs[6]
        if doubles_iiaa:
            for right_ex in csfs[Q:Q+n_iiaa]:
                k,l,c,d = right_ex
                row[Q] = 0.0
                Q += 1
        if doubles_iiab:
            for right_ex in csfs[Q:Q+n_iiab]:
                k,l,c,d = right_ex
                row[Q] = 0.0
                Q +=1
        if doubles_ijaa:
            for right_ex in csfs[Q:Q+n_ijaa]:
                k,l,c,d = right_ex
                row[Q] = 0.0
                Q += 1
        if doubles_ijab_A:
            for right_es in csfs[Q:Q+n_ijab_A]:
                k,l,c,d = right_ex
                row[Q] = 0.0
                Q += 1
        if doubles_ijab_B:
            for right_ex in csfs[Q:Q+n_ijab_B]:
                row[Q] = 0.0
                Q += 1
        return row, 1           
    except:
        raise Exception("Something went wronh while computing row %i"%(P))
    return row, 0

def comp_oeprop_ijaa(mo_oeprop, csfs, num_csfs, options, P):
    singles, full_cis, doubles, doubles_options = options
    (doubles_iiaa, doubles_iiab, doubles_ijaa,
    doubles_ijab_A, doubles_ijab_B) = doubles_options
    N = sum(num_csfs)
    row = np.zeros(N)
    i,j,a,b = csfs[P]
    try:
        row[0.0] = 0.0
        Q = 1
        if singles:
            n_ia = num_csfs[1]
            for right_ex in csfs[Q:Q+n_ia]:
                k,l,c,d = right_ex
                row[Q] = 0.0
                Q += 1
        n_iiaa = num_csfs[2]
        n_iiab = num_csfs[3]
        n_ijaa = num_csfs[4]
        n_ijab_A = num_csfs[5]
        n_ijab_B = num_csfs[6]
        if doubles_iiaa:
            for right_ex in csfs[Q:Q+n_iiaa]:
                k,l,c,d = right_ex
                row[Q] = 0.0
                Q += 1
        if doubles_iiab:
            for right_ex in csfs[Q:Q+n_iiab]:
                k,l,c,d = right_ex
                row[Q] = 0.0
                Q +=1
        if doubles_ijaa:
            for right_ex in csfs[Q:Q+n_ijaa]:
                k,l,c,d = right_ex
                row[Q] = 0.0
                Q += 1
        if doubles_ijab_A:
            for right_es in csfs[Q:Q+n_ijab_A]:
                k,l,c,d = right_ex
                row[Q] = 0.0
                Q += 1
        if doubles_ijab_B:
            for right_ex in csfs[Q:Q+n_ijab_B]:
                row[Q] = 0.0
                Q += 1
        return row, 1           
    except:
        raise Exception("Something went wronh while computing row %i"%(P))
    return row, 0

def comp_oeprop_ijab_A(mo_oeprop, csfs, num_csfs, options, P):
    singles, full_cis, doubles, doubles_options = options
    (doubles_iiaa, doubles_iiab, doubles_ijaa,
    doubles_ijab_A, doubles_ijab_B) = doubles_options
    N = sum(num_csfs)
    row = np.zeros(N)
    i,j,a,b = csfs[P]
    try:
        row[0.0] = 0.0
        Q = 1
        if singles:
            n_ia = num_csfs[1]
            for right_ex in csfs[Q:Q+n_ia]:
                k,l,c,d = right_ex
                row[Q] = 0.0
                Q += 1
        n_iiaa = num_csfs[2]
        n_iiab = num_csfs[3]
        n_ijaa = num_csfs[4]
        n_ijab_A = num_csfs[5]
        n_ijab_B = num_csfs[6]
        if doubles_iiaa:
            for right_ex in csfs[Q:Q+n_iiaa]:
                k,l,c,d = right_ex
                row[Q] = 0.0
                Q += 1
        if doubles_iiab:
            for right_ex in csfs[Q:Q+n_iiab]:
                k,l,c,d = right_ex
                row[Q] = 0.0
                Q +=1
        if doubles_ijaa:
            for right_ex in csfs[Q:Q+n_ijaa]:
                k,l,c,d = right_ex
                row[Q] = 0.0
                Q += 1
        if doubles_ijab_A:
            for right_es in csfs[Q:Q+n_ijab_A]:
                k,l,c,d = right_ex
                row[Q] = 0.0
                Q += 1
        if doubles_ijab_B:
            for right_ex in csfs[Q:Q+n_ijab_B]:
                row[Q] = 0.0
                Q += 1
        return row, 1           
    except:
        raise Exception("Something went wronh while computing row %i"%(P))
    return row, 0

def comp_oeprop_ijab_B(mo_oeprop, csfs, num_csfs, options, P):
    singles, full_cis, doubles, doubles_options = options
    (doubles_iiaa, doubles_iiab, doubles_ijaa,
    doubles_ijab_A, doubles_ijab_B) = doubles_options
    N = sum(num_csfs)
    row = np.zeros(N)
    i,j,a,b = csfs[P]
    try:
        row[0.0] = 0.0
        Q = 1
        if singles:
            n_ia = num_csfs[1]
            for right_ex in csfs[Q:Q+n_ia]:
                k,l,c,d = right_ex
                row[Q] = 0.0
                Q += 1
        n_iiaa = num_csfs[2]
        n_iiab = num_csfs[3]
        n_ijaa = num_csfs[4]
        n_ijab_A = num_csfs[5]
        n_ijab_B = num_csfs[6]
        if doubles_iiaa:
            for right_ex in csfs[Q:Q+n_iiaa]:
                k,l,c,d = right_ex
                row[Q] = 0.0
                Q += 1
        if doubles_iiab:
            for right_ex in csfs[Q:Q+n_iiab]:
                k,l,c,d = right_ex
                row[Q] = 0.0
                Q +=1
        if doubles_ijaa:
            for right_ex in csfs[Q:Q+n_ijaa]:
                k,l,c,d = right_ex
                row[Q] = 0.0
                Q += 1
        if doubles_ijab_A:
            for right_es in csfs[Q:Q+n_ijab_A]:
                k,l,c,d = right_ex
                row[Q] = 0.0
                Q += 1
        if doubles_ijab_B:
            for right_ex in csfs[Q:Q+n_ijab_B]:
                row[Q] = 0.0
                Q += 1
        return row, 1           
    except:
        raise Exception("Something went wronh while computing row %i"%(P))
    return row, 0

def comp_oeprop_matrix(mo_oeprop, csfs, num_csfs, options, ncore=4):
    singles, full_cis, doubles, doubles_options = options
    (doubles_iiaa, doubles_iiab, doubles_ijaa, 
    doubles_ijab_A, doubles_ijab_B) = doubles_options
    N = sum(num_csfs)
    csf_oeprop = []
    P = 0 
    row_hf, log_hf = comp_oeprop_hf(mo_oeprop, csfs, num_csfs, options)
    csf_oeprop += [row_hf]
    P += 1
    if singles:
        n_ia = num_csfs[1]
        pfunc_oeprop_ia = partial(comp_oeprop_ia, mo_oeprop, csfs, num_csfs, options)
        Plist_ia = list(range(P, P+n_ia))
        rows_ia, log_ia = multproc_comp_rows(pfunc_oeprop_ia, Plist_ia, ncore=ncore)
        csf_oeprop += rows_ia
        P += n_ia
    if doubles:
        if doubles_iiaa:
            n_iiaa = num_csfs[2]
            pfunc_oeprop_iiaa = partial(comp_oeprop_iiaa, mo_oeprop, csfs, num_csfs, options)
            Plist_iiaa = list(range(P, P+n_iiaa))
            rows_iiaa, log_iiaa = multproc_comp_rows(pfunc_oeprop_iiaa, Plist_iiaa, ncore=ncore)
            csf_oeprop += rows_iiaa
            P += n_iiaa
        if doubles_iiab:
            n_iiab = num_csfs[3]
            pfunc_oeprop_iiab = partial(comp_oeprop_iiab, mo_oeprop, csfs, num_csfs, options)
            Plist_iiab = list(range(P, P+n_iiab))
            rows_iiab, log_iiab = multproc_comp_rows(pfunc_oeprop_iiab, Plist_iiab, ncore=ncore)
            csf_oeprop += rows_iiab
            P += n_iiab
        if doubles_ijaa:
            n_ijaa = num_csfs[4]
            pfunc_oeprop_ijaa = partial(comp_oeprop_ijaa, mo_oeprop, csfs, num_csfs, options)
            Plist_ijaa = list(range(P, P+n_ijaa))
            rows_ijaa, log_ijaa = multproc_comp_rows(pfunc_oeprop_ijaa, Plist_ijaa, ncore=ncore)
            csf_oeprop += rows_ijaa
            P += n_ijaa
        if doubles_ijab_A:
            n_ijab_A = num_csfs[5]
            pfunc_oeprop_ijab_A = partial(comp_oeprop_ijab_A, mo_oeprop, csfs, num_csfs, options)
            Plist_ijab_A = list(range(P, P+n_ijab_A))
            rows_ijab_A, log_ijab_A = multproc_comp_rows(pfunc_oeprop_ijab_A, Plist_ijab_A, ncore=ncore)
            csf_oeprop += rows_ijab_A
            P += n_ijab_A
        if doubles_ijab_B:
            n_ijab_B = num_csfs[6]
            pfunc_oeprop_ijab_B = partial(comp_oeprop_ijab_B, mo_oeprop, csfs, num_csfs, options)
            Plist_ijab_B = list(range(P, P+n_ijab_B))
            rows_ijab_B, log_ijab_B = multproc_comp_rows(pfunc_oeprop_ijab_B, Plist_ijab_B, ncore=ncore)
            csf_oeprop += rows_ijab_B
            P += n_ijab_B
    if P != N:
        raise Exception("Error: posval not equal to CSFs")
    csf_oeprop = np.array(csf_oeprop)
    return csf_oeprop