#!/usr/bin/env python
#
# Author : Sai Vijay Mocherla <vijaysai.mocherla@gmail.com>
#
""" rcisd.py
A python program with implementation of CISD in terms of CSF basis.

References:
[1] Woźniak, A. P., Przybytek, M., Lewenstein, M., et al.
J. Chem. Phys. 156, 174106 (2022); https://doi.org/10.1063/5.0087384
"""
import os
import numpy as np
from functools import partial
from pyci.utils.multproc import pool_jobs
# import pyci.lib.configint.rcisd as lib_rcisd

# input : eps, Ca, mo_oeints, mo_erints
# methods : 
#   1. Setup CI calculation options: singles, doubles, full_cis and active space.
#   2. Calculate operators in CSF basis using explicit formulaes from ref[1]
#       a. Hamiltonian 
#       b. calculate OEPROPs(dipoles, charges etc)
#       c. parallel routines   
# TODO:   
#   - Cythonize comp_row functions() and import to use parallisation funcs 
#   - Caculate 1-RDM from a state in CSF basis
#
SQRT2 = 1.4142135623730950
SQRT3 = 1.7320508075688772
SQRT1b2 = 0.7071067811865476   
SQRT3b2 = 1.2247448713915890
SQRT3b4 = 0.8660254037844386

def generate_csfs(orbinfo, active_space, options):
    nocc, nmo = orbinfo
    act_occ, act_vir = active_space
    occ_list = range(nocc-act_occ, nocc)
    vir_list = range(nocc, nocc+act_vir)
    csfs = [(0,0,0,0)]
    num_csfs = [1,0,0,0,0,0,0]
    if options['singles']:
        if options['full_cis']:
            D_ia = [(i,0,a,0) for i in range(nocc) for a in range(nocc,nmo)]
        else:    
            D_ia = [(i,0,a,0) for i in occ_list for a in vir_list]
        csfs.extend(D_ia)
        num_csfs[1] = len(D_ia)
    if options['doubles']:
        if options['doubles_iiaa']:
            D_iiaa = [(i,i,a,a) for i in occ_list for a in vir_list]
            csfs.extend(D_iiaa)
            num_csfs[2] = len(D_iiaa)
        if options['doubles_iiab']:      
            D_iiab = [(i,i,a,b) for i in occ_list for a in vir_list for b in vir_list if a>b]
            csfs.extend(D_iiab)
            num_csfs[3] = len(D_iiab)
        if options['doubles_ijaa']:
            D_ijaa = [(i,j,a,a) for i in occ_list for j in occ_list for a in vir_list if i>j]
            csfs.extend(D_ijaa)
            num_csfs[4] = len(D_ijaa)
        D_ijab = [(i,j,a,b) for i in occ_list for j in occ_list 
                            for a in vir_list for b in vir_list if i>j and a>b]
        if options['doubles_ijab_A']:
            D_ijab_A = D_ijab        
            csfs.extend(D_ijab_A)
            num_csfs[5] = len(D_ijab_A)
        if options['doubles_ijab_B']:        
            D_ijab_B = D_ijab
            csfs.extend(D_ijab_B)
            num_csfs[6] = len(D_ijab_B)
    return csfs, num_csfs


from scipy.sparse.linalg import eigsh
from scipy.linalg.lapack import dsyev

class CISD(object):
    options = { 'singles' : True,
                'full_cis' : True,
                'doubles' : True,
                'doubles_iiaa' : True,
                'doubles_iiab' : True,
                'doubles_ijaa' : True,
                'doubles_ijab_A' : True,
                'doubles_ijab_B' : True}
    
    def __init__(self, molecule, active_space=[], options={}, ncore=4):
        self.mol = mol = molecule
        self.scratch = mol.scratch
        for key in options.keys():
            self.options[key] = options[key]
        if len(active_space) == 0:
            active_space = mol.active_space
        self.csfs, self.num_csfs = generate_csfs(mol.orbinfo, 
                                                active_space, 
                                                self.options)
        self.ncore = ncore               
    
    def save_hcisd(self):
        HCISD = self.comp_hcisd(ncore=self.ncore)
        np.savez('cimat.npz', HCISD=HCISD)
        return 0

    def energy(self, HCISD, return_wfn=False):
        HCISD0 = HCISD - self.mol.scf_energy*np.eye(sum(self.num_csfs)) 
        energy, wfn = eigsh(HCISD0, k=1, which='SM')
        energy += self.mol.scf_energy
        energy = energy[0]
        if return_wfn:
            return energy, wfn
        else:
            return energy
    
    def get_eigen(self, HCISD):
        HCISD0 = HCISD - self.mol.scf_energy*np.eye(sum(self.num_csfs)) 
        sol = dsyev(HCISD0)
        vals, vecs = sol[0], sol[1]
        vals = vals + self.mol.scf_energy
        return vals, vecs
    
    def get_all_dipoles(self):
        mo_dpx, mo_dpy, mo_dpz = self.mol.get_mo_dpints()
        if 'dipoles' not in self.mol.properties:
            raise Exception("dipoles are not in the list of properties to be computed.")
        csf_dpy  = self.comp_oeprop(mo_dpy, ncore=self.ncore)
        csf_dpx  = self.comp_oeprop(mo_dpx, ncore=self.ncore)
        csf_dpz  = self.comp_oeprop(mo_dpz, ncore=self.ncore)
        return csf_dpx, csf_dpy, csf_dpz
    
    def save_dpx(self):
        mo_dpx = np.load(self.scratch+'mo_dpints.npz')['dpx_moints']
        csf_dpx = self.comp_oeprop(mo_dpx, ncore=self.ncore)
        np.savez('dpx.npz', csf_dpx=csf_dpx)
        del csf_dpx
        return 0

    def save_dpy(self):
        mo_dpy = np.load(self.scratch+'mo_dpints.npz')['dpy_moints']    
        csf_dpy  = self.comp_oeprop(mo_dpy, ncore=self.ncore)
        np.savez('dpy.npz', csf_dpy=csf_dpy)
        del csf_dpy
        return 0
    
    def save_dpz(self):
        mo_dpz = np.load(self.scratch+'mo_dpints.npz')['dpz_moints']
        csf_dpz  = self.comp_oeprop(mo_dpz, ncore=self.ncore)
        np.savez('dpz.npz', csf_dpz=csf_dpz)
        del csf_dpz
        return 0

    def get_all_quadrupoles(self):
        mo_qdxx, mo_qdxy, mo_qdxz, mo_qdyy, mo_qdyz, mo_qdzz = self.mol.get_mo_qdints()
        if 'quadrupoles' not in self.mol.properties:
            raise Exception("quadrupoles are not in the list of properties to be computed.")
        csf_qdxx  = self.comp_oeprop(mo_qdxx, ncore=self.ncore)
        csf_qdxy  = self.comp_oeprop(mo_qdxy, ncore=self.ncore)
        csf_qdxz  = self.comp_oeprop(mo_qdxz, ncore=self.ncore)
        csf_qdyy  = self.comp_oeprop(mo_qdyy, ncore=self.ncore)
        csf_qdyz  = self.comp_oeprop(mo_qdyz, ncore=self.ncore)
        csf_qdzz  = self.comp_oeprop(mo_qdzz, ncore=self.ncore)
        return csf_qdxx, csf_qdxy, csf_qdxz, csf_qdyy, csf_qdyz, csf_qdzz

    def get_dipole(self, mo_dipole):
        csf_dipole  = self.comp_oeprop(mo_dipole, ncore=self.ncore)
        return csf_dipole

    def get_quadrupole(self, mo_quadrupole):
        csf_quadrupole = self.comp_oeprop(mo_quadrupole, ncore=self.ncore)
        return csf_quadrupole
    
    def comp_hrow_hf(self):
        mol = self.mol
        mo_eps = mol.mo_eps[0]
        mo_erints = mol.mo_erints
        scf_energy = mol.scf_energy 
        csfs = self.csfs 
        num_csfs = self.num_csfs 
        options = self.options
        N = sum(num_csfs)
        E0 = scf_energy
        row = np.zeros(N, dtype=np.float64)
        try:
            row[0] = E0
            Q = 1
            if options['singles']:
                n_ia = num_csfs[1]
                Q += n_ia
            if options['doubles']:
                # then do doubles
                n_iiaa = num_csfs[2]
                n_iiab = num_csfs[3]
                n_ijaa = num_csfs[4]
                n_ijab_A = num_csfs[5]
                n_ijab_B = num_csfs[6]
                if options['doubles_iiaa']:
                    for right_ex in csfs[Q:Q+n_iiaa]:
                        k,l,c,d = right_ex
                        row[Q] = mo_erints[c,k,c,k]
                        Q += 1
                if options['doubles_iiab']:
                    for right_ex in csfs[Q:Q+n_iiab]:
                        k,l,c,d = right_ex
                        row[Q] = SQRT2*mo_erints[c,k,d,k]
                        Q += 1 
                if options['doubles_ijaa']:            
                    for right_ex in csfs[Q:Q+n_ijaa]:
                        k,l,c,d = right_ex
                        row[Q] = SQRT2*mo_erints[c,k,c,l]
                        Q += 1
                if options['doubles_ijab_A']:
                    for right_ex in csfs[Q:Q+n_ijab_A]:
                        # A 
                        k,l,c,d = right_ex
                        row[Q] = SQRT3*(mo_erints[c,k,d,l] - mo_erints[c,l,d,k])
                        Q += 1
                if options['doubles_ijab_B']:
                    for  right_ex in csfs[Q:Q+n_ijab_B]:
                        # B 
                        k,l,c,d = right_ex
                        row[Q] = mo_erints[c,k,d,l] + mo_erints[c,l,d,k]
                        Q += 1
            return row
        except :
            raise Exception("Something went wrong while computing row %i"%(0))

    # calculate rows for singles csf
    def comp_hrow_ia(self, P):
        mol = self.mol
        mo_eps = mol.mo_eps[0]
        mo_erints = mol.mo_erints
        scf_energy = mol.scf_energy 
        csfs = self.csfs 
        num_csfs = self.num_csfs 
        options = self.options
        N = sum(num_csfs)
        E0 = scf_energy
        row = np.zeros(N, dtype=np.float64)
        i,j,a,b = csfs[P]
        try:    
            Q = 1
            n_ia = num_csfs[1]    
            for right_ex in csfs[Q:Q+n_ia]:
                k,l,c,d = right_ex
                row[Q] = ((i==k)*(a==c)*(E0 + mo_eps[a] - mo_eps[i])
                                    +2*mo_erints[a,i,c,k] - mo_erints[c,a,k,i]) 
                Q += 1
            if options['doubles']:
                # then do doubles
                n_iiaa = num_csfs[2]
                n_iiab = num_csfs[3]
                n_ijaa = num_csfs[4]
                n_ijab_A = num_csfs[5]
                n_ijab_B = num_csfs[6]
                if options['doubles_iiaa']:
                    for right_ex in csfs[Q:Q+n_iiaa]:
                        k,l,c,d = right_ex
                        row[Q] = SQRT2 * ((i==k)*mo_erints[c,a,c,i]
                                             - (a==c)*mo_erints[k,a,k,i])
                        Q += 1
                if options['doubles_iiab']:
                    for right_ex in csfs[Q:Q+n_iiab]:
                        k,l,c,d = right_ex
                        row[Q] =  ((i==k)*(mo_erints[d,a,c,i] + mo_erints[c,a,d,i])
                                        - (a==c)*mo_erints[k,d,k,i]
                                        - (a==d)*mo_erints[k,c,k,i])
                        Q += 1
                if options['doubles_ijaa']:
                    for right_ex in csfs[Q:Q+n_ijaa]:
                        k,l,c,d = right_ex
                        row[Q] = ((i==k)*mo_erints[c,a,c,l]
                                    + (i==l)*mo_erints[c,a,c,k]
                                    - (a==c)*(mo_erints[a,l,k,i] + mo_erints[a,k,l,i]))
                        Q += 1
                if options['doubles_ijab_A']:        
                    for right_ex in csfs[Q:Q+n_ijab_A]:
                        # A 
                        k,l,c,d = right_ex
                        row[Q] = SQRT3b2 * ((i==k)*(mo_erints[a,c,d,l] - mo_erints[a,d,c,l])
                                               - (i==l)*(mo_erints[a,c,d,k] - mo_erints[a,d,c,k])
                                               + (a==c)*(mo_erints[d,k,l,i] - mo_erints[d,l,k,i])
                                               - (a==d)*(mo_erints[c,k,l,i] - mo_erints[c,l,k,i]))
                        Q += 1
                if options['doubles_ijab_B']:
                    for  right_ex in csfs[Q:Q+n_ijab_B]:
                        # B 
                        k,l,c,d = right_ex
                        row[Q] = SQRT1b2 * ((i==k)*(mo_erints[a,c,d,l] + mo_erints[a,d,c,l])
                                               + (i==l)*(mo_erints[a,c,d,k] + mo_erints[a,d,c,k])
                                               - (a==c)*(mo_erints[d,k,l,i] + mo_erints[d,l,k,i])
                                               - (a==d)*(mo_erints[c,k,l,i] + mo_erints[c,l,k,i]))
                        Q += 1
        except:
            raise Exception("Something went wrong while computing row %i" % (P))
        return row

    # calculate rows for doubles csf
    def comp_hrow_iiaa(self, P):
        mol = self.mol
        mo_eps = mol.mo_eps[0]
        mo_erints = mol.mo_erints
        scf_energy = mol.scf_energy 
        csfs = self.csfs 
        num_csfs = self.num_csfs 
        options = self.options
        N = sum(num_csfs)
        E0 = scf_energy
        row = np.zeros(N, dtype=np.float64)
        i,j,a,b = csfs[P]
        try:
            row[0] = mo_erints[a,i,a,i]
            Q = 1
            if options['singles']:
                n_ia = num_csfs[1]    
                for right_ex in csfs[Q:Q+n_ia]:
                    k,l,c,d = right_ex
                    row[Q] = SQRT2 * ((k==i)*mo_erints[a,c,a,k]
                                        - (c==a)*mo_erints[i,c,i,k])
                    Q += 1
            n_iiaa = num_csfs[2]
            n_iiab = num_csfs[3]
            n_ijaa = num_csfs[4]
            n_ijab_A = num_csfs[5]
            n_ijab_B = num_csfs[6]
            if options['doubles_iiaa']:
                for right_ex in csfs[Q:Q+n_iiaa]:
                    k,l,c,d = right_ex
                    row[Q] = ((i==k)*(a==c) *(E0 - 2*mo_eps[i] + 2*mo_eps[a] 
                                            - 4*mo_erints[a,a,i,i] + 2*mo_erints[a,i,a,i]) 
                                + (i==k)*mo_erints[c,a,c,a]
                                + (a==c)*mo_erints[k,i,k,i])
                    Q += 1
            if options['doubles_iiab']:
                for right_ex in csfs[Q:Q+n_iiab]:
                    k,l,c,d = right_ex
                    row[Q] = SQRT2*( (i==k)*(a==c)*(mo_erints[a,i,d,i] - 2*mo_erints[a,d,i,i])
                                        + (i==k)*(a==d)*(mo_erints[a,i,c,i] - 2*mo_erints[a,c,i,i])
                                        + (i==k)*mo_erints[a,d,a,c])
                    Q += 1
            if options['doubles_ijaa']: 
                for right_ex in csfs[Q:Q+n_ijaa]:
                    k,l,c,d = right_ex
                    row[Q] = SQRT2*( (i==k)*(a==c)*(mo_erints[a,i,a,l] - 2*mo_erints[a,a,l,i])
                                        + (i==l)*(a==c)*(mo_erints[a,i,a,k] - 2*mo_erints[a,a,k,i])
                                        + (a==c)*mo_erints[k,i,l,i])
                    Q += 1
            if options['doubles_ijab_A']:
                for right_ex in csfs[Q:Q+n_ijab_A]:
                    # A 
                    k,l,c,d = right_ex
                    row[Q] = SQRT3*( (i==k)*(a==c)*mo_erints[a,i,d,l]
                                        - (i==k)*(a==d)*mo_erints[a,i,c,l]
                                        - (i==l)*(a==c)*mo_erints[a,i,d,k]
                                        + (i==l)*(a==d)*mo_erints[a,i,c,k])
                    Q += 1
            if options['doubles_ijab_B']:
                for  right_ex in csfs[Q:Q+n_ijab_B]:
                    # B 
                    k,l,c,d = right_ex
                    row[Q] = ((i==k)*(a==c)*(mo_erints[a,i,d,l] - 2*mo_erints[a,d,l,i])
                            + (i==k)*(a==d)*(mo_erints[a,i,c,l] - 2*mo_erints[a,c,l,i])
                            + (i==l)*(a==c)*(mo_erints[a,i,d,k] - 2*mo_erints[a,d,k,i])
                            + (i==l)*(a==d)*(mo_erints[a,i,c,k] - 2*mo_erints[a,c,k,i]))
                    Q += 1
        except:
            raise Exception("Something went wrong while computing row %i" % (P))
        return row

    def comp_hrow_iiab(self, P):
        mol = self.mol
        mo_eps = mol.mo_eps[0]
        mo_erints = mol.mo_erints
        scf_energy = mol.scf_energy 
        csfs = self.csfs 
        num_csfs = self.num_csfs 
        options = self.options
        N = sum(num_csfs)
        E0 = scf_energy
        row = np.zeros(N, dtype=np.float64)
        i,j,a,b = csfs[P]
        try :
            row[0] = SQRT2*mo_erints[a,i,b,i]
            Q = 1
            if options['singles']:
                n_ia = num_csfs[1]    
                for right_ex in csfs[Q:Q+n_ia]:
                    k,l,c,d = right_ex
                    row[Q] = ((k==i)*(mo_erints[b,c,a,k] + mo_erints[a,c,b,k])
                                    - (c==a)*mo_erints[i,b,i,k]
                                    - (c==b)*mo_erints[i,a,i,k])
                    Q += 1
            n_iiaa = num_csfs[2]
            n_iiab = num_csfs[3]
            n_ijaa = num_csfs[4]
            n_ijab_A = num_csfs[5]
            n_ijab_B = num_csfs[6]
            if options['doubles_iiaa']:
                for right_ex in csfs[Q:Q+n_iiaa]:
                    k,l,c,d = right_ex
                    row[Q] = SQRT2*(((k==i)*(c==a)*(mo_erints[c,k,b,k] - 2*mo_erints[c,b,k,k]))
                                + (k==i)*(c==b)*(mo_erints[c,k,a,k] - 2*mo_erints[c,a,k,k])
                                + (k==i)*mo_erints[c,b,c,a])  
                    Q += 1
            if options['doubles_iiab']:                
                for right_ex in csfs[Q:Q+n_iiab]:
                    k,l,c,d = right_ex
                    row[Q] = ((i==k)*(a==c)*(b==d)*(E0 - 2*mo_eps[k] + mo_eps[c] + mo_eps[d])
                                + (k==i)*(c==a)*(mo_erints[d,k,b,k] - 2*mo_erints[d,b,k,k])
                                + (k==i)*(c==b)*(mo_erints[d,k,a,k] - 2*mo_erints[d,a,k,k])
                                + (k==i)*(d==a)*(mo_erints[c,k,b,k] - 2*mo_erints[c,b,k,k])
                                + (k==i)*(d==b)*(mo_erints[c,k,a,k] - 2*mo_erints[c,a,k,k])
                                + (k==i)*(mo_erints[c,a,d,b] + mo_erints[c,b,d,a])
                                + (c==a)*(d==b)*(mo_erints[i,k,i,k]))
                    Q += 1
            if options['doubles_ijaa']: 
                for right_ex in csfs[Q:Q+n_ijaa]:
                    k,l,c,d = right_ex
                    row[Q] =   ((i==k)*(a==c)*(mo_erints[a,l,b,i] - 2*mo_erints[a,b,l,i])
                            +(i==k)*(b==c)*(mo_erints[b,l,a,i] - 2*mo_erints[b,a,l,i])
                            +(i==l)*(a==c)*(mo_erints[a,k,b,i] - 2*mo_erints[a,b,k,i])
                            +(i==l)*(b==c)*(mo_erints[b,k,a,i] - 2*mo_erints[b,a,k,i]))
                    Q += 1
            if options['doubles_ijab_A']:
                for right_ex in csfs[Q:Q+n_ijab_A]:
                    # A 
                    k,l,c,d = right_ex
                    row[Q] = SQRT3b2*  ((i==k)*(a==c)*mo_erints[b,i,d,l] 
                                        - (i==k)*(a==d)*mo_erints[b,i,c,l]
                                        + (i==k)*(b==c)*mo_erints[a,i,d,l]
                                        - (i==k)*(b==d)*mo_erints[a,i,c,l]
                                        - (i==l)*(a==c)*mo_erints[b,i,d,k]
                                        + (i==l)*(a==d)*mo_erints[b,i,c,k]
                                        - (i==l)*(b==c)*mo_erints[a,i,d,k]
                                        + (i==l)*(b==d)*mo_erints[a,i,c,k])
                    Q += 1
            if options['doubles_ijab_B']:
                for  right_ex in csfs[Q:Q+n_ijab_B]:
                    # B 
                    k,l,c,d = right_ex
                    row[Q] = SQRT1b2*((i==k)*(a==c)*(mo_erints[b,i,d,l]- 2*mo_erints[b,d,l,i])
                                            +(i==k)*(a==d)*(mo_erints[b,i,c,l]- 2*mo_erints[b,c,l,i])
                                            +(i==k)*(b==c)*(mo_erints[a,i,d,l]- 2*mo_erints[a,d,l,i])
                                            +(i==k)*(b==d)*(mo_erints[a,i,c,l]- 2*mo_erints[a,c,l,i])
                                            +(i==l)*(a==c)*(mo_erints[b,i,d,k]- 2*mo_erints[b,d,k,i])
                                            +(i==l)*(a==d)*(mo_erints[b,i,c,k]- 2*mo_erints[b,c,k,i])
                                            +(i==l)*(b==c)*(mo_erints[a,i,d,k]- 2*mo_erints[a,d,k,i])
                                            +(i==l)*(b==d)*(mo_erints[a,i,c,k]- 2*mo_erints[a,c,k,i])
                                            +(a==c)*(b==d)*2*mo_erints[k,i,l,i])
                    Q += 1
        except:
            raise Exception("Something went wrong while computing row %i" % (P))
        return row

    def comp_hrow_ijaa(self, P):
        mol = self.mol
        mo_eps = mol.mo_eps[0]
        mo_erints = mol.mo_erints
        scf_energy = mol.scf_energy 
        csfs = self.csfs 
        num_csfs = self.num_csfs 
        options = self.options
        N = sum(num_csfs)
        E0 = scf_energy
        row = np.zeros(N, dtype=np.float64)
        i,j,a,b = csfs[P]
        try:
            row[0] = SQRT2*mo_erints[a,i,a,j]
            Q = 1
            if options['singles']:
                n_ia = num_csfs[1]    
                for right_ex in csfs[Q:Q+n_ia]:
                    k,l,c,d = right_ex
                    row[Q] =((k==i)*mo_erints[a,c,a,j]
                           + (k==j)*mo_erints[a,c,a,i]
                           - (c==a)*(mo_erints[c,j,i,k] + mo_erints[c,i,j,k]))
                    Q += 1
            n_iiaa = num_csfs[2]
            n_iiab = num_csfs[3]
            n_ijaa = num_csfs[4]
            n_ijab_A = num_csfs[5]
            n_ijab_B = num_csfs[6]
            if options['doubles_iiaa']:
                for right_ex in csfs[Q:Q+n_iiaa]:
                    k,l,c,d = right_ex
                    row[Q] =  SQRT2*(((k==i)*(c==a)*(mo_erints[c,k,c,j]- 2*mo_erints[c,c,j,k]))
                                         + (k==j)*(c==a)*(mo_erints[c,k,c,i] - 2*mo_erints[c,c,i,k])
                                         + (c==a)*mo_erints[i,k,j,k])
                    Q += 1
            if options['doubles_iiab']:
                for right_ex in csfs[Q:Q+n_iiab]:
                    k,l,c,d = right_ex
                    row[Q] =   ((k==i)*(c==a)*(mo_erints[c,j,d,k] - 2*mo_erints[c,d,j,k])
                            +(k==i)*(d==a)*(mo_erints[d,j,c,k] - 2*mo_erints[d,c,j,k])
                            +(k==j)*(c==a)*(mo_erints[c,i,d,k] - 2*mo_erints[c,d,i,k])
                            +(k==j)*(d==a)*(mo_erints[d,i,c,k] - 2*mo_erints[d,c,i,k]))
                    Q += 1
            if options['doubles_ijaa']: 
                for right_ex in csfs[Q:Q+n_ijaa]:
                    k,l,c,d = right_ex
                    row[Q] = ((i==k)*(j==l)*(a==c)*(E0-mo_eps[i]-mo_eps[j]+2*mo_eps[a])
                            + (i==k)*(a==c)*(mo_erints[a,l,a,j] -2*mo_erints[a,a,l,j])
                            + (i==l)*(a==c)*(mo_erints[a,k,a,j] -2*mo_erints[a,a,k,j])
                            + (j==k)*(a==c)*(mo_erints[a,l,a,i] -2*mo_erints[a,a,l,i])
                            + (j==l)*(a==c)*(mo_erints[a,k,a,i] -2*mo_erints[a,a,k,i])
                            + (a==c)*(mo_erints[k,i,l,j] + mo_erints[l,i,k,j])
                            + (i==k)*(j==l)*(mo_erints[c,a,c,a]))
                    Q += 1
            if options['doubles_ijab_A']:
                for right_ex in csfs[Q:Q+n_ijab_A]:
                    # A 
                    k,l,c,d = right_ex
                    row[Q] = SQRT3b2*((i==k)*(a==c)*(mo_erints[a,j,d,l])
                                        - (i==k)*(a==d)*(mo_erints[a,j,c,l])
                                        - (i==l)*(a==c)*(mo_erints[a,j,d,k])
                                        + (i==l)*(a==d)*(mo_erints[a,j,c,k])
                                        + (j==k)*(a==c)*(mo_erints[a,i,d,l])
                                        - (j==k)*(a==d)*(mo_erints[a,i,c,l])
                                        - (j==l)*(a==c)*(mo_erints[a,i,d,k])
                                        + (j==l)*(a==d)*(mo_erints[a,i,c,k]))
                    Q += 1
            if options['doubles_ijab_B']:
                for  right_ex in csfs[Q:Q+n_ijab_B]:
                    # B 
                    k,l,c,d = right_ex
                    row[Q] = SQRT1b2*((i==k)*(a==c)*(mo_erints[a,j,d,l] -2*mo_erints[a,d,j,l])
                                        + (i==k)*(a==d)*(mo_erints[a,j,c,l] -2*mo_erints[a,c,j,l])
                                        + (i==l)*(a==c)*(mo_erints[a,j,d,k] -2*mo_erints[a,d,j,k])
                                        + (i==l)*(a==d)*(mo_erints[a,j,c,k] -2*mo_erints[a,c,j,k])
                                        + (j==k)*(a==c)*(mo_erints[a,i,d,l] -2*mo_erints[a,d,i,l])
                                        + (j==k)*(a==d)*(mo_erints[a,i,c,l] -2*mo_erints[a,c,i,l])
                                        + (j==l)*(a==c)*(mo_erints[a,i,d,k] -2*mo_erints[a,d,i,k])
                                        + (j==l)*(a==d)*(mo_erints[a,i,c,k] -2*mo_erints[a,c,i,k])
                                        + (i==k)*(j==l)*2*mo_erints[c,a,d,a])
                    Q += 1
        except:
            raise Exception("Something went wrong while computing row %i" % (P))
        return row

    def comp_hrow_ijab_A(self, P):
        mol = self.mol
        mo_eps = mol.mo_eps[0]
        mo_erints = mol.mo_erints
        scf_energy = mol.scf_energy 
        csfs = self.csfs 
        num_csfs = self.num_csfs 
        options = self.options
        N = sum(num_csfs)
        E0 = scf_energy
        row = np.zeros(N, dtype=np.float64)
        i,j,a,b = csfs[P]
        try :
            row[0] = SQRT3*(mo_erints[a,i,b,j] - mo_erints[a,j,b,i])
            Q = 1
            if options['singles']:
                n_ia = num_csfs[1]    
                for right_ex in csfs[Q:Q+n_ia]:
                    k,l,c,d = right_ex
                    row[Q] = SQRT3b2 * ((k==i)*(mo_erints[c,a,b,j] - mo_erints[c,b,a,j])
                                           - (k==j)*(mo_erints[c,a,b,i] - mo_erints[c,b,a,i])
                                           + (c==a)*(mo_erints[b,i,j,k] - mo_erints[b,j,i,k])
                                           - (c==b)*(mo_erints[a,i,j,k] - mo_erints[a,j,i,k]))
                    Q += 1
            n_iiaa = num_csfs[2]
            n_iiab = num_csfs[3]
            n_ijaa = num_csfs[4]
            n_ijab_A = num_csfs[5]
            n_ijab_B = num_csfs[6]
            if options['doubles_iiaa']:
                for right_ex in csfs[Q:Q+n_iiaa]:
                    k,l,c,d = right_ex
                    row[Q] = SQRT3*( (k==i)*(c==a)*mo_erints[c,k,b,j]
                                        - (k==i)*(c==b)*mo_erints[c,k,a,j]
                                        - (k==j)*(c==a)*mo_erints[c,k,b,i]
                                        + (k==j)*(c==b)*mo_erints[c,k,a,i])
                    Q += 1
            if options['doubles_iiab']:
                for right_ex in csfs[Q:Q+n_iiab]:
                    k,l,c,d = right_ex
                    row[Q] = SQRT3b2 * ((k==i)*(c==a)*mo_erints[d,k,b,j] 
                                        - (k==i)*(c==b)*mo_erints[d,k,a,j]
                                        + (k==i)*(d==a)*mo_erints[c,k,b,j]
                                        - (k==i)*(d==b)*mo_erints[c,k,a,j]
                                        - (k==j)*(c==a)*mo_erints[d,k,b,i]
                                        + (k==j)*(c==b)*mo_erints[d,k,a,i]
                                        - (k==j)*(d==a)*mo_erints[c,k,b,i]
                                        + (k==j)*(d==b)*mo_erints[c,k,a,i])
                    Q += 1
            if options['doubles_ijaa']: 
                for right_ex in csfs[Q:Q+n_ijaa]:
                    k,l,c,d = right_ex
                    row[Q] = SQRT3b2*((k==i)*(c==a)*(mo_erints[c,l,b,j])
                                        - (k==i)*(c==b)*(mo_erints[c,l,a,j])
                                        - (k==j)*(c==a)*(mo_erints[c,l,b,i])
                                        + (k==j)*(c==b)*(mo_erints[c,l,a,i])
                                        + (l==i)*(c==a)*(mo_erints[c,k,b,j])
                                        - (l==i)*(c==b)*(mo_erints[c,k,a,j])
                                        - (l==j)*(c==a)*(mo_erints[c,k,b,i])
                                        + (l==j)*(c==b)*(mo_erints[c,k,a,i]))

                    Q += 1
            if options['doubles_ijab_A']:
                for right_ex in csfs[Q:Q+n_ijab_A]:
                    # A 
                    k,l,c,d = right_ex
                    row[Q] = ((i==k)*(j==l)*(a==c)*(b==d)*(E0-mo_eps[i]-mo_eps[j]+mo_eps[a]+mo_eps[b])
                            +(i==k)*(a==c)*(1.5*mo_erints[b,j,d,l] - mo_erints[b,d,l,j])
                            -(i==k)*(a==d)*(1.5*mo_erints[b,j,c,l] - mo_erints[b,c,l,j])
                            -(i==k)*(b==c)*(1.5*mo_erints[a,j,d,l] - mo_erints[a,d,l,j])
                            +(i==k)*(b==d)*(1.5*mo_erints[a,j,c,l] - mo_erints[a,c,l,j])
                            -(i==l)*(a==c)*(1.5*mo_erints[b,j,d,k] - mo_erints[b,d,k,j])
                            +(i==l)*(a==d)*(1.5*mo_erints[b,j,c,k] - mo_erints[b,c,k,j])
                            +(i==l)*(b==c)*(1.5*mo_erints[a,j,d,k] - mo_erints[a,d,k,j])
                            -(i==l)*(b==d)*(1.5*mo_erints[a,j,c,k] - mo_erints[a,c,k,j])
                            -(j==k)*(a==c)*(1.5*mo_erints[b,i,d,l] - mo_erints[b,d,l,i])
                            +(j==k)*(a==d)*(1.5*mo_erints[b,i,c,l] - mo_erints[b,c,l,i])
                            +(j==k)*(b==c)*(1.5*mo_erints[a,i,d,l] - mo_erints[a,d,l,i])
                            -(j==k)*(b==d)*(1.5*mo_erints[a,i,c,l] - mo_erints[a,c,l,i])
                            +(j==l)*(a==c)*(1.5*mo_erints[b,i,d,k] - mo_erints[b,d,k,i])
                            -(j==l)*(a==d)*(1.5*mo_erints[b,i,c,k] - mo_erints[b,c,k,i])
                            -(j==l)*(b==c)*(1.5*mo_erints[a,i,d,k] - mo_erints[a,d,k,i])
                            +(j==l)*(b==d)*(1.5*mo_erints[a,i,c,k] - mo_erints[a,c,k,i])
                            +(i==k)*(j==l)*(mo_erints[a,c,d,b]- mo_erints[a,d,c,b])
                            +(a==c)*(b==d)*(mo_erints[i,k,l,j]- mo_erints[i,l,k,j]))
                    Q += 1
            if options['doubles_ijab_B']:
                for  right_ex in csfs[Q:Q+n_ijab_B]:
                    # B 
                    k,l,c,d = right_ex
                    row[Q] = SQRT3b4* ((i==k)*(a==c)*(mo_erints[b,j,d,l])
                                            +(i==k)*(a==d)*(mo_erints[b,j,c,l])
                                            -(i==k)*(b==c)*(mo_erints[a,j,d,l])
                                            -(i==k)*(b==d)*(mo_erints[a,j,c,l])
                                            +(i==l)*(a==c)*(mo_erints[b,j,d,k])
                                            +(i==l)*(a==d)*(mo_erints[b,j,c,k])
                                            -(i==l)*(b==c)*(mo_erints[a,j,d,k])
                                            -(i==l)*(b==d)*(mo_erints[a,j,c,k])
                                            -(j==k)*(a==c)*(mo_erints[b,i,d,l])
                                            -(j==k)*(a==d)*(mo_erints[b,i,c,l])
                                            +(j==k)*(b==c)*(mo_erints[a,i,d,l])
                                            +(j==k)*(b==d)*(mo_erints[a,i,c,l])
                                            -(j==l)*(a==c)*(mo_erints[b,i,d,k])
                                            -(j==l)*(a==d)*(mo_erints[b,i,c,k])
                                            +(j==l)*(b==c)*(mo_erints[a,i,d,k])
                                            +(j==l)*(b==d)*(mo_erints[a,i,c,k]))
                    Q += 1
        except:
            raise Exception("Something went wrong while computing row %i" % (P))
        return row

    def comp_hrow_ijab_B(self, P):
        mol = self.mol
        mo_eps = mol.mo_eps[0]
        mo_erints = mol.mo_erints
        scf_energy = mol.scf_energy 
        csfs = self.csfs 
        num_csfs = self.num_csfs 
        options = self.options
        N = sum(num_csfs)
        E0 = scf_energy
        row = np.zeros(N, dtype=np.float64)
        i,j,a,b = csfs[P]
        try:
            row[0] = mo_erints[a,i,b,j] + mo_erints[a,j,b,i]
            Q = 1
            if options['singles']:
                n_ia = num_csfs[1]    
                for right_ex in csfs[Q:Q+n_ia]:
                    k,l,c,d = right_ex
                    row[Q] = SQRT1b2 * ((k==i)*(mo_erints[c,a,b,j] + mo_erints[c,b,a,j])
                                           + (k==j)*(mo_erints[c,a,b,i] + mo_erints[c,b,a,i])
                                           - (c==a)*(mo_erints[b,i,j,k] + mo_erints[b,j,i,k])
                                           - (c==b)*(mo_erints[a,i,j,k] + mo_erints[a,j,i,k]))
                    Q += 1
            n_iiaa = num_csfs[2]
            n_iiab = num_csfs[3]
            n_ijaa = num_csfs[4]
            n_ijab_A = num_csfs[5]
            n_ijab_B = num_csfs[6]
            if options['doubles_iiaa']:
                for right_ex in csfs[Q:Q+n_iiaa]:
                    k,l,c,d = right_ex
                    row[Q] = ((k==i)*(c==a)*(mo_erints[c,k,b,j] - 2*mo_erints[c,b,j,k])
                            + (k==i)*(c==b)*(mo_erints[c,k,a,j] - 2*mo_erints[c,a,j,k])
                            + (k==j)*(c==a)*(mo_erints[c,k,b,i] - 2*mo_erints[c,b,i,k])
                            + (k==j)*(c==b)*(mo_erints[c,k,a,i] - 2*mo_erints[c,a,i,k]))

                    Q += 1
            if options['doubles_iiab']:
                for right_ex in csfs[Q:Q+n_iiab]:
                    k,l,c,d = right_ex
                    row[Q] = SQRT1b2*((k==i)*(c==a)*(mo_erints[d,k,b,j]- 2*mo_erints[d,b,j,k])
                                        + (k==i)*(c==b)*(mo_erints[d,k,a,j]- 2*mo_erints[d,a,j,k])
                                        + (k==i)*(d==a)*(mo_erints[c,k,b,j]- 2*mo_erints[c,b,j,k])
                                        + (k==i)*(d==b)*(mo_erints[c,k,a,j]- 2*mo_erints[c,a,j,k])
                                        + (k==j)*(c==a)*(mo_erints[d,k,b,i]- 2*mo_erints[d,b,i,k])
                                        + (k==j)*(c==b)*(mo_erints[d,k,a,i]- 2*mo_erints[d,a,i,k])
                                        + (k==j)*(d==a)*(mo_erints[c,k,b,i]- 2*mo_erints[c,b,i,k])
                                        + (k==j)*(d==b)*(mo_erints[c,k,a,i]- 2*mo_erints[c,a,i,k])
                                        + (c==a)*(d==b)*2*mo_erints[i,k,j,k])
                    Q += 1
            if options['doubles_ijaa']: 
                for right_ex in csfs[Q:Q+n_ijaa]:
                    k,l,c,d = right_ex
                    row[Q] = SQRT1b2*((k==i)*(c==a)*(mo_erints[c,l,b,j] -2*mo_erints[c,b,l,j])
                                        + (k==i)*(c==b)*(mo_erints[c,l,a,j] -2*mo_erints[c,a,l,j])
                                        + (k==j)*(c==a)*(mo_erints[c,l,b,i] -2*mo_erints[c,b,l,i])
                                        + (k==j)*(c==b)*(mo_erints[c,l,a,i] -2*mo_erints[c,a,l,i])
                                        + (l==i)*(c==a)*(mo_erints[c,k,b,j] -2*mo_erints[c,b,k,j])
                                        + (l==i)*(c==b)*(mo_erints[c,k,a,j] -2*mo_erints[c,a,k,j])
                                        + (l==j)*(c==a)*(mo_erints[c,k,b,i] -2*mo_erints[c,b,k,i])
                                        + (l==j)*(c==b)*(mo_erints[c,k,a,i] -2*mo_erints[c,a,k,i])
                                        + (k==i)*(l==j)*2*mo_erints[a,c,b,c])
                    Q += 1
            if options['doubles_ijab_A']:
                for right_ex in csfs[Q:Q+n_ijab_A]:
                    # A 
                    k,l,c,d = right_ex
                    row[Q] = SQRT3b4* ((k==i)*(c==a)*(mo_erints[d,l,b,j])
                                            +(k==i)*(c==b)*(mo_erints[d,l,a,j])
                                            -(k==i)*(d==a)*(mo_erints[c,l,b,j])
                                            -(k==i)*(d==b)*(mo_erints[c,l,a,j])
                                            +(k==j)*(c==a)*(mo_erints[d,l,b,i])
                                            +(k==j)*(c==b)*(mo_erints[d,l,a,i])
                                            -(k==j)*(d==a)*(mo_erints[c,l,b,i])
                                            -(k==j)*(d==b)*(mo_erints[c,l,a,i])
                                            -(l==i)*(c==a)*(mo_erints[d,k,b,j])
                                            -(l==i)*(c==b)*(mo_erints[d,k,a,j])
                                            +(l==i)*(d==a)*(mo_erints[c,k,b,j])
                                            +(l==i)*(d==b)*(mo_erints[c,k,a,j])
                                            -(l==j)*(c==a)*(mo_erints[d,k,b,i])
                                            -(l==j)*(c==b)*(mo_erints[d,k,a,i])
                                            +(l==j)*(d==a)*(mo_erints[c,k,b,i])
                                            +(l==j)*(d==b)*(mo_erints[c,k,a,i]))
                    Q += 1
            if options['doubles_ijab_B']:
                for  right_ex in csfs[Q:Q+n_ijab_B]:
                    # B 
                    k,l,c,d = right_ex
                    row[Q] = ((i==k)*(j==l)*(a==c)*(b==d)*(E0-mo_eps[i]-mo_eps[j]+mo_eps[a]+mo_eps[b])
                            +(i==k)*(a==c)*(0.5*mo_erints[b,j,d,l] - mo_erints[b,d,l,j])
                            +(i==k)*(a==d)*(0.5*mo_erints[b,j,c,l] - mo_erints[b,c,l,j])
                            +(i==k)*(b==c)*(0.5*mo_erints[a,j,d,l] - mo_erints[a,d,l,j])
                            +(i==k)*(b==d)*(0.5*mo_erints[a,j,c,l] - mo_erints[a,c,l,j])
                            +(i==l)*(a==c)*(0.5*mo_erints[b,j,d,k] - mo_erints[b,d,j,k])
                            +(i==l)*(a==d)*(0.5*mo_erints[b,j,c,k] - mo_erints[b,c,k,j])
                            +(i==l)*(b==c)*(0.5*mo_erints[a,j,d,k] - mo_erints[a,d,k,j])
                            +(i==l)*(b==d)*(0.5*mo_erints[a,j,c,k] - mo_erints[a,c,k,j])
                            +(j==k)*(a==c)*(0.5*mo_erints[b,i,d,l] - mo_erints[b,d,l,i])
                            +(j==k)*(a==d)*(0.5*mo_erints[b,i,c,l] - mo_erints[b,c,l,i])
                            +(j==k)*(b==c)*(0.5*mo_erints[a,i,d,l] - mo_erints[a,d,l,i])
                            +(j==k)*(b==d)*(0.5*mo_erints[a,i,c,l] - mo_erints[a,c,l,i])
                            +(j==l)*(a==c)*(0.5*mo_erints[b,i,d,k] - mo_erints[b,d,k,i])
                            +(j==l)*(a==d)*(0.5*mo_erints[b,i,c,k] - mo_erints[b,c,k,i])
                            +(j==l)*(b==c)*(0.5*mo_erints[a,i,d,k] - mo_erints[a,d,k,i])
                            +(j==l)*(b==d)*(0.5*mo_erints[a,i,c,k] - mo_erints[a,c,k,i])
                            +(i==k)*(j==l)*(mo_erints[a,c,d,b] + mo_erints[a,d,c,b])
                            +(a==c)*(b==d)*(mo_erints[i,k,j,l] + mo_erints[i,l,k,j]))
                    Q += 1
        except:
            raise Exception("Something went wrong while computing row %i" % (P))
        return row

    def comp_oeprop_hf(self, mo_oeprop, mo_oeprop_trace):
        csfs = self.csfs
        num_csfs = self.num_csfs
        options = self.options
        N = sum(num_csfs)
        row = np.zeros(N, dtype=np.float64)
        i,j,a,b = csfs[0]
        try:
            row[0] = mo_oeprop_trace
            Q = 1
            if options['singles']:
                n_ia = num_csfs[1]
                for right_ex in csfs[Q:Q+n_ia]:
                    k,l,c,d = right_ex
                    row[Q] = SQRT2 * mo_oeprop[k,c]
                    Q += 1
            n_iiaa = num_csfs[2]
            n_iiab = num_csfs[3]
            n_ijaa = num_csfs[4]
            n_ijab_A = num_csfs[5]
            n_ijab_B = num_csfs[6]
            if options['doubles_iiaa']:
                Q += n_iiaa
            if options['doubles_iiab']:
                Q += n_iiab
            if options['doubles_ijaa']:
                Q += n_ijaa
            if options['doubles_ijab_A']:
                Q += n_ijab_A
            if options['doubles_ijab_B']:
                Q += n_ijab_B
        except:
            raise Exception("Something went wronh while computing row %i"%(0))
        return row           

    def comp_oeprop_ia(self, mo_oeprop, mo_oeprop_trace, P):
        csfs = self.csfs
        num_csfs = self.num_csfs
        options = self.options
        N = sum(num_csfs)
        row = np.zeros(N, dtype=np.float64)
        i,j,a,b = csfs[P]
        try:
            row[0] = SQRT2 * mo_oeprop[i,a]
            Q = 1
            if options['singles']:
                n_ia = num_csfs[1]
                for right_ex in csfs[Q:Q+n_ia]:
                    k,l,c,d = right_ex
                    row[Q] = ((i==k)*(a==c)*mo_oeprop_trace 
                             - (a==c)*mo_oeprop[k,i]
                             + (i==k)*mo_oeprop[a,c])
                    Q += 1
            n_iiaa = num_csfs[2]
            n_iiab = num_csfs[3]
            n_ijaa = num_csfs[4]
            n_ijab_A = num_csfs[5]
            n_ijab_B = num_csfs[6]
            if options['doubles_iiaa']:
                for right_ex in csfs[Q:Q+n_iiaa]:
                    k,l,c,d = right_ex
                    row[Q] = (i==k)*(a==c)*SQRT2*mo_oeprop[i,a]
                    Q += 1
            if options['doubles_iiab']:
                for right_ex in csfs[Q:Q+n_iiab]:
                    k,l,c,d = right_ex
                    row[Q] = ((i==k)*(a==c)*mo_oeprop[i,d]
                            + (i==k)*(a==d)*mo_oeprop[i,c])
                    Q +=1
            if options['doubles_ijaa']:
                for right_ex in csfs[Q:Q+n_ijaa]:
                    k,l,c,d = right_ex
                    row[Q] = ((i==k)*(a==c)*mo_oeprop[l,a]
                            + (i==l)*(a==c)*mo_oeprop[k,a])
                    Q += 1
            if options['doubles_ijab_A']:
                for right_ex in csfs[Q:Q+n_ijab_A]:
                    k,l,c,d = right_ex
                    row[Q] = SQRT3b2*((i==k)*(a==c)*mo_oeprop[l,d]
                                         - (i==k)*(a==d)*mo_oeprop[l,c]
                                         - (i==l)*(a==c)*mo_oeprop[k,d]
                                         + (i==l)*(a==d)*mo_oeprop[k,c])
                    Q += 1
            if options['doubles_ijab_B']:
                for right_ex in csfs[Q:Q+n_ijab_B]:
                    k,l,c,d = right_ex
                    row[Q] = SQRT1b2*((i==k)*(a==c)*mo_oeprop[l,d]
                                         + (i==k)*(a==d)*mo_oeprop[l,c]
                                         + (i==l)*(a==c)*mo_oeprop[k,d]
                                         + (i==l)*(a==d)*mo_oeprop[k,c])
                    Q += 1
        except:
            raise Exception("Something went wronh while computing row %i"%(P))
        return row           

    def comp_oeprop_iiaa(self, mo_oeprop, mo_oeprop_trace, P):
        csfs = self.csfs
        num_csfs = self.num_csfs
        options = self.options
        N = sum(num_csfs)
        row = np.zeros(N, dtype=np.float64)
        i,j,a,b = csfs[P]
        try:
            Q = 1
            if options['singles']:
                n_ia = num_csfs[1]
                for right_ex in csfs[Q:Q+n_ia]:
                    k,l,c,d = right_ex
                    row[Q] = (k==i)*(c==a)*SQRT2*mo_oeprop[k,c]
                    Q += 1
            n_iiaa = num_csfs[2]
            n_iiab = num_csfs[3]
            n_ijaa = num_csfs[4]
            n_ijab_A = num_csfs[5]
            n_ijab_B = num_csfs[6]
            if options['doubles_iiaa']:
                for right_ex in csfs[Q:Q+n_iiaa]:
                    k,l,c,d = right_ex
                    row[Q] = (i==k)*(a==c)*(mo_oeprop_trace
                                        - 2*mo_oeprop[i,i]
                                        + 2*mo_oeprop[a,a])
                    Q += 1
            if options['doubles_iiab']:
                for right_ex in csfs[Q:Q+n_iiab]:
                    k,l,c,d = right_ex
                    row[Q] = SQRT2*((i==k)*(a==c)*mo_oeprop[a,d]
                                       + (i==k)*(a==d)*mo_oeprop[a,c])
                    Q +=1
            if options['doubles_ijaa']:
                for right_ex in csfs[Q:Q+n_ijaa]:
                    k,l,c,d = right_ex
                    row[Q] = -SQRT2*((i==k)*(a==c)*mo_oeprop[l,i]
                                        + (i==l)*(a==c)*mo_oeprop[k,i])
                    Q += 1
            if options['doubles_ijab_A']:
                Q += n_ijab_A
            if options['doubles_ijab_B']:
                Q += n_ijab_B
        except:
            raise Exception("Something went wronh while computing row %i"%(P))
        return row           

    def comp_oeprop_iiab(self, mo_oeprop, mo_oeprop_trace, P):
        csfs = self.csfs
        num_csfs = self.num_csfs
        options = self.options
        N = sum(num_csfs)
        row = np.zeros(N, dtype=np.float64)
        i,j,a,b = csfs[P]
        try:        
            Q = 1
            if options['singles']:
                n_ia = num_csfs[1]
                for right_ex in csfs[Q:Q+n_ia]:
                    k,l,c,d = right_ex
                    row[Q] = ((k==i)*(c==a)*mo_oeprop[k,b]
                            + (k==i)*(c==b)*mo_oeprop[k,a])
                    Q += 1
            n_iiaa = num_csfs[2]
            n_iiab = num_csfs[3]
            n_ijaa = num_csfs[4]
            n_ijab_A = num_csfs[5]
            n_ijab_B = num_csfs[6]
            if options['doubles_iiaa']:
                for right_ex in csfs[Q:Q+n_iiaa]:
                    k,l,c,d = right_ex
                    row[Q] = SQRT2*((k==i)*(c==a)*mo_oeprop[c,b]
                                       + (k==i)*(c==b)*mo_oeprop[c,a])
                    Q += 1
            if options['doubles_iiab']:
                for right_ex in csfs[Q:Q+n_iiab]:
                    k,l,c,d = right_ex
                    row[Q] = ((i==k)*(a==c)*(b==d)*(mo_oeprop_trace - 2*mo_oeprop[i,i])
                            + (i==k)*(a==c)*mo_oeprop[b,d]
                            + (i==k)*(a==d)*mo_oeprop[b,c]
                            + (i==k)*(b==c)*mo_oeprop[a,d]
                            + (i==k)*(b==d)*mo_oeprop[a,c])
                    Q +=1
            if options['doubles_ijaa']:
                Q += n_ijaa
            if options['doubles_ijab_A']:
                Q += n_ijab_A
            if options['doubles_ijab_B']:
                for right_ex in csfs[Q:Q+n_ijab_B]:
                    k,l,c,d = right_ex
                    row[Q] = -SQRT2*((i==k)*(a==c)*(b==d)*mo_oeprop[l,i]
                                        + (i==l)*(a==c)*(b==d)*mo_oeprop[k,i])
                    Q += 1
        except:
            raise Exception("Something went wronh while computing row %i"%(P))
        return row           

    def comp_oeprop_ijaa(self, mo_oeprop, mo_oeprop_trace, P):
        csfs = self.csfs
        num_csfs = self.num_csfs
        options = self.options
        N = sum(num_csfs)
        row = np.zeros(N, dtype=np.float64)
        i,j,a,b = csfs[P]
        try:
            Q = 1
            if options['singles']:
                n_ia = num_csfs[1]
                for right_ex in csfs[Q:Q+n_ia]:
                    k,l,c,d = right_ex
                    row[Q] = ((k==i)*(c==a)*mo_oeprop[j,c]
                            + (k==j)*(c==a)*mo_oeprop[i,c])
                    Q += 1
            n_iiaa = num_csfs[2]
            n_iiab = num_csfs[3]
            n_ijaa = num_csfs[4]
            n_ijab_A = num_csfs[5]
            n_ijab_B = num_csfs[6]
            if options['doubles_iiaa']:
                for right_ex in csfs[Q:Q+n_iiaa]:
                    k,l,c,d = right_ex
                    row[Q] = -SQRT2*((k==i)*(c==a)*mo_oeprop[j,k]
                                        + (k==j)*(c==a)*mo_oeprop[i,k])
                    Q += 1
            if options['doubles_iiab']:
                Q += n_iiab
            if options['doubles_ijaa']:
                for right_ex in csfs[Q:Q+n_ijaa]:
                    k,l,c,d = right_ex
                    row[Q] = ((i==k)*(j==l)*(a==c)*(mo_oeprop_trace + 2*mo_oeprop[a,a])
                             -(i==k)*(a==c)*mo_oeprop[l][j]
                             -(i==l)*(a==c)*mo_oeprop[k][j]
                             -(j==k)*(a==c)*mo_oeprop[l][i]
                             -(j==l)*(a==c)*mo_oeprop[k][i])
                    Q += 1
            if options['doubles_ijab_A']:
                Q += n_ijab_A
            if options['doubles_ijab_B']:
                for right_ex in csfs[Q:Q+n_ijab_B]:
                    k,l,c,d = right_ex
                    row[Q] = SQRT2*((i==k)*(j==l)*(a==c)*mo_oeprop[a,d]
                                        +(i==k)*(j==l)*(a==d)*mo_oeprop[a,c])
                    Q += 1
        except:
            raise Exception("Something went wronh while computing row %i"%(P))
        return row           

    def comp_oeprop_ijab_A(self, mo_oeprop, mo_oeprop_trace, P):
        csfs = self.csfs
        num_csfs = self.num_csfs
        options = self.options
        N = sum(num_csfs)
        row = np.zeros(N, dtype=np.float64)
        i,j,a,b = csfs[P]
        try:  
            Q = 1
            if options['singles']:
                n_ia = num_csfs[1]
                for right_ex in csfs[Q:Q+n_ia]:
                    k,l,c,d = right_ex
                    row[Q] = SQRT3b2*((k==i)*(c==a)*mo_oeprop[j,b]
                                         - (k==i)*(c==b)*mo_oeprop[j,a]
                                         - (k==j)*(c==a)*mo_oeprop[i,b]
                                         + (k==j)*(c==b)*mo_oeprop[i,a])
                    Q += 1
            n_iiaa = num_csfs[2]
            n_iiab = num_csfs[3]
            n_ijaa = num_csfs[4]
            n_ijab_A = num_csfs[5]
            n_ijab_B = num_csfs[6]
            if options['doubles_iiaa']:
                Q += n_iiaa
            if options['doubles_iiab']:
                Q += n_iiab
            if options['doubles_ijaa']:
                Q += n_ijaa
            if options['doubles_ijab_A']:
                for right_ex in csfs[Q:Q+n_ijab_A]:
                    k,l,c,d = right_ex
                    row[Q] = ((i==k)*(j==l)*(a==c)*(b==d)*mo_oeprop_trace
                            - (i==k)*(a==c)*(b==d)*mo_oeprop[l,j]
                            + (i==l)*(a==c)*(b==d)*mo_oeprop[k,j]
                            + (j==k)*(a==c)*(b==d)*mo_oeprop[l,i]
                            - (j==l)*(a==c)*(b==d)*mo_oeprop[k,i]
                            + (i==k)*(j==l)*(a==c)*mo_oeprop[b,d]
                            - (i==k)*(j==l)*(a==d)*mo_oeprop[b,c]
                            - (i==k)*(j==l)*(b==c)*mo_oeprop[a,d]
                            + (i==k)*(j==l)*(b==d)*mo_oeprop[a,c])
                    Q += 1
            if options['doubles_ijab_B']:
                Q += n_ijab_B
        except:
            raise Exception("Something went wronh while computing row %i"%(P))
        return row           

    def comp_oeprop_ijab_B(self, mo_oeprop, mo_oeprop_trace, P):
        csfs = self.csfs
        num_csfs = self.num_csfs
        options = self.options
        N = sum(num_csfs)
        row = np.zeros(N, dtype=np.float64)
        i,j,a,b = csfs[P]
        try:  
            Q = 1
            if options['singles']:
                n_ia = num_csfs[1]
                for right_ex in csfs[Q:Q+n_ia]:
                    k,l,c,d = right_ex
                    row[Q] = SQRT1b2*((k==i)*(c==a)*mo_oeprop[j,b]
                                         + (k==i)*(c==b)*mo_oeprop[j,a]
                                         + (k==j)*(c==a)*mo_oeprop[i,b]
                                         + (k==j)*(c==b)*mo_oeprop[i,a])

                    Q += 1
            n_iiaa = num_csfs[2]
            n_iiab = num_csfs[3]
            n_ijaa = num_csfs[4]
            n_ijab_A = num_csfs[5]
            n_ijab_B = num_csfs[6]
            if options['doubles_iiaa']:
                Q += n_iiaa
            if options['doubles_iiab']:
                for right_ex in csfs[Q:Q+n_iiab]:
                    k,l,c,d = right_ex
                    row[Q] = -SQRT2*((k==i)*(c==a)*(d==b)*mo_oeprop[j,k]
                                        + (k==j)*(c==a)*(d==b)*mo_oeprop[i,k])
                    Q +=1
            if options['doubles_ijaa']:
                for right_ex in csfs[Q:Q+n_ijaa]:
                    k,l,c,d = right_ex
                    row[Q] = SQRT2*((k==i)*(l==j)*(c==a)*mo_oeprop[c,b]
                                        +(k==i)*(l==j)*(c==b)*mo_oeprop[c,a])
                    Q += 1
            if options['doubles_ijab_A']:
                Q += n_ijab_A
            if options['doubles_ijab_B']:
                for right_ex in csfs[Q:Q+n_ijab_B]:
                    k,l,c,d = right_ex
                    row[Q] = ((i==k)*(j==l)*(a==c)*(b==d)*mo_oeprop_trace
                            - (i==k)*(a==c)*(b==d)*mo_oeprop[l,j]
                            - (i==l)*(a==c)*(b==d)*mo_oeprop[k,j]
                            - (j==k)*(a==c)*(b==d)*mo_oeprop[l,i]
                            - (j==l)*(a==c)*(b==d)*mo_oeprop[k,i]
                            + (i==k)*(j==l)*(a==c)*mo_oeprop[b,d]
                            + (i==k)*(j==l)*(a==d)*mo_oeprop[b,c]
                            + (i==k)*(j==l)*(b==c)*mo_oeprop[a,d]
                            + (i==k)*(j==l)*(b==d)*mo_oeprop[a,c])

                    Q += 1
        except:
            raise Exception("Something went wrong while computing row %i"%(P))
        return row           

    def comp_hcisd(self, ncore=4):
        mol = self.mol
        options = self.options
        print('Computing CISD Hamiltonian matrix with {ncore:d} cores \n'.format(ncore=ncore))
        num_csfs = self.num_csfs 
        N = sum(num_csfs)
        hcisd = []
        P = 0
        row_hf = self.comp_hrow_hf()
        hcisd += [row_hf]
        P += 1
        if options['singles']:
            n_ia = num_csfs[1]
            Plist_ia = list(range(P,P+n_ia))
            rows_ia = pool_jobs(self.comp_hrow_ia, Plist_ia, ncore)
            hcisd += rows_ia
            P += n_ia
        if options['doubles']:
            if options['doubles_iiaa']:
                n_iiaa = num_csfs[2]
                Plist_iiaa = list(range(P,P+n_iiaa))
                rows_iiaa = pool_jobs(self.comp_hrow_iiaa, Plist_iiaa, ncore)
                hcisd += rows_iiaa
                P += n_iiaa
            if options['doubles_iiab']:        
                n_iiab = num_csfs[3]
                Plist_iiab = list(range(P,P+n_iiab))
                rows_iiab = pool_jobs(self.comp_hrow_iiab, Plist_iiab, ncore)
                hcisd += rows_iiab
                P += n_iiab
            if options['doubles_ijaa']:
                n_ijaa = num_csfs[4]
                Plist_ijaa = list(range(P,P+n_ijaa))
                rows_ijaa = pool_jobs(self.comp_hrow_ijaa, Plist_ijaa, ncore)
                hcisd += rows_ijaa
                P += n_ijaa
            if options['doubles_ijab_A']:
                n_ijab_A = num_csfs[5]
                Plist_ijab_A = list(range(P,P+n_ijab_A))
                rows_ijab_A = pool_jobs(self.comp_hrow_ijab_A, Plist_ijab_A, ncore)
                hcisd += rows_ijab_A
                P += n_ijab_A
            if options['doubles_ijab_B']:        
                n_ijab_B = num_csfs[6]
                Plist_ijab_B = list(range(P,P+n_ijab_B))
                rows_ijab_B = pool_jobs(self.comp_hrow_ijab_B, Plist_ijab_B, ncore)
                hcisd += rows_ijab_B
                P += n_ijab_B
        if P != N:
            raise Exception("ERROR: posval not equal nCSFs")
        hcisd = np.array(hcisd, dtype=np.float64)
        return hcisd

    def comp_oeprop(self, mo_oeprop, ncore=4):
        mol = self.mol
        csfs = self.csfs 
        num_csfs = self.num_csfs
        options = self.options
        nocc, nmo = mol.orbinfo
        N = sum(num_csfs)
        mo_oeprop_trace = np.sum(np.diag(mo_oeprop)[:nocc])
        csf_oeprop = []
        P = 0 
        row_hf = self.comp_oeprop_hf(mo_oeprop, mo_oeprop_trace)
        csf_oeprop += [row_hf]
        P += 1
        if options['singles']:
            n_ia = num_csfs[1]
            comp_oeprop_ia = partial(self.comp_oeprop_ia, mo_oeprop, mo_oeprop_trace)
            Plist_ia = list(range(P, P+n_ia))
            rows_ia = pool_jobs(comp_oeprop_ia, Plist_ia, ncore)
            csf_oeprop += rows_ia
            P += n_ia
        if options['doubles']:
            if options['doubles_iiaa']:
                n_iiaa = num_csfs[2]
                comp_oeprop_iiaa = partial(self.comp_oeprop_iiaa, mo_oeprop, mo_oeprop_trace)
                Plist_iiaa = list(range(P, P+n_iiaa))
                rows_iiaa = pool_jobs(comp_oeprop_iiaa, Plist_iiaa, ncore)
                csf_oeprop += rows_iiaa
                P += n_iiaa
            if options['doubles_iiab']:
                n_iiab = num_csfs[3]
                comp_oeprop_iiab = partial(self.comp_oeprop_iiab, mo_oeprop, mo_oeprop_trace)
                Plist_iiab = list(range(P, P+n_iiab))
                rows_iiab = pool_jobs(comp_oeprop_iiab, Plist_iiab, ncore)
                csf_oeprop += rows_iiab
                P += n_iiab
            if options['doubles_ijaa']:
                n_ijaa = num_csfs[4]
                comp_oeprop_ijaa = partial(self.comp_oeprop_ijaa, mo_oeprop, mo_oeprop_trace)
                Plist_ijaa = list(range(P, P+n_ijaa))
                rows_ijaa = pool_jobs(comp_oeprop_ijaa, Plist_ijaa, ncore)
                csf_oeprop += rows_ijaa
                P += n_ijaa
            if options['doubles_ijab_A']:
                n_ijab_A = num_csfs[5]
                comp_oeprop_ijab_A = partial(self.comp_oeprop_ijab_A, mo_oeprop, mo_oeprop_trace)
                Plist_ijab_A = list(range(P, P+n_ijab_A))
                rows_ijab_A = pool_jobs(comp_oeprop_ijab_A, Plist_ijab_A, ncore)
                csf_oeprop += rows_ijab_A
                P += n_ijab_A
            if options['doubles_ijab_B']:
                n_ijab_B = num_csfs[6]
                comp_oeprop_ijab_B = partial(self.comp_oeprop_ijab_B, mo_oeprop, mo_oeprop_trace)
                Plist_ijab_B = list(range(P, P+n_ijab_B))
                rows_ijab_B = pool_jobs(comp_oeprop_ijab_B, Plist_ijab_B, ncore)
                csf_oeprop += rows_ijab_B
                P += n_ijab_B
        if P != N:
            raise Exception("Error: posval not equal to CSFs")
        csf_oeprop = np.array(csf_oeprop, dtype=np.float64)
        return csf_oeprop