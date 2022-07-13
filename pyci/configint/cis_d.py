#!/usr/bin/env python
#
# Author : Sai Vijay Mocherla <vijaysai.mocherla@gmail.com>
#
#   TO-DO:
#       1. check and maks sure erints in all funcs are in chemists notation.
#       2. Simplify tensor computations further to reduce computational overhead
#       3. clean up API 
""" CIS(D) Spin MO version
"""
from multiprocessing import Pool
from itertools import product
from opt_einsum import contract 
import numpy as np

class CIS_D:
    """ A class to compute perturbative corrections from doubles(D) for enegies from configuration interation singles(CIS).
    """
    def __init__(self, cis_eigvals, cis_eigvecs, mo_eps, mo_so_erints, orbinfo):
        """input args :
            cis_eigvals: eigenvalues of the CIS hamiltonian,
            cis_eigvecs: eigenvectors of the CIS hamiltonian,
            mo_eps: energies of the Molecular Orbitals(MOs) used for CI calculation
            mo_so_erints: electron repulsion integrals tensor in SO basis using chemists notation.
            orbinfo: a tuple of the form (nocc, nvir, nmo)
            nocc: no. of occupied orbitals
            nvir: no. of virtual orbitals
            nmo: no. of molecular orbitals
        """
        self.nocc, self.nmo = orbinfo
        self.nvir, self.nel = self.nmo- self.nocc, 2*self.nocc
        self.nso = 2*self.nmo
        self.cis_iterlist = list(product(range(self.nocc), range(self.nocc,self.nmo)))
        self.occ_list = range(self.nel)
        self.vir_list = range(self.nel, self.nso)
        self.ecis = cis_eigvals
        self.ccis = cis_eigvecs[1:, :]
        self.delta = self.comp_delta(mo_eps)
        self.g_tensor = mo_so_erints - mo_so_erints.transpose(0, 1, 3, 2)
        self.w_tensor = - self.g_tensor[self.nel:, self.nel:, :self.nel, :self.nel]/self.delta  # g_vvoo / delta[vvoo]
        
    def comp_delta(self, mo_eps):
        """Computes the tensor delta_{ab}^{rs} with orbital energy differences
        """
        so_mo_eps = np.kron(mo_eps, np.array([1.0, 1.0], dtype=np.float64))
        delta = np.full(([self.nso]*4), 1.0, dtype=np.float64)
        iterlist = list(product(self.vir_list, self.vir_list, self.occ_list, self.occ_list))
        for conf in iterlist:
            a, b, i, j = conf
            delta[conf] = so_mo_eps[a] + so_mo_eps[b] - so_mo_eps[i] - so_mo_eps[j]
        delta = delta[self.nel:, self.nel:, :self.nel, :self.nel]  # reshaping delta to vvoo
        return delta

    def comp_e0mp2(self):
        """Computes MP2 correction for ground-state energy
        """
        E0_mp2 = 0.2500*contract('abij,ijab', self.w_tensor, self.g_tensor[:self.nel, :self.nel, self.nel:, self.nel:], optimize=True)
        return E0_mp2

    def comp_utensor(self, Ci):
        """Computes U_{ab}^{rs} tensor for each SA-CIS state
        """
        utensor = (  contract('abcj,ic->abij', self.g_tensor[self.nel:, self.nel:, self.nel:, :self.nel], Ci, optimize=True)
                   - contract('abci,jc->abij', self.g_tensor[self.nel:, self.nel:, self.nel:, :self.nel], Ci, optimize=True)
                   + contract('kaij,kb->abij', self.g_tensor[:self.nel, self.nel:, :self.nel, :self.nel], Ci, optimize=True)
                   - contract('kbij,ka->abij', self.g_tensor[:self.nel, self.nel:, :self.nel, :self.nel], Ci, optimize=True))
        return utensor

    def comp_varray(self, Ci):
        """Computes V_{a}^{r} array for each SA-CIS state
        """
        term1 = contract('jkbc,ib,cajk->ia', self.g_tensor[:self.nel, :self.nel, self.nel:, self.nel:], Ci, self.w_tensor, optimize=True)
        term2 = contract('jkbc,ja,cbik->ia', self.g_tensor[:self.nel, :self.nel, self.nel:, self.nel:], Ci, self.w_tensor, optimize=True)
        term3 = contract('jkbc,jb,acik->ia', self.g_tensor[:self.nel, :self.nel, self.nel:, self.nel:], Ci, self.w_tensor, optimize=True)
        varray = 0.5*(term1 + term2 + (2*term3))
        return varray

    def get_idx(self, occ_idx, vir_idx):
        """Get basis index of single det in cis eigenvectors
        """
        idx = (occ_idx*self.nvir) + (vir_idx-self.nocc)
        return idx

    def spin_block_ci(self, Ci):
        """Spin blocks Ci into spin blocked form
        """
        new_Ci = Ci.reshape(self.nocc, self.nvir)
        spin_Ci = np.kron(new_Ci, np.array([[1.0, 0.0], [0.0, 1.0]]))  # spin blocking the cis vector
        return spin_Ci

    def comp_dcorr(self, i):
        Ei = self.ecis[i]
        Ci = self.spin_block_ci(1/np.sqrt(2) * self.ccis[:, i])
        Ui = self.comp_utensor(Ci)
        Vi = self.comp_varray(Ci)
        Ei_d_corr = -0.25*contract('ijab,abij', Ui.transpose(2,3,0,1), Ui/(self.delta-Ei), optimize=True)
        Ei_t_corr = np.sum(Ci*Vi)
        Ei_corr = Ei_d_corr + Ei_t_corr
        print('%3.4f \t %3.4f \t %3.4f \t %3.4f' % (Ei_d_corr, Ei_t_corr, Ei_corr, Ei+Ei_corr))
        return Ei_corr

    def comp_cis_d(self, nevals, ncores=2):
        """Computes of doubles substitution correction to CIS eigen energies.
        The calculation is parallelised with multiprocessing.Pool
        -----------------------------------------------------------------------------
        nvals: no. of first few eigen values for which corrections are to be computed.
        ncore: no. of cores to be used for computations(default = 2).
        """
        if type(nevals) is int:
            if nevals > len(self.ecis):
                raise Exception('nvals can not be greater than no. of CIS eigen values')
        # print('Ui.shape, self_energy.shape, Vi.shape, Ci.shape')
        print('Ei_d_corr \t Ei_t_corr \t Ei_corr \t Ei_cis(d)')
        with Pool(processes=ncores) as pool:
            async_object = pool.map_async(self.comp_dcorr, range(nevals))
            ecis_d = self.ecis[:nevals] + np.array(async_object.get(), dtype=np.float64)
            pool.close()
            pool.join()
        return ecis_d