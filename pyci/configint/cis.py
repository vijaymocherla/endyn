#!/usr/bin/env python
#
# Author : Sai Vijay Mocherla <vijaysai.mocherla@gmail.com>
#
"""Simple RCIS - restricted configuration interaction singles
"""
import numpy as np
from itertools import product
from multiprocessing import Pool


def comp_cis_hamiltonian(orbinfo, mo_eps, mo_eris, parallelised=False, ncore=1):
    """Computes the configuration-interaction singles(CIS) hamiltonian in the
    spin-adapted singlet determinant basis.
    ! Note: The Hamiltonian needs scaled to SCF ground state energy, therefore 
    the eigen values need to be scaled back.
    --------------------------------------------------------------------------
    mo_eps : array of molecular orbtial(MO) energies from SCF calculations.
    mo_eris : 4-d array of 2e- integrals using chemists notation in MO basis.
    orbinfo : (nocc, nvir, nmo) tuple to pass orbital information.
    ---------------------------------------------------------------------------
    """
    if parallelised:
        multp_obj = multp_cis(orbinfo, mo_eps, mo_eris)
        HCIS = multp_obj.comp_hcis(ncore)
    else:    
        nocc, nvir, nmo = orbinfo
        excitation_singles = list(product(range(nocc), range(nocc, nmo)))
        nDets = (nocc * nvir) + 1
        HCIS = np.zeros((nDets, nDets))
        for P, L_ex in enumerate(excitation_singles):
            a, r = L_ex
            for Q, R_ex in enumerate(excitation_singles):
                b, s = R_ex
                HCIS[P+1, Q+1] = (((mo_eps[r] - mo_eps[a]) * (a == b) * (r == s))
                                - mo_eris[r, s, b, a]
                                + (2 * mo_eris[r, a, b, s]))
    return HCIS



class multp_cis:
    def __init__(self, orbinfo, mo_eps, mo_eris):
        """Parallelised implementation to get the HCIS Matrix
        """
        nel, nbf, nmo = orbinfo
        nocc, nvir = int(nel/2), int((nmo-nel)/2)
        self.excitation_singles = list(product(range(nocc), range(nocc, nmo)))
        self.nDets = (nocc * nvir) + 1
        self.mo_eris = mo_eris
        self.mo_eps = mo_eps

    def comp_cis_mat_row(self, P):
        row = np.zeros(self.nDets)
        try:
            a, r = self.excitation_singles[P]
            for Q, R_ex in enumerate(self.excitation_singles):
                b, s = R_ex
                row[Q+1] = (((self.mo_eps[r] - self.mo_eps[a]) * (a == b) * (r == s))
                            + 2*self.mo_eris[r, a, b, s] - self.mo_eris[r, s, b, a])
            return row, 1
        except :
            raise Exception("Something went wrong while computing row %i" % (P+1))
            return row, 0

    def comp_hcis(self, ncore):
        HCIS = np.zeros((self.nDets, self.nDets))
        with Pool(processes=ncore) as pool:
            async_object = pool.map_async(self.comp_cis_mat_row, range(self.nDets-1))
            rows_data = async_object.get()
            pool.close()
            pool.join() 
        for i in range(HCIS.shape[0]-1):
            HCIS[i+1] = rows_data[i][0]     
        # checking if all process finished succesfully
        row_log = [rows_data[i][1] for i in range(len(rows_data))]
        if sum(row_log) != int(len(self.excitation_singles)):
            for idx, val in enumerate(row_log):
                if val != 1:
                    print('Something went wrong while computing row %i' % (idx+1))
        else:
            print("All rows were computed successfully! \n")
        return HCIS


def comp_cis_edipole_r(orbinfo, mo_edipole_r):
    """Computes electric dipole operator for a particular cartesian 
    coordinate(say < -r >), in spin adapted CSF basis for CIS states.
    """
    nocc, nvir, nmo = orbinfo
    nDets = (nocc * nvir) + 1
    excitation_singles = list(product(range(nocc), range(nocc, nocc+nvir)))
    cis_edipole_r = np.zeros((nDets, nDets))
    diag_sum = 2 * np.sum(np.diag(mo_edipole_r)[:nocc])
    cis_edipole_r[0, 0] = diag_sum
    for P, L_ex in enumerate(excitation_singles):
        a, r = L_ex  # left excited slater-determinant
        # 1st row elements
        cis_edipole_r[0][P + 1] = np.sqrt(2)*mo_edipole_r[a, r]
        # 1st column elements
        cis_edipole_r[P + 1][0] = np.sqrt(2)*mo_edipole_r[r, a]
        for Q, R_ex in enumerate(excitation_singles):
            b, s = R_ex  # right excited slater-determinant
            cis_edipole_r[P + 1, Q + 1] = (((a == b)*(r == s)*diag_sum)
                                            - (r == s)*mo_edipole_r[a, b]
                                            + (a == b)*mo_edipole_r[r, s])
    return cis_edipole_r


def comp_cis_edipoles(orbinfo, mo_edipoles):
    """Compute electric dipole operators for cartesian coordinates i,e.
        -<x>, -<y>, -<z> in spin adapted CSF basis for CIS states.
    """
    cis_edipoles = [comp_cis_edipole_r(orbinfo, mo_edipole)
                    for mo_edipole in mo_edipoles]
    return cis_edipoles
