#!/usr/bin/env python
#
# Author : Sai Vijay Mocherla <vijaysai.mocherla@gmail.com>
#
"""CISD bitstring implementation
"""

import numpy as np
import gc
from multiprocessing import Pool
from pyci.configint.bitstrings import bitDet, SlaterCondon
from pyci.configint.csf import CSF
from pyci.configint.csf import gen_singlet_singles, gen_singlet_doubles

class CISD:
    def __init__(self, orbinfo, mo_eps, mo_coeff, mo_erints, mo_edipoles, active_space):
        self.orbinfo = orbinfo
        self.mo_eps = mo_eps
        self.mo_coeff = mo_coeff
        self.mo_erints = mo_erints
        self.mo_edipoles = mo_edipoles
        self.csf_list, self.exc_list = CISD.gen_csfs(self.orbinfo, active_space )
        self.nCSFs = len(self.csf_list)
        self.SlaterCondonRule = SlaterCondon(orbinfo, mo_eps, mo_coeff, mo_erints).comp_hmatrix_elem

    
    @staticmethod
    def gen_csfs(orbinfo, active_space=(1,1), full_cis=True):
        """Generates list of CSFs for given active space options
        """
        nel, nbf, nmo = orbinfo
        nocc, nvir = int(nel/2), int((nmo-nel)/2)
        act_occ, act_vir = active_space
        occ_list = range(nocc-act_occ, nocc)
        vir_list = range(nocc, nocc+act_vir)
        refDet = bitDet(alpha_orblist=list(range(nocc)), beta_orblist=list(range(nocc)))
        csf_list = [CSF({refDet : 1.0})]
        exc_list = [(0)]
        # Singles
        if full_cis:
            csf_list, exc_list = gen_singlet_singles(refDet, range(nocc), range(nocc,nmo), csf_list, exc_list)
        else:
            csf_list, exc_list = gen_singlet_singles(refDet, occ_list, vir_list, csf_list, exc_list)
        # Doubles
        csf_list, exc_list = gen_singlet_doubles(refDet, occ_list, vir_list, csf_list, exc_list)
        return csf_list, exc_list
    
    def comp_cisd_hmatrix_elem(self, csf1, csf2):
        """Evaluates H-matrix elements using Slater-Condon rule
        """
        matrix_elem = sum([ci*cj*self.SlaterCondonRule(I,J) 
                            for ci, I in list(zip(csf1.coeff, csf1.Dets))
                            for cj, J in list(zip(csf2.coeff, csf2.Dets))])
        return matrix_elem
    
    def comp_cisd_hmatrix_row(self, P):
        """Computes only elements of upper triangular matrix in each row
        """
        row = np.zeros(self.nCSFs)
        try:
            for Q in range(P, self.nCSFs):
                csf1, csf2 = self.csf_list[P], self.csf_list[Q]
                row[Q] = self.comp_cisd_hmatrix_elem(csf1, csf2)
            return row, 1
        except :
            raise Exception("Something went wrong while computing row %i" % (P+1))
            return row, 0

    def comp_hcisd(self, ncore):
        """ Parallel computation HCISD
        """
        with Pool(processes=ncore) as pool:
            async_object = pool.map_async(self.comp_cisd_hmatrix_row, list(range(self.nCSFs)))
            rows_data = async_object.get()
            pool.close()
            pool.join() 
        # checking if all process finished succesfully
        row_log = [rows_data[i][1] for i in range(len(rows_data))]
        if sum(row_log) != self.nCSFs:
            for idx, val in enumerate(row_log):
                if val != 1:
                    print('Something went wrong while computing row %i' % (idx+1))
        else:
            print("All rows were computed successfully! \n")    
        HCISD = np.empty((self.nCSFs,self.nCSFs))
        for i in range(self.nCSFs):
            HCISD[i] = rows_data[i][0]
        del rows_data
        gc.collect()
        for P in range(self.nCSFs):
            for Q in range(P, self.nCSFs):
                HCISD[Q,P] = HCISD[P,Q]
        return HCISD
    
    def make_rdm1(self, active_space):
        nel, nbf, nmo = self.orbinfo
        nocc, nvir = int(nel/2), int((nmo-nel)/2)
        act_occ, act_vir = active_space
        rdm1 = np.zeros((nmo, nmo))
        return rdm1
