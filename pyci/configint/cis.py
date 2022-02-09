import numpy as np
from itertools import product
from multiprocessing import Pool, Process


def comp_cis_hamiltonian(eps, mo_erints, orbinfo, parallelised=False, ncore=1):
    """Computes the configuration-interaction singles(CIS) hamiltonian in the
    spin-adapted singlet determinant basis.
    ! Note: The Hamiltonian needs scaled to SCF ground state energy, therefore 
    the eigen values need to be scaled back.
    --------------------------------------------------------------------------
    eps : array of molecular orbtial(MO) energies from SCF calculations.
    mo_erints : 4-d array of 2e- integrals using chemists notation in MO basis.
    orbinfo : (nocc, nvir, nmo) tuple to pass orbital information.
    ---------------------------------------------------------------------------
    """
    if parallelised:
        multp_obj = multp_cis(eps, mo_erints, orbinfo)
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
                HCIS[P+1, Q+1] = (((eps[r] - eps[a]) * (a == b) * (r == s))
                                - mo_erints[r, s, b, a]
                                + (2 * mo_erints[r, a, b, s]))
    return HCIS



class multp_cis:
    def __init__(self, eps, mo_erints, orbinfo):
        """Parallelised implementation to get the HCIS Matrix
        """
        nocc, nvir, nmo = orbinfo
        self.excitation_singles = list(product(range(nocc), range(nocc, nmo)))
        self.nDets = (nocc * nvir) + 1
        self.mo_erints = mo_erints
        self.eps = eps

    def comp_cis_mat_row(self, P):
        row = np.zeros(self.nDets)
        try:
            a, r = self.excitation_singles[P]
            for Q, R_ex in enumerate(self.excitation_singles):
                b, s = R_ex
                row[Q+1] = (((self.eps[r] - self.eps[a]) * (a == b) * (r == s))
                                  - self.mo_erints[r, s, b, a]
                                  + (2*self.mo_erints[r, a, b, s]))
                #print(self.HCIS[P+1, Q+1])                                  
            return row, 1
        except :
            raise Exception("Something went wrong while computing row %i" % (P+1))
            return row, 0

    def comp_hcis(self, ncore):
        HCIS = np.zeros((self.nDets, self.nDets))
        with Pool(processes=ncore) as pool:
            rows_data = pool.map(self.comp_cis_mat_row, range(self.nDets-1))
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


def comp_cis_edipole_r(mo_edipole_r, nocc, nvir):
    """Computes electric dipole operator for a particular cartesian 
    coordinate(say < -r >), in spin adapted CSF basis for CIS states.
    """
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
            cis_edipole_r[P + 1, Q + s1] = (((a == b)*(r == s)*diag_sum)
                                            - (r == s)*mo_edipole_r[a, b]
                                            + (a == b)*mo_edipole_r[r, s])
    return cis_edipole_r


def comp_cis_edipoles(mo_edipoles, nocc, nvir):
    """Compute electric dipole operators for cartesian coordinates i,e.
    -<x>, -<y>, -<z> in spin adapted CSF basis for CIS states.
    """
    cis_edipoles = [gen_cis_edipoles(mo_edipole, nocc, nvir)
                    for mo_edipole in mo_edipoles]
    return cis_edipoles
