import numpy as np
from itertools import product
from multiprocessing import Pool, Process


def comp_cis_hamiltonian(eps, mo_erints, orbinfo):
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


def multp_comp_cis_hamiltonian(eps, mo_erints, orbinfo, ncore):
    """Parallelised implementation to get the HCIS Matrix
    """
    nocc, nvir, nmo = orbinfo
    excitation_singles = list(product(range(nocc), range(nocc, nmo)))
    nDets = (nocc * nvir) + 1
    HCIS = np.zeros((nDets, nDets))

    def multp_cis_mat_row(P):
        try:
            row = []
            a, r = excitation_singles[P]
            for Q, R_ex in enumerate(excitation_singles):
                b, s = R_ex
                HCIS[P+1, Q+1] = (((eps[r] - eps[a]) * (a == b) * (r == s))
                                  - mo_erints[r, s, b, a]
                                  + (2 * mo_erints[r, a, b, s]))
            return 1
        except RuntimeError("Something went wrong while computing row %i" % (P+1)):
            return 0

    with Pool(processes=ncore) as pool:
        row_log = pool.map(multp_cis_mat_row, range(len(excitation_singles)))
    # checking if all process finished succesfully
    if sum(row_log) != int(len(excitation_singles)-1):
        for idx, val in enumerate(row_log):
            if val != 0:
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
