import numpy as np
from itertools import product
from multiprocessing import Pool

class cis_d:
    """ A class to compute perturbative corrections from doubles(D) for enegies from configuration interation singles(CIS).
    """
    def __init__(self, cis_eigvals, cis_eigvecs, eps, mo_erints, orbinfo, charge_cloud=False):
        """input args : 
            cis_eigvals: eigenvalues of the CIS hamiltonian,
            cis_eigvecs: eigenvectors of the CIS hamiltonian,
            eps: energies of the Molecular Orbitals(MOs) used for CI calculation
            mo_erints: electron repulsion integrals tensor in MO basis .
            charge_cloud: "True" if 2 e-integrals are in chemists notation.
            orbinfo: a tuple of the form (nocc, nvir, nmo)
            nocc: no. of occupied orbitals
            nvir: no. of virtual orbitals
            nmo: no. of molecular orbitals 
        """
        self.nocc, self.nvir, self.nmo = orbinfo
        self.ecis = cis_eigvals
        self.ccis = cis_eigvecs[1:, :]
        self.occ_list = range(self.nocc)
        self.vir_list = range(self.nocc, self.nmo)
        if charge_cloud:
            mo_erints = mo_erints.transpose(0, 2, 1, 3)
        self.mo_erints = mo_erints
        self.delta = self.comp_delta(eps)
        self.vvvo = mo_erints[self.nocc:self.nmo, self.nocc:self.nmo, self.nocc:self.nmo, :self.nocc]
        self.ovoo = mo_erints[:self.nocc, self.nocc:self.nmo, :self.nocc, :self.nocc]

    def comp_e0mp2(self):
        vvoo = self.mo_erints[self.nocc:self.nmo, self.nocc:self.nmo, :self.nocc, :self.nocc]
        oovv = vvoo.transpose(2, 3, 0, 1)
        E0_mp2 = (2*np.einsum('abrs,rsab', oovv, -vvoo/self.delta, optimize=True)
                  - np.einsum('abrs,rsba', oovv, -vvoo/self.delta, optimize=True))
        return E0_mp2

    def comp_delta(self, eps):
        delta = np.full((self.nmo, self.nmo, self.nmo, self.nmo), 1.0)
        iterlist = list(product(self.vir_list, self.vir_list, self.occ_list, self.occ_list))
        for conf in iterlist:
            r, s, a, b = conf
            delta[conf] = eps[r] + eps[s] - eps[a] - eps[b]
        return delta[self.nocc:self.nmo, self.nocc:self.nmo, :self.nocc, :self.nocc]

    def comp_utensor(self, Ci):
        return utensor

    def comp_vterm1(self):
        vvoo = self.mo_erints[self.nocc:self.nmo, self.nocc:self.nmo, :self.nocc, :self.nocc]
        oovv = vvoo.transpose(2, 3, 0, 1)
        i11 = np.einsum('bcst,trbc->sr', oovv, vvoo, optimize=True)
        i12 = np.einsum('bcst,trcb->sr', oovv, vvoo, optimize=True)
        i13 = np.einsum('bcts,trbc->sr', oovv, vvoo, optimize=True)
        i14 = np.einsum('bcts,trcb->sr', oovv, vvoo, optimize=True)
        vterm1 = 2*(i11 + i14) - 4*(i12 + i13)
        return vterm1

    def comp_vterm2(self):
        vvoo = self.mo_erints[self.nocc:self.nmo, self.nocc:self.nmo, :self.nocc, :self.nocc]
        oovv = vvoo.transpose(2, 3, 0, 1)
        i21 = np.einsum('bcst,tsac->ab', oovv, vvoo, optimize=True)
        i22 = np.einsum('bcst,tsca->ab', oovv, vvoo, optimize=True)
        i23 = np.einsum('bcts,tsac->ab', oovv, vvoo, optimize=True)
        i24 = np.einsum('bcts,tsca->ab', oovv, vvoo, optimize=True)
        vterm2 = 2*(i21 + i24) - 4*(i22 + i23)
        return vterm2

    def comp_vterm3(self):
        vvoo = self.mo_erints[self.nocc:self.nmo, self.nocc:self.nmo, :self.nocc, :self.nocc]
        oovv = vvoo.transpose(2, 3, 0, 1)
        i31 = np.einsum('bcst,rtac->bsra', oovv, vvoo, optimize=True)
        i32 = np.einsum('bcst,rtca->bsra', oovv, vvoo, optimize=True)
        i33 = np.einsum('bcts,rtac->bsra', oovv, vvoo, optimize=True)
        i34 = np.einsum('bcts,rtca->bsra', oovv, vvoo, optimize=True)
        vterm3 = 8*i31 - 4*(i32 + i33) + 2*i34
        return vterm3

    def comp_varray(self, Ci):
        varray = 0.5*(np.einsum('as,sr->ar', Ci, self.vterm1, optimize=True)
                      + np.einsum('br,ab->ar', Ci, self.vterm2, optimize=True)
                      + 2*np.einsum('bs,bsra->ar', Ci, self.vterm3, optimize=True))
        return varray

    def comp_utensor(self, Ci):
        utensor = 4*(np.einsum('rstb,at->rsab', self.vvvo, Ci, optimize=True)
                     - np.einsum('rsta,bt->rsab', self.vvvo, Ci, optimize=True)
                     + np.einsum('crab,cs->rsab', self.ovoo, Ci, optimize=True)
                     - np.einsum('csab,cr->rsab', self.ovoo, Ci, optimize=True))
        return utensor

    def get_idx(self, occ_idx, vir_idx):
        """Get basis index of single det in cis eigenvectors
        """
        idx = (occ_idx*self.nvir) + (vir_idx-self.nocc)
        return idx

    def reshape_ci(self, Ci):
        """Reshapes eigen vector Ci, such each row corresponds to
           a coefficients of a single from a particular occ. orb
        """
        new_ci = np.zeros((self.nocc, self.nvir))
        iterlist = list(product(self.occ_list, self.vir_list))
        for state in iterlist:
            i, j = state
            new_ci[i, (j-self.nocc-1)] = Ci[self.get_idx(i, j)]
        print(np.allclose(Ci.reshape(self.nocc, self.nvir), new_ci))
        return new_ci

    def comp_dcorr(self, i):
        """Computes doubles correction for the ith cis eigen value
        """
        Ei = self.ecis[i]
        Ci = 1/np.sqrt(2) * self.ccis[:, i]
        Ci = self.reshape_ci(Ci)
        Ui = self.comp_utensor(Ci)
        Vi = self.comp_varray(Ci)
        # print(Ui.shape, self.delta.shape, Vi.shape, Ci.shape)
        Ei_corr = -0.25*np.sum((Ui**2 / (self.delta - Ei))) + np.sum(Ci*Vi)
        return Ei_corr

    def comp_cis_d(self, nvals, ncore=4):
        """Computes of doubles substitution correction to CIS eigen energies.
        The calculation is parallelised with multiprocessing.Pool
        -----------------------------------------------------------------------------
        nvals: no. of first few eigen values for which corrections are to be computed.
        ncore: no. of cores to be used for computations(default = 4).
        """
        if nvals > len(self.ecis):
            raise Exception('nvals can not be greater than no. of CIS eigen values')
        self.vterm1 = self.comp_vterm1()
        self.vterm2 = self.comp_vterm2()
        self.vterm3 = self.comp_vterm3()
        print('Ui.shape, self_energy.shape, Vi.shape, Ci.shape')
        with Pool(ncore) as p:
            E_CIS_D = p.map(self.comp_dcorr, range(nvals))
        return E_CIS_D
