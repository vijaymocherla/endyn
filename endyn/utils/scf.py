#!/usr/bin/env python
#
# Author : Sai Vijay Mocherla <vijaysai.mocherla@gmail.com>
#
""" Self-Consistent Field(SCF) procedure accelerated 
    with Direct Inversion of the Iterative Subspace(DIIS)
"""
import os
import numpy as np
from endyn.utils import AOint
from endyn.linalg import blas, lapack


def orthonormalize(S):
    # check_overlap = np.allclose(np.diag(S), np.ones(S.shape[0]))
    # if not check_overlap:
    #     raise Exception('Improper overlap matrix was given.')
    svals, svecs = np.linalg.eigh(S)
    X = lapack.dinv(np.diag(np.sqrt(svals)))
    X = blas.dmul_mmm(svecs, X, svecs.T)
    S_p = blas.dmul_mmm(X.T, S, X)
    # Checking if orthonormalised
    check_overlap = np.allclose(S_p, np.eye(S.shape[0]), atol=1e-08)
    if not check_overlap:
        raise Exception("There's something wrong with the transformation.")
    return S_p, X


def SCF(AOint, scfoptions={}):
    options = {
        'maxiter': 75,
        'e_convergence': 1e-12,
        'd_convergence': 1e-10,
        'diis': True,
        'diis_iter': 15,
    }
    options.update(scfoptions)
    if not os.path.isfile(AOint.scratch+'ao_oeints.npz'):
        AOint.save_ao_oeints()
    if not os.path.isfile(AOint.scratch+'ao_erints.npz'):
        AOint.save_ao_erints()
    S, T, V = AOint.get_ao_oeints()
    I = AOint.get_ao_erints()
    H = T + V
    S_p, X = orthonormalize(S)
    F_p = blas.dmul_mmm(X.T, H, X)
    eps_p, C_p = np.linalg.eigh(F_p)
    C = blas.dmul_mm(X, C_p)
    nel = (AOint.wfn.nalpha() + AOint.wfn.nbeta())
    nmo = AOint.wfn.nmo()
    if nel % 2:
        raise Exception('Open-Shell system passed RHF algorithm.')
    ndocc = int(nel/2)
    options['nmo'] = nmo
    options['ndocc'] = ndocc
    C_occ = C[:, :ndocc]
    D = blas.dmul_mm(C_occ, C_occ.T)
    MAXITER = options.get('maxiter')
    # int(N) if you want to switch of DIIS after 'N' scf cycles
    DIIS_ITER = options.get('diis_iter')
    E_CONV = options.get('e_convergence')
    D_CONV = options.get('e_convergence')
    E_old = 0.0
    D_old = np.zeros(D.shape)
    SCF_E = 0.0
    # begin Iterations
    print('==> Starting SCF Iterations <==\n')
    # Trial & Residual Vector Lists
    F_list = []
    DIIS_RESID = []
    # ==> SCF Iterations w/ DIIS <==
    for scf_iter in range(1, MAXITER + 1):
        # Build Fock matrix
        J = np.einsum('pqrs,rs->pq', I, D, optimize=True)
        K = np.einsum('prqs,rs->pq', I, D, optimize=True)
        F = H + 2*J - K
        if scf_iter <= DIIS_ITER:
            diis_bool = True
            # Build DIIS Residual
            diis_r = X.dot(F.dot(D).dot(S) - S.dot(D).dot(F)).dot(X)
            # Append trial & residual vectors to lists
            F_list.append(F)
            DIIS_RESID.append(diis_r)
        else:
            diis_bool = False
        # Compute RHF energy
        SCF_E = np.einsum('pq,pq->', (H + F), D, optimize=True)
        dE = SCF_E - E_old
        dRMS = np.mean(diis_r**2)**0.5
        if diis_bool:
            diis_str = '   DIIS is ON'
        else:
            diis_str = '   DIIS is OFF'
        print(('SCF Iteration %3d: Energy = %4.16f dE = % 1.5E dRMS = %1.5E' %
               (scf_iter, SCF_E, dE, dRMS)) + diis_str)
        # Checking if SCF Converged?
        # if (abs(dE) <= E_CONV):
        #     break
        if np.allclose(D, D_old, rtol=D_CONV):
            break
        E_old = SCF_E
        D_old = D
        if scf_iter >= 2:
            # Build B matrix
            B_dim = len(F_list) + 1
            B = np.empty((B_dim, B_dim))
            B[-1, :] = -1
            B[:, -1] = -1
            B[-1, -1] = 0
            for i in range(len(F_list)):
                for j in range(len(F_list)):
                    B[i, j] = np.einsum('ij,ij->', DIIS_RESID[i],
                                        DIIS_RESID[j], optimize=True)
            # Build RHS of Pulay equation
            rhs = np.zeros((B_dim))
            rhs[-1] = -1
            # Solve Pulay equation for c_i's with NumPy
            coeff = lapack.dsolve(B, rhs)
            # Build DIIS Fock matrix
            F = np.zeros_like(F)
            for x in range(coeff.shape[0] - 1):
                F += coeff[x] * F_list[x]

        # Compute new orbital guess with DIIS Fock matrix
        F_p = X.dot(F).dot(X)
        e, C_p = np.linalg.eigh(F_p)
        C = X.dot(C_p)
        ndocc = options.get('ndocc')
        C_occ = C[:, :ndocc]
        D = np.einsum('pi,qi->pq', C_occ, C_occ, optimize=True)
        # MAXITER exceeded?
        if (scf_iter == MAXITER):
            raise Exception("Maximum number of SCF iterations exceeded.")
    # Post iterations
    print('\nSCF converged.')
    print('Final RHF Energy: %.12f [Eh]' % SCF_E)
    return SCF_E, C
