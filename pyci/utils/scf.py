#!/usr/bin/env python
#
# Author : Sai Vijay Mocherla <vijaysai.mocherla@gmail.com>
#
""" Self-Consistent Field(SCF) procedure accelerated with Direct Inversion 
    of the Iterative Subspace(DIIS)
"""
import numpy as np
from pyci.utils import AOint
### TO DO : 
#    - Repurpose following code into an SCF function
#    - Add options to control DIIS
#    - Check future compatibility for EDIIS and other variational relaxation   
# S = np.load('../data/h2o_oeints.npz')['overlap']
# T = np.load('../data/h2o_oeints.npz')['kinetic']
# V = np.load('../data/h2o_oeints.npz')['potential']
# I = np.load('../data/h2o_erints.npz')['erints']

def orthonormalize(S, atol=1e-8):
    # Diagonalise the Overlap matrix
    svals, svecs = np.linalg.eigh(S)
    # Taking the inverse square root
    X = np.linalg.inv(np.diag(np.sqrt(svals)))
    # Using the eigen vector back transform from eigen basis to previous basis.
    X = np.einsum('aI,IJ,Jb', svecs, X, svecs.T)
    # Orthonormalising S
    S_p = np.einsum('ij,jk,kl->il', X.T, S, X)
    orthonormalised = np.allclose(S_p, np.eye(S.shape[0]), atol=atol)
    if orthonormalised:
        print("There is a chance for diagonalisation")
    else:
        raise Exception("There's something wrong with symmetric orthonormalisation")
    return S_p, X

def get_density(H, X, nocc):
    # Transforming the Fock matrix
    F_p = np.einsum('ij,jk,kl->il', X.T, H, X)
    # Diagonalising the Fock matrix eigenvalues and eigenvectors
    e, C_p = np.linalg.eigh(F_p)
    # Transform C_p back into AO basis
    C = np.dot(X, C_p)
    # slicing out occupied orbitals
    C_occ = C[:, :nocc]
    # Building density matrix from C_occ
    D = np.einsum('pi,qi->pq', C_occ, C_occ, optimize=True)
    return D

def scf(S,T,V,I, orbinfo):
    nocc, nmo = orbinfo
    S_p, X = orthonormalize(S)
    H = T + V
    D = get_density(H, X, nocc)
    # ==> SCF Iterations <==
    # Maximum SCF iterations
    MAXITER = 50
    # Energy convergence criterion
    E_conv = 1.0e-10
    D_conv = 1.0e-8
    # Pre-iteration energy declarations
    SCF_E = 0.0
    E_old = 0.0
    # Trial & Residual Vector Lists
    diis_iter = MAXITER # int(N) if you want to switch of DIIS after 'N' scf cycles 
    F_list = []
    DIIS_RESID = []
    # ==> SCF Iterations w/ DIIS <==
    print('==> Starting SCF Iterations <==\n')
    # Begin Iterations
    for scf_iter in range(1, MAXITER + 1):
        # Build Fock matrix
        J = np.einsum('pqrs,rs->pq', I, D, optimize=True)
        K = np.einsum('prqs,rs->pq', I, D, optimize=True)
        F = H + 2*J - K
        if scf_iter <= diis_iter:
            #print('DIIS is ON \n')
            diis_bool = True
            # Build DIIS Residual
            diis_r = X.dot(F.dot(D).dot(S) - S.dot(D).dot(F)).dot(X)
            # Append trial & residual vectors to lists
            F_list.append(F)
            DIIS_RESID.append(diis_r)
        else:
            #print('DIIS is OFF \n')
            diis_bool = False
        # Compute RHF energy
        SCF_E = np.einsum('pq,pq->', (H + F), D, optimize=True)
        dE = SCF_E - E_old
        dRMS = np.mean(diis_r**2)**0.5
        if diis_bool:
            diis_str = '   DIIS is ON'
        else:
            diis_str = '   DIIS is OFF'    
        print(('SCF Iteration %3d: Energy = %4.16f dE = % 1.5E dRMS = %1.5E' % (scf_iter, SCF_E, dE, dRMS)) + diis_str )
        
        # SCF Converged?
        if (abs(dE) < E_conv):
            break
        E_old = SCF_E
        
        if scf_iter >= 2:
            # Build B matrix
            B_dim = len(F_list) + 1
            B = np.empty((B_dim, B_dim))
            B[-1, :] = -1
            B[:, -1] = -1
            B[-1, -1] = 0
            for i in range(len(F_list)):
                for j in range(len(F_list)):
                    B[i, j] = np.einsum('ij,ij->', DIIS_RESID[i], DIIS_RESID[j], optimize=True)
    
            # Build RHS of Pulay equation 
            rhs = np.zeros((B_dim))
            rhs[-1] = -1
            
            # Solve Pulay equation for c_i's with NumPy
            coeff = np.linalg.solve(B, rhs)
            
            # Build DIIS Fock matrix
            F = np.zeros_like(F)
            for x in range(coeff.shape[0] - 1):
                F += coeff[x] * F_list[x]
        
        # Compute new orbital guess with DIIS Fock matrix
        F_p =  X.dot(F).dot(X)
        e, C_p = np.linalg.eigh(F_p)
        C = X.dot(C_p)
        C_occ = C[:, :ndocc]
        D = np.einsum('pi,qi->pq', C_occ, C_occ, optimize=True)
        
        # MAXITER exceeded?
        if (scf_iter == MAXITER):
            raise Exception("Maximum number of SCF iterations exceeded.")
    # Post iterations
    print('\nSCF converged.')
    print('Final RHF Energy: %.8f [Eh]' % SCF_E)
    return SCF_E