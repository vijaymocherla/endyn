#!/usr/bin/env python
#
# Author : Sai Vijay Mocherla
#
"""cranknicholson.py
"""

import numpy as np
from time import perf_counter
from pyci.utils import units
from pyci.linalg.lapack import dmat_inv, zmat_inv
#from scipy.linalg import blas
from pyci.linalg import blas
ALPHA = 1.0+0j

def project_matrix_eigbasis(matrix, eigvecs):
    """Projects a matrix from CSF basis => EIGEN basis 
    """
    # matrix_eig = np.einsum('iA, AB, Bj -> ij', np.conjugate(self.eigvecs.T), matrix, self.eigvecs, optimize=True) 
    matrix_eig = blas.zmul_mmm(eigvecs.T, matrix, eigvecs)
    return matrix_eig

def project_vec_eig(y, eigvecs):
    """Projects y from CSF basis => EIGEN basis 
    """
    y_eig = blas.zmul_mv(np.conjugate(eigvecs), y)
    return y_eig

def project_vec_csf(y, eigvecs):
    """Projects y from EIGEN basis => CSF basis 
    """
    y_csf = blas.zmul_mv(eigvecs.T, y)
    return y_csf

def ops_expt(yi_dag, operator, yi):
    # NOTE:
    # In the following case for the operation (vec.matrix.vec), 
    # the implement scheme is 10-100x faster than the version
    # where the array.flags are not check for memory form
    # #
    temp = blas.zgemm(ALPHA, operator.T, yi.T, trans_a=True)[:,0]
    temp = blas.zgemm(ALPHA, yi_dag.T, temp, trans_a=True)
    expt = np.real(temp)[0][0]
    return expt


def _calc_expectations(yi_eig, ti, ops_list, eigvecs, y0_csf, fobj, ncols):
    ops_expectations = []
    yi_csf = project_vec_csf(yi_eig, eigvecs)
    norm = abs(np.sum(np.conjugate(yi_csf.T, dtype=np.cdouble) * yi_csf))
    autocorr = abs(np.sum(np.conjugate(yi_csf.T, dtype=np.cdouble) * y0_csf))
    for operator in ops_list:
        expectation = ops_expt(np.conjugate(yi_csf), operator, yi_csf) 
        ops_expectations.append(np.real(expectation))
    ti_fs = ti / units.fs_to_au
    fobj.write((" {:>16.16f} "*(ncols)+"\n").format(ti_fs, norm, autocorr, *ops_expectations).encode("utf-8"))
    return 0

def CrankNicholson(eigvals, eigvecs, field_func_eig, y0, time_params, 
                ncore=4, ops_list=[], ops_headers=[], 
                print_nstep= 1, outfile='tdprop.txt'):
    """An implementation of a Caley propagator or the Crank-Nicholson method.
    """
    Hfunc_eig = lambda t: np.diag(eigvals) - field_func_eig(t)
    ndim = y0.shape[0]
    y0_csf = y0
    y0_eig = project_vec_eig(y0, eigvecs)   # initial state in EIGEN basis
    t0, tf, dt = time_params       # time params
    yi_eig, ti = y0_eig, t0
    fobj= open(outfile, 'wb', buffering=0)
    ncols = 3 + len(ops_list)
    fobj.write((" {:<19} "*(ncols)+"\n").format('time_fs', 'norm', 'autocorr', *ops_headers).encode("utf-8"))
    _calc_expectations(yi_eig, ti, ops_list, eigvecs, y0_csf, fobj, ncols)
    start = perf_counter()
    I_matrix = np.identity(ndim)
    H_plus = lambda t: I_matrix + (1j*dt/2)*Hfunc_eig(t)
    H_minus = lambda t: I_matrix - (1j*dt/2)*Hfunc_eig(t)
    while ti <= tf:
        for i in range(print_nstep):
            Umatrix = blas.zmul_mm(zmat_inv(H_plus(ti+dt/2)), H_minus(ti+dt/2)) 
            yi_eig =  blas.zmul_mv(Umatrix, yi_eig)
            ti += dt
        _calc_expectations(yi_eig, ti, ops_list, eigvecs, y0_csf, fobj, ncols)
    fobj.close()
    stop = perf_counter()
    print('Time taken %3.3f seconds' % (stop-start))    
    return 0
