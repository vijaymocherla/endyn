#!/usr/bin/env python
#
# Author : Sai Vijay Mocherla <vijaysai.mocherla@gmail.com>
#
"""splitoperator.py
"""
# NOTE:   
# ----
# Check `Fi.flags` and `yi.flags` before passing arrays to 
# methods from scipy.linalg.blas as, it can give significant speedups.
# >>> scheme1 = blas.zgemm(ALPHA, Fi, yi)
# >>> scheme2 = blas.zgemm(ALPHA, Fi.T, yi.T, trans_a=True) 
# scheme2 is ~1.5x faster than scheme1 as the 2d numpy array
# Fi can be NOT F_CONTIGUOUS but is C_CONTIGUOUS.
# For more information see: https://scipy.github.io/old-wiki/pages/PerformanceTips
# ----

import numpy as np 
from scipy.linalg import expm, blas
from time import perf_counter
from pyci.utils import units

ALPHA = 1.0+0j

def project_matrix_eigbasis(matrix, eigvecs):
    """Projects a matrix from CSF basis => EIGEN basis 
    """
    # matrix_eig = np.einsum('iA, AB, Bj -> ij', np.conjugate(self.eigvecs.T), matrix, self.eigvecs, optimize=True) 
    temp = blas.zgemm(ALPHA, matrix.T, eigvecs, trans_a=True)
    matrix_eig = blas.zgemm(ALPHA, np.conjugate(eigvecs.T), temp)
    return matrix_eig

def project_vec_eig(y, eigvecs):
    """Projects y from CSF basis => EIGEN basis 
    """
    y_eig = blas.zgemm(ALPHA, np.conjugate(eigvecs), y, trans_a=True)[:,0]
    return y_eig

def project_vec_csf(y, eigvecs):
    """Projects y from EIGEN basis => CSF basis 
    """
    y_csf = blas.zgemm(ALPHA, eigvecs.T, y, trans_a=True)[:,0]
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


def SplitOperator(eigvals, eigvecs, field_func, y0, time_params, 
                  ncore=4, ops_list=[], ops_headers=[], 
                  print_nstep= 1, outfile='tdprop.txt'):
    """Given eigenvalues and eigen-vectors(in a certain basis), the methods of 
    exact_prop() aids in exact time-propagation.  
    """
    y0_csf = y0
    y0_eig = project_vec_eig(y0, eigvecs)   # initial state in EIGEN basis
    t0, tf, dt = time_params       # time params
    yi_eig, ti = y0_eig, t0
    fobj= open(outfile, 'wb', buffering=0)
    ncols = 3 + len(ops_list)
    fobj.write((" {:<19} "*(ncols)+"\n").format('time_fs', 'norm', 'autocorr', *ops_headers).encode("utf-8"))
    _calc_expectations(yi_eig, ti, ops_list, eigvecs, y0_csf, fobj, ncols)
    start = perf_counter()
    while ti <= tf:
        for i in range(print_nstep):
            exp_field = project_matrix_eigbasis(expm(1j*field_func(ti)*dt), eigvecs)
            yn_eig = np.exp(-1j*eigvals*dt) * yi_eig
            yn_eig = blas.zgemm(ALPHA, exp_field.T, yn_eig, trans_a=True)[:,0]
            ti = ti + dt
        _calc_expectations(yi_eig, ti, ops_list, eigvecs, y0_csf, fobj, ncols)
    fobj.close()
    stop = perf_counter()
    print('Time taken %3.3f seconds' % (stop-start))    
    return 0