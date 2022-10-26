#!/usr/bin/env python
#
# Author : Sai Vijay Mocherla <vijaysai.mocherla@gmail.com>
#
"""splitoperator.py
"""

import numpy as np
from scipy.linalg import expm, blas
from time import perf_counter
from pyci.utils import units
from pyci.linalg.blas import zmul_mm, zmul_mmm
from pyci.linalg.blas import zmul_mv, zmul_zdotc


def _calc_expectations(ops_list, psi_i, psi_0):
    ops_expt = []
    norm = zmul_zdotc(psi_i, psi_i).real
    autocorr = zmul_zdotc(psi_i, psi_0).real
    for operator in ops_list:
        expt = zmul_zdotc(psi_i, zmul_mv(operator, psi_i)).real
        ops_expt.append(expt)
    return ops_expt, norm, autocorr


def SplitOperator(eigvals, eigvecs, func, psi_0, time_params,
                  ncore=4, ops_list=[], ops_headers=[],
                  print_nstep=1, outfile='tdprop.txt'):
    """ Given eigenvalues and eigen-vectors(in a certain basis), the methods of 
        exact_prop() aids in exact time-propagation.  
    """
    psi_0_eig = zmul_mv(eigvecs, psi_0)   # projecting psi_0 into EIGEN basis
    t0, tf, dt = time_params    # time params
    psi_i_eig, ti = psi_0_eig, t0
    fobj = open(outfile, 'wb', buffering=0)
    ncols = 3 + len(ops_list)
    fobj.write((" {:<19} "*(ncols)+"\n").format('time_fs',
               'norm', 'autocorr', *ops_headers).encode("utf-8"))
    ops_expt, norm, autocorr = _calc_expectations(ops_list, psi_0, psi_0)
    ti_fs = ti / units.fs_to_au
    fobj.write((" {:>16.16f} "*(ncols)+"\n").format(ti_fs, norm, autocorr, *ops_expt).encode("utf-8"))
    start = perf_counter()
    while ti <= tf:
        for i in range(print_nstep):
            exp_field = zmul_mmm(eigvecs.T, expm(1j*func(ti)*dt), eigvecs)
            psi_i_eig = np.exp(-1j*eigvals*dt) * psi_i_eig
            psi_i_eig = zmul_mv(exp_field, psi_i_eig)
            ti += dt
        psi_i = zmul_mv(eigvecs.T, psi_i_eig)
        ops_expt, norm, autocorr = _calc_expectations(ops_list, psi_i, psi_0)
        ti_fs = ti / units.fs_to_au
        fobj.write((" {:>16.16f} "*(ncols)+"\n").format(ti_fs, norm, autocorr, *ops_expt).encode("utf-8"))
    fobj.close()
    stop = perf_counter()
    print('Time taken %3.3f seconds' % (stop-start))
    return 0
