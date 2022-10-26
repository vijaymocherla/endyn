#!/usr/bin/env python
#
# Author : Sai Vijay Mocherla
#
"""cranknicholson.py
"""

import numpy as np
from time import perf_counter
from pyci.utils import units
from pyci.linalg.lapack import zsolve
from pyci.linalg.blas import zmul_mm, zmul_mmm
from pyci.linalg.blas import zmul_mv, zmul_zdotc


def _calc_expectations(yi_eig, ti, ops_list, eigvecs, y0_csf, fobj, ncols):
    ops_expt = []
    yi_csf = zmul_mv(eigvecs.T, yi_eig)
    norm = zmul_zdotc(yi_csf, yi_csf).real
    autocorr = zmul_zdotc(yi_csf, y0_csf).real
    for operator in ops_list:
        expt = zmul_zdotc(yi_csf, zmul_mv(operator, yi_csf)).real
        ops_expt.append(expt)
    ti_fs = ti / units.fs_to_au
    fobj.write((" {:>16.16f} "*(ncols)+"\n").format(ti_fs, norm, autocorr, *ops_expt).encode("utf-8"))
    return 0

def CrankNicholson(eigvals, eigvecs, field_func_eig, y0, time_params, 
                ncore=4, ops_list=[], ops_headers=[], 
                print_nstep= 1, outfile='tdprop.txt'):
    """An implementation of a Caley propagator or the Crank-Nicholson method.
    """
    Hfunc_eig = lambda t: np.diag(eigvals) - field_func_eig(t)
    ndim = y0.shape[0]
    y0_csf = y0
    y0_eig = zmul_mv(eigvecs.T, y0)
   # initial state in EIGEN basis
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
            A = H_plus(ti+dt/2)
            b = zmul_mv(H_minus(ti+dt/2), yi_eig)
            yi_eig = zsolve(A, b)
            ti += dt
        _calc_expectations(yi_eig, ti, ops_list, eigvecs, y0_csf, fobj, ncols)
    fobj.close()
    stop = perf_counter()
    print('Time taken %3.3f seconds' % (stop-start))    
    return 0