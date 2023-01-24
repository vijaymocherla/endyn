#!/usr/bin/env python
#
# Author: Sai Vijay Mocherla <vijaysai.mocherla@gmail.com>
#
"""rungekutta.py
"""

import numpy as np
from time import perf_counter
from endyn.utils import units
from threadpoolctl import threadpool_limits
from endyn.linalg.blas import zmul_mv, zmul_zdotc, zmul_zdotu


def _calc_expectations(ops_list, psi_i, psi_0):
    ops_expts = []
    norm = np.abs(zmul_zdotc(psi_i, psi_i))
    autocorr = np.abs(zmul_zdotc(psi_i, psi_0))
    for operator in ops_list:
        expt = zmul_zdotc(psi_i, zmul_mv(operator, psi_i)).real
        ops_expts.append(expt)
    return ops_expts, norm, autocorr


def RK4(func, psi_0, time_params, ncore=4, ops_list=[], ops_headers=[],
        print_nstep=1, outfile='tdprop.txt'):
    """ Fixed step-size implementation of fourth order runge-kutta 
    """
    t0, tf, dt = time_params
    psi_i, ti = np.copy(psi_0), t0
    fobj = open(outfile, 'wb', buffering=0)
    ncols = 3 + len(ops_list)
    fobj.write((" {:<19} "*(ncols)+"\n").format('time_fs',
               'norm', 'autocorr', *ops_headers).encode("utf-8"))
    ti_fs = ti / units.fs_to_au
    ops_expt, norm, autocorr = _calc_expectations(ops_list, psi_i, psi_0)
    fobj.write((" {:>16.16f} "*(ncols)+"\n").format(ti_fs, norm, autocorr, *ops_expt).encode("utf-8"))
    start = perf_counter()
    with threadpool_limits(limits=ncore, user_api='blas'):
        while ti <= tf:
            for i in range(print_nstep):
                Fi = func(ti)
                k1 = zmul_mv(Fi, psi_i)
                k2 = zmul_mv(Fi, (psi_i + (dt*0.5)*k1))
                k3 = zmul_mv(Fi, (psi_i + (dt*0.5)*k2))
                k4 = zmul_mv(Fi, (psi_i + (dt*1.0)*k3))
                psi_i += ((dt/6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4))
                ti += dt
            ti_fs = ti / units.fs_to_au
            ops_expt, norm, autocorr  = _calc_expectations(ops_list, psi_i, psi_0)
            fobj.write((" {:>16.16f} "*(ncols)+"\n").format(ti_fs, norm, autocorr, *ops_expt).encode("utf-8"))
    stop = perf_counter()
    print('Time taken %3.3f seconds' % (stop-start))
    return 0
