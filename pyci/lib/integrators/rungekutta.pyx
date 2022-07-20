#!/usr/bin/env python
#
#   Author: Sai Vijay Mocherla <vijaysai.mocherla@gmail.com>
#
import numpy as np 
from time import perf_counter
from scipy.linalg import blas
from pyci.utils import units

ALPHA = 1.0+0j


def ops_expt(yi_dag, operator, yi):
    #NOTE:
    # In the following case for the operation (vec.matrix.vec), 
    # the implement scheme is 10-100x faster than the version
    # where the array.flags are not check for memory form
    # #
    temp = blas.zgemm(ALPHA, operator.T, yi.T, trans_a=True)[:,0]
    temp = blas.zgemm(ALPHA, yi_dag.T, temp, trans_a=True)
    expt = np.real(temp)[0][0]
    return expt

def _calc_expectations(ops_list, yi, ti):
    ops_expts = []
    yi_dag = np.conjugate(yi)
    ti_fs = ti / units.fs_to_au
    norm = np.real(np.sum(yi_dag * yi))
    for operator in ops_list:
        expt = ops_expt(yi_dag, operator, yi)
        ops_expts.append(expt)
    return ti_fs, norm, ops_expts

def RK4(func, y0, time_params, ncore=4, ops_list=[], ops_headers=[], 
        print_nstep= 1, outfile='tdprop.txt'):
    """Fixed step-size implementation of fourth order runge-kutta 
    """
    t0, tf, dt = time_params        
    yi, ti = y0, t0
    fobj = open(outfile, 'wb', buffering=0)
    ncols = 2 + len(ops_list)
    fobj.write((" {:<19} "*(ncols)+"\n").format('time_fs', 'norm', *ops_headers).encode("utf-8"))
    ti_fs, norm, ops_expt = _calc_expectations(ops_list, yi, ti)
    fobj.write((" {:>16.16f} "*(ncols)+"\n").format(ti_fs, norm, *ops_expt).encode("utf-8"))
    start = perf_counter()
    while ti <= tf:
        # NOTE:   
        # check `Fi.flags` and `yi.flags` before passing arrays to 
        # methods from scipy.linalg.blas as, it can give significant speedups.
        # >>> scheme1 = blas.zgemm(ALPHA, Fi, yi)
        # >>> scheme2 = blas.zgemm(ALPHA, Fi.T, yi.T, trans_a=True) 
        # scheme2 is ~1.5x faster than scheme1 as the 2d numpy array
        # Fi can be NOT F_CONTIGUOUS but is C_CONTIGUOUS.
        # For more information see: https://scipy.github.io/old-wiki/pages/PerformanceTips
        # 
        for i in range(print_nstep):    
            Fi = func(ti)
            k1 = blas.zgemm(ALPHA, Fi.T, yi.T, trans_a=True)[:,0]
            k2 = blas.zgemm(ALPHA, Fi.T, (yi + (dt/2.0)*k1).T, trans_a=True)[:,0]
            k3 = blas.zgemm(ALPHA, Fi.T, (yi + (dt/2.0)*k2).T, trans_a=True)[:,0]
            k4 = blas.zgemm(ALPHA, Fi.T, (yi + dt*k3).T, trans_a=True)[:,0]
            yi += ((dt/6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4))
            ti = ti + dt
        ti_fs, norm, ops_expt = _calc_expectations(ops_list, yi, ti)
        fobj.write((" {:>16.16f} "*(ncols)+"\n").format(ti_fs, norm, *ops_expt).encode("utf-8"))
    stop = perf_counter()
    print('Time taken %3.3f seconds' % (stop-start))    
    return 0