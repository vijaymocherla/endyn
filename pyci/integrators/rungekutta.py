#!/usr/bin/env python
#
#   Author: Sai Vijay Mocherla <vijaysai.mocherla@gmail.com>
#
import os
os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
import numpy as np 
from time import perf_counter
from opt_einsum import contract
from pyci.utils import units

class RK4(object):
    """Fixed step-size implementation of fourth order runge-kutta 
    """
    def __init__(self, func, y0, time_params):
        self.func = func
        self.y0 = np.array(y0, dtype=np.cdouble)
        self.t0, self.tf, self.dt = time_params

    def _rk4_step(self, yi, ti):
        k1 = contract('ij,i', self.func(ti), yi, optimize=True)
        k2 = contract('ij,i', self.func(ti), yi+((self.dt/2.0) * k1), optimize=True)
        k3 = contract('ij,i', self.func(ti), yi+((self.dt/2.0) * k2), optimize=True)
        k4 = contract('ij,i', self.func(ti), yi+(self.dt * k3), optimize=True) 
        yn = yi + ((self.dt/6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4))
        tn = ti + self.dt
        return(yn, tn)

    def _time_propagation(self, ops_list=[], ops_headers=[], 
                        print_nstep= 1, outfile='tdprop.txt'):
        yi, ti = self.y0, self.t0
        iterval = int(0)
        fobj= open(outfile, 'w', buffering=10)
        ncols = 2 + len(ops_list)
        fobj.write((" {:>16} "*(ncols)+"\n").format('time_fs', 'norm', *ops_headers))
        RK4._calc_expectations(ops_list, yi, ti, fobj, ncols)
        y_list = []
        t_list = []
        start = perf_counter()
        while ti <= self.tf:
            if iterval == print_nstep:
                iterval = int(0)
                RK4._calc_expectations(ops_list, yi, ti, fobj, ncols)
                t_list.append(ti)
                y_list.append(yi)
            yi, ti = self._rk4_step(yi, ti)
            iterval += int(1)
        fobj.close()
        stop = perf_counter()
        y_array = np.array(y_list, dtype=np.cdouble)
        t_array = np.array(t_list, dtype=np.float64) 
        np.savez('wfn_log.npz', t_log=t_array, psi_log=y_array )
        print('Time taken %3.3f seconds' % (stop-start))    
        return 0 
        
    @staticmethod
    def _calc_expectations(ops_list, yi, ti, fobj, ncols):
        ops_expectations = []
        norm = abs(np.sum(np.conjugate(yi.T, dtype=np.cdouble) * yi))
        for operator in ops_list:
            expectation = np.real(contract("i,ij,j->", np.conjugate(yi.T, dtype=np.cdouble), operator, yi, optimize=True))
            ops_expectations.append(expectation)
        ti_fs = ti / units.fs_to_au
        fobj.write((" {:>16.16f} "*(ncols)+"\n").format(ti_fs, norm, *ops_expectations))
        return 0