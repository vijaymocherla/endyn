#!/usr/bin/env python
#
# Author : Sai Vijay Mocherla <vijaysai.mocherla@gmail.com>
#
"""splitoperator.py
"""

import numpy as np
from time import perf_counter
from pyci.utils import units
from pyci.linalg.blas import zmul_mv, zmul_mm, zmul_zdotc

def _calc_expectations(yi_csf, ops_list, y0_csf):
    ops_expectations = []
    norm = np.abs(zmul_zdotc(yi_csf, yi_csf))
    autocorr = np.abs(zmul_zdotc(yi_csf, y0_csf))
    for operator in ops_list:
        expectation = zmul_zdotc(yi_csf, zmul_mv(operator, yi_csf)).real
        ops_expectations.append(expectation)
    return norm, autocorr, ops_expectations

def SplitOperator(eigvals, eigvecs, y0, time_params, 
                  field_params={}, 
                  dpx=[], dpy=[], dpz=[],
                  ncore=4, ops_list=[], ops_headers=[],
                  print_nstep=1, outfile='tdprop.txt'):
    """Given eigenvalues and eigen-vectors(in a certain basis), the methods of 
    exact_prop() aids in exact time-propagation.  
    """
    # Initial params
    t0, tf, dt = time_params
    y0_csf = y0
    y0_eig = zmul_mv(eigvecs.T, y0_csf)
    yi_eig, ti = y0_eig, t0
    # Feilds options
    options = {'t_start': 0.0,
               't_stop': 0.0,
               'Fx': False, 
               'Fy': False, 
               'Fz': False,
               'Ex': lambda t: 0.0, 
               'Ey': lambda t: 0.0,
               'Ez': lambda t: 0.0,}
    options.update(field_params)
    if options['Fx']:
        if dpx == []:
            raise Exception('Fx set to True but no x dipole was given')
        Dx, Vx = np.linalg.eigh(dpx)
        Ux = zmul_mm(Vx.T, eigvecs)
        UxT = zmul_mm(eigvecs.T, Vx)
        def Fx(t):
            Ex = options['Ex']
            Fx = zmul_mm(np.diag(np.exp(1j*dt*Ex(t)*Dx)), Ux)
            Fx = zmul_mm(UxT, Fx)
            return Fx
    if options['Fy']:
        if dpy == []:
            raise Exception('Fy set to True but no y dipole was given')
        Dy, Vy = np.linalg.eigh(dpy)
        Uy = zmul_mm(Vy.T, eigvecs)
        UyT = zmul_mm(eigvecs.T, Vy)
        def Fy(t):
            Ey = options['Ey']
            Fy = zmul_mm(np.diag(np.exp(1j*dt*Ey(t)*Dy)), Uy)
            Fy = zmul_mm(UyT, Fy)
            return Fy
    if options['Fz']:
        if dpz == []:
            raise Exception('Fz set to True but no z dipole was given')
        Dz, Vz = np.linalg.eigh(dpz)
        Uz = zmul_mm(Vz.T, eigvecs)
        UzT = zmul_mm(eigvecs.T, Vz)
        def Fz(t):
            Ez = options['Ez']
            Fz = zmul_mm(np.diag(np.exp(1j*dt*Ez(t)*Dz)), Uz)
            Fz = zmul_mm(UzT, Fz)
            return Fz
    # Setting up output file
    fobj = open(outfile, 'wb', buffering=0)
    ncols = 3 + len(ops_list)
    fobj.write((" {:<19} "*(ncols)+"\n").format('time_fs', 'norm', 'autocorr', *ops_headers).encode("utf-8"))
    norm, autocorr, ops_expectations = _calc_expectations(y0_csf, ops_list, y0_csf)
    ti_fs = ti / units.fs_to_au
    fobj.write((" {:>16.16f} "*(ncols)+"\n").format(ti_fs, norm, autocorr, *ops_expectations).encode("utf-8"))
    start = perf_counter()
    while ti <= tf:    
        for i in range(print_nstep):
            if ti >= options['t_start'] and ti <= options['t_stop']:
                yi_eig = np.exp(-1j*eigvals*dt) * yi_eig
                if options['Fx']:
                    yi_eig = zmul_mv(Fx(ti), yi_eig)
                if options['Fy']:
                    yi_eig = zmul_mv(Fy(ti), yi_eig)
                if options['Fz']:
                    yi_eig = zmul_mv(Fz(ti), yi_eig)
            else:
                yi_eig = np.exp(-1j*eigvals*dt) * yi_eig
            ti = ti + dt
        yi_csf = zmul_mv(eigvecs, yi_eig)
        norm, autocorr, ops_expectations = _calc_expectations(yi_csf, ops_list, y0_csf)
        ti_fs = ti / units.fs_to_au
        fobj.write((" {:>16.16f} "*(ncols)+"\n").format(ti_fs,
                   norm, autocorr, *ops_expectations).encode("utf-8"))
    fobj.close()
    stop = perf_counter()
    print('Time taken %3.3f seconds' % (stop-start))
    return 0
