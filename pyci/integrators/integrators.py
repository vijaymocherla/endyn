#!/usr/bin/env python
#
# Author: Sai Vijay Mocherla <vijaysai.mocherla@gmail.com>
#
# Single File with all the time propagators

import numpy as np
from time import perf_counter
from pyci.utils import units
from threadpoolctl import threadpool_limits
from pyci.linalg.blas import zmul_mm, zmul_mmm, zmul_mv
from pyci.linalg.blas import zmul_zdotc, zmul_zdotu


METHODS = {'runge kutta' : RungeKutta,
           'split operator' : SplitOperator,
           'crank nicholson' : CrankNicholson,
           'exact propagation': ExactPropagator,}

MESSAGES = {0: "Propagator successfully reached the end of the time interval",
            1: "Error occured during propagation, please check outfile."}

def _calc_expectations(ops_list, psi_i, psi_0):
    """ Calculate expectation values of a given list of operators
        
        Parameters
        ----------
        ops_list : list[] of (n,n) arrays
             Operators whose expectation values are to calculated.
        psi_i : array_like, shape (n,)
             Wave function at time ti
        psi_0 : array_like, shape (n,)
             Initial wavefunction at t=0.
        
        Returns
        -------
        ops_expts : list of floats
             List of expectation values of given operators
        norm : float
             Norm of the wavefunction at ti (psi_i)
        autocorr : float 
             Survival probability of the initial state psi_0
    """
    ops_expts = []
    norm = zmul_zdotc(psi_i, psi_i).real
    autocorr = np.abs(zmul_zdotu(psi_i, psi_0))
    for operator in ops_list:
        expt = zmul_zdotc(psi_i, zmul_mv(operator, psi_i)).real
        ops_expts.append(expt)
    return ops_expts, norm, autocorr


class Propagator:
    def __init__(psi_0, time_params, ):
        return message