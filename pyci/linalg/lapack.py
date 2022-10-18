#!/usr/bin/env python
#
# Author : Sai Vijay Mocherla
#

import numpy as np
from scipy.linalg.lapack import dgetrf, dgetri, zgetrf, zgetri

def zmat_inv(A):
    """Computes the inverse of a matrix of complex-doubles
    """
    if not np.isfortran(A):
        A = A.T
    lu, piv, info = zgetrf(A)
    Ainv, info = zgetri(lu,piv)
    return Ainv

def dmat_inv(A):
    """Computes the inverse of a matrix of doubles
    """
    if not np.isfortran(A):
        A = A.T
    lu, piv, info = dgetrf(A)
    Ainv, info = dgetri(lu, piv)
    return Ainv