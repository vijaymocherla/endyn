#!/usr/bin/env python
#
# Author : Sai Vijay Mocherla
#

import numpy as np
from scipy.linalg.lapack import dgetrf, dgetri, zgetrf, zgetri
from scipy.linalg.lapack import dgesv, zgesv
from scipy.linalg.lapack import dgesvx, zgesvx

def zinv(A):
    """Computes the inverse of a matrix of complex-doubles
    """
    if not np.isfortran(A):
        A = A.T
    lu, piv, info = zgetrf(A)
    Ainv, info = zgetri(lu,piv)
    if info != 0:
        raise Exception("Given martix is singular!")
    return Ainv

def dinv(A):
    """Computes the inverse of a matrix of doubles
    """
    if not np.isfortran(A):
        A = A.T
    lu, piv, info = dgetrf(A)
    Ainv, info = dgetri(lu, piv)
    if info != 0:
        raise Exception("Given martix is singular!")
    return Ainv

def zsolve(A, b, expert=False):
    if not np.isfortran(A):
        A = A.T
    if not np.isfortran(b):
        b = b.T
    if expert:
        sol = zgesvx(A, b)
        lu, ipiv = sol[1], sol[2]
        x, info = sol[7], sol[11]
    else:
        sol = zgesv(A, b)
        lu, ipiv, x, info = sol 
    if info != 0:
        raise Exception("Couldn't solve for x")
    return x

def dsolve(A, b, expert=False):
    if not np.isfortran(A):
        A = A.T
    if not np.isfortran(b):
        b = b.T
    if expert:
        sol = dgesvx(A, b)
        lu, ipiv = sol[1], sol[2]
        x, info = sol[7], sol[11]
    else:
        lu, piv, x, info = dgesv(A, b)
    if info != 0:
        raise Exception("Couldn't solve for x")
    return x