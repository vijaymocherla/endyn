#!/usr/bin/env python
#
# Author : Sai Vijay Mocherla <vijaysai.mocherla@gmail.com>
#
# NOTE:   
# ---
# Check `Fi.flags` and `yi.flags` before passing arrays to 
# methods from scipy.linalg.blas as, it can give significant speedups.
# >>> scheme1 = blas.zgemm(ALPHA, Fi, yi)
# >>> scheme2 = blas.zgemm(ALPHA, Fi.T, yi.T, trans_a=True) 
# scheme2 is ~1.5x faster than scheme1 as the 2d numpy array
# Fi can be NOT F_CONTIGUOUS but is C_CONTIGUOUS.
# For more information see: https://scipy.github.io/old-wiki/pages/PerformanceTips
# ---

import numpy as np
from scipy.linalg.blas import zgemv, dgemv
from scipy.linalg.blas import zgemm, dgemm
from scipy.linalg.blas import ddot, zdotu, zdotc

def chkf90(array): 
    return np.asfortranarray(array)

def zmul_mmm(A, B, C):
    """Matrix-Matrix multiplication for complex doubles
    """
    A, B, C = chkf90(A), chkf90(B), chkf90(C)
    D = zgemm(1.0+0j, A, B)
    D = zgemm(1.0+0j, D, C)  
    return D

def zmul_mm(A, B):
    """Matrix-Matrix multiplication for complex doubles
    """
    A, B = chkf90(A), chkf90(B)
    C = zgemm(1.0+0j, A, B)
    return C

def zmul_mv(A, x):
    """Matrix-vector multiplication for complex doubles
    """
    A, x = chkf90(A), chkf90(x)
    y = zgemv(1.0+0j, A, x)
    return y

def zmul_zdotc(x, y):
    """ Returns x^{\dagger} . y
        Note: This returns the inner product <x|y>
    """
    x, y = chkf90(x), chkf90(y)
    dot = zdotc(x, y)
    return dot

def zmul_zdotu(x, y):
    """ Returns x^{T} . y
    """
    x, y = chkf90(x), chkf90(y)
    dot = zdotu(x, y)
    return dot

def dmul_mm(A, B):
    """Matrix-vector multiplication for doubles
    """
    A, B = chkf90(A), chkf90(B)
    C = dgemm(1.0, A, B)
    return C

def dmul_mv(A, x):
    """Matrix-Matrix multiplication for doubles
    """
    A, x = chkf90(A), chkf90(x)
    y = dgemv(1.0, A, x)
    return y

def dmul_dot(x, y):
    """Returns dot product of two vectors.
    """
    x, y = chkf90(x), chkf90(y)
    dot = ddot(x, y)
    return dot

def dmul_mmm(A, B, C):
    """Matrix-Matrix multiplication for doubles
    """
    A, B, C = chkf90(A), chkf90(B), chkf90(C)
    D = dgemm(1.0, A, B)
    D = dgemm(1.0, D, C)
    return D
