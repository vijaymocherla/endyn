# Helper functions to transform from AO to MO basis 
import numpy as np


def eri_ao2mo(Ca, ao_erints, greedy=False):
    if greedy:
        # TODO Check precision issues involved if greedy=True
        size = Ca.shape[0]
        mo_erints = np.dot(Ca.T, ao_erints.reshape(size, -1))
        mo_erints = np.dot(mo_erints.reshape(-1, size), Ca)
        mo_erints = mo_erints.reshape(size, size, size, size).transpose(1, 0, 3, 2)
        mo_erints = np.dot(Ca.T, mo_erints.reshape(size, -1))
        mo_erints = np.dot(mo_erints.reshape(-1, size), Ca)
        mo_erints = mo_erints.reshape(size, size, size, size).transpose(1, 0, 3, 2)
    else:
        mo_erints = np.einsum('pqrs,pI,qJ,rK,sL->IJKL', ao_erints, Ca, Ca, Ca, Ca, optimize=True)
    return mo_erints

def matrix_ao2mo(Ca, matrix):
    mo_matrix = np.einsum('pq,pI,qJ->IJ', matrix, Ca)
    return mo_matrix

