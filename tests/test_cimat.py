#!/usr/bin/env python
#
# Author : Sai Vijay Mocherla <vijaysai.mocherla@gmail.com>
#
import os 
import sys
os.environ['OMP_NUM_THREADS'] = '11'
os.environ["OPENBLAS_NUM_THREADS"] = "11"  
os.environ["MKL_NUM_THREADS"] = "11" 
os.environ["VECLIB_MAXIMUM_THREADS"] = "11" 
os.environ["NUMEXPR_NUM_THREADS"] = "11" 
import numpy as np
import pyci
import psi4
from threadpoolctl import threadpool_limits
from time import perf_counter 

# Lets intiate AOint() a subclass of psi4utils()
molfile = '.xyz/He.xyz'
with threadpool_limits(limits=22, user_api='blas'):
      mol = pyci.utils.molecule('d-aug-cc-pv5z', molfile=molfile, wd='./', ncore=22, psi4mem='210 Gb', numpymem=210, 
                  custom_basis=False, store_wfn=False, properties=['dipoles'], psi4options={'puream': False}) 

print('Begining CISD calculation .....\n')
start = perf_counter()
from pyci.configint.rcisd import CISD
with threadpool_limits(limits=22, user_api='blas'):
      cisd = CISD(mol, options={'doubles':True}, ncore=22)
      ndim = sum(cisd.num_csfs)
      print(cisd.num_csfs, ndim)
      cisd.save_hcisd()
stop = perf_counter()
print('Completed generating Hamiltonian matrix of size ({ndim:d},{ndim:d}) in {time:3.2f} seconds\n'.format(ndim=ndim, time=(stop-start)))
# Calculating and saving dipoles in CSF basis
print('Completed the task and saved data in {time:3.2f}\n'.format(time=(stop-start)))
print('Calculating Dipoles in CSF basis\n')
start = perf_counter()
cisd.save_dpx()
cisd.save_dpy()
cisd.save_dpz()
stop = perf_counter()
print('Completed generating Dipole matrices in {time:3.2f}\n'.format(time=(stop-start)))

# print CISD or CIS groud-state energy for a reference 
start = perf_counter()
print('Diagonalizing HCISD matrix.....\n')
HCISD = np.load('cimat.npz')['HCISD']
cisd_pyci = cisd.energy(HCISD)
if cisd.options['doubles']:
    cisd_psi4 = psi4.energy('CISD')
else:
    cisd_psi4 = psi4.energy('scf')
print("pyci CISD E0: {e:16.16f}".format(e=cisd_pyci))
print("psi4 CISD E0: {e:16.16f}".format(e=cisd_psi4))
print("dE : {dE:1.2E}\n".format(dE=abs(cisd_psi4 - cisd_pyci)))
# Getting all the eigenvalues and eigen vectors
start = perf_counter()
with threadpool_limits(limits=22, user_api='blas'):
      vals, vecs = cisd.get_eigen(HCISD)
del HCISD
np.savez('hdata.npz', eigvals=vals, eigvecs=vecs, scf_energy=mol.scf_energy, mo_eps=mol.mo_eps[0], csfs=cisd.csfs, num_csfs=cisd.num_csfs)
del vals, vecs
stop = perf_counter()
print('Completed Diagonalization task and saved data in {time:3.2f}\n'.format(time=(stop-start)))
