import psi4
import numpy as np
import gc
import cis
import cis_d
import sys

# Setting psi4 environment variables
print("\nSetting up PSI4 environment variables\n")
psi4.core.set_output_file('output.dat', False)
psi4.set_memory('2 Gb')
psi4.set_options({'basis':        sys.argv[2],
                  'scf_type':     'pk',
                  'reference':    'rhf',
                  'e_convergence': 1e-12,
                  'd_convergence': 1e-10})

# Reading input from .xyz file into a string 
print("\nReading input from .xyz file\n")
with open(sys.argv[1], 'r') as inp_file:
    mol_str = inp_file.read()
mol = psi4.geometry(mol_str)
print('\n***\n'
      + '\n'
      + mol_str
      + '\n'
      + '\n***\n')

# Running SCF to get Ground State Hartree-Fock WaveFunction
scf_e, scf_wfn = psi4.energy('SCF', return_wfn=True)
print('Total Electronic energy: %3.8f' % (scf_e - mol.nuclear_repulsion_energy()))
print('Nuclear repulsion energy: %3.8f' % mol.nuclear_repulsion_energy())
print('SCF Ground state energy: %3.8f' % scf_e)
Ca = scf_wfn.Ca_subset('AO', 'ALL')
Cb = scf_wfn.Cb_subset('AO', 'ALL')
eps = np.array(scf_wfn.epsilon_a_subset('AO', 'ALL'))
# Get basis and orbital information
nbf = scf_wfn.basisset().nbf()
nmo = scf_wfn.nmo()
nocc = scf_wfn.nalpha()
nvirt = nmo - nocc

# Getting AO integrals for computing CIS Hamiltonian
mints = psi4.core.MintsHelper(scf_wfn.basisset())
# The are 2-electron repulsion intergrals in AO basis
ao_erints = np.asarray(mints.ao_eri())
# transforming to MO basis 
C = np.array(Ca)
mo_erints = np.einsum('pqrs,pI,qJ,rK,sL->IJKL',
                      ao_erints, C, C, C, C,
                      optimize=True)

# Computing the CIS Hamiltonian
orbinfo = (nocc, nvirt, nmo)
HCIS = cis.comp_cis_hamiltonian(eps, mo_erints, orbinfo)

# Computing the cis eigenvalues and eigen-vectors
ECIS, CCIS = np.linalg.eigh(HCIS)
# ECIS = ECIS + (scf_e - mol.nuclear_repulsion_energy())
mo_so_erints = np.asarray(mints.mo_spin_eri(Ca, Cb))
mo_so_erints = mo_so_erints.transpose(0, 2, 1, 3)
cis_d_corr = cis_d.cis_d(ECIS, CCIS, eps, mo_so_erints, orbinfo)
# MP2 energy calculation
print('\nComputing MP2 corrections\n')
mp2_e = psi4.energy('mp2')
print('PSI4 MP2 Ground state energy: %3.8f' % mp2_e)
print('CALC MP2 Ground state energy: %3.8f' % (scf_e+cis_d_corr.comp_e0mp2()))
print('\n')
# Computing doubles corrections for cis states
print('\nComputing pertubative (D) corrections for CIS\n')
n = int(sys.argv[3])
ECIS_D = cis_d_corr.comp_cis_d(n, ncore=3)
print(' \t  ECIS  \t  D_corr  \t  ECIS_D')
for i in range(n):
    print('%i \t  %3.5f  \t  %3.5f  \t  %3.5f' %
          (i, ECIS[i], ECIS_D[i], (ECIS[i]+ECIS_D[i])))
          #(i, ECIS[i]*27.2114, ECIS_D[i]*27.2114, (ECIS[i]+ECIS_D[i])*27.2114))
