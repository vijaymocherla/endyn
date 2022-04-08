import numpy as np
import pyci
# Lets intiate AOint() a subclass of psi4utils() 
aoint = pyci.AOint('6-31g*', 'licn.xyz')
# saves all the necessary integrals as .npz files
aoint.save_all_aoints()
# run scf and get info for canonical HF orbitals
aoint.save_mo_info()
# lets load  ao_erints, mo_info and try to get the CIS matrix
ao_erints = np.load('ao_erints.npz')['electron_repulsion_aoints'] 
eps_a = np.load('mo_scf_info.npz')['eps_a']
Ca = np.load('mo_scf_info.npz')['Ca'] 
# lets convert our erints from AO basis to MO basis
mo_erints = aoint.eri_ao2mo(Ca, ao_erints)
del ao_erints
aoint.get_orb_info(aoint.wfn)

HCIS = pyci.comp_cis_hamiltonian(eps_a, mo_erints, (8, 37, 45), parallelised=True, ncore=10)
ECIS, CCIS = np.linalg.eigh(HCIS)
mo_so_erints = aoint.eri_mo2so(Ca, Ca)
cis_d_class = pyci.CIS_D(ECIS, CCIS, eps_a, mo_so_erints, (8, 37, 45))
e_cis_d = cis_d_class.comp_cis_d(10, ncores=10)