import numpy as np
import pyci
import psi4
# Lets intiate AOint() a subclass of psi4utils() 
aoint = pyci.utils.AOint('6-31g*', '../.xyz/LiCN.xyz')
# saves all the necessary integrals as .npz files
aoint.save_all_aoints()
# run scf and get info for canonical HF orbitals
aoint.save_mo_info()
# lets load  ao_erints, mo_info and try to get the CIS matrix
ao_erints = np.load('.scratch/ao_erints.npz')['electron_repulsion_aoints'] 
eps_a = np.load('.scratch/mo_scf_info.npz')['eps_a']
Ca = np.load('.scratch/mo_scf_info.npz')['Ca'] 
# lets convert our erints from AO basis to MO basis
mo_erints = aoint.eri_ao2mo(Ca, ao_erints, greedy=True)
del ao_erints

nbf, nmo, nso, na, nb, nocc, nvirt = aoint.get_orb_info(aoint.wfn)
orbinfo = (nocc, nmo)
active_space = (nocc,nvirt)
singles = True
full_cis = True
doubles = False
doubles_iiaa = False
doubles_iiab = False
doubles_ijaa = False
doubles_ijab_A = False
doubles_ijab_B = False
options = [singles, full_cis, doubles,
           doubles_iiaa, doubles_iiab,doubles_ijaa,
           doubles_ijab_A, doubles_ijab_B]
csfs, num_csfs = pyci.configint.rcisd.generate_csfs(orbinfo, active_space, options)
num_csfs, sum(num_csfs)
scf_energy = aoint.scf_energy
HCIS = pyci.configint.rcisd.comp_hcisd(eps_a, mo_erints, scf_energy, orbinfo, active_space, options, ncore=2)
mo_so_erints = aoint.eri_mo2so_psi4(Ca, Ca)
mo_so_erints = mo_so_erints.transpose(0,2,1,3)
cis_d_class = pyci.configint.CIS_D(ECIS, CCIS, eps_a, mo_so_erints, (nocc, nmo))
# MP2 energy calculation
print('\nComputing MP2 corrections\n')
mp2_e = psi4.energy('mp2')
print('PSI4 MP2 Ground state energy: %3.16f' % mp2_e)
print('CALC MP2 Ground state energy: %3.16f' % (scf_energy+cis_d_class.comp_e0mp2()))
print('Difference : %3.2E'%(scf_energy+cis_d_class.comp_e0mp2() - mp2_e))
e_cis_d = cis_d_class.comp_cis_d(50, ncores=4)