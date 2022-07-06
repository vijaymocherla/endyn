#!/usr/bin/python envs
#
# Author : Sai Vijay Mocherla <vijaysai.mocherla@gmail.com>
# 
# TO-DO: 
#   1. code up eri_mo2so() MO to SO integral transform routine.
#   2. clean up psi4utils API
#
"""
"""
import psi4
import numpy as np
from opt_einsum import contract
import os 

class psi4utils:
    """Helper class to get AO integrals and other abinitio data from PSI4 
       for time-dependent configuration interaction(TDCI) calculations.
    """
    def __init__(self, basis, molfile, psi4mem='2 Gb', wd='./'):
        self.scratch = scratch = wd+'.scratch/'
        if not os.path.isdir(self.scratch):
            os.system('mkdir '+wd+'.scratch')
        psi4.core.clean()
        psi4.core.set_output_file(scratch+'psi4_output.dat', False) # psi4 output
        psi4.set_memory(psi4mem) # psi4 memory 
        self.numpy_memory = 2 # numpy memory
        self.options_dict = { 'basis': basis,
                            'reference' : 'rhf',
                            'scf_type' : 'pk',
                            'e_convergence' : 1e-12,
                            'd_convergence' : 1e-10}            
        psi4.set_options(self.options_dict)               
        self.mol = psi4.geometry(psi4utils.get_mol_str(molfile))
        self.wfn = psi4.core.Wavefunction.build(self.mol, psi4.core.get_global_option('basis'))
        if self.wfn.basisset().has_puream():
            self.options_dict['puream'] = 'true'
            psi4.set_options(self.options_dict)
            self.wfn = psi4.core.Wavefunction.build(self.mol, psi4.core.get_global_option('basis'))   
        self.mints = psi4.core.MintsHelper(self.wfn.basisset())

    

    @staticmethod    
    def get_mol_str(molfile):
        """Reads input geometry provided in .xyz file into a string.
        """
        with open(molfile, 'r') as input_file:
            mol_str = input_file.read()
        return mol_str

    @staticmethod    
    def check_mem_eri(nbf, numpy_memory):
        """Checks memory requirements to store an ERI tensor
        """
        eri_tensor_size = (nbf**4) * 8.e-9
        print("Size of the ERI tensor will be %4.2f Gb\n" % (eri_tensor_size))
        memory_footprint = eri_tensor_size * 1.5
        if eri_tensor_size > numpy_memory:
            psi4.core.clean()
            raise Exception("Estimated memory utlisation (%4.2f Gb) exceeds \
                            alloted memory limit (%4.2f Gb)" % (memory_footprint, numpy_memory))
        else:
            return 1                     

    @staticmethod
    def get_orb_info(wfn):    
        # Print basis and orbital information
        nbf = wfn.basisset().nbf()
        nmo = wfn.nmo()
        nalpha = wfn.nalpha()
        nbeta = wfn.nbeta()
        nso = 2 * nmo
        nocc = nalpha 
        nvir = nmo - nocc
        print(' ORBITAL INFORMATION  \n'
               '-----------------------\n'
               'Basis functions   : %i  \n' % nbf    + 
               'Molecular orbitals: %i  \n' % nmo    +
               'Spin Orbitals     : %i  \n' % nso    +
               'Alpha Orbitals    : %i  \n' % nalpha + 
               'Beta Orbitals     : %i  \n' % nbeta  + 
               'Occupied Orbitals : %i  \n' % nocc   +
               'Virtual Orbitals  : %i  \n' % nvir)
        return(nbf, nmo, nso, nalpha, nbeta, nocc, nvir)

    @staticmethod
    def eri_ao2mo(Ca, ao_erints, greedy=False):
        if greedy:
            # TODO Check precision issues involved if greedy=True
            print("!!!Warning: Using greedy eri_ao2mo transform\n")
            size = Ca.shape[0]
            mo_erints = np.dot(Ca.T, ao_erints.reshape(size, -1))
            mo_erints = np.dot(mo_erints.reshape(-1, size), Ca)
            mo_erints = mo_erints.reshape(size, size, size, size).transpose(1, 0, 3, 2)
            mo_erints = np.dot(Ca.T, mo_erints.reshape(size, -1))
            mo_erints = np.dot(mo_erints.reshape(-1, size), Ca)
            mo_erints = mo_erints.reshape(size, size, size, size).transpose(1, 0, 3, 2)
        else:
            mo_erints = contract('pqrs,pI,qJ,rK,sL->IJKL', ao_erints, Ca, Ca, Ca, Ca, optimize=True)
        return mo_erints

    @staticmethod
    def matrix_ao2mo(Ca, matrix):
        mo_matrix = contract('pq,pI,qJ->IJ', matrix, Ca, Ca, optimize=True)
        return mo_matrix
    
    @staticmethod
    def eri_mo2so(mo_erints):
        # TO-DO : code mo to so transform for eris (current implementation is bad)
        print("!!!Warning: Using unstable MO to SO transform eri_mo2so(), instead use eri_mo2so_psi4()\n")
        dim = mo_erints.shape[0]
        mo_so_eri=np.zeros((dim*2,dim*2,dim*2,dim*2), np.float64)
        for p in range(1, dim*2 + 1):
            for q in range(1, dim*2 + 1):
                for r in range(1, dim*2 + 1):
                    for s in range(1, dim*2 + 1):
                        mo_so_eri[p-1, q-1, r-1, s-1] = (((q/2)==(p/2))*((s/2)==(r/2))*mo_erints[p//2 - 1, q//2 - 1, r//2 - 1, s//2 - 1])
        return mo_so_eri

    def eri_mo2so_psi4(self, Ca, Cb):
        """Returns MO spin eri tensor in chemist's notation
        """
        Ca_psi4_matrix = psi4.core.Matrix.from_array(Ca)
        Cb_psi4_matrix = psi4.core.Matrix.from_array(Cb)
        mo_spin_eri_tensor = np.asarray(self.mints.mo_spin_eri(Ca_psi4_matrix, Cb_psi4_matrix), dtype=np.float64)
        return mo_spin_eri_tensor


class AOint(psi4utils):
    """A module to get AO integrals using psi4
    """
    def __init__(self, basis, molfile, psi4mem='2 Gb', scratch='./'):
        psi4utils.__init__(self, basis, molfile, psi4mem, scratch)
        
    def save_ao_oeints(self):
        """Saves S, T and V integrals in AO basis as a .npz file.
           For example, you can load the saved integrals as follows:
           ```py
           >>> ao_oeints_data = np.load('ao_oeints.npz')
           >>> S = ao_oeints_data['overlap_aoints']
           >>> T = ao_oeints_data['kinetic_aoints']
           >>> V = ao_oeints_data['potential_aoints']
           ```
        """
        # one-electron integrals 
        overlap_aoints = np.asarray(self.mints.ao_overlap(), dtype=np.float64)
        kinetic_aoints = np.asarray(self.mints.ao_kinetic(), dtype=np.float64)
        potential_aoints = np.asarray(self.mints.ao_potential(), dtype=np.float64)
        np.savez(self.scratch+'ao_oeints.npz', 
                overlap_aoints=overlap_aoints,
                kinetic_aoints=kinetic_aoints,
                potential_aoints=potential_aoints)
        return 1

    def get_ao_oeints(self):
        """ Returns S, T, V in AO basis
        """
        ao_oeints = np.load(self.scratch+'ao_oeints.npz')
        ao_overlap = ao_oeints['overlap_aoints']
        ao_kinetic = ao_oeints['kinetic_aoints']
        ao_potential = ao_oeints['potential_aoints']
        return ao_overlap, ao_kinetic, ao_potential
    
    def get_ao_erints(self):
        """ Returns ERIs in AO basis
        """
        ao_eris = np.load(self.scratch + 'ao_erints.npz')['electron_repulsion_aoints']
        return ao_eris

    def save_ao_erints(self): 
        """Saves 2-electron repulsion integrals in AO basis as a .npz file.
           For example, you can load the saved integrals as follows:
           ```py
           >>> ao_erints_data = np.load('ao_erints.npz')
           >>> ao_erints = ao_erints_data['electron_repulsion_aoints']
           ``` 
        """   
        nbf = self.wfn.basisset().nbf() 
        if psi4utils.check_mem_eri(nbf, self.numpy_memory): 
            electron_repulsion_aoints = np.asarray(self.mints.ao_eri())
            np.savez(self.scratch+'ao_erints.npz', electron_repulsion_aoints=electron_repulsion_aoints)
        return 1    

    def save_ao_dpints(self):
        """Saves dipole integrals in AO basis as a .npz file.
           For example, you can load the saved integrals as follows:
           ```py
           >>> ao_dipoles_data = np.load('ao_dpints.npz')
           >>> ao_dpx = ao_dipoles_data['dpx_aoints']
           >>> ao_dpy = ao_dipoles_data['dpy_aoints']
           >>> ao_dpz = ao_dipoles_data['dpz_aoints']
           ```
        """
        dpx_aoints, dpy_aoints, dpz_aoints =  np.asarray(self.mints.ao_dipole(), dtype=np.float64)
        np.savez(self.scratch+'ao_dpints.npz',
                dpx_aoints=dpx_aoints,
                dpy_aoints=dpy_aoints,
                dpz_aoints=dpz_aoints)
        return 1

    def get_ao_dpints(self):
        """ Returns dipole integrals in AO basis
        """
        ao_dipoles_data = np.load(self.scratch+'ao_dpints.npz')
        ao_dpx = ao_dipoles_data['dpx_aoints']
        ao_dpy = ao_dipoles_data['dpy_aoints']
        ao_dpz = ao_dipoles_data['dpz_aoints']
        return ao_dpx, ao_dpy, ao_dpz

    def save_ao_qdints(self):
        """Saves dipole integrals in AO basis as a .npz file.
           For example, you can load the saved integrals as follows:
           ```py
           >>> ao_dipoles_data = np.load('ao_qdints.npz')
           >>> ao_qdxx = ao_dipoles_data['qdxx_aoints']
           >>> ao_qdxy = ao_dipoles_data['qdxy_aoints']
           >>> ao_qdxz = ao_dipoles_data['qdxz_aoints']
           >>> ao_qdyy = ao_dipoles_data['qdyy_aoints']
           >>> ao_qdyz = ao_dipoles_data['qdyz_aoints']
           >>> ao_qdzz = ao_dipoles_data['qdzz_aoints']

           ```
        """
        (qdxx_aoints, qdxy_aoints, qdxz_aoints, 
        qdyy_aoints, qdyz_aoints, qdzz_aoints) =  np.asarray(self.mints.ao_quadrupole(), dtype=np.float64)
        np.savez(self.scratch+'ao_qdints.npz',
                qdxx_aoints=qdxx_aoints,
                qdxy_aoints=qdxy_aoints,
                qdxz_aoints=qdxz_aoints,
                qdyy_aoints=qdyy_aoints,
                qdyz_aoints=qdyz_aoints,
                qdzz_aoints=qdzz_aoints)
        return 1

    def get_ao_qdints(self):
        ao_dipoles_data = np.load(self.scratch+'ao_qdints.npz')
        ao_qdxx = ao_dipoles_data['qdxx_aoints']
        ao_qdxy = ao_dipoles_data['qdxy_aoints']
        ao_qdxz = ao_dipoles_data['qdxz_aoints']
        ao_qdyy = ao_dipoles_data['qdyy_aoints']
        ao_qdyz = ao_dipoles_data['qdyz_aoints']
        ao_qdzz = ao_dipoles_data['qdzz_aoints']
        return ao_qdxx, ao_qdxy, ao_qdxz, ao_qdyy, ao_qdyz, ao_qdzz, 

    def save_all_aoints(self):
        self.save_ao_erints()
        self.save_ao_oeints()
        self.save_ao_dpints()
        return 1    
    
    
    def save_mo_info(self):
        """Runs an SCF calculation and saves info about molecular orbitals.  
           Saved MO energies and coefficients can be accessed as follows:
           ```py
           >>> mo_info_data = np.load('mo_scf_info.npz')
           >>> eps_a = mo_info_data['eps_a']
           >>> Ca = mo_info_data['Ca'] 
           ```
        """
        self.scf_energy, self.scf_wfn = psi4.energy('scf', return_wfn=True)
        print('Ground state SCF Energy : %3.8f \n' % self.scf_energy)        # MO coefficients and energies
        print('Nuclear repulsion energy : %3.8f \n' % self.mol.nuclear_repulsion_energy())
        print('Total electronic energy : %3.8f \n' % (self.scf_energy - self.mol.nuclear_repulsion_energy()))
        eps_a = np.array(self.scf_wfn.epsilon_a_subset('AO', 'ALL'), dtype=np.float64)
        Ca = np.array(self.scf_wfn.Ca_subset('AO','ALL'), dtype=np.float64)
        # beta orbitals
        Cb = np.array(self.scf_wfn.Cb_subset('AO','ALL'), dtype=np.float64)
        eps_b = np.array(self.scf_wfn.epsilon_b_subset('AO', 'ALL'), dtype=np.float64)
        np.savez(self.scratch+'mo_scf_info.npz', eps_a=eps_a, Ca=Ca, eps_b=eps_b, Cb=Cb)
        return 1