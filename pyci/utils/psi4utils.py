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
from pickletools import optimize
import psi4
import numpy as np
from opt_einsum import contract
import os 
from scipy.linalg import blas

ALPHA = 1.0

class psi4utils:
    """Helper class to get AO integrals and other abinitio data from PSI4 
       for time-dependent configuration interaction(TDCI) calculations.
    """
    def __init__(self, basis, molfile,wd='./', 
            ncore=2, psi4mem='2 Gb', numpymem=2,
            custom_basis=False, basis_dict=None):
        self.scratch = wd+'.scratch/'
        if not os.path.isdir(self.scratch):
            os.system('mkdir '+wd+'.scratch')
        psi4.core.clean()
        psi4.core.set_output_file(self.scratch+'psi4_output.dat', False) # psi4 output
        psi4.core.set_num_threads(ncore)
        psi4.set_memory(psi4mem) # psi4 memory 
        self.numpy_memory = numpymem # numpy memory
        self.mol = psi4.geometry(psi4utils.get_mol_str(molfile))
        if custom_basis:
            basis = 'userdef'
            if basis_dict == None:
                raise Exception("custom_basis set as True, but no valid basis_dict was passed!")
            else:
                psi4utils.set_custombasis(basis_dict)
        self.options_dict = {'basis': basis,
                            'reference' : 'rhf',
                            'scf_type' : 'pk',
                            'e_convergence' : 1e-12,
                            'd_convergence' : 1e-10}            
        psi4.set_options(self.options_dict)               
        self.wfn = psi4.core.Wavefunction.build(self.mol, psi4.core.get_global_option('basis'))
        if self.wfn.basisset().has_puream():
            self.options_dict['puream'] = 'true'
            psi4.set_options(self.options_dict)
            self.wfn = psi4.core.Wavefunction.build(self.mol, psi4.core.get_global_option('basis'))   
        self.mints = psi4.core.MintsHelper(self.wfn.basisset())
    
    @staticmethod
    def set_custombasis(basis_dict):        
        def basisspec_psi4_yo__mybasis(mol, role):
            basstrings = {}
            mol.set_basis_all_atoms("ALLATOMS",)
            basstrings['allatoms'] = basis_dict['basstring']
            return basstrings
        psi4.qcdb.libmintsbasisset.basishorde['USERDEF'] = basisspec_psi4_yo__mybasis

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
            mo_erints = blas.dgemm(ALPHA, Ca.T, ao_erints.reshape(size, -1).T, trans_b=True)
            mo_erints = blas.dgemm(ALPHA, mo_erints.reshape(-1, size), Ca.T, trans_b=True)
            mo_erints = mo_erints.reshape(size, size, size, size).transpose(1, 0, 3, 2)
            mo_erints = blas.dgemm(ALPHA, Ca.T, mo_erints.reshape(size, -1).T, trans_b=True)
            mo_erints = blas.dgemm(ALPHA, mo_erints.reshape(-1, size), Ca.T, trans_b=True)
            mo_erints = mo_erints.reshape(size, size, size, size).transpose(1, 0, 3, 2)
        else:
            mo_erints = contract('pqrs,pI,qJ,rK,sL->IJKL', ao_erints, Ca, Ca, Ca, Ca, optimize=True)
        return mo_erints

    @staticmethod
    def matrix_ao2mo(Ca, matrix):
        mo_matrix = contract('pq,pI,qJ->IJ', matrix, Ca, Ca, optimize=True)
        return mo_matrix
    
    @staticmethod
    def get_mo_so_eri(mo_eps, mo_coeff, ao_erints):
        """Returns MO spin eri tensor in chemist's notation
        """
        # TO-DO : code mo to so transform for erints (current implementation is bad)
        print("!!!Warning: Using unstable MO to SO transform eri_mo2so(), instead use eri_mo2so_psi4()\n")
        Ca, Cb = mo_coeff
        eps_a, eps_b = mo_eps
        eps = np.append(eps_a, eps_b)
        C = np.block([[      Ca,         np.zeros(Cb.shape)],
                      [np.zeros(Ca.shape),          Cb     ]
                      ])
        C = C[:, eps.argsort()]
        # spin blocking erints 
        ao_so_eri = np.kron(np.eye(2), np.kron(np.eye(2), ao_erints).T)
        mo_so_eri = contract('pqrs,pI,qJ,rK,sL->IJKL', ao_so_eri, C, C, C, C, optimize=True)
        return mo_so_eri

    def eri_mo2so_psi4(self, Ca, Cb):
        """Returns MO spin eri tensor in chemist's notation
        """
        Ca_psi4_matrix = psi4.core.Matrix.from_array(Ca)
        Cb_psi4_matrix = psi4.core.Matrix.from_array(Cb)
        mo_spin_eri_tensor = np.array(self.mints.mo_spin_eri(Ca_psi4_matrix, Cb_psi4_matrix), dtype=np.float64)
        return mo_spin_eri_tensor


class AOint(psi4utils):
    """A module to get AO integrals using psi4
    """
    def __init__(self, basis, molfile,wd='./', 
            ncore=2, psi4mem='2 Gb', numpymem=2,
            custom_basis=False, basis_dict=None):
        psi4utils.__init__(self, basis, molfile, wd, ncore, psi4mem, numpymem, custom_basis, basis_dict)
        
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
        """ Returns erints in AO basis
        """
        ao_erints = np.load(self.scratch + 'ao_erints.npz')['electron_repulsion_aoints']
        return ao_erints

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
        return ao_qdxx, ao_qdxy, ao_qdxz, ao_qdyy, ao_qdyz, ao_qdzz 

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

    def get_mo_info(self):
        eps_a = np.load(self.scratch+'mo_scf_info.npz')['eps_a']
        eps_b = np.load(self.scratch+'mo_scf_info.npz')['eps_b']
        Ca = np.load(self.scratch+'mo_scf_info.npz')['Ca'] 
        Cb = np.load(self.scratch+'mo_scf_info.npz')['Cb']
        return (eps_a, eps_b), (Ca, Cb) 

class molecule(object):
    def __init__(self, basis, molfile, wd='./', 
                 ncore=2, psi4mem='2 Gb', numpymem=2, 
                 custom_basis=False, basis_dict=None,
                 store_wfn=False, properties=[]):
        aoint = AOint(basis, molfile, wd, ncore, psi4mem, 
                    numpymem, custom_basis, basis_dict)
        aoint.save_all_aoints()
        aoint.save_mo_info()
        self.mo_eps, self.mo_coeff = aoint.get_mo_info()
        Ca = self.mo_coeff[0]
        nbf, nmo, nso, na, nb, nocc, nvirt = aoint.get_orb_info(aoint.wfn)        
        self.scf_energy = aoint.scf_energy
        self.orbinfo = (nocc, nmo)
        self.active_space = (nocc,nvirt)
        if store_wfn:
            self.wfn = aoint.wfn
        ao_erints = aoint.get_ao_erints()
        self.mo_erints = aoint.eri_ao2mo(Ca, ao_erints, greedy=False)
        del ao_erints
        self.properties = properties
        if 'dipoles' in self.properties:
            ao_dpx, ao_dpy, ao_dpz = aoint.get_ao_dpints()
            self.mo_dpx = aoint.matrix_ao2mo(Ca, ao_dpx)
            self.mo_dpy = aoint.matrix_ao2mo(Ca, ao_dpy)
            self.mo_dpz = aoint.matrix_ao2mo(Ca, ao_dpz)
            del ao_dpx, ao_dpy, ao_dpz
        if 'quadrupoles' in self.properties:
            (ao_qdxx, ao_qdxy, ao_qdxz, 
             ao_qdyy, ao_qdyz, ao_qdzz) = aoint.get_ao_qdints()
            self.mo_qdxx = aoint.matrix_ao2mo(Ca, ao_qdxx)
            self.mo_qdxy = aoint.matrix_ao2mo(Ca, ao_qdxy)
            self.mo_qdxz = aoint.matrix_ao2mo(Ca, ao_qdxz)
            self.mo_qdyy = aoint.matrix_ao2mo(Ca, ao_qdyy)
            self.mo_qdyz = aoint.matrix_ao2mo(Ca, ao_qdyz)
            self.mo_qdzz = aoint.matrix_ao2mo(Ca, ao_qdzz)
            del ao_qdxx, ao_qdxy, ao_qdxz, ao_qdyy, ao_qdyz, ao_qdzz
        