#!/usr/bin/python envs
#
# Author : Sai Vijay Mocherla <vijaysai.mocherla@gmail.com>
#
import psi4
import numpy as np


class psi4utils:
    """Helper class to get AO integrals and other abinitio data from PSI4 
       for time-dependent configuration interaction(TDCI) calculations.
    """
    def __init__(self, basis, molfile, psi4mem='2 Gb', scratch='./'):
        self.scratch = scratch
        psi4.core.set_output_file('psi4_output.dat', False) # psi4 output
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

    def eri_mo2so(self, Ca, Cb):
        """Returns MO spin eri tensor in physicist's notation
        """
        Ca_psi4_matrix = psi4.core.Matrix.from_array(Ca)
        Cb_psi4_matrix = psi4.core.Matrix.from_array(Cb)
        mo_spin_eri_tensor = np.asarray(self.mints.mo_spin_eri(Ca_psi4_matrix, Cb_psi4_matrix))
        return mo_spin_eri_tensor

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

    @staticmethod
    def matrix_ao2mo(Ca, matrix):
        mo_matrix = np.einsum('pq,pI,qJ->IJ', matrix, Ca, Ca, optimize=True)
        return mo_matrix
    
    @staticmethod
    def eri_mo2so(Ca, Cb, mo_eri):
        mo_so_eri = []
        return mo_so_eri



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
        overlap_aoints = np.asarray(self.mints.ao_overlap())
        kinetic_aoints = np.asarray(self.mints.ao_kinetic())
        potential_aoints = np.asarray(self.mints.ao_potential())
        np.savez(self.scratch+'ao_oeints.npz', 
                overlap_aoints=overlap_aoints,
                kinetic_aoints=kinetic_aoints,
                potential_aoints=potential_aoints)
        return 1
    
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
        dpx_aoints, dpy_aoints, dpz_aoints =  np.asarray(self.mints.ao_dipole())
        np.savez(self.scratch+'ao_dpints.npz',
                dpx_aoints=dpx_aoints,
                dpy_aoints=dpy_aoints,
                dpz_aoints=dpz_aoints)
        return 1

    def save_ao_dpints(self):
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
        qdyy_aoints, qdyz_aoints, qdzz_aoints) =  np.asarray(self.mints.ao_quadrupole())
        np.savez(self.scratch+'ao_qdints.npz',
                qdxx_aoints=qdxx_aoints,
                qdxy_aoints=qdxy_aoints,
                qdxz_aoints=qdxz_aoints,
                qdyy_aoints=qdyy_aoints,
                qdyz_aoints=qdyz_aoints,
                qdzz_aoints=qdzz_aoints)
        return 1


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
        eps_a = np.array(self.scf_wfn.epsilon_a_subset('AO', 'ALL'))
        Ca = np.array(self.scf_wfn.Ca_subset('AO','ALL'))
        # beta orbitals
        # Cb = np.array(scf_wfn.Cb_subset('AO','ALL'))
        # eps_b = np.array(scf_wfn.epsilon_b_subset('AO', 'ALL'))
        np.savez(self.scratch+'mo_scf_info.npz', eps_a=eps_a, Ca=Ca)
        return 1