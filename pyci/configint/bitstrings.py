#!/usr/bin/env python
#
# Author : Sai Vijay Mocherla <vijaysai.mocherla@gmail.com>
#
"""
bitstrings.py 

Helper modules for configuration interaction calculations
using bitstrings operations to implement slater-condon rules.

Reference: 
- Smith, D. G., Burns, L. A., Sherrill, C. D. & et al. Psi4NumPy. 
    JCTC, 14(7), 3504-3511 (2018). 
    (For examples on implementing bit-string operations see the Determinant()
     Configuration-Interaction/helper_CI.py)
- Knowles, P. J., & Handy, N. C. 
    Computer physics communications, 54(1), 75-83 (1989).
"""

import numpy as np

class bitDet(object):
    """A class for Slater Determinants represented as Bit-strings 
    """
    def __init__(self, alpha_orblist=None, beta_orblist=None, alpha_bitstr=0, 
                beta_bitstr=0):
        """ Creates a Slater bitDets from lists of alpha and beta indices
        """
        if alpha_orblist != None and alpha_bitstr == 0:
            alpha_bitstr = bitDet.orblist2bitstr(alpha_orblist)
        if beta_orblist != None and beta_bitstr == 0:
            beta_bitstr = bitDet.orblist2bitstr(beta_orblist)     
        # Setting alpha, beta string attributes
        self.alpha_bitstr = alpha_bitstr
        self.beta_bitstr = beta_bitstr

    @staticmethod
    def orblist2bitstr(orblist):
        """Converts a list of MO indices to a bitstring
        """
        if len(orblist) == 0:
            return 0
        orblist = sorted(orblist, reverse=True)
        iPre = orblist[0]
        bitstr = 1
        for i in orblist:
            bitstr <<= iPre - i
            bitstr |= 1  # bitwise OR
            iPre = i
        bitstr <<= iPre  # LEFT-SHFIT by iPre bits
        return bitstr

    @staticmethod
    def bitstr2orblist(bitstr):
        """Converts a bitstring to a list of corresponding MO indices
        """
        i = 0 
        orblist = []
        while bitstr != 0:
            if bitstr & 1 == 1:  # bitwise AND
                orblist.append(i)
            bitstr >>= 1  # RIGHT-SHIFT by 1 bit
            i += 1
        return orblist

    @staticmethod
    def countbits(bitstr):
        """Counts number of orbitals in a bitstring
        """
        count = 0 
        while bitstr != 0:
            if bitstr & 1 == 1:
                count += 1
            bitstr >>= 1
        return count
    
    @staticmethod
    def orbpositions(bitstr, orblist): # redundant function
        """Returns positions of orbitals in a bit-determinant
        """
        count = 0 
        idx = 0
        positions = []
        for i in orblist:
            while idx < i:
                if bitstr & 1 == 1:
                    count += 1
                bitstr >>= 1
                idx += 1
            positions.append(count)
            continue
        return positions
    
    @staticmethod
    def countorbitals(orblist):
        return len(orblist)

    def __orbstr__(self):
        alist = bitDet.bitstr2orblist(self.alpha_bitstr)
        blist = bitDet.bitstr2orblist(self.beta_bitstr)
        return "|"+str(alist)+","+str(blist)+">"

    def __bitstr__(self):
        astr = bin(self.alpha_bitstr)
        bstr = bin(self.beta_bitstr)
        return "|"+astr+","+bstr+">"

    def copy(self):
        """ Deep copy of determinant object
        """
        return bitDet(alpha_bitstr=self.alpha_bitstr, beta_bitstr=self.beta_bitstr)

    # Creation and Annhilation operators for Alpha and Beta electrons
    def add_alpha(self, orbidx):
        """Adds an alpha electron(up-spin) into an MO with index=orbidx 
        """
        self.alpha_bitstr |= 1 << orbidx

    def add_beta(self, orbidx):
        """Adds an beta electron(down-spin) into an MO with index=orbidx
        """
        self.beta_bitstr |= 1 << orbidx

    def remove_alpha(self, orbidx):
        """Removes an alpha electron from an MO with index=orbidx
        """
        self.alpha_bitstr &= ~(1 << orbidx)

    def remove_beta(self, orbidx):
        """Removes an beta electron from an MO with index=orbidx
        """
        self.beta_bitstr &= ~(1 << orbidx)



class SlaterCondon:
    """ A sub-module implementing Slater-Condon rules
        explicitly in MO indices.
    """
    def __init__(self, orbinfo, mo_eps, mo_coeff, mo_erints):
        # currently only implemented for closed shell system
        nel, nbf, nmo = orbinfo
        nocc, nvir = int(nel/2), int((nmo-nel)/2)
        self.occ_list = range(nocc)
        self.mo_eps = mo_eps
        self.mo_coeff = mo_coeff
        self.mo_erints = mo_erints
        self.mo_fock_matrix = np.diag(mo_eps)
        

    @staticmethod
    def common_orblist(det1, det2):
        """Return a list of common orbitals
        """    
        alpha_orbs = bitDet.bitstr2orblist(det1.alpha_bitstr & det2.alpha_bitstr)
        beta_orbs = bitDet.bitstr2orblist(det1.beta_bitstr & det2.alpha_bitstr)
        return alpha_orbs, beta_orbs
    
    @staticmethod
    def diff_orblist(det1, det2):
        """Returns lists of unique [alpha, beta] orbitals in det1 and det2  
        """
        alpha_common = det1.alpha_bitstr & det2.alpha_bitstr
        beta_common = det1.beta_bitstr & det2.beta_bitstr
        det1_orbs = [(det1.alpha_bitstr ^ alpha_common), (det1.beta_bitstr ^ beta_common)]
        det2_orbs = [(det2.alpha_bitstr ^ alpha_common), (det2.beta_bitstr ^ beta_common)]
        det1_diff_orblist = [bitDet.bitstr2orblist(bitstr) for bitstr in det1_orbs]
        det2_diff_orblist = [bitDet.bitstr2orblist(bitstr) for bitstr in det2_orbs]
        return det1_diff_orblist, det2_diff_orblist
    
    @staticmethod
    def num_diff_orb(det1, det2):
        """Returns number of different alpha, beta orbitals b/w det1 & det2
        """
        num_diff_alpha = bitDet.countbits(det1.alpha_bitstr ^ det2.alpha_bitstr)
        num_diff_beta = bitDet.countbits(det1.beta_bitstr ^ det2.beta_bitstr)
        return int(num_diff_alpha/2) , int(num_diff_beta/2)
    
    @staticmethod
    def overlap(det1, det2):
        """Returns inner product < det1 | det2 >
        """
        overlap = 0 
        num_diff_alpha, num_diff_beta = SlaterCondon.num_diff_orb(det1, det2)
        num_difforb = num_diff_alpha + num_diff_beta
        if num_difforb == 0:   # differ 0 orbs
            overlap += 0
        elif num_difforb == 1: # differ 1 orbs
            overlap += 0 
        elif num_difforb == 2: # differ 2 orbs
            overlap += 0 
        return overlap

    def one_elec_overlap(self, num_diff, det1_orbs, det2_orbs, one_eprop):
        """Returns <det1 | O_{1} | det2 >
        """
        alpha1, beta1 = det1_orbs
        alpha2, beta2 = det2_orbs
        one_elec_overlap = 0 # if it differs by 2 or more 
        if num_diff == 0:   # differ 0 orbs
            one_elec_overlap = 2*sum(np.diag(one_eprop))
        elif num_diff == 1: # differ 1 orbs
            if alpha1 != [] and alpha2 != []:
                m, p = alpha1[0], alpha2[0]
                one_elec_overlap = one_eprop[m,p]
            elif beta1 != [] and beta2 != []:
                one_elec_overlap = one_eprop[beta1[0], beta2[0]]  
        return one_elec_overlap

    def two_elec_overlap(self, num_diff, det1_orbs, det2_orbs, two_eprop):
        """Returns < det1 | O_{2} | det2 >
        """
        two_elec_overlap = 0 
        alpha1, beta1 = det1_orbs
        alpha2, beta2 = det2_orbs
        if num_diff == 0:   # differ 0 orbs
            two_elec_overlap = sum([2*two_eprop(a,b,a,b) - two_eprop(a,b,b,a) 
                                for a in self.occ_list for b in self.occ_list])
        elif num_diff == 1: # differ 1 orbs 
            if alpha1 != [] and alpha2 != []:
                m,p = alpha1[0], alpha2[0]
                two_elec_overlap = sum([2*two_eprop(m,a,p,a) - two_eprop(m,a,a,p)
                                        for a in self.occ_list])                    
            elif beta1 != [] and beta2 != []:
                m,p = beta1[0], beta2[0]
                two_elec_overlap = sum([2*two_eprop(m,a,p,a) - two_eprop(m,a,a,p)
                                        for a in self.occ_list])                    
        elif num_diff == 2: # differ 2 orbs
            # needs a loop to sort spins to get matrix elements
            # if alpha1==[] and alpha2==[]:
            #     m,n = beta1
            #     p,q = beta2
            #     two_elec_overlap = two_eprop[m,n,p,q] - two_eprop[m,n,q,p]  
            # elif beta1==[] and beta2==[]:
            #     m,n = alpha1
            #     p,q = alpha2  
            #     two_elec_overlap = two_eprop[m,n,p,q] - two_eprop[m,n,q,p]  
            # elif alpha1==[] and alpha2!=[] or beta1==[] and beta2!=[]:
            #     two_elec_overlap = 0
            # else:
            #     two_elec_overlap = 0
        return two_elec_overlap

    # def three_elec_overlap(self, num_diff, det1_orbs, det2_orbs, three_eprop):
    #     """Returns < det1 | O_{3} | det2 >
    #     """
    #     #three_elec_overlap = 0 
    #     #alpha1, beta1 = det1_orbs
    #     #alpha2, beta2 = det2_orbs
    #     #if num_difforb == 0:   # differ 0 orbs
    #     #     three_elec_overlap += 0
    #     #elif num_difforb == 1: # differ 1 orbs
    #     #     three_elec_overlap += 0 
    #     #elif num_difforb == 2: # differ 2 orbs
    #     #     three_elec_overlap += 0 
    #     return three_elec_overlap

    def comp_hmatrix_elem(self, det1, det2):
        """Returns matrix element < det1 | H | det2 >
        """
        hmatrix_elem = 0 
        diff_alpha, diff_beta = SlaterCondon.num_diff_orb(det1, det2)
        num_diff = diff_alpha + diff_beta
        det1_orbs, det2_orbs = SlaterCondon.diff_orblist(det1,det2)
        hmatrix_elem += self.one_elec_overlap(num_diff, det1_orbs, det2_orbs, self.mo_fock_matrix) 
        hmatrix_elem += self.two_elec_overlap(num_diff, det1_orbs, det2_orbs, self.mo_erints) 
        return hmatrix_elem