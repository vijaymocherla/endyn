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
- Knowles, P. J., & Handy, N. C. 
    Computer physics communications, 54(1), 75-83 (1989).
"""


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
    """ A sub-module implementing Slater-Condon rules.
    """
    def __init__(self, mo_energies, mo_coeff, mo_erints):
        self.eps = mo_energies
        self.Ca = mo_coeff
        self.mo_erints = mo_erints

    @staticmethod
    def common_orblist(det1, det2):
        """Return a list of common orbitals
        """    
        alpha_orbs = bitDet.bitstr2orblist(det1.alpha_bitstr & det2.alpha_bitstr)
        beta_orbs = bitDet.bitstr2orblist(det1.beta_bitstr & det2.alpha_bitstr)
        return alpha_orbs, beta_orbs
    
    @staticmethod
    def num_diff_orb(det1, det2):
        diff_alpha = bitDet.countbits(det1.alpha_bitstr ^ det2.alpha_bitstr)
        diff_beta = bitDet.countbits(det1.beta_bitstr ^ det2.beta_bitstr)        
        return diff_alpha/2 , diff_beta/2
    
    @staticmethod
    def overlap(det1, det2):
        """Returns inner product < det1 | det2 >
        """
        overlap = 0 
        diff_alpha, diff_beta = SlaterCondon.num_diff_orb(det1, det2)
        num_difforb = diff_alpha + diff_beta
        if num_difforb == 0:   # differ 0 orbs
            overlap += 0
        elif num_difforb == 1: # differ 1 orbs
            overlap += 0 
        elif num_difforb == 2: # differ 2 orbs
            overlap += 0 
        return overlap

    @staticmethod
    def one_elec_overlap(det1, det2, one_eprop):
        """ Returns <det1 | O_{1} | det2 >
        """
        one_elec_overlap = 0 
        diff_alpha, diff_beta = SlaterCondon.num_diff_orb(det1, det2)
        num_difforb = diff_alpha + diff_beta
        if num_difforb == 0:   # differ 0 orbs
            one_elec_overlap += 0
        elif num_difforb == 1: # differ 1 orbs
            one_elec_overlap += 0 
        elif num_difforb == 2: # differ 2 orbs
            one_elec_overlap += 0 
        return one_elec_overlap

    @staticmethod
    def two_elec_overlap(det1, det2, two_eprop):
        """Returns < det1 | O_{2} | det2 >
        """
        two_elec_overlap = 0 
        diff_alpha, diff_beta = SlaterCondon.num_diff_orb(det1, det2)
        num_difforb = diff_alpha + diff_beta
        if num_difforb == 0:   # differ 0 orbs
            two_elec_overlap += 0
        elif num_difforb == 1: # differ 1 orbs
            two_elec_overlap += 0 
        elif num_difforb == 2: # differ 2 orbs
            two_elec_overlap += 0 
        return two_elec_overlap

    @staticmethod
    def three_elec_overlap(det1, det2, three_eprop):
        """Returns < det1 | O_{3} | det2 >
        """
        three_elec_overlap = 0 
        diff_alpha, diff_beta = SlaterCondon.num_diff_orb(det1, det2)
        num_difforb = diff_alpha + diff_beta
        if num_difforb == 0:   # differ 0 orbs
            three_elec_overlap += 0
        elif num_difforb == 1: # differ 1 orbs
            three_elec_overlap += 0 
        elif num_difforb == 2: # differ 2 orbs
            three_elec_overlap += 0 
        return three_elec_overlap

class CSF(object):
    def __init__(self, csf_dict, spin=0):
        self.dets = csf_dict.keys()
        self.coeff = csf_dict.values()
        self.Ns = 2*spin + 1

    def overlap(self, another):
        overlap = 0         
        return overlap
        















