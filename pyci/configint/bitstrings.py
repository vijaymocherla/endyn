"""
Modules for implementing Configuration Interaction calculations
in terms of bitstrings operations.

Reference: 
- Smith, D. G., Burns, L. A., Sherrill, C. D. & et al. Psi4NumPy. 
    JCTC, 14(7), 3504-3511 (2018). 
- Knowles, P. J., & Handy, N. C. 
    Computer physics communications, 54(1), 75-83 (1989).
"""


class bitDet(object):
    """A class for Slater determinants represented as Bit-strings 
    """
    
    def __init__(self, alpha_orblist=None, beta_orblist=None, alpha_bitstr=0, beta_bitstr=0):
        """ Creates a Slater Determinants from lists of alpha and beta indices
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
        while bits != 0:
            if bits & 1 == 1:
                count += 1
            bits >>= 1
        return count
    
    
    @staticmethod
    def countorbitals(orblist):
        return len(orblist)


    @staticmethod
    def get_num_common_orbs(det1, det2):
        """Returns number of common orbitals between 2 determinant bitstrings
        """
        num_common_orbs = bitDet.countbits((det1 ^ det2))
        return num_common_orbs      


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

