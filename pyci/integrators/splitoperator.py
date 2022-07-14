from pickletools import optimize
import numpy as np 
from scipy.linalg import expm
from time import perf_counter
from opt_einsum import contract
from pyci.utils import units

class SplitOperator(object):
    """Given eigenvalues and eigen-vectors(in a certain basis), the methods of 
    exact_prop() aids in exact time-propagation.  
    """
    def __init__(self, eigen_values, eigen_vectors, field_func, y0, time_params):
        self.eigvals = eigen_values         # eigvals[i] => i-th eigenvalue
        self.eigvecs = eigen_vectors        # eigvecs[:,i] => i-th eigen-vector 
        self.field_func = field_func
        self.y0_eig = self.project_vec_eig(y0)   # initial state in EIGEN basis
        self.t0, self.tf, self.dt = time_params # time params

    def project_matrix_eigbasis(self, matrix):
        """Projects a matrix from CSF basis => EIGEN basis 
        """
        matrix_eig = contract('iA, AB, Bj -> ij', np.conjugate(self.eigvecs.T), matrix, self.eigvecs, optimize=True) 
        return matrix_eig

    def project_vec_eig(self, y):
        """Projects y from CSF basis => EIGEN basis 
        """
        y_eig = contract('ij,j', np.conjugate(self.eigvecs).T, y, optimize=True)
        return y_eig
    
    def project_vec_csf(self, y):
        """Projects y from EIGEN basis => CSF basis 
        """
        y_csf = contract('ij,j', self.eigvecs, y, optimize=True)
        return y_csf
    
    def _exact_prop_step(self, yi_eig, ti):
        exp_field = self.project_matrix_eigbasis(expm(1j*self.field_func(ti)*self.dt))
        yn_eig = np.exp(-1j*self.eigvals*self.dt) * yi_eig
        yn_eig = contract('ij,j', exp_field, yn_eig, optimize=True)
        tn = ti + self.dt
        return(yn_eig, tn)

    def _time_propagation(self, ops_list=[], ops_headers=[], 
                        print_nstep= 1, outfile='tdprop.txt',
                        save_wfn=False):
        yi_eig, ti = self.y0_eig, self.t0
        iterval = int(0)
        fobj= open(outfile, 'w', buffering=10)
        ncols = 2 + len(ops_list)
        fobj.write((" {:>20} "*(ncols)+"\n").format('time_fs', 'norm', *ops_headers))
        self._calc_expectations(ops_list, yi_eig, ti, fobj, ncols)
        start = perf_counter()
        y_list = []
        t_list = []
        while ti <= self.tf:
            if iterval == print_nstep:
                iterval = int(0)
                self._calc_expectations(ops_list, yi_eig, ti, fobj, ncols)
                if save_wfn:    
                    yi_csf = self.project_vec_csf(yi_eig) 
                    t_list.append(ti)
                    y_list.append(yi_csf)
            yi_eig, ti = self._exact_prop_step(yi_eig, ti)
            iterval += int(1)
        fobj.close()
        stop = perf_counter()
        if save_wfn:
            y_array = np.array(y_list, dtype=np.cdouble)
            t_array = np.array(t_list, dtype=np.float64) 
            np.savez('wfn_log.npz', t_log=t_array, psi_log=y_array)
        print( 'Time taken %3.3f seconds' % (stop-start))    
        return 0
    
    def _calc_expectations(self, ops_list, yi_eig, ti, fobj, ncols):
        ops_expectations = []
        yi_csf = self.project_vec_csf(yi_eig)
        norm = abs(np.sum(np.conjugate(yi_eig.T, dtype=np.cdouble) * yi_csf))
        for operator in ops_list:
            expectation = np.real(contract("i,ij,j->", 
                                np.conjugate(yi_csf.T, dtype=np.cdouble),
                                operator, yi_csf, optimize=True))
            ops_expectations.append(expectation)
        ti_fs = ti / units.fs_to_au
        fobj.write((" {:>16.16f} "*(ncols)+"\n").format(ti_fs, norm, *ops_expectations))
        return 0