from pickletools import optimize
import numpy as np 
from time import perf_counter
from opt_einsum import contract

class SplitOperator(object):
    """Given eigenvalues and eigen-vectors(in a certain basis), the methods of 
    exact_prop() aids in exact time-propagation.  
    """
    def __init__(self, eigen_values, eigen_vectors, field_func, y0, time_params):
        self.eigvals = eigen_values         # eigvals[i] => i-th eigenvalue
        self.eigvecs = eigen_vectors        # eigvecs[:,i] => i-th eigen-vector 
        self.feild_func = field_func
        self.y0 = y0                                # initial state in CSF basis 
        self.y0_eigen = self.project_vec_eig(y0)   # initial state in EIGEN basis
        self.t0, self.tf, self.dt = time_params # time params
        self.y_list = [self.y0]
        self.t_list = [self.t0]

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
    
    def _exac_prop_step(self, ti):
        tn = ti + self.dt
        yn_eigen = np.e**(-1j*self.eigvals*tn) * self.y0_eigen
        yn_csf = self.project_vec_csf(yn_eigen)
        return(yn_csf, tn)

    def _time_propagation(self, check_norm=False):
        yi, ti = self.y0, self.t0
        i = 0 
        start = perf_counter()
        while ti < self.tf:
            #print('t = %2.3f' % ti)
            yi, ti = self._exac_prop_step(ti)
            self.y_list.append(yi)
            self.t_list.append(ti)
            i += 1
        stop = perf_counter()
        print( 'Time taken %3.3f seconds' % (stop-start))    
        return self.y_list, self.t_list 