import numpy as np 
from time import perf_counter
from scipy.linalg import blas
from pyci.utils import units
from threadpoolctl import threadpool_limits

ALPHA = 1+0j

class ExactProp(object):
    """Given eigenvalues and eigen-vectors(in a certain basis), the methods of 
    exact_prop() aids in exact time-propagation.  
    """
    def __init__(self, eigen_values, eigen_vectors, y0, time_params, ncore=4):
        self.eigvals = eigen_values         # eigvals[i] => i-th eigenvalue
        self.eigvecs = eigen_vectors        # eigvecs[:,i] => i-th eigen-vector 
        self.y0 = y0                                # initial state in CSF basis 
        self.y0_eigen = self.project_vec_eig(y0)   # initial state in EIGEN basis
        self.t0, self.tf, self.dt = time_params # time params
        self.ncore = ncore

    def project_vec_eig(self, y):
        """Projects y from CSF basis => EIGEN basis 
        """
        y_eig = blas.zgemm(ALPHA, np.conjugate(self.eigvecs).T, y)
        return y_eig
    
    def project_vec_csf(self, y):
        """Projects y from EIGEN basis => CSF basis 
        """
        y_csf = blas.zgemm(ALPHA, self.eigvecs, y)
        return y_csf
    
    def _exac_prop_step(self, yi_eig, ti, print_nstep):
        tn = ti + self.dt
        yn_eig = np.exp(-1j*self.eigvals*tn) * yi_eig
        yn_csf = self.project_vec_csf(yn_eig)
        return(yn_csf, tn)
    
    def _calc_expectations(self, ops_list, yi_eig, ti):
        ops_expectations = []
        yi_csf = self.project_vec_csf(yi_eig)
        yi_dag_csf = np.conjugate(yi_csf)
        ti_fs = ti / units.fs_to_au
        norm = np.real(np.sum(yi_dag_csf * yi_csf))
        for operator in ops_list:
            temp = blas.zgemm(ALPHA, operator, yi_csf) 
            temp = blas.zgemm(ALPHA, yi_dag_csf, temp, trans_a=True)
            expectation = np.real(temp)[0][0]
            ops_expectations.append(expectation)
        return ti_fs, norm, ops_expectations

    def _time_propagation(self, ops_list=[], ops_headers=[], print_nstep= 1, 
                        outfile='tdprop.txt', save_wfn=False):
        with threadpool_limits(limits=self.ncore, user_api='blas'):
            yi_eig, ti = self.y0, self.t0
            fobj= open(outfile, 'wb', buffering=0)
            ncols = 2 + len(ops_list)
            fobj.write((" {:<19} "*(ncols)+"\n").format('time_fs', 'norm', *ops_headers).encode("utf-8"))
            ti_fs, norm, ops_expt = self._calc_expectations(ops_list, yi_eig, ti)
            fobj.write((" {:>16.16f} "*(ncols)+"\n").format(ti_fs, norm, *ops_expt).encode("utf-8"))
            y_list = []
            t_list = []
            start = perf_counter()
            while ti < self.tf:
                yi, ti = self._exac_prop_step(yi, ti)
                self.y_list.append(yi)
                self.t_list.append(ti)
            stop = perf_counter()
            fobj.write((" {:>16.16f} "*(ncols)+"\n").format(ti_fs, norm, *ops_expectations))
            print( 'Time taken %3.3f seconds' % (stop-start))    
        return self.y_list, self.t_list 