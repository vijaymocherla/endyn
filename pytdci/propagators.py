import numpy as np 
from time import perf_counter
from opt_einsum import contract

class RK4(object):
    """Fixed step-size implementation of fourth order runge-kutta 
    """
    def __init__(self, func, y0, t0, dt, t_bound):
        self.func = func
        self.y0 = y0
        self.t0 = t0
        self.dt = dt
        self.t_bound = t_bound
        self.y_list = [self.y0]
        self.t_list = [self.t0]


    def _rk4_step(self, yi, ti):
        k1 = contract('ij,i', self.func, yi, optimize=True)
        k2 = contract('ij,i', self.func, yi+(self.dt/2 * k1), optimize=True) 
        k3 = contract('ij,i', self.func, yi+(self.dt/2 * k2), optimize=True) 
        k4 = contract('ij,i', self.func, yi+(self.dt * k3), optimize=True) 
        yn = yi + (self.dt/6 * (k1 + 2*k2 + 2*k3 + k4))
        tn = ti + self.dt
        return(yn, tn)


    def _time_propagation(self, check_norm=False):
        yi, ti = self.y0, self.t0
        i = 0 
        start = perf_counter()
        while ti <= self.t_bound:
            yi, ti = self._rk4_step(yi, ti)
            self.y_list.append(yi)
            self.t_list.append(ti)
            i += 1
        stop = perf_counter()
        print( 'Time taken %3.3f seconds' % (stop-start))    
        return self.y_list, self.t_list   


class ExactProp(object):
    """Given eigenvalues and eigen-vectors(in a certain basis), the methods of 
    exact_prop() aids in exact time-propagation.  
    """
    def __init__(self, eigen_values, eigen_vectors,  y0, t0, dt, t_bound):
        self.eigvals = eigen_values         # eigvals[i] => i-th eigenvalue
        self.eigvecs = eigen_vectors        # eigvecs[:,i] => i-th eigen-vector 
        self.y0 = y0                                # initial state in CSF basis 
        self.y0_eigen = self.project_eigbasis(y0)   # initial state in EIGEN basis
        self.t0 = t0
        self.t_bound = t_bound
        self.dt = dt
        self.y_list = [self.y0]
        self.t_list = [self.t0]


    def project_eigbasis(self, y):
        """Projects y from CSF basis => EIGEN basis 
        """
        y_eig = contract('ij,j', np.conjugate(self.eigvecs).T, y, optimize=True)
        return y_eig
    

    def project_csfbasis(self, y):
        """Projects y from EIGEN basis => CSF basis 
        """
        y_csf = contract('ij,j', self.eigvecs, y, optimize=True)
        return y_csf
    

    def _exac_prop_step(self, ti):
        tn = ti + self.dt
        yn_eigen = np.e**(-1j*self.eigvals*tn) * self.y0_eigen
        yn_csf = self.project_csfbasis(yn_eigen)
        return(yn_csf, tn)


    def _time_propagation(self, check_norm=False):
        yi, ti = self.y0, self.t0
        i = 0 
        start = perf_counter()
        while ti < self.t_bound:
            #print('t = %2.3f' % ti)
            yi, ti = self._exac_prop_step(ti)
            self.y_list.append(yi)
            self.t_list.append(ti)
            i += 1
        stop = perf_counter()
        print( 'Time taken %3.3f seconds' % (stop-start))    
        return self.y_list, self.t_list 



