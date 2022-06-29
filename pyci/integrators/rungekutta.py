import numpy as np 
from time import perf_counter
from opt_einsum import contract

class RK4(object):
    """Fixed step-size implementation of fourth order runge-kutta 
    """
    def __init__(self, func, y0, t0, tf, dt):
        self.func = func
        print(self.func)
        self.y0 = y0
        self.t0 = t0
        self.dt = dt
        self.tf = tf
        self.y_list = [self.y0]
        self.t_list = [self.t0]

    def _rk4_step(self, yi, ti):
        k1 = contract('ij,i', self.func(ti), yi, optimize=True)
        k2 = contract('ij,i', self.func(ti), yi+(self.dt/2 * k1), optimize=True) 
        k3 = contract('ij,i', self.func(ti), yi+(self.dt/2 * k2), optimize=True) 
        k4 = contract('ij,i', self.func(ti), yi+(self.dt * k3), optimize=True) 
        yn = yi + (self.dt/6 * (k1 + 2*k2 + 2*k3 + k4))
        tn = ti + self.dt
        return(yn, tn)

    def _time_propagation(self, check_norm=False):
        yi, ti = self.y0, self.t0
        i = 0 
        start = perf_counter()
        while ti <= self.tf:
            yi, ti = self._rk4_step(yi, ti)
            self.y_list.append(yi)
            self.t_list.append(ti)
            i += 1
        stop = perf_counter()
        print( 'Time taken %3.3f seconds' % (stop-start))    
        return self.y_list, self.t_list   
 
