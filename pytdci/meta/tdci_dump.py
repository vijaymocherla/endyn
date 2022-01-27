import psi4
import numpy as np
numpy_memory = 16
psi4.core.set_output_file('output1.dat', False)
mol = psi4.geometry("""
C        0.000000    0.000000   -0.136477                                                                                                        
Li       0.000000    0.000000   -2.085527                                           
N        0.000000    0.000000    1.010778

units angstrom
""")

psi4.set_options({'basis':        '6-31g*',
                  'scf_type':     'pk',
                  'reference':    'rhf',
                  'e_convergence': 1e-14,
                  'd_convergence': 1e-8
                })

# Running SCF to get Ground State Hartree-Fock WaveFunction
scf_e, scf_wfn = psi4.energy('SCF', return_wfn=True)
print(scf_e, mol.nuclear_repulsion_energy())

# MO coefficients and energies
Ca = np.array(scf_wfn.Ca_subset('AO','ALL'))
eps = np.array(scf_wfn.epsilon_a_subset('AO', 'ALL'))

# Get basis and orbital information
nbf = scf_wfn.basisset().nbf()
nso = 2 * nbf
nalpha = scf_wfn.nalpha()
nbeta = scf_wfn.nbeta()
nocc = nalpha 
nso = 2 * nbf
nvir = nbf - nocc
nmo = nbf
print( ' ORBITAL INFORMATION  \n'
               '-----------------------\n'
               'Basis functions   : %i  \n' % nbf    + 
               'Spin Orbitals     : %i  \n' % nso    +
               'Molecular Orbitals: %i  \n' % nmo    +
               'Alpha Orbitals    : %i  \n' % nalpha + 
               'Beta Orbitals     : %i  \n' % nbeta  + 
               'Occupied Orbitals : %i  \n' % nocc   +
               'Virtual Orbitals  : %i  \n' % nvir    
            )

mints = psi4.core.MintsHelper(scf_wfn.basisset())
# The are 2-electron repulsion intergrals in AO basis
ao_erints = np.asarray(mints.ao_eri())
# This is still in chemists notation
mo_erints = np.einsum('pqrs,pI,qJ,rK,sL->IJKL', ao_erints, Ca, Ca, Ca, Ca, optimize=True)
# converting mo_erints to physicists notation
mo_erints = mo_erints.transpose(0,2,1,3)

import ci
params = (nocc,nvir)
HCIS = ci.gen_cis_hamiltonian(eps, mo_erints, params)

# MO_vecs = Ca.T
# for i in range(MO_vecs.shape[0]):
#     MO_vecs[i] = MO_vecs[i]/np.sqrt(np.sum(MO_vecs[i]**2))
# MO_amps = MO_vecs**2
# C = np.sum(MO_amps[:, :14], axis=1)*100
# N = np.sum(MO_amps[:, 30:], axis=1)*100
# orbs_list = []
# for i in range(nbf):
#     if C[i]>30.0 and N[i]>30.0:
#         orbs_list.append(i)
#         print('State %3.i with Energy %04.2f has %4.2f %% on C and %4.2f %% on N \n' % (i, eps[i], C[i], N[i]))

# Pulses
def sin_pulse(t, params):
    muij, sigma, w = params
    f0 = np.pi/(sigma*muij)
    fsin = f0 * np.sin(np.pi*t/(2*sigma))**2 * np.sin((w)*t)
    return fsin

def cos_pulse(t, params):
    muij, sigma, w = params
    f0 = np.pi/(sigma*muij)
    tp = sigma
    fcos = f0 * np.cos(np.pi*(t-tp)/(2*sigma))**2 * np.cos(w*(t-tp))
    return fcos  

fs_2_au = 41.341374575751 

t = np.linspace(0,2*fs_2_au, 2000)
sigma = 0.5 * fs_2_au

params = 1, sigma , 1.5
sinp = []
cosp = []
for ti in t:
    if ti <= 1*fs_2_au:
        sinp.append(sin_pulse(ti, params))
        cosp.append(cos_pulse(ti, params))
    else:
        sinp.append(0.0)
        cosp.append(0.0)
import matplotlib.pyplot as plt
figs, axs = plt.subplots(2)
axs[0].plot(t/fs_2_au, sinp, color='blue', label=r'$\sin^2$ envelop')
axs[1].plot(t/fs_2_au, cosp, color='orange', label=r'$\cos^2$ envelop')
axs[0].legend()
axs[1].legend()     

ao_dipoles = mints.ao_dipole()

ao_dipoles = np.array(ao_dipoles)
dpx = ao_dipoles[0]
dpx = np.einsum('iP,jQ,ij->PQ', Ca, Ca, dpx, optimize=True)
dpx = cis.gen_cis_edipole_r(dpx, nocc, nvir)
dpy = ao_dipoles[1]
dpy = np.einsum('iP,jQ,ij->PQ', Ca, Ca, dpy, optimize=True)
dpy = cis.gen_cis_edipole_r(dpy, nocc, nvir)

def cos_x_pulse(t, dpx, params):
    V = -dpx*cos_pulse(t, params)
    return V

def sin_x_pulse(t, dpx, params):
    V = -dpx*sin_pulse(t, params)
    return V

from timeprop import RK4
func = -1j*HCIS
y0 = np.zeros(297)
y0[0] = 1.0
t0 = 0.0 
dt = 1e-5* fs_2_au  # in a.u. or 0.02 atto-second 
t_bound = 6*  fs_2_au # in a.u
propagator = RK4(func, y0, t0, dt, t_bound)     

import time

sigma = 0.5 * fs_2_au
xpulse = []
ypulse = []
# pulse 1
params = 0.086 , sigma, 1.0
yi, ti = propagator.y0, propagator.t0
i = 0 
start = time.time()
while ti <= 1.0*fs_2_au:
    propagator.func = -1j*(HCIS + sin_x_pulse(ti, dpx, params))
    xpulse.append(sin_pulse(ti,params))
    ypulse.append(0)
    #print('%i, %3.8f'%(i, ti))
    yi, ti = propagator._rk4_step(yi, ti)
    propagator.y_list.append(yi)
    propagator.t_list.append(ti)
    i += 1
stop = time.time()
print( 'Time taken %3.3f seconds' % (stop-start))    
propagator.func = func
start = time.time()
while ti <= 2.0*fs_2_au:
    #print('%i, %3.8f'%(i, ti))
    xpulse.append(0)
    ypulse.append(0)
    yi, ti = propagator._rk4_step(yi, ti)
    propagator.y_list.append(yi)
    propagator.t_list.append(ti)
    i += 1
stop = time.time()
print( 'Time taken %3.3f seconds' % (stop-start))   
# pulse 2
start = time.time()
params = 1.0 , sigma, 1.0
while ti <= 3.0*fs_2_au:
    xpulse.append(0)
    ypulse.append(sin_pulse(ti,params))
    propagator.func = -1j*(HCIS + sin_x_pulse(ti, dpy, params))
    #print('%i, %3.8f'%(i, ti))
    yi, ti = propagator._rk4_step(yi, ti)
    propagator.y_list.append(yi)
    propagator.t_list.append(ti)
    i += 1
stop = time.time()
print( 'Time taken %3.3f seconds' % (stop-start))    
propagator.func = func
start = time.time()
while ti <= propagator.t_bound:
    #print('%i, %3.8f'%(i, ti))
    xpulse.append(0)
    ypulse.append(0)
    yi, ti = propagator._rk4_step(yi, ti)
    propagator.y_list.append(yi)
    propagator.t_list.append(ti)
    i += 1
stop = time.time()
print( 'Time taken %3.3f seconds' % (stop-start))   

psi_array = np.array(propagator.y_list)
t_fs = 0.02418884254*np.array(propagator.t_list)

import matplotlib as mpl
mpl.rc('font', size=14)
mpl.rc('text', usetex=True)
mpl.rc('font', family='sans-serif', serif='Helvetica')
energy = []
for yi in psi_array:
    energy.append(np.einsum('i,ij,j', np.conjugate(yi), HCIS, yi, optimize=True))
    s1 = 0
amps1 =  np.conjugate(psi_array[:, s1])*psi_array[:, s1]
# amps2 =  np.conjugate(psi_array[:, s2])*psi_array[:, s2]
plt.rc('text', usetex=True)
fig, ax1 = plt.subplots(figsize=(12,8))
ax1.plot(t_fs,amps1, label=r'$\Psi_{0}^{HF}$')
ax1.plot(t_fs,energy, label=r'$\langle E \rangle$')
ax1.set_ylabel(r'$|\Psi_{}|^2$', fontsize=20)
ax1.set_xlabel(r' $t$ $(fs)$', fontsize=20)
ax1.legend()
ax2 = ax1.twinx()
ax2.plot(t_fs[:600001], xpulse, '-.' ,
         color='green',
         label=(r'$\mu_{ij}$= %3.2f a.u,  $\sigma$ = %3.2f a.u,    $\omega$ =%3.2f a.u'%(0.086,sigma, 0.086)))
ax2.plot(t_fs[:600001], ypulse, '-.' ,
         color='red',
         label=(r'$\mu_{ij}$= %3.2f a.u,  $\sigma$ = %3.2f a.u,    $\omega$ =%3.2f a.u'%(1.0, sigma, 0.044)))

ax2.set_ylabel(r'$f(t)$', fontsize=18)
#ax2.set_ylim(-0.5, 0.5)
ax2.legend(loc='lower right')

# plt.text(2.,1.1, fontsize=20, s=r'$\Psi(t_{0})\  = \ ^{1}\Psi_{8}^{28}$',
#          bbox=dict(facecolor='none', edgecolor='blue'))
#plt.title(r'$\pi\rightarrow \pi^{*}$ in LiCN')
plt.savefig('2laser_pulse.png',dpi=100)
plt.show()
