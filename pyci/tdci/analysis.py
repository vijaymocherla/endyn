#!/usr/bin/env python
#
# Author : Sai Vijay Mocherla <vijaysai.mocherla@gmail.com>
#
"""analysis.py
A module with functions to analyse output from tdci.py 
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy import fft
from pyci.utils.units import fs_to_au
# some matplolib settings to get production level plots
import matplotlib as mpl
mpl.rc('text', usetex=True) # Switches 'ON' LaTex, comment out to switch 'OFF'
mpl.rc('font', family='sans-serif', serif='Computer Modern')
mpl.rc('font', size=14)


def get_prop_data(filename):
    prop_data = np.loadtxt(filename, skiprows=1)
    time_fs = prop_data[:, 0]
    norm = prop_data[:, 1]
    autocorr = prop_data[:, 2]
    X, Y, Z = prop_data[:, 3], prop_data[:, 4], prop_data[:, 5]
    energy = prop_data[:, 6]
    # Fx, Fy, Fz = prop_data[:, 7], prop_data[:, 8], prop_data[:, 9]
    dipoles = (X,Y,Z)
    # fields = (Fx, Fy, Fz)
    return time_fs, norm, autocorr, dipoles, energy

def calc_moments(time_fs, observable):
    time = time_fs * fs_to_au
    velocity = np.gradient(observable, time)
    acceleration = np.gradient(velocity, time)
    return velocity, acceleration

def calc_fft(time_fs, observable, dt_fs=1e-4):
    dt_fs = time_fs[2]-time_fs[1]
    dt = dt_fs * fs_to_au
    time = time_fs * fs_to_au
    freq = 2*np.pi*fft.fftfreq(len(time), dt) 
    obs_fft = fft.fft(observable)
    return freq, obs_fft

def calc_Gobs(time_fs, observable, dt_fs=1e-4, return_moments=False):
    freq, obs_fft= calc_fft(time_fs, observable, dt_fs=dt_fs)
    G_obs = abs(1/((time_fs[-1]-time_fs[0])*fs_to_au) * obs_fft)**2 
    if return_moments:
        vel, acc = calc_moments(time_fs, observable)
        freq, obs_vel_fft= calc_fft(time_fs, vel, dt_fs=dt_fs)
        freq, obs_acc_fft= calc_fft(time_fs, acc, dt_fs=dt_fs)
        G_obs_vel = abs(1/((time_fs[-1]-time_fs[0])*fs_to_au) * obs_vel_fft)**2 
        G_obs_acc = abs(1/((time_fs[-1]-time_fs[0])*fs_to_au) * obs_acc_fft)**2 
        return freq, G_obs, G_obs_vel, G_obs_acc
    else:
        return freq, G_obs, [], []

def plot_hhg(freq, G_obs, G_obs_vel, G_obs_acc, axes, label, xmax=15, w0=0.056961578478002):
    n = int(freq.shape[0]/2)
    hhg =  (G_obs)[:n]
    hhg_vel =  (G_obs_vel)[:n]
    hhg_acc =  (G_obs_acc)[:n]

    index_w0 = max(range(len(hhg)), key=hhg.__getitem__)
    index_w0_vel = max(range(len(hhg_vel)), key=hhg_vel.__getitem__)
    index_w0_acc = max(range(len(hhg_acc)), key=hhg_vel.__getitem__)

    ho = freq[:n]/w0 
    ho_vel = freq[:n]/w0 
    ho_acc = freq[:n]/w0 

    axes[0].plot(ho, hhg, linewidth=1.0, color='red', label=label+r': $\frac{1}{|t_{f}- t_{i}|}\int \langle {D_{z}}(t)\rangle e^{i\omega t} dt$')
    axes[0].legend(frameon=False, loc='upper right')
    axes[0].set_yscale('log')
    axes[0].set_xlim(0,xmax)
    axes[0].tick_params(labelbottom=False)
    axes[1].plot(ho_vel, hhg_vel, linewidth=1.0, color='blue', label=label+r': $\frac{1}{|t_{f}- t_{i}|}\int \langle \dot{D_{z}}(t)\rangle e^{i\omega t} dt$')
    axes[1].legend(frameon=False, loc='upper right')
    axes[1].set_yscale('log')
    axes[1].set_xlim(0,xmax)
    axes[1].tick_params(labelbottom=False)
    axes[1].set_ylabel(r'Signal in (a.u.)', size=18)
    axes[2].plot(ho_acc, hhg_acc, linewidth=1.0, color='orange', label=label+r': $\frac{1}{|t_{f}- t_{i}|}\int \langle \ddot{D_{z}}(t)\rangle e^{i\omega t} dt$')
    axes[2].legend(frameon=False, loc='upper right')
    axes[2].set_yscale('log')
    axes[2].set_xlim(0,xmax)
    axes[2].set_xlabel(r'harmonic order ($\omega / \omega_{0}$)', size=18)


