"""
Utility library functions
"""

import time
import os
import dill
import sys

import numpy as np
import sympy as sym
import matplotlib.pyplot as plt

#from scipy.interpolate import interp1d
from sympy.physics.quantum import TensorProduct as kp
#from sympy.utilities.lambdify import lambdify, implemented_function

from scipy.integrate import solve_ivp

def get_phase(t,sol_arr,skipn,system1):

    phase1 = np.zeros(len(t[::skipn]))
    for i in range(len(t[::skipn])):
        d1 = np.linalg.norm(sol_arr[::skipn][i,:]-system1.lc['dat'],axis=1)
        phase1[i] = np.argmin(d1)/len(system1.lc['dat'])
    return t[::skipn],2*np.pi*phase1


def freq_est(t,y,transient=.5,width=50,prominence=0,return_idxs=False):
    """ for use only with the frequency ratio plots"""
    peak_idxs = sp.signal.find_peaks(y,width=width,prominence=prominence)[0]
    peak_idxs = peak_idxs[int(len(peak_idxs)*transient):]
    freq = 2*np.pi/np.mean(np.diff(t[peak_idxs]))
    
    if return_idxs:
        return freq,peak_idxs
    else:
        return freq
