"""
library for various functions
"""

import numpy as np


def gau(x,s=1):
    return np.exp(-(x/s)**2)

def p(x,s=1):
    """
    Unnormalized periodized Gaussian.
    """
    
    x = np.mod(x,2*np.pi)
    tot = 0
    for i in range(-10,10+1):
        tot -= np.exp(-(((x+i*2*np.pi))/s)**2)
    return tot

x_temp,dx_temp = np.linspace(-np.pi,np.pi,20000,retstep=True,endpoint=False)
pbar = np.sum(p(x_temp))*dx_temp/(2*np.pi)

def g1(x,s=1,scale=1):
    """
    Periodized Gaussian with subtracted mean.
    """
    
    x = np.mod(x,2*np.pi)
    tot = 0
    for i in range(-10,10+1):
        tot -= np.exp(-(((x+i*2*np.pi))/s)**2)
    return scale*(tot - pbar)


def g2(x,s=1,scale=1):
    """
    Periodized Gaussian with subtracted mean.
    """
    
    x = np.mod(x,2*np.pi)
    tot = 0
    for i in range(-10,10+1):
        tot -= np.exp(-(((x+i*2*np.pi))/s)**2)
    return scale*(tot - pbar)
