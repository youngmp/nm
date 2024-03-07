"""
Thalamic model with forcing
"""

# user-defined
from nmCoupling import nmCoupling as nm
from response import Response as rsp

from cgl_thalamic import rhs_thal

import numpy as np
import sympy as sym
from sympy import Matrix
import scipy as sp

import matplotlib.pyplot as plt

import os
import argparse

def rhs(t,z,pdict,option='val',idx=''):
    """
    Right-hand side of the Thalamic model from Wilson and Ermentrout
    RSTA 2019 and Rubin and Terman JCNS 2004
        
        
    Parameters
        
        t : float or sympy object.
            time
        z : array or list of floats or sympy objects.
            state variables of the thalamic model v, h, r, w.
        pdict : dict of flots or sympy objects.
            parameter dictionary pdict[key], val. key is always a string
            of the parameter. val is either the parameter value (float) or 
            the symbolic version of the parameter key.
        option : string.
            Set to 'val' when inputs, t, z, pdict are floats. Set to
            'sym' when inputs t, z, pdict are sympy objects. The default
            is 'val'.
            
    Returns
            
        numpy array or sympy Matrix
        returns numpy array if option == 'val'
        returns sympy Matrix if option == 'sym'
            
    """
    idx = str(idx)

    if option in ['val','value']:
        exp = np.exp
    elif option in ['sym','symbolic']:
        exp = sym.exp
            
    v,h,r = z

    omt = pdict['om'+idx]
    om_fix = pdict['om_fix'+idx]
        
    v *= 100
    r /= 100
    
    ah = 0.128*exp(-(v+46)/18)  #
    bh = 4/(1+exp(-(v+23)/5))  #
    
    minf = 1/(1+exp(-(v+37)/7))  #
    hinf = 1/(1+exp((v+41)/4))  #
    rinf = 1/(1+exp((v+84)/4))  #
    pinf = 1/(1+exp(-(v+60)/6.2))  #
    
    tauh = 1/(ah+bh)  #
    taur = 28+exp(-(v+25)/10.5)  #
    
    iL = pdict['gL'+idx]*(v-pdict['eL'+idx])  #
    ina = pdict['gna'+idx]*(minf**3)*h*(v-pdict['ena'+idx])  #
    ik = pdict['gk'+idx]*((0.75*(1-h))**4)*(v-pdict['ek'+idx])  #
    it = pdict['gt'+idx]*(pinf**2)*r*(v-pdict['et'+idx])  #    
    
    dv = (-iL-ina-ik-it+pdict['ib'+idx])/pdict['c'+idx]
    dh = (hinf-h)/tauh
    dr = (rinf-r)/taur
    
    if option in ['val','value']:
        return om_fix*omt*np.array([dv/100,dh,dr*100])
    elif option in ['sym','symbolic']:
        return om_fix*omt*Matrix([dv/100,dh,dr*100])

    
def main():
    pass


if __name__ == "__main__":
    main()
