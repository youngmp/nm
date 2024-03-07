"""
CGL with forcing
"""

# user-defined
from nmCoupling import nmCoupling as nm
from response import Response as rsp

import numpy as np
import sympy as sym
from sympy import Matrix
import scipy as sp

from extensisq import SWAG, CK5, Ts5

import pathos
from pathos.pools import ProcessPool
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp

from cgl_thalamic import rhs_cgl

import os
import argparse


def rhs(t,z,pdict,option='value',idx=''):
    """
    Right-hand side of the Complex Ginzburgh-Landau (CGL) model from
    Wilson and Ermentrout RSTA 2019 

    Parameters

        t : float or sympy object.
            time
        z : array or list of floats or sympy objects.
            state variables of the cgl model, x,y,wc
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
    
    if option in ['val','value']:
        exp = np.exp
    elif option in ['sym','symbolic']:
        exp = sym.exp

    idx = str(idx)
    x,y = z

    R2 = x**2 + y**2
    mu = pdict['mu'+idx];sig = pdict['sig'+idx];rho = pdict['rho'+idx]
    omc = pdict['om'+idx]
    om_fix = pdict['om_fix'+idx]

    if option in ['val','value']:
        C = 1#2*np.pi
        return om_fix*omc*np.array([C*sig*x*(1-R2)-y*(C*(1+rho*(R2-1))),
                                    C*sig*y*(1-R2)+x*(C*(1+rho*(R2-1)))])
    
    elif option in ['sym','symbolic']:
        C = 1#2*sym.pi
        return om_fix*omc*Matrix([C*sig*x*(1-R2)-y*(C*(1+rho*(R2-1))),
                                  C*sig*y*(1-R2)+x*(C*(1+rho*(R2-1)))])


def rhs_cgl_old(t,z,pdict,option='value',idx=''):
    """
    Right-hand side of the Complex Ginzburgh-Landau (CGL) model from
    Wilson and Ermentrout RSTA 2019 

    Parameters

        t : float or sympy object.
            time
        z : array or list of floats or sympy objects.
            state variables of the cgl model, x,y,wc
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
    
    if option in ['val','value']:
        exp = np.exp
    elif option in ['sym','symbolic']:
        exp = sym.exp

    idx = str(idx)
    x,y = z
    R2 = x**2 + y**2
    mu = pdict['mu'+idx];sig = pdict['sig'+idx];rho = pdict['rho'+idx]
    omc = pdict['om'+idx]
    om_fix = pdict['om_fix'+idx]

    if option in ['val','value']:
        return om_fix*omc*np.array([sig*x*(mu-R2)-y*(1+rho*(R2-mu)),
                                    sig*y*(mu-R2)+x*(1+rho*(R2-mu))])
    
    elif option in ['sym','symbolic']:
        return om_fix*omc*Matrix([sig*x*(mu-R2)-y*(1+rho*(R2-mu)),
                                  sig*y*(mu-R2)+x*(1+rho*(R2-mu))])


def ff(t):
    return np.sin(t)
    
def main():

    pd1 = {'sig':.08,'rho':.12,'mu':1,
           'om':1,'om_fix':1}
    
    system1 = rsp(var_names=['x','y'],
                  pardict=pd1,rhs=rhs,
                  init=np.array([1,0,2*np.pi]),
                  TN=2000,
                  idx=0,
                  model_name='cglf0',
                  forcing_fn=ff,
                  
                  recompute_list=[],
                  g_forward=False,
                  z_forward=False,
                  i_forward=False,
                  i_bad_dx=True,
                  max_iter=100,
                  rel_tol=1e-9,
                  trunc_order=4,
                  coupling=coupling)
    
    a = nm(system1,None,
           #recompute_list=recompute_list,
           _n=('om0',1),_m=('om1',1),
           NP=200,NH=200)

    
    fig,axs = plt.subplots(3,1)

    x = np.linspace(0,2*np.pi*a._m[1],1000)
    for i in range(3):
        axs[i].axhline(0,0,2*np.pi*a._m[1],color='gray')
        axs[i].plot(x,system1.h['lam'][i](x))

    plt.savefig('figs_temp/h_diffs.png')
    plt.close()
    
    
if __name__ == "__main__":
    
    __spec__ = None
    main()
