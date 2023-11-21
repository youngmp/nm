"""
Example: Thalamic model from Wilson and Ermentrout RSTA 2019,
Rubin and Terman JCNS 2004

note state variables must be unique
model names must be unique as they are used in
expansion functions g,z,i.
see load_coupling_expansions
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
            

def coupling_cgl(vars_pair,pdict,option='val',idx=''):
    """
        
    Synaptic coupling function between oscillators.
    
    This function is $G_2(X,Y)$

    Parameters

        vars_pair : list or array
            contains state variables from oscillator A and B, e.g.,
            (x,y,wc,v,h,r,wt)
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
            returns numpy array if option == 'val'. 
            returns sympy Matrix if option == 'sym'

    """
    x,y,qc,x2,y2,qc2 = vars_pair
    idx = str(idx)
    omc = pdict['om'+idx]
    om_fix = pdict['om_fix'+idx]

    if option in ['val','value']:
        return -om_fix*omc*np.array([qc2*(x-pdict['esyn'+idx]),0,0])
    elif option in ['sym','symbolic']:
        return -om_fix*omc*Matrix([qc2*(x-pdict['esyn'+idx]),0,0])
    

def main():
    pd1 = {'q':1,'d':.9,'sig':.1,'rho':.15,'mu':.15,
           'om':1,'om_fix':1,'alc':3,'bec':2,'esyn':0}
    
    system1 = rsp(var_names=['x','y','w'],
                  pardict=pd1,rhs=rhs_cgl,
                  init=np.array([.333,0,0,2*np.pi]),
                  TN=2000,
                  idx=0,
                  model_name='cgl0',
                  
                  recompute_list=['het'],
                  z_forward=False,
                  i_forward=False,
                  i_bad_dx=False,
                  coupling=coupling_cgl)
    

    pd2 = {'q':1,'d':.9,'sig':.1,'rho':.15,'mu':.15,
           'om':1,'om_fix':1,'alc':3,'bec':2,'esyn':0}
    
    system2 = rsp(var_names=['x','y','w'],
                  pardict=pd2,rhs=rhs_cgl,
                  init=np.array([.333,0,0,2*np.pi]),
                  TN=2000,
                  idx=1,
                  model_name='cgl1',
                  
                  recompute_list=['het'],
                  z_forward=False,
                  i_forward=False,
                  i_bad_dx=False,
                  coupling=coupling_cgl)

    recompute_list=['p_cgl0',
                    'p_cgl1',
                    'k_cgl0',
                    'k_cgl1',
                    'p_data_cgl0',
                    'p_data_cgl1',
                    'h_data_cgl0',
                    'h_data_cgl1',
                    ]


    a = nm(system1,system2,
           recompute_list=['h_cgl0',
                           'h_cgl1'],
           #recompute_list=recompute_list,
           _n=('om0',1),_m=('om1',2),
           NP=100)


    fig,axs = plt.subplots(3,1)

    x = np.linspace(0,2*np.pi*a._m[1],1000)
    for i in range(3):
        axs[i].axhline(0,0,2*np.pi*a._m[1],color='gray')
        axs[i].plot(x,system2.h['lam'][i](x) - system1.h['lam'][i](-x))

    plt.savefig('figs_temp/h_diffs.png')
    plt.close()

    
if __name__ == "__main__":
    
    __spec__ = None
    main()
