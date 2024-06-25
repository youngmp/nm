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
from cgl1 import rhs

import numpy as np
import sympy as sym
from sympy import Matrix
import scipy as sp

from extensisq import SWAG, CK5, Ts5

import pathos
from pathos.pools import ProcessPool
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp



import os
import argparse
import copy


def coupling(vars_pair,pdict,option='val',idx=''):
    x1,y1,x2,y2 = vars_pair
    idx = str(idx)
    omc = pdict['om'+idx]
    om_fix = pdict['om_fix'+idx]

    d = pdict['d'+idx]

    if option in ['val','value']:
        return om_fix*omc*np.array([x2-x1-d*(y2-y1),
                                    y2-y1+d*(x2-x1)])
    
    elif option in ['sym','symbolic']:
        return om_fix*omc*Matrix([x2-x1-d*(y2-y1),
                                  y2-y1+d*(x2-x1)])
    

def main():
    pd1 = {'q':1,'d':.9,'sig':.08,'rho':.12,'mu':1,
           'om':1,'om_fix':1}
    pd2 = copy.deepcopy(pd1)
    
    system1 = rsp(var_names=['x','y'],
                  pardict=pd1,rhs=rhs,
                  init=np.array([1,0,2*np.pi]),
                  TN=2000,
                  idx=0,
                  model_name='cgl0',
                  
                  recompute_list=[],
                  z_forward=False,
                  i_forward=False,
                  i_bad_dx=[False,True,False,False],
                  coupling=coupling)

    system2 = copy.deepcopy(system1)

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
           #recompute_list=recompute_list,
           _n=('om0',1),_m=('om1',2),
           NP=100)


    
if __name__ == "__main__":
    
    __spec__ = None
    main()
