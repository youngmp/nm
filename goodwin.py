"""
Example: Goodwin circadian oscillator from Gonze et al Biophys J 2005 

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

def rhs(t,z,pdict,option='value'):
    """
    Right-hand side of the Goodwin oscillator from
    Gonze et al Biophys J 2005
    
    Parameters
        
        t : float or sympy object.
            time
        z : array or list of floats or sympy objects.
            state variables of the thalamic model x,y,z,v
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
    
    x,y,z,v = z
    
    p = pdict
    n = p['n']
    
    dx = p['v1']*p['k1']**n/(p['k1']**n+z**n) - p['v2']*x/(p['k2']+x) + p['L']
    dy = p['k3']*x - p['v4']*y/(p['k4']+y)
    dz = p['k5']*y - p['v6']*z/(p['k6']+z)
    dv = p['k7']*x - p['v8']*v/(p['k8']+v)
    
    if option == 'value':
        return np.array([dx,dy,dz,dv])
    elif option == 'sym':
        return Matrix([dx,dy,dz,dv])

def coupling(vars_pair,pdict,option='value'):
    """
    
    Ccoupling function between Goodwin oscillators
    
    E.g.,this Python function is the function $G(x_i,x_j)$
    in the equation
    $\\frac{dx_i}{dt} = F(x_i) + \\varepsilon G(x_i,x_j)$
    
    Parameters
    
        vars_pair : list or array
            contains state variables from oscillator A and B, e.g.,
            x1,y1,z1,v1,x2,y2,z2,v2
        pdict : dict of flots or sympy objects.
            parameter dictionary pdict[key], val. key is always a string
            of the parameter. val is either the parameter value (float) or 
            the symbolic version of the parameter key.
        option : string.
            Set to 'val' when inputs, t, z, pdict are floats. Set to
            'sym' when inputs t, z, pdict are sympy objects. The default
            is 'val'.
    
    Returns
        
        * numpy array or sympy Matrix
            * returns numpy array if option == 'val'. 
            returns sympy Matrix if option == 'sym'
    
    """
    x1,y1,z1,v1,x2,y2,z2,v2 = vars_pair
    
    K = pdict['K']
    kc = pdict['kc']
    F = 0.5*(v1+v2)
    
    if option == 'value':
        return np.array([K*F/(kc+K*F),0,0,0])
    elif option == 'sym':
        return Matrix([K*F/(kc+K*F),0,0,0])
    

def main():
    
    var_names = ['x','y','z','v']
    
    # parameters from Wilson 2020
    pardict = {'v1':.84,'v2':.42,'v4':.35,'v6':.35,'v8':1,
               'k1':1,'k2':1,'k3':.7,'k4':1,'k5':.7,
               'k6':1,'k7':.35,'k8':1,'K':0.5,'kc':1,
               'n':6,'L':0}
    
    kwargs = {'g_forward':True,'z_forward':False,'i_forward':False,
              'i_bad_dx':[False,True,False,False,False,False],
              'dense':True,
              'dir':'home+goodwin_dat/',
              'trunc_order':5,
              'trunc_deriv':5,
              'NA':2000,
              'p_iter':20,
              'max_iter':200,
              'TN':2000,
              'rtol':1e-13,
              'atol':1e-13,
              'rel_tol':1e-10,
              'method':'LSODA',
              'processes':4,
              'chunksize':10000}
    
    
    T_init = 24.2
    LC_init = np.array([.3882,.523,1.357,.4347,T_init])
    
    a = StrongCoupling(rhs,coupling,LC_init,var_names,pardict,**kwargs)
    
    # plot H functions
    phi = np.linspace(0,a.T,a.NA)
    for k in range(a.trunc_order+1):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(phi,a.hodd['dat'][k])
        
        ax.set_title('Goodwin H Function Order '+str(k))
        ax.set_xlabel(r'$\phi$')
        ax.set_ylabel(r'$-2\mathcal{H}^{('+str(k)+r')}_\text{odd}$')
        #ax.set_ylim(-1000,1000)
        #plt.tight_layout()
        #plt.show(block=True)
        plt.savefig('goodwin_hodd'+str(k)+'.pdf')
        #time.sleep(.1)

if __name__ == "__main__":
    __spec__ = None
    main()
