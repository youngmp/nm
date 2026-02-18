
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
            
    v,h,r,w = z

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
    ainf = 1/(1+exp(-(v-pdict['vt'+idx])/pdict['sigmat'+idx]))
    
    tauh = 1/(ah+bh)  #
    taur = 28+exp(-(v+25)/10.5)  #
    
    iL = pdict['gL'+idx]*(v-pdict['eL'+idx])  #
    ina = pdict['gna'+idx]*(minf**3)*h*(v-pdict['ena'+idx])  #
    ik = pdict['gk'+idx]*((0.75*(1-h))**4)*(v-pdict['ek'+idx])  #
    it = pdict['gt'+idx]*(pinf**2)*r*(v-pdict['et'+idx])  #
    
    dv = (-iL-ina-ik-it+pdict['ib'+idx])/pdict['c'+idx]
    dh = (hinf-h)/tauh
    dr = (rinf-r)/taur
    dw = pdict['alpha'+idx]*(1-w)*ainf-pdict['beta'+idx]*w
    
    if option in ['val','value']:
        return om_fix*omt*np.array([dv/100,dh,dr*100,dw])
    elif option in ['sym','symbolic']:
        return om_fix*omt*Matrix([dv/100,dh,dr*100,dw])


def coupling(vars_pair,pdict,option='val',idx='',c_sign=1):
    """
    
    Synaptic coupling function between Thalamic oscillators.
        
    E.g.,this Python function is the function $G(x_i,x_j)$
    in the equation
    $\\frac{dx_i}{dt} = F(x_i) + \\varepsilon G(x_i,x_j)$

    Parameters

        vars_pair : list or array
            contains state variables from oscillator A and B, e.g.,
            vA, hA, rA, wA, vB, hB, rB, wB  
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
    
    v,h,r,w,v2,h2,r2,w2 = vars_pair
    idx = str(idx)
    omt = pdict['om'+idx]
    om_fix = pdict['om_fix'+idx]
    del1 = pdict['del'+idx]
    if idx == 1:
        del1 = 0

    if option in ['val','value']:
        return -c_sign*om_fix*omt*np.array([w2*(v*100-pdict['esyn'+idx])+del1*100/om_fix,0,0,0])/pdict['c'+idx]/100
    
    elif option in ['sym','symbolic']:
        return -c_sign*om_fix*omt*Matrix([w2*(v*100-pdict['esyn'+idx])+del1*100/om_fix,0,0,0])/pdict['c'+idx]/100

def main():
    pd1 = {'gL':0.05,'gna':3,'gk':5,
           'gt':5,'eL':-70,'ena':50,
           'ek':-90,'et':0,'esyn':0,
           'c':1,'alpha':3,'beta':2,
           'sigmat':0.8,'vt':-20,'del':0,
           'ib':3.5,'om':1,'om_fix':1}
    
    # default period must be 2*np.pi
    kws1 = {'var_names':['v','h','r','w'],
            'pardict':pd1,
            'rhs':rhs,
            'coupling':coupling,
            'init':np.array([-.64,0.71,0.25,0,6]),
            'TN':2000,
            'trunc_order':1,
            'z_forward':False,
            'i_forward':False,
            'i_bad_dx':[False,True,False,False,False,False],
            'max_iter':20,
            'rtol':1e-12,
            'atol':1e-12,
            'rel_tol':1e-9,
            'save_fig':False,
            'lc_prominence':.05,
            'factor':1}

    system1 = rsp(idx=0,model_name='thal0_test',**kws1)
    #system2 = rsp(idx=1,model_name='thal1_test',**kws1)

    #a11_p0 = nm(system1,system2,recompute_list=['k_thal0','k_thal1'],
    #            _n=('om0',1),_m=('om1',1),NH=1000,save_fig=False,del1=0)


if __name__ == "__main__":
    main()
