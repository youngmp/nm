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

import os
import argparse

def rhs_thal(t,z,pdict,option='val',idx=''):
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
            
    v,h,r,q = z

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
    #print(pinf)
    
    tauh = 1/(ah+bh)  #
    taur = 28+exp(-(v+25)/10.5)  #
    
    iL = pdict['gL'+idx]*(v-pdict['eL'+idx])  #
    ina = pdict['gna'+idx]*(minf**3)*h*(v-pdict['ena'+idx])  #
    ik = pdict['gk'+idx]*((0.75*(1-h))**4)*(v-pdict['ek'+idx])  #
    it = pdict['gt'+idx]*(pinf**2)*r*(v-pdict['et'+idx])  #    
    
    dv = (-iL-ina-ik-it+pdict['ib'+idx])/pdict['c'+idx]
    dh = (hinf-h)/tauh
    dr = (rinf-r)/taur
    dq = pdict['alpha'+idx]*(1-q)*ainf-pdict['beta'+idx]*q
    
    if option in ['val','value']:
        return om_fix*omt*np.array([dv/100,dh,dr*100,dq])
    elif option in ['sym','symbolic']:
        return om_fix*omt*Matrix([dv/100,dh,dr*100,dq])

def rhs_cgl(t,z,pdict,option='value',idx=''):
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
    x,y,qc = z
    R2 = x**2 + y**2
    mu = pdict['mu'+idx];sig = pdict['sig'+idx];rho = pdict['rho'+idx]
    alc = pdict['alc'+idx];bec = pdict['bec'+idx]
    omc = pdict['om'+idx]
    om_fix = pdict['om_fix'+idx]
    
    acinf = 1/(1+exp(-125*(x-0.2)))

    if option in ['val','value']:
        return om_fix*omc*np.array([sig*x*(mu-R2)-y*(1+rho*(R2-mu)),
                                    sig*y*(mu-R2)+x*(1+rho*(R2-mu)),
                                    alc*(1-qc)*acinf-bec*qc])
    
    elif option in ['sym','symbolic']:
        return om_fix*omc*Matrix([sig*x*(mu-R2)-y*(1+rho*(R2-mu)),
                                  sig*y*(mu-R2)+x*(1+rho*(R2-mu)),
                                  alc*(1-qc)*acinf-bec*qc])
            
def coupling_thal(vars_pair,pdict,option='val',idx=''):
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
    v,h,r,qt,x,y,qc = vars_pair
    idx = str(idx)
    omt = pdict['om'+idx]
    om_fix = pdict['om_fix'+idx]
    
    if option in ['val','value']:
        return -om_fix*omt*np.array([qc*(v-pdict['esyn'+idx]),
                                     0,0,0])/pdict['c'+idx]
    elif option in ['sym','symbolic']:
        return -om_fix*omt*Matrix([qc*(v-pdict['esyn'+idx]),
                                   0,0,0])/pdict['c'+idx]

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
    x,y,qc,v,h,r,qt = vars_pair
    idx = str(idx)
    omc = pdict['om'+idx]
    om_fix = pdict['om_fix'+idx]

    if option in ['val','value']:
        return -om_fix*omc*np.array([qt*(x-pdict['esyn'+idx]),0,0])
    elif option in ['sym','symbolic']:
        return -om_fix*omc*Matrix([qt*(x-pdict['esyn'+idx]),0,0])

def eval_lam_arr(pts,lam=None):
    """
    use this for vectorizing lamban until i figure out something better
    """
    
    pts = list(pts)
    sol = list(map(lam,pts))
    
    return sol


def normalize_period(system):
    """
    given a limit cycle, find the om_fix that
    yields a 2*pi periodic solution.

    then find om that yields 2*pi*n cycles per 2pi period.
    """
    
    #system['pardict']['om'] = (2*np.pi)/system['T']
    system['pardict']['om_fix'] = system['T']/(2*np.pi)    
    system['T'] = 2*np.pi

    return system
    

def main():

    # parameter dictionary for system 1
    pd1 = {'gL':0.05,'gna':3,'gk':5,
           'gt':5,'eL':-70,'ena':50,
           'ek':-90,'et':0,'esyn':-1,
           'c':1,'alpha':3,'beta':2,
           'sigmat':0.8,'vt':-20,
           'ib':15,'omt':1,'omt_fix':1}
    
    system1 = rsp(var_names=['v','h','r','qt'],
                  pardict=pd1,rhs=rhs_thal,
                  init=np.array([-.64,0.71,0.25,0,5]),
                  TN=2000,model_name='thal',
                  om_fix='omt_fix', # necessary evil
                  pars_for_fname='ib='+str(pd1['ib']),
                  
                  recompute_list=[],
                  z_forward=[False,True,True,True],
                  i_forward=[True,False,False,False],
                  i_bad_dx=[False,True,False,False],
                  i_jac_eps=[1e-3,1e-3,1e-1],
                  coupling=coupling_thal)
    

    pd2 = {'q':1,'d':.9,'sig':.1,'rho':.15,'mu':.15,
           'omc':1,'omc_fix':1,'alc':3,'bec':2,'esyn':0}
    
    system2 = rsp(var_names=['x','y','qc'],
                  pardict=pd2,rhs=rhs_cgl,
                  init=np.array([.333,0,0,2*np.pi]),
                  TN=2000,model_name='cgl',
                  om_fix='omc_fix', # necessary evil
                  
                  recompute_list=[],
                  z_forward=False,
                  i_forward=False,
                  i_bad_dx=False,
                  coupling=coupling_cgl)

    recompute_list=[#'p_data_thal',
                    #'p_data_cgl',
                    'h_data_thal',
                    'h_data_cgl',
                    'p_thal',
                    'p_cgl',
                    'h_thal',
                    'h_cgl']
    
    a = nm(system1,system2,
           #recompute_list=[],
           recompute_list=recompute_list,
           _n=('omt',1),_m=('omc',2),
           mesh_h1=150,mesh_h2=101)

    fig,axs = plt.subplots(3,1)

    x = np.linspace(0,2*np.pi*a._m[1],1000)
    for i in range(3):
        axs[i].axhline(0,0,2*np.pi*a._m[1],color='gray')
        axs[i].plot(x,system2.h['lam'][i](x) - system1.h['lam'][i](x))

    plt.savefig('figs_temp/h_diffs.png')
    plt.close()
    
    
if __name__ == "__main__":
    
    __spec__ = None
    main()
