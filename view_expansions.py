# view expansions for a given system. user can choose.

from nmCoupling import nmCoupling as nm
from response import Response as rsp

import numpy as np
import sympy as sym
from sympy import Matrix
import scipy as sp

from cgl_thalamic import rhs_cgl, rhs_thal
import cgl2
import cgl1
import thal1

from argparse import ArgumentDefaultsHelpFormatter as ADHF
from argparse import ArgumentParser

def model_cgl1():

    pd2 = {'om':1,'amp':1,'om_fix':1}
    # default period must be 2*np.pi
    system2 = rsp(var_names=[],
                  pardict=pd2,rhs=None,init=None,
                  coupling=None,
                  model_name='f1',
                  forcing_fn=np.sin,
                  idx=1,
                  TN=0)

    pd1 = {'q':1,'d':.9,'sig':.1,'rho':.15,'mu':.15,
           'om':1,'om_fix':1,'alc':3,'bec':2,'esyn':0}
    
    system1 = rsp(var_names=['x','y','w'],
                  pardict=pd1,rhs=rhs_cgl,
                  init=np.array([.333,0,0,2*np.pi]),
                  TN=2000,
                  idx=0,
                  model_name='cgl0',
                  
                  recompute_list=[],
                  z_forward=False,
                  i_forward=False,
                  i_bad_dx=False,
                  coupling=cgl1.coupling_cgl)

    return system1,system2


def model_cgl2():
    
    pd1 = {'q':1,'d':.9,'sig':.1,'rho':.15,'mu':.15,
           'om':1,'om_fix':1,'alc':3,'bec':2,'esyn':0}
    
    system1 = rsp(var_names=['x','y','w'],
                  pardict=pd1,rhs=rhs_cgl,
                  init=np.array([.333,0,0,2*np.pi]),
                  TN=2000,
                  idx=0,
                  model_name='cgl0',
                  
                  recompute_list=[],
                  z_forward=False,
                  i_forward=False,
                  i_bad_dx=False,
                  coupling=cgl2.coupling_cgl)

    pd2 = {'q':1,'d':.9,'sig':.1,'rho':.15,'mu':.15,
           'om':1,'om_fix':1,'alc':3,'bec':2,'esyn':0}
    
    system2 = rsp(var_names=['x','y','w'],
                  pardict=pd2,rhs=rhs_cgl,
                  init=np.array([.333,0,0,2*np.pi]),
                  TN=2000,
                  idx=1,
                  model_name='cgl1',
                  
                  recompute_list=[],
                  z_forward=False,
                  i_forward=False,
                  i_bad_dx=False,
                  coupling=cgl2.coupling_cgl)

    return system1,system2

def model_thal1():
    
    pd1 = {'gL':0.05,'gna':3,'gk':5,
           'gt':5,'eL':-70,'ena':50,
           'ek':-90,'et':0,'esyn':-1,
           'c':1,'alpha':3,'beta':2,
           'sigmat':0.8,'vt':-20,
           'ib':8.5,'om':1,'om_fix':1}

    system1 = rsp(var_names=['v','h','r','qt'],
                  pardict=pd1,rhs=rhs_thal,
                  init=np.array([-.64,0.71,0.25,0,5]),
                  TN=2000,
                  idx=0,
                  model_name='thalf0',

                  recompute_list=[],
                  z_forward=False,
                  i_forward=False,
                  i_bad_dx=False,
                  coupling=thal1.coupling_thal)
    
    pd2 = {'om':1,'amp':1,'om_fix':1,'esyn':0,'c':1}
    ff = lambda x: np.sin(x)+.2
    
    # default period must be 2*np.pi
    system2 = rsp(var_names=[],
                  pardict=pd2,rhs=None,init=None,
                  coupling=None,
                  model_name='thal_force1',
                  forcing_fn=ff,
                  idx=1,
                  TN=0)

    return system1,system2

def model(name):

    if name == 'cgl2':
        return model_cgl2()

    if name == 'cgl1':
        return model_cgl1()

    if name == 'thal1':
        return model_thal1()
    
def neat_out(d,num=1):
    # d is dictionary of sym terms
    s = ''
    for i in range(len(d)):
        s += '\t'*num + 'Order' +str(i)+': '+str(d[i])+'\n'

    return s
    
def main():

    d = 'View specific expansions in models'
    parser = ArgumentParser(description=d,formatter_class=ADHF)

    parser.add_argument('-t','--terms',default='all',type=str,
                        help='pick terms to display')

    parser.add_argument('-m','--model',default='cgl2',type=str,
                        help='pick model pair')

    args = parser.parse_args()
    print('args',args)

    
    system1,system2 = model(args.model)

    

    a = nm(system1,system2,
           recompute_list=[],
           _n=('om0',1),_m=('om1',1),
           NP=100)

    args.terms = args.terms.lower()
    if args.terms == 'all':
        args.terms = 'kgp'

    for letter in args.terms:
        if letter == 'k':
            print('K:')
            keys = system1.K['sym'].keys()
            for key in keys:
                print('\t '+key+':')
                print(neat_out(system1.K['sym'][key],2))
            
        if letter == 'g':
            print('G:')
            keys = system1.K['sym'].keys()
            for key in keys:
                print('\t '+key+':' +str(system1.G['sym'][key]))
            print()

        if letter == 'p':
            print('p:')
            print(neat_out(system1.p['sym']))
            print()

        
        if letter == 'h':
            print('H:')
            print(neat_out(system1.h['sym']))
            print()


if __name__ == "__main__":
    main()
