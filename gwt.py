"""
Example: Goodwin + thalamic

"""

import numpy as np
import sympy as sym
from sympy import Matrix
import scipy as sp

import matplotlib.pyplot as plt

import os
import argparse

def coupling_gw(vars_pair,pdict,option='value',idx=''):
    """
    Coupling function for gw in gwt (goodwin-thalamic)
    """

    x1,y1,z1,v1,v2,h2,r2,w2 = vars_pair
    idx = str(idx)

    K = pdict['K'+idx]
    #vc = pdict['eps']
    kc = pdict['kc'+idx]
    F = (v1 + w2)/2
    del1 = pdict['del'+idx]

    om = pdict['om'+idx]
    om_fix = pdict['om_fix'+idx]
    
    if option in ['value','val']:
        #return om*om_fix*np.array([K*F/(kc+K*F),0,0,0])
        return om*om_fix*np.array([K*F/kc + del1,0,0,0])
    elif option in ['sym','symbolic']:
        #return om*om_fix*Matrix([K*F/(kc+K*F),0,0,0])
        return om*om_fix*Matrix([K*F/kc + del1,0,0,0])


def coupling_thal(vars_pair,pdict,option='val',idx=''):
    """
    Coupling function for thal in gwt (goodwin-thalamic)
    """
    
    v1,h1,r1,w1,x2,y2,z2,v2 = vars_pair
    idx = str(idx)
    omt = pdict['om'+idx]
    om_fix = pdict['om_fix'+idx]
    del1 = pdict['del'+idx]
    
    if option in ['val','value']:
        return -om_fix*omt*np.array([v2*(v1*100-pdict['esyn'+idx])+del1*100,
                                     0,0,0])/pdict['c'+idx]/100
    elif option in ['sym','symbolic']:
        return -om_fix*omt*Matrix([v2*(v1*100-pdict['esyn'+idx])+del1*100,
                                   0,0,0])/pdict['c'+idx]/100
