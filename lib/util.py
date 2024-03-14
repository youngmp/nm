"""
Utility library functions
"""

import time
import os
import dill
import sys

import numpy as np
import scipy as sp
import sympy as sym
import matplotlib.pyplot as plt

from scipy.optimize import bisect

from . import rhs

rhs_avg_ld = rhs.rhs_avg_ld
rhs_avg = rhs.rhs_avg

#from scipy.interpolate import interp1d
from sympy.physics.quantum import TensorProduct as kp
#from sympy.utilities.lambdify import lambdify, implemented_function

from scipy.integrate import solve_ivp

def get_phase(t,sol_arr,skipn,system1):
    """
    Get the brute-force phase estimate over time
    t: time array
    sol_arr: solution array (must include all dimensions of 
    one and only one oscillator)
    skipn: number of points to skip
    system1: object of response functions
    """
    
    phase1 = np.zeros(len(t[::skipn]))
    for i in range(len(t[::skipn])):
        d1 = np.linalg.norm(sol_arr[::skipn][i,:]-system1.lc['dat'],axis=1)
        phase1[i] = np.argmin(d1)/len(system1.lc['dat'])
    return t[::skipn],2*np.pi*phase1


def freq_est(t,y,transient=.5,width=50,prominence=0,return_idxs=False):
    """ 
    Estimate the oscillator frequency
    For use only with the frequency ratio plots
    """
    peak_idxs = sp.signal.find_peaks(y,width=width,prominence=prominence)[0]
    peak_idxs = peak_idxs[int(len(peak_idxs)*transient):]
    freq = 2*np.pi/np.mean(np.diff(t[peak_idxs]))
    
    if return_idxs:
        return freq,peak_idxs
    else:
        return freq

def pl_exist_ld(eps,del1,a,th_init=0,return_data=False):
    """
    
    """
    sys1 = a.system1
    
    if eps == 0:
        return -1
    err = 1
    
    th_temp = np.linspace(0, 2*np.pi, 200)
    rhs = rhs_avg_ld(0,th_temp,a,eps,del1)

    # print('eps,del,sign',eps,del1,np.abs(np.sum(np.sign(rhs))))

    if return_data:
        return th_temp,rhs

    
    
    if np.abs(np.sum(np.sign(rhs))) < len(rhs):
        return 1
    else:
        return -1
    
def get_tongue_ld(del_list,a,deps=.002,max_eps=.3,min_eps=0):
    """
    Get the Arnold tongue for the low-dim reduced model
    (where the isostable coordinate was eliminated)
    """
    ve_exist = np.zeros(len(del_list))
    
    for i in range(len(del_list)):
        print(np.round((i+1)/len(del_list),2),'    ',end='\r')

        if np.isnan(ve_exist[i-1]):
            eps = 0
        else:
            eps = max(ve_exist[i-1] - 2*deps,0)
        while not(pl_exist_ld(eps,del_list[i],a)+1)\
        and eps <= max_eps:
            eps += deps
            #print('existloop',eps)
        if eps >= max_eps:
            ve_exist[i] = np.nan
        else:
            deps2 = deps
            flag = False
            while not(flag) and deps2 < .2:
                #print('while loop',deps2)
                try:
                    out = bisect(pl_exist_ld,0,eps+deps2,args=(del_list[i],a))
                    flag = True
                except ValueError:
                    deps2 += .001
            if flag:
                ve_exist[i] = out
            else:
                ve_exist[i] = np.nan
    print('')
    return del_list,ve_exist

def pl_exist(eps,del1,a,th_init=0,return_data=False,
             pmin=-.25,pmax=.25):
    """
    Check existence of phase-locked solutions in forced system
    given planar reduction.
    """
    system1 = a.system1
    
    if eps == 0:
        return -1
    # every intersection point must be within eps of a point on the other
    # contour path
    err = 1
    
    # cluster together intersection points so that the
    # original points in each flat cluster have a
    # cophenetic_distance < cluster_size
    # from stackoverflow
    cluster_size = 100
    
    th_temp = np.linspace(0, 2*np.pi, 200)
    ps_temp = np.linspace(pmin, pmax, 200)
    
    TH,PS = np.meshgrid(th_temp,ps_temp)

    Z1,Z2 = rhs_avg(0,[TH,PS],a,eps,del1)

    contour1 = plt.contour(TH,PS,Z1,levels=[0],linewidths=.5,colors='k')
    contour2 = plt.contour(TH,PS,Z2,levels=[0],linewidths=.5,colors='b')

    if return_data:
        return contour1,contour2

    plt.close()

    points1 = contour_points(contour1)
    points2 = contour_points(contour2)
    
    if isinstance(points1, int) or isinstance(points2, int):
        
        return -1
        
    else:
        intersection_points = intersection(points1, points2, err)

        if len(intersection_points) == 0:
            return -1
        else:
            return 1
            print('intersection_points',intersection_points)
            intersection_points = cluster(intersection_points, cluster_size)

def get_contour_data(cs):
    x = None;y = None
    for item in cs.collections:
        for i in item.get_paths():
            v = i.vertices
            x = v[:, 0]
            y = v[:, 1]

    if (x is None)*(y is None):
        return [np.nan],[np.nan]
    
    return x,y
