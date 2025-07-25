"""
Utility library functions
"""
from . import rhs

import time
import os
import dill
import sys

import numpy as np
import scipy as sp
import sympy as sym
import matplotlib.pyplot as plt

import matplotlib.lines as mlines
import matplotlib.patches as mpatches

from scipy.optimize import bisect

from copy import deepcopy

from .rhs import _redu_c

rhs_avg_ld = rhs.rhs_avg_1df
rhs_avg = rhs.rhs_avg_2d

#from scipy.interpolate import interp1d
from sympy.physics.quantum import TensorProduct as kp
#from sympy.utilities.lambdify import lambdify, implemented_function

from scipy.integrate import solve_ivp
kw_bif = {'method':'LSODA','dense_output':True,'rtol':1e-9,'atol':1e-9}

def add_arrow_to_line2D(
        axes, line, arrow_locs=[0.2, 0.4, 0.6, 0.8],
        arrowstyle='-|>', arrowsize=1.2, transform=None,
        tail_space=50):

    if not isinstance(line, mlines.Line2D):
        raise ValueError("expected a matplotlib.lines.Line2D object")
    x, y = line.get_xdata(), line.get_ydata()

    arrow_kw = {
        "arrowstyle": arrowstyle,
        "mutation_scale": 10 * arrowsize,
    }

    color = line.get_color()
    use_multicolor_lines = isinstance(color, np.ndarray)
    #if use_multicolor_lines:
    #    raise NotImplementedError("multicolor lines not supported")
    #else:
    arrow_kw['color'] = color

    linewidth = line.get_linewidth()
    if isinstance(linewidth, np.ndarray):
        raise NotImplementedError("multiwidth lines not supported")
    else:
        arrow_kw['linewidth'] = linewidth

    if transform is None:
        transform = axes.transData

    arrows = []
    for loc in arrow_locs:
        s = np.nancumsum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))
        n = np.searchsorted(s, s[-1] * loc)
        arrow_tail = (x[n-tail_space], y[n-tail_space])
        arrow_head = (np.nanmean(x[n:n + 2]),
                      np.nanmean(y[n:n + 2]))
        p = mpatches.FancyArrowPatch(
            arrow_tail, arrow_head, transform=transform,
            **arrow_kw)
        axes.add_patch(p)
        arrows.append(p)
        #print(loc)
    return arrows



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
        idx1 = np.argmin(d1)
        
        phase1[i] = idx1/len(system1.lc['dat'])
        
    return t[::skipn],2*np.pi*phase1


def freq_est(t,y,transient=.5,width=10,prominence=.05,return_idxs=False):
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

def pl_exist_1d(eps,del1,a,th_init=0,return_data=False):
    """
    
    """
    sys1 = a.system1
    
    if eps == 0:
        return -1
    err = 1
    
    th_temp = np.linspace(0, 2*np.pi, 200)
    rhs = rhs_avg_ld(0,th_temp,a,eps,del1)

    if return_data:
        return th_temp,rhs
    
    if np.abs(np.sum(np.sign(rhs))) < len(rhs):
        return 1
    else:
        return -1
    
def get_tongue_1d(del_list,a,deps=.002,max_eps=.3,min_eps=0):
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

def is_stable(J):
    u,v = np.linalg.eig(J)
    print('eigs',u)
    print('')
    # if all real parts are negative, return true
    if np.sum(np.real(u)<0) == len(u):
        return True
    else:
        return False

def jac_2d(y,a,eps,del1,h=.01):
    """
    Jacobian for 2d averaged system (forcing only)
    """
    print('y',y,'eps',eps,'del1',del1)
    dx = np.array([h,0])
    dy = np.array([0,h])

    args = (a,eps,del1)
    
    c1 = (rhs.rhs_avg_2d(0,y+dx,*args)-rhs.rhs_avg_2d(0,y,*args))/h
    c2 = (rhs.rhs_avg_2d(0,y+dy,*args)-rhs.rhs_avg_2d(0,y,*args))/h

    return np.array([c1,c2])


def pl_exist_2d(eps,del1,a,th_init=0,return_data=False,
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
    err = .1
    
    # cluster together intersection points so that the
    # original points in each flat cluster have a
    # cophenetic_distance < cluster_size
    # from stackoverflow
    cluster_size = 100
    
    th_temp = np.linspace(0, 2*np.pi, 1000)
    ps_temp = np.linspace(pmin, pmax, 1000)
    
    TH,PS = np.meshgrid(th_temp,ps_temp)

    Z1,Z2 = rhs_avg(0,[TH,PS],a,eps,del1)
    fig_temp,axs_temp = plt.subplots()

    contour1 = axs_temp.contour(TH,PS,Z1,levels=[0],linewidths=.5,colors='k')
    contour2 = axs_temp.contour(TH,PS,Z2,levels=[0],linewidths=.5,colors='b')

    plt.close(fig_temp)
    
    if return_data:
        points1 = contour_points(contour1)
        points2 = contour_points(contour2)

        if isinstance(points1, int) or isinstance(points2, int):
            data = np.array([[np.nan,np.nan]])

        else:
            intersection_points = intersection(points1, points2, err)

            if len(intersection_points) == 0:
                data = np.array([[np.nan,np.nan]])
            
            else:
                data = cluster(intersection_points,cluster_size)

        return contour1,contour2,data
    else:
        return contour1,contour2

def get_tongue_2d(del_list,a,deps=.002,max_eps=.3,min_eps=0):

    ve_exist = np.zeros(len(del_list))
    
    for i in range(len(del_list)):
        print(np.round((i+1)/len(del_list),2),'    ',end='\r')

        if np.isnan(ve_exist[i-1]):
            eps = 0
        else:
            eps = max(ve_exist[i-1] - 2*deps,0)
        while not(pl_exist(eps,del_list[i],a)+1)\
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
                    out = bisect(pl_exist,0,eps+deps2,args=(del_list[i],a))
                    flag = True
                except ValueError:
                    deps2 += .001
            if flag:
                ve_exist[i] = out
            else:
                ve_exist[i] = np.nan
    print('')
    return del_list,ve_exist

import collections
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial as spatial
import scipy.spatial.distance as dist
import scipy.cluster.hierarchy as hier


def intersection(points1, points2, eps):
    tree = spatial.KDTree(points1)
    distances, indices = tree.query(points2, k=1, distance_upper_bound=eps)
    intersection_points = tree.data[indices[np.isfinite(distances)]]
    return intersection_points


def cluster(points, cluster_size):
    dists = dist.pdist(points, metric='sqeuclidean')
    linkage_matrix = hier.linkage(dists, 'average')
    groups = hier.fcluster(linkage_matrix, cluster_size, criterion='distance')
    return np.array([points[cluster].mean(axis=0)
                     for cluster in clusterlists(groups)])


def contour_points(contour, steps=1):
    for linecol in contour.collections:
        
        if len(linecol.get_paths()) == 0:
            return 0
    
    return np.row_stack([path.interpolated(steps).vertices
                         for linecol in contour.collections
                         for path in linecol.get_paths()])

def clusterlists(T):
    '''
    http://stackoverflow.com/a/2913071/190597 (denis)
    T = [2, 1, 1, 1, 2, 2, 2, 2, 2, 1]
    Returns [[0, 4, 5, 6, 7, 8], [1, 2, 3, 9]]
    '''
    groups = collections.defaultdict(list)
    for i, elt in enumerate(T):
        groups[elt].append(i)
    return sorted(groups.values(), key=len, reverse=True)


def period_is_multi(sol,idx:int=0,prominence:float=.25):
    """
    check if period is uneven during n:m locking
    assumes transients have decayed
    """

    tn = len(sol.y.T)
    peak_idxs = sp.signal.find_peaks(sol.y.T[:,idx],
                                     prominence=prominence)[0]
    maxidx1 = peak_idxs[-2]    
    maxidx_prev1 = peak_idxs[-3]

    maxidx2 = peak_idxs[-3]    
    maxidx_prev2 = peak_idxs[-4]

    
    def sol_min(t):
        return -sol.sol(t)[idx]

    # get stronger estimates of max values
    # use optimization on ode solution
    pad1lo = (sol.t[maxidx1]-sol.t[maxidx1-1])/2
    pad1hi = (sol.t[maxidx1+1]-sol.t[maxidx1])/2
    bounds1 = [sol.t[maxidx1]-pad1lo,sol.t[maxidx1]+pad1hi]
    res1 = sp.optimize.minimize_scalar(sol_min,bounds=bounds1)

    pad2lo = (sol.t[maxidx_prev1]-sol.t[maxidx_prev1-1])/2
    pad2hi = (sol.t[maxidx_prev1+1]-sol.t[maxidx_prev1])/2
    bounds2 = [sol.t[maxidx_prev1]-pad2lo,sol.t[maxidx_prev1]+pad2hi]
    res2 = sp.optimize.minimize_scalar(sol_min,bounds=bounds2)

    T_init = res1.x - res2.x

    
    pass

def get_period(sol,idx:int=0,prominence:float=.25,
               idx_shift:int=0)->float:
    """
    sol: solution object from solve_ivp
    idx: index of desired variable
    nm: n:m locking numbers
    """
    
    tn = len(sol.y.T)
    peak_idxs = sp.signal.find_peaks(sol.y.T[:,idx],prominence=prominence)[0]
    n_half = int(len(peak_idxs)/2)

    # instead of always using the last points
    # start with the max idx after 50% of the simulation as a starting point.

    # sorted indices of peak values
    peak_idxs_sorted = np.argsort(sol.y.T[peak_idxs[n_half:],idx])[::-1]

    if peak_idxs_sorted[0] == 0:
        idx_temp = peak_idxs_sorted[1]
    else:
        idx_temp = peak_idxs_sorted[0]
        
    maxidx = peak_idxs[n_half:][idx_temp]
    maxidx_prev = peak_idxs[n_half:][idx_temp-1]

    def sol_min(t):
        return -sol.sol(t)[idx]

    # get stronger estimates of max values
    # use optimization on ode solution
    pad1t = (sol.t[maxidx]-sol.t[maxidx-5])/2
    bounds1 = [sol.t[maxidx]-pad1t,sol.t[maxidx]+pad1t]
    res1 = sp.optimize.minimize_scalar(sol_min,bounds=bounds1)

    pad2t = (sol.t[maxidx_prev]-sol.t[maxidx_prev-5])/2
    bounds2 = [sol.t[maxidx_prev]-pad2t,sol.t[maxidx_prev]+pad2t]
    res2 = sp.optimize.minimize_scalar(sol_min,bounds=bounds2)

    print('res period',res1.x,res2.x)

    T_init = res1.x - res2.x
    
    return T_init,res1.x


def get_period_all(sol,idx:int=0,prominence:float=.25,
                   idx_shift:int=0,n:int=0)->float:
    """
    Gets each peak-to-peak period T_i of a given oscillator,
    for i = 1,...,n, where n is the number of peak to peak
    periods that add up to 2pi (the oscillator locking number).
    
    sol: solution object from solve_ivp
    idx: index of desired variable
    n: multiple of desired period

    Returns n numbers. T1,...Tn

    It's just really rare for there to be nonequal peaks with different periods.
    Small eps means peak-to-peak periods and amplitudes are generally the same.
    Greater eps means peak-to-peak periods differ along with amplitudes.
    So picking the largest peak amplitude (after transients) is a reasonable
    starting point.
    """
    
    tn = len(sol.y.T)
    peak_idxs = sp.signal.find_peaks(sol.y.T[:,idx],prominence=prominence)[0]

    # start with the max idx after 50% of the simulation as a starting point.
    n_half = int(len(peak_idxs)/2)
    
    # get sorted indices of peak values
    peak_idxs_sorted = np.argsort(sol.y.T[peak_idxs[n_half:],idx])[::-1]

    # avoids getting a peak at zero or last index and having issues with opt.
    
    choice = 0
    idx_temp = peak_idxs_sorted[choice]
    while idx_temp in [0,n_half-1,n_half-2,n_half-3,n_half-4]:
        choice += 1
        idx_temp = peak_idxs_sorted[choice]

    # the max idx is taken to be at the largest of all peaks
    # also get next n peaks
    maxidxs = peak_idxs[n_half:][idx_temp:idx_temp+n]
    maxidxs_prev = peak_idxs[n_half:][idx_temp-1:idx_temp-1+n]

    def sol_min(t):
        return -sol.sol(t)[idx]

    
    Tlist = []

    for maxidx,maxidx_prev in zip(maxidxs,maxidxs_prev):
       # get stronger estimates of max values
       # use optimization on ode solution
       pad1t = (sol.t[maxidx]-sol.t[maxidx-5])/2
       bounds1 = [sol.t[maxidx]-pad1t,sol.t[maxidx]+pad1t]
       res1 = sp.optimize.minimize_scalar(sol_min,bounds=bounds1)

       pad2t = (sol.t[maxidx_prev]-sol.t[maxidx_prev-5])/2
       bounds2 = [sol.t[maxidx_prev]-pad2t,sol.t[maxidx_prev]+pad2t]
       res2 = sp.optimize.minimize_scalar(sol_min,bounds=bounds2)

       #print('res period',res1.x,res2.x)

       T_init = res1.x - res2.x

       Tlist.append((T_init,res1.x))
    
    return Tlist





def get_phase_diff_f(rhs,phi0,a,eps,del1,u_sign:int=-1,
                     max_time=500,prominence=.1):
    """
    find initial condition on limit cycle. For forced systems only.
    does not extract phase, only state variables and period.

    u_sign: simple way to make sure that phase zero starts at appropriate
    place for certain forcing functions.
    """
    
    init1 = list(a.system1.lc['dat'][int(phi0*a.system1.TN/(2*np.pi)),:])
    init = np.array(init1+[2*np.pi])
    
    # run for a while and get period
    t = np.arange(0,max_time,.01)
    sol = solve_ivp(rhs,[0,max_time],init[:-1],args=(a,eps,del1),t_eval=t,
                    **kw_bif)
    
    # get period estimate using original system
    period_est1,time_est1 = get_period(sol,idx_shift=0)
    period_est2,time_est2 = get_period(sol,idx_shift=1)

    if period_est1 > period_est2:
        period_est = period_est1
        time_est = time_est1
    else:
        period_est = period_est2
        time_est = time_est2
    
    peak_idxs1 = sp.signal.find_peaks(sol.y.T[:,0],prominence=prominence)[0]

    # phase zero of the forcing function starts at time zero
    # so this part is trivially easy to compute.
    # take -abs(mod(sol.t,2*pi) - 2*pi). then peaks will be at zeros.
    peaks_at_zero_phase = -abs(mod(sol.t,2*pi) - 2*pi)
    
    peak_idxs2 = sp.signal.find_peaks(peaks_at_zero_phase)[0]

    if len(peak_idxs1) > len(peak_idxs2):
        peak_idxs1 = peak_idxs1[:len(peak_idxs2)]
    else:
        peak_idxs2 = peak_idxs2[:len(peak_idxs1)]

    # get initial condition at zero phase of forcing function
    def u_min(t):
        return a.system1.forcing_fn(t*(a._m[1]+del1))

    ress = []
    for shift in range(a._m[1]):
        bounds1 = [time_est-period_est*(1+2*shift)*a._n[1]/2/a._m[1],
                   time_est+period_est*(1-2*shift)*a._n[1]/2/a._m[1]]
        #print(time_est,period_est,shift)
        #print(bounds1)
        res1 = sp.optimize.minimize_scalar(u_min,bounds=bounds1)
        ress.append(res1.x)

    ress = np.asarray(ress)

    # diff taken in time not phase, so order is reversed.
    phis = np.mod(2*np.pi*(ress - time_est)/(period_est),2*np.pi)
    print('times full vs force',time_est,res1.x,phis,'eps',eps,'per',period_est)

    
    if True:

        fig,axs = plt.subplots(3,1,figsize=(4,2))

        axs[0].scatter(t[peak_idxs1],(t[peak_idxs2]-t[peak_idxs1]*a.om),
                       s=5,color='gray',alpha=.5,label='Full')

        axs[1].plot(sol.t,sol.y[0])
        axs[1].plot(t,u_temp)

        axs[1].scatter(t[peak_idxs1],sol.y[0,peak_idxs1])
        axs[1].scatter(t[peak_idxs2],u_temp[peak_idxs2])

        axs[2].plot(sol.t,sol.y[0])
        axs[2].plot(t,u_temp)
        axs[2].set_xlim(time_est-10,time_est+10)

        #axs[0].set_xlim(t[peak_idxs1][-1]-50,t[peak_idxs1][-1])
        axs[1].set_xlim(t[peak_idxs1][-1]-50,t[peak_idxs1][-1])
        
        axs[0].set_ylim(0,2*np.pi)
        
        #print('times full vs force',t[peak_idxs1][-1],t[peak_idxs2][-1],
        #      t[peak_idxs2][-1]-t[peak_idxs1][-1]*a.om)
        
        plt.savefig('figs_temp/phi0_eps='+str(eps)+'del='+str(del1)+'.png')
        plt.close()

    
    return np.asarray(phis)


def get_phase_diff_f_v2(rhs,phi0,a,eps,del1,max_time=500):
    """
    get phase difference given phase-locking for a forced system.

    rhs: right-hand side of full system
    phi0: initial phase of full system
    
    """
    
    init = list(a.system1.lc['dat'][int(phi0*a.system1.TN/(2*np.pi)),:])
    #init = np.array(init1+[2*np.pi])
    
    # run for a while and get period
    t = np.arange(0,max_time,.01)
    sol = solve_ivp(rhs,[0,max_time],init,args=(a,eps,del1),t_eval=t,**kw_bif)

    
    # get period estimates
    Tlist = get_period_all(sol,n=a._n[1])
    print('eps=',eps,'; Tlist=',Tlist)

    total_period = 0
    for i in range(len(Tlist)):
        total_period += Tlist[i][0]

    # since the forcing phase always starts with zero at time zero
    # and does not change over time (excxept for delta)
    # we can just take the times of the peaks of the full model
    # mod 2pi+delta.

    # phase diff for each peak-to-peak period

    phis = []
    for (period_est,time_est) in Tlist:

        # diff taken in time not phase, so order is reversed.
        phi = np.mod(a._n[1]*2*np.pi*(-time_est)/total_period,2*np.pi)

        phis.append(phi)

    
    return np.asarray(phis)



def bif1d(a,eps,del1,domain,rhs=_redu_c,miter=None):
    """
    for coupling only
    gets stable and unstable points for a given epsilon and delta
    """

    if miter is None:
        y = rhs(0,domain,a,eps,del1)
    else:
        y = rhs(0,domain,a,eps,del1,miter)
            
    # get all zeros
    z1 = domain[1:][(y[1:]>0)*(y[:-1]<=0)]
    z2 = domain[1:][(y[1:]<0)*(y[:-1]>=0)]

    return z1,z2




def get_initial_phase_diff_c(phi0,a,eps,del1,max_time=1000,_full_rhs=None,
                             prominence=0.1):

    # get first phase estimate based on phi0
    init1 = list(a.system1.lc['dat'][int(phi0*a.system1.TN/(2*np.pi)),:])
    init2 = list(a.system2.lc['dat'][0,:])
    init = init1+init2
    
    # run for a while and get period and steady-state phase difference
    sol = solve_ivp(_full_rhs,[0,max_time],init,
                    args=(a,eps),**kw_bif)

    # get period estimate and time at which period was estimated.
    period_est = 0

    # this is in case the ISIs are unequal.
    # adding up the ISIs between n+1 peaks makes the period more accurate.
    for i in range(a._n[1]):
        pp,tt = get_period(sol,idx_shift=-i,prominence=prominence)
        period_est += pp

        if i == 0:
            time_est = tt
    
    # get solution values at time_est.
    init = list(sol.sol(time_est))
    init = np.array(init+[period_est])
    
    # return solution values at steady-state phase difference.
    return init


def get_phase_lock_full(Y,a,del1,eps_init,eps_final,deps,
                        max_iter:int=20,_full_rhs=None,
                        fudge=.01,prominence=.05):
    
    """
    "fudge' extends the integration a tiny amount to ensure that the peak
    of the solution is not exactly at the edge (which makes find_peaks fail)
    """
    
    Y_init = deepcopy(Y)
    dy_init = np.ones(len(Y))/5
    eps_range = np.arange(eps_init,eps_final,deps)
    ts = np.zeros([len(eps_range),a._n[1]+a._m[1]])
    t_diffs = np.zeros([len(eps_range),a._n[1]+a._m[1]])
    
    for i,eps in enumerate(eps_range):
        
        dy = deepcopy(dy_init)
        
        counter = 0
        print('eps iter',i,eps,end='                \n')
        
        while np.linalg.norm(dy) > 1e-6 and\
        counter <= max_iter:# and np.linalg.norm(dy)<1:
            
            args_temp = {'fun':_full_rhs,'args':(a,eps,del1),
                         'dense_output':True,**kw_bif}
            dy,solb = get_dy_nm_coupling(Y,args_temp,_full_rhs=_full_rhs,
                                    return_sol=True)
            Y += dy
            
            str1 = 'iter={}, rel. err ={:.2e}'
            end = '                                              \r'
            printed = str1.format(counter,np.linalg.norm(dy))
            print(printed,end=end)
            counter += 1

            if False:
                fig,axs = plt.subplots()
                axs.plot(solb.t,solb.y.T[:,0])
                axs.plot(solb.t,solb.y.T[:,4])
                axs.set_title('counter'+str(counter))
                plt.show()

    
        if (counter >= max_iter) or (np.linalg.norm(dy) >= 1):
            ts[i,:] = np.nan
            Y = Y_init
            dy = np.ones(len(Y))/5
        
        else:
            ff = 10*max(a._n[1],a._m[1])
            solf = solve_ivp(y0=Y[:-1],t_span=[0,ff*Y[-1]+Y[-1]/4],**args_temp)
            
            idx1 = sp.signal.find_peaks(solf.y.T[:,0],prominence=prominence)[0]
            idx2 = sp.signal.find_peaks(solf.y.T[:,4],prominence=prominence)[0]

            if len(idx2) == 0:
                Y = Y_init
                dy = dy_init
                ts[i,:] = np.nan
                
            else:
                
                # get periods and tdiff for oscillator 1
                for ll in range(a._n[1]):
                    ts[i,ll] = get_period(solf,idx=0,idx_shift=-ll,prominence=prominence)[0]
                    t_diffs[i,ll] = solf.t[idx2[0]]-solf.t[idx1[ll]]

                # get periods for oscillator 2
                for ll in range(a._m[1]):
                    idxn = len(a.system1.var_names)
                    ts[i,a._n[1]+ll] = get_period(solf,idx=idxn,idx_shift=-ll,prominence=prominence)[0]
                    print('t1,t2',solf.t[idx2[ll]],solf.t[idx1[0]])
                    t_diffs[i,a._n[1]+ll] = solf.t[idx2[ll]]-solf.t[idx1[0]]
                
                                
                print('periods',ts[i,:])
        
        
    return eps_range,ts,t_diffs


def follow_phase_diffs(init=0,a=None,del1=0,eps_init=0.02,
                       eps_final=0.0005,deps=-.0005,
                       recompute=False,bifdir='bif1d/',
                       _full_rhs=None,max_iter=20,max_time=1000,
                       prominence=0.05):
    
    # branch init
    fname_template1 = 'td_branch_init={}_nm={}{}_del={}_ei={}_ef={}_de={}.txt'
    fname_template2 = 'ts_branch_init={}_nm={}{}_del={}_ei={}_ef={}_de={}.txt'
    fname1 = fname_template1.format(init,a._n[1],a._m[1],del1,eps_init,eps_final,deps)
    fname2 = fname_template2.format(init,a._n[1],a._m[1],del1,eps_init,eps_final,deps)

    print('fname1',fname1)
    print('fname2',fname2)
    # force save to curent working directory (where the script was called)
    # using absolute path.
    bifdir = os.getcwd()+'/'+bifdir
    if not(os.path.exists(bifdir)):
        os.makedirs(bifdir)
        
    fname1 = bifdir + fname1
    fname2 = bifdir + fname2

    kw1 = {'a':a,'del1':del1,'eps_init':eps_init,'eps_final':eps_final,'deps':deps,
           '_full_rhs':_full_rhs,'max_iter':max_iter,'prominence':prominence}

    if not(os.path.isfile(fname1)) or not(os.path.isfile(fname2)) or recompute:
        init = get_initial_phase_diff_c(init,a,eps_init,del1,max_time=max_time,
                                        _full_rhs=_full_rhs,prominence=prominence)
        print('init',init)
        eps_range,ts,t_diffs = get_phase_lock_full(Y=deepcopy(init),**kw1)
        
        data1 = np.zeros([len(eps_range),1+a._n[1]+a._m[1]])
        data1[:,0] = eps_range
        data1[:,1:] = ts

        data2 = np.zeros([len(eps_range),1+a._n[1]+a._m[1]])
        data2[:,0] = eps_range
        data2[:,1:] = t_diffs
        
        np.savetxt(fname1,data1)
        np.savetxt(fname2,data2)
        
    else:
        data1 = np.loadtxt(fname1)
        data2 = np.loadtxt(fname2)

    return data1,data2



def get_dy_nm_coupling(Y,args_temp,eps_pert=1e-4,eps_pert_time=1e-4,_full_rhs=None,
                       return_sol=False):
    """
    get dy for Newton's method for estimating a phase locked state
    """
    
    J = np.zeros([len(Y),len(Y)])

    for jj in range(len(Y)-1):
        pert = np.zeros(len(Y)-1);pert[jj] = eps_pert

        t_span = [0,Y[-1]]
        solp = solve_ivp(y0=Y[:-1]+pert,t_span=t_span,**args_temp)
        solm = solve_ivp(y0=Y[:-1]-pert,t_span=t_span,**args_temp)

        # Jacobian
        J[:-1,jj] = (solp.y.T[-1,:]-solm.y.T[-1,:])/(2*eps_pert)

    J[:-1,:-1] = J[:-1,:-1] - np.eye(len(Y)-1)

    t_spanp = [0,Y[-1]+eps_pert_time]
    t_spanm = [0,Y[-1]-eps_pert_time]
    solpt = solve_ivp(y0=Y[:-1],t_span=t_spanp,**args_temp)
    solmt = solve_ivp(y0=Y[:-1],t_span=t_spanm,**args_temp)

    J[:-1,-1] = (solpt.y.T[-1,:]-solmt.y.T[-1,:])/(2*eps_pert_time)
    J[-1,-1] = 0
    J[-1,:-1] = _full_rhs(0,Y[:-1],*args_temp['args'])

    solb = solve_ivp(y0=Y[:-1],t_span=t_span,**args_temp)

    
    b = np.array(list(Y[:-1]-solb.y.T[-1,:])+[0])
    dy = np.linalg.solve(J,b)

    if return_sol:
        return dy,solb
    else:
        return dy



def get_dy_nm_force(rhs,Y,args_temp,eps_pert=1e-4,eps_pert_time=1e-4,
           return_sol=False):
    """
    get dy for phase locking state in full model with forcing
    """
    J = np.zeros([len(Y),len(Y)])

    for jj in range(len(Y)-1):
        pert = np.zeros(len(Y)-1);pert[jj] = eps_pert

        t_span = [0,Y[-1]]
        solp = solve_ivp(y0=Y[:-1]+pert,t_span=t_span,**args_temp)
        solm = solve_ivp(y0=Y[:-1]-pert,t_span=t_span,**args_temp)

        # Jacobian
        J[:-1,jj] = (solp.y.T[-1,:]-solm.y.T[-1,:])/(2*eps_pert)

    J[:-1,:-1] = J[:-1,:-1] - np.eye(len(Y)-1)

    t_spanp = [0,Y[-1]+eps_pert_time]
    t_spanm = [0,Y[-1]-eps_pert_time]
    solpt = solve_ivp(y0=Y[:-1],t_span=t_spanp,**args_temp)
    solmt = solve_ivp(y0=Y[:-1],t_span=t_spanm,**args_temp)

    J[:-1,-1] = (solpt.y.T[-1,:]-solmt.y.T[-1,:])/(2*eps_pert_time)
    J[-1,-1] = 0
    J[-1,:-1] = rhs(0,Y[:-1],*args_temp['args'])

    solb = solve_ivp(y0=Y[:-1],t_span=t_span,**args_temp)

    b = np.array(list(Y[:-1]-solb.y.T[-1,:])+[0])
    dy = np.linalg.solve(J,b)

    if return_sol:
        return dy,solb
    else:
        return dy

    

def run_bif1d_f(rhs,Y,a,del1,eps_init,eps_final,deps,
                max_iter:int=100,tol:float=1e-10,
                u_sign=-1,mult=1):
    """
    1d bifurcation diagram for the full model with forcing
    """

    print('u_sign -- must be set manually',u_sign)

    Y[-1] *= mult

    print('integrating bif. over {} periods'.format(mult))
    
    Y_init = deepcopy(Y)
    dy_init = np.ones(len(Y))/5
    eps_range = np.arange(eps_init,eps_final,deps)
    phase_diffs = np.zeros([len(eps_range),2])

    osc2_idx = len(a.system1.var_names)
    
    for i,eps in enumerate(eps_range):
        
        dy = deepcopy(dy_init)
        
        counter = 0
        print('eps iter',i,eps,end='                    \n')
        
        while np.linalg.norm(dy) > tol and\
        counter <= max_iter and\
        np.linalg.norm(dy)<1:
            
            args_temp = {'fun':rhs,'args':(a,eps,del1),**kw_bif}
            dy,solb = get_dy_nm_force(rhs,Y,args_temp,return_sol=True,
                             eps_pert=1e-2,eps_pert_time=1e-2)
            
            if False:
                solt = solve_ivp(t_span=[0,50],y0=Y[:-1],**args_temp)
                ut = a.system1.forcing_fn(solt.t*(a._m[1]+args_temp['args'][2]))
                a = args_temp['args'][0]
                u = a.system1.forcing_fn(solb.t*(a._m[1]+args_temp['args'][2]))
                fig,axs = plt.subplots()
                axs.plot(solb.t,solb.y[0])
                axs.plot(solt.t,solt.y[0])
                axs.plot(solb.t,u)
                axs.plot(solt.t,ut)
                axs.set_title('eps={}, del1={}'.format(args_temp['args'][1],args_temp['args'][2]))
                plt.savefig('figs_temp/newton_f_eps={}_del1={}_{}.png'.format(args_temp['args'][1],args_temp['args'][2],counter))
                plt.close()

            Y += dy
            print('dy',dy,'Y',Y)
            
            str1 = 'iter={}, LC rel. err ={:.2e}'
            end = '                                       \r'
            printed = str1.format(counter,np.linalg.norm(dy))
            print(printed,end=end)
            counter += 1
            
            time.sleep(.2)
    
        if (counter >= max_iter) or (np.linalg.norm(dy) >= 1):
            phase_diffs[i,:] = np.nan
            Y = Y_init
            dy = np.ones(len(Y))/5
        
        else:
            solf = solve_ivp(y0=Y[:-1],t_span=[0,Y[-1]],**args_temp)

            idx1 = np.argmax(solf.y.T[:,0])
            u = a.system1.forcing_fn(solf.t*(a._m[1]+del1))
            idx2 = sp.signal.find_peaks(u_sign*u)[0]

            if False:

                fig,axs = plt.subplots()
                axs.plot(solf.t,solf.y[0])
                axs.plot(solf.t,u)

                axs.scatter(solf.t[idx1],solf.y[0,idx1])
                axs.scatter(solf.t[idx2],u[idx2])
                
                axs.set_title('eps={}, del1={}'.format(eps,del1))
                plt.savefig('figs_temp/{}.png'.format(counter))
                plt.close()


            if len(idx2) == 0:
                Y = Y_init
                dy = dy_init
                t_diff = phase_diffs[i,:] = np.nan
                
            else:
                t_diff = solf.t[idx2]-solf.t[idx1]
                period = Y[-1]/mult
                #print('ts',solf.t[idx1],solf.t[idx2])
                print('t_diff',t_diff)
                phase_diffs[i,:] = 2*np.pi*np.mod(t_diff,period)/period
        
        
    return eps_range,phase_diffs


def dy_r3d(rhs,Y,args_temp,h=1e-6,return_sol=False):
    """
    get dy for phase locking state in 3d reduction
    """
    Y = np.array(deepcopy(Y))
    J = np.zeros([len(Y),len(Y)])
    F = rhs(0,Y,*args_temp['args'])
    
    for i in range(len(Y)): # column by colun
        pert = np.zeros(len(Y));pert[i] = h
        d = (rhs(0,Y+pert,*args_temp['args']) - F)/h
        J[:,i] = d

    dy = np.linalg.solve(J,-F)

    return dy
    
def load_phase_lock_r3d(Y,a,eps,rhs,rel_tol=1e-7,max_iter=30,
                        return_init=True,save=True,
                        recompute=False,bifdir=''):

    if save:
        del1 = a.system1.pardict['del0']
        fname = 'pl1_init={}_n={}_m={}_del={}_eps={}.txt'
        fname = fname.format(Y,a._n[1],a._m[1],del1,eps)

        # force save to curent working directory (where the script was called)
        # using absolute path.
        bifdir = os.getcwd()+'/'+bifdir
        if not(os.path.exists(bifdir)):
            os.makedirs(bifdir)

        fname = bifdir + fname
        isfile = os.path.isfile(fname)

        if not(isfile) or recompute:

            phase_diff,Y_init = phase_lock_r3d(Y,a,eps,rhs,rel_tol=1e-7,
                                               max_iter=30,return_init=True)

            dat = np.zeros(1+len(Y_init))
            dat[0] = phase_diff
            dat[1:] = Y_init
            np.savetxt(fname,dat)
            
        else:
            dat = np.loadtxt(fname)
            phase_diff = dat[0]
            Y_init = dat[1:]
    
    else:
        phase_diff,Y_init = phase_lock_r3d(Y,a,eps,rhs,rel_tol=1e-7,
                                           max_iter=30,return_init=True)

    return phase_diff,Y_init
    
def phase_lock_r3d(Y,a,eps,rhs,rel_tol=1e-7,max_iter=30,
                   return_init=True,bifdir=''):

    """
    Get phase-locked state given init Y, eps and del1.
    del1 is contained in object a's parameter dict.
    for reduced 3d model only.
    """

    Y_init = deepcopy(Y)
    dy = deepcopy(np.ones(len(Y))/5)

    counter = 0

    while np.linalg.norm(dy) > rel_tol and counter <= max_iter:

        args_temp = {'fun':rhs,'args':(a,eps),**kw_bif}
        dy = dy_r3d(rhs,Y_init,args_temp)
        Y_init += dy

        str1 = 'iter={}, rel. err ={:.2e}, Y={}'
        end = '                                              \r'
        printed = str1.format(counter,np.linalg.norm(dy),Y_init)
        print(printed,end=end)
        counter += 1

    if (counter >= max_iter) or (np.linalg.norm(dy) >= 10):
        phase_diff = np.nan
        dy = np.ones(len(Y_init))/5

    else:
        phase_diff = Y_init[0]
        print('t_diff',np.mod(phase_diff,2*np.pi))
        

    if return_init:
        # trim init for more reliable file name
        Y_init[0] = np.mod(Y_init[0],2*np.pi)
        for i in range(len(Y_init)):
            Y_init[i] = np.round(Y_init[i],4)
        return phase_diff,Y_init
    else:
        return phase_diff


def get_pl_range_r3d(Y,a,eps_tup,rhs,rel_tol=1e-7,max_iter=100,
                     return_init=True):

    eps_range = np.arange(*eps_tup)
    phase_diffs = np.zeros(len(eps_range))
    init = deepcopy(Y)
    
    for i,eps in enumerate(eps_range):
        print('eps iter',i,eps,end='                \n')
        phase_diff,init = phase_lock_r3d(init,a,eps,rhs,
                                         rel_tol=rel_tol,
                                         max_iter=max_iter,
                                         return_init=return_init)
        #phase_diffs[i,0] = eps_range[i]
        phase_diffs[i] = phase_diff
        
    return eps_range,phase_diffs


def follow_locking_3d(init,a,eps_tup,bifdir='',
                      recompute=False,rhs=None,max_iter=100):
    """
    For phase locking in 3d reduced averaged model.
    """

    print('eps_tup',eps_tup)

    del1 = a.system1.pardict['del0']
    # branch init
    fname_template = ('branch_init={}_n={}_m={}_'+
                      'del={}_ei={}_ef={}_de={}.txt')
    fname = fname_template.format(init,a._n[1],a._m[1],del1,*eps_tup)
    
    # force save to curent working directory using absolute path.
    bifdir = os.getcwd()+'/'+bifdir
    if not(os.path.exists(bifdir)):
        os.makedirs(bifdir)
    fname = bifdir + fname

    kw1 = {'a':a, 'eps_tup':eps_tup, 'rhs':rhs, 'max_iter':max_iter}

    if not(os.path.isfile(fname)) or recompute:
        print(init,a,eps_tup,rhs)
        _,init3d = phase_lock_r3d([init,0,0],a,eps_tup[0],rhs,max_iter=max_iter)
        eps_range,phase_diffs = get_pl_range_r3d(Y=deepcopy(init3d),
                                                 **kw1)
        
        data = np.zeros([len(eps_range),2])
        data[:,0] = eps_range;data[:,1] = phase_diffs
        np.savetxt(fname,data)
        
    else:
        data = np.loadtxt(fname)
        eps_range = data[:,0];phase_diffs = data[:,1]
        
    return data


def _get_sol(rhs,y0,t,args,recompute=False,data_dir='sols',idx='',
             kw_sim={'rtol':1e-7,'atol':1e-7,'method':'LSODA'}):
    """
    load or generate solution.
    args must be in the order obj,eps,del1
    """
    a = args[0];eps = args[1];del1 = a.del1

    data_dir = os.getcwd()+'/'+data_dir
    if not(os.path.isdir(data_dir)):
        os.mkdir(data_dir)

    rhs_name = rhs.__name__

    y0 = np.round(y0,4)
    
    ratio = str(a._n[1])+str(a._m[1])
    raw = ('{}/sol_{}_y0={}_ratio={}_T={}_dt={}_e={}_d={}_idx={}.txt')
    fpars = [data_dir,rhs_name,y0,ratio,t[-1],t[1]-t[0],eps,del1,idx]
    args1 = {'t_eval':t,'t_span':[0,t[-1]],'args':(*args,),**kw_sim}

    fname = raw.format(*fpars)
    file_dne  = not(os.path.isfile(fname))

    if file_dne or recompute:
        solf = solve_ivp(rhs,y0=y0,**args1)
        y = solf.y.T
        np.savetxt(fname,y)
    else:
        y = np.loadtxt(fname)

    return y


def _get_sol_3d(rhs,y0,t,args,recompute=True,data_dir='sols',idx='',
                kw_sim={'rtol':1e-7,'atol':1e-7,'method':'LSODA'}):
    """
    load or generate solution.
    args must be in the order obj,eps,del1
    """
    a = args[0];eps = args[1];del1 = a.del1

    data_dir = os.getcwd()+'/'+data_dir
    if not(os.path.isdir(data_dir)):
        os.mkdir(data_dir)

    rhs_name = rhs.__name__

    y0 = np.round(y0,4)
    
    ratio = str(a._n[1])+str(a._m[1])
    raw = ('{}/sol_{}_y0={}_ratio={}_T={}_dt={}_e={}_d={}_idx={}.txt')
    fpars = [data_dir,rhs_name,y0,ratio,t[-1],t[1]-t[0],eps,del1,idx]
    args1 = {'t_eval':t,'t_span':[0,t[-1]],'args':(*args,),**kw_sim}

    fname = raw.format(*fpars)
    file_dne  = not(os.path.isfile(fname))

    if file_dne or recompute:
        solf = solve_ivp(rhs,y0=y0,**args1)
        y = solf.y.T
        np.savetxt(fname,y)
    else:
        y = np.loadtxt(fname)

    return y
