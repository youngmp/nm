"""
Generate figures for strong coupling paper

TODO: INCLUDE BOTH THALAMIC EXAMPLES. NEED TO SHOW
METHOD STILL WORKS WELL NEAR BIFURCATIONS?
BORING EXAMPLE SO MAYBE PUT IN APPENDIX.
"""


# user-defined
#import nmCoupling_old as nm
import nmCoupling as nm
import response as rp

import cgl1 as c1
#import cgl2 as c2

import thal1 as t1
import thal2 as t2

import gw2

import gt

from lib.util import (follow_phase_diffs, dy_r3d, phase_lock_r3d,
                      dy_r3d, get_pl_range_r3d, follow_locking_3d)

from lib.rhs import _full, _redu_3dc_gw, _redu_3dc_thal
from lib.functions import *
from lib.plot_util import *

from lib.rhs import _redu_c

rhs_avg_1dc = _redu_c

nmc = nm.nmCoupling;rsp = rp.Response

import os
from matplotlib.gridspec import GridSpec
import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt
import string
import sympy as sym
import scipy as sp



from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from matplotlib import cm
from scipy.signal import find_peaks
from scipy.integrate import solve_ivp
#from matplotlib.legend_handler import HandlerBase
from scipy.optimize import brentq, root, bisect



import matplotlib.patheffects as pe
import copy
from copy import deepcopy

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['pgf.texsystem'] = 'pdflatex'
#matplotlib.rcParams['font.size'] = 12


preamble = (r'\usepackage{amsmath}'
            r'\usepackage{siunitx}'
            r'\usepackage{bm}'
            r'\newcommand{\ve}{\varepsilon}')

matplotlib.rcParams['text.latex.preamble'] = preamble
fontsize = 12

labels = ['A','B','C','D','E','F','G','H','I','J']

pd_cgl_template = {'sig':.08,'rho':.12,'mu':1,'om':1,'om_fix':1}

kw_cgl_template = {'var_names':['x','y'],
                   'init':np.array([1,0,2*np.pi]),
                   'TN':2000,
                   'idx':0,
                   'model_name':'cglf0',
                   'trunc_order':3,
                   'recompute_list':[],
                   'g_forward':False,
                   'z_forward':False,
                   'i_forward':[False,True,True,True,True,True,True],
                   'i_bad_dx':[False,True,False,False,False,False,False],
                   'max_iter':20,
                   'rtol':1e-12,
                   'atol':1e-12,
                   'rel_tol':1e-9,
                   'forcing_fn':g1}

pd_thal_template = {'gL':0.05,'gna':3,'gk':5,
                    'gt':5,'eL':-70,'ena':50,
                    'ek':-90,'et':0,'esyn':0,
                    'c':1,'alpha':3,'beta':2,
                    'sigmat':0.8,'vt':-20,
                    'ib':8.5,'om':1,'om_fix':1,
                    'del':0}

# default period must be 2*np.pi
kw_thal_template = {'var_names':['v','h','r'],
                    'rhs':t1.rhs,
                    'init':np.array([-.64,0.71,0.25,10.6]),
                    'TN':10000,
                    'idx':0,
                    'model_name':'thalf0',
                    'trunc_order':3,
                    'recompute_list':[],
                    'z_forward':False,
                    'i_forward':False,
                    'i_bad_dx':[False,True,False,False],
                    'max_iter':200,
                    'rtol':1e-13,
                    'atol':1e-13,
                    'rel_tol':1e-9,
                    'forcing_fn':g1,
                    'factor':0.5,
                    'save_fig':False,
                    }

pd_gw_template = {'v1':.84,'v2':.42,'v4':.35,'v6':.35,'v8':1,
                  'k1':1,'k2':1,'k3':.7,'k4':1,'k5':.7,
                  'k6':1,'k7':.35,'k8':1,'K':0.5,'kc':1,
                  'n':6,'L':0,'eps':0,'del':0,'om':1,'om_fix':1}

kw_gw_template = {'var_names':['x','y','z','v'],
                  'rhs':gw2.rhs,
                  'coupling':gw2.coupling,
                  'init':np.array([.3882,.523,1.357,.4347,24.2]),
                  'TN':2000,
                  'trunc_order':2,
                  'z_forward':False,
                  'i_forward':[False,True,False,False],
                  'i_bad_dx':[False,True,False,False],
                  'max_iter':20,
                  'rtol':1e-12,
                  'atol':1e-12,
                  'rel_tol':1e-9,
                  'save_fig':False,
                  'lc_prominence':0.05,
                  'factor':0.5}


kw_sim = {'rtol':1e-7,'atol':1e-7,'method':'LSODA'}

pi_label_short = [r"$0$", r"$\pi$", r"$2\pi$"]
pi_label_half = [r"$0$", r"$\pi/2$", r"$\pi$"]
pi_label_third = [r"$0$", r"$\pi/3$", r"$2\pi/3$"]



def forcing_fn():

    fig,axs = plt.subplots(1,2,figsize=(8,2))

    x1 = np.linspace(-5,5,100)
    x2 = np.linspace(0,4*np.pi,300)
    
    g1m = lambda x:-g1(x)
    fnlist = [gau,g1m]
    xlist = [x1,x2]
    title_add = ['Gaussian','Zero-Mean Periodized Gaussian']

    for i in range(len(fnlist)):
        axs[i].plot(xlist[i],fnlist[i](xlist[i]))
        axs[i].set_title(labels[i],loc='left')

        tt = axs[i].get_title()
        tt += title_add[i]
        axs[i].set_title(tt)
        axs[i].set_xlabel(r'$t$')

        axs[i].margins(x=0)

    xticks = np.arange(0,x2[-1]+np.pi,np.pi)
    axs[1].set_xticks(xticks)

    tlabels = [0]
    tlabels += [r'${}\pi$'.format(i+1) for i in range(len(xticks[1:]))]
    print(tlabels)
    #print([0].append())
    axs[1].set_xticklabels(tlabels)

    plt.tight_layout()
    

    return fig

    

def load_tongue(n,m,pd1,pd2,kws1,kws2,model_name='cglf0',dir1='',
                recompute=False,dlo=-17,dhi=3,Nd=30):
    """
    load tongue data
    """
    
    ratio = str(n)+str(m)
    rawname = '{}/{}_ratio={}_d0={}_de={}_dN={}.txt'
    fname = rawname.format(dir1,model_name,ratio,dlo,dhi,Nd)
    
    file_dne = not(os.path.isfile(fname))

    if file_dne or recompute:

        system1 = rsp(pardict=copy.deepcopy(pd1),**copy.deepcopy(kws1))
        system2 = rsp(pardict=copy.deepcopy(pd2),**copy.deepcopy(kws2))
        
        a = nmc(system1,system2,
                _n=('om0',n),_m=('om1',m),
                #recompute_list = ['p_data_cglf0','h_data_cglf0'],
                NP=400,NH=400)

        dtemp = np.linspace(dlo,dhi,Nd)
        in1 = 2**dtemp

        in1 = np.append([0],in1)# add zero to avoid discontinuities
        del_list1,ve_list1 = get_tongue(-in1[::-1],system1,system2,a,
                                        deps=.001,max_eps=.3)
        del_list2,ve_list2 = get_tongue(in1[1:],system1,system2,a,
                                        deps=.001,max_eps=.3)

        del_list = np.concatenate([del_list1,del_list2])
        ve_list = np.concatenate([ve_list1,ve_list2])

        data = np.zeros([len(del_list),2])
        data[:,0] = del_list
        data[:,1] = ve_list

        np.savetxt(fname,data)
        
    else:
        data = np.loadtxt(fname)

    return data





def _full_cgl1(t,y,a,eps=0,del1=0):
    pd1 = a.system1.pardict
    omx = a._n[1];omf = a._m[1]
    ofix = pd1['om_fix0']
    u = a.system1.forcing_fn(t*(del1+omf))
    
    out1 = c1.rhs_old2(t,y,pd1,'val',0) + eps*ofix*omx*np.array([u,0])
    return np.array(list(out1))

def _full_thal1(t,y,a,eps=0,del1=0):
    pd1 = a.system1.pardict
    omx = a._n[1];omf = a._m[1]
    ofix = pd1['om_fix0']
    u = a.system1.forcing_fn(t*(del1+omf))
    
    out1 = t1.rhs(t,y,pd1,'val',0) + eps*ofix*omx*np.array([u,0,0])
    return np.array(list(out1))

def _full_thal2(t,y,a,eps=0,del1=0):
    pd1 = a.system1.pardict;pd2 = a.system2.pardict
    y1 = y[:4];y2 = y[4:]
    
    c1 = t2.coupling(y,pd1,'val',0)
    c2 = t2.coupling(list(y2)+list(y1),pd2,'val',1)
    
    out1 = t2.rhs(t,y1,pd1,'val',0) + eps*(del1 +c1)
    out2 = t2.rhs(t,y2,pd2,'val',1) + eps*c2
    return np.array(list(out1)+list(out2))

def _full_gt2(t,y,a,eps=0,del1=0):
    pd1 = a.system1.pardict;pd2 = a.system2.pardict
    y1 = y[:4];y2 = y[4:]
    
    c1 = gt.coupling_tm(y,pd1,'val',0)
    c2 = gt.coupling_gw(list(y2)+list(y1),pd2,'val',1)
    
    out1 = t2.rhs(t,y1,pd1,'val',0) + eps*(del1 + c1)
    out2 = gt.rhs(t,y2,pd2,'val',1) + eps*c2
    return np.array(list(out1)+list(out2))


def _full_gw2(t,y,a,eps,del1):
    pd1 = a.system1.pardict;pd2 = a.system2.pardict
    om_fix = pd1['om_fix0']
    y1 = y[:4];y2 = y[4:]
    out1 = gw2.rhs(t,y1,pd1,'val',0) + eps*gw2.coupling(y,pd1,'val',0)
    out2 = gw2.rhs(t,y2,pd2,'val',1) + eps*gw2.coupling(list(y2)+list(y1),pd2,'val',1)
    return np.array(list(out1)+list(out2))

labeltempf = [r'$\psi$',r'$\mathcal{H}$',r'$t$']
labeltempc = [r'$\mathcal{H}$',r'$t$']

def _setup_trajectories_plot(mode='f',labels=True,
                             wspace=0.1,hspace=0.05,
                             padw=0.04,padh=0.05,
                             lo=0.075,hi=0.96):
    """
    plot arranged as 0 top left, 1 top right, 2 bottom left, 3 bottom right
    each object in axs is a 2x3 set of axis objects.

    mode: 'f' is for forcing, 'c' is for coupling
    """

    if mode == 'f':
        w = 8;h = 6
        nr1 = 3;nc1 = 2
    elif mode == 'c':
        w = 8;h = 6
        nr1 = 2;nc1 = 2
    
    fig = plt.figure(figsize=(w,h))

    kws = {'wspace':wspace,'hspace':hspace,'nrows':nr1,'ncols':nc1}

    gs1 = fig.add_gridspec(left=lo, right=0.5-padw,bottom=0.5+padh,top=hi,**kws)
    axs1 = [[fig.add_subplot(gs1[i,j]) for j in range(nc1)] for i in range(nr1)]

    gs2 = fig.add_gridspec(left=0.5+padw, right=hi,bottom=0.5+padh,top=hi,**kws)
    axs2 = [[fig.add_subplot(gs2[i,j]) for j in range(nc1)] for i in range(nr1)]

    gs3 = fig.add_gridspec(left=lo, right=.5-padw,bottom=lo,top=.5-padh,**kws)
    axs3 = [[fig.add_subplot(gs3[i,j]) for j in range(nc1)] for i in range(nr1)]

    gs4 = fig.add_gridspec(left=.5+padw, right=hi,bottom=lo,top=.5-padh,**kws)
    axs4 = [[fig.add_subplot(gs4[i,j]) for j in range(nc1)] for i in range(nr1)]

    axs = [np.asarray(axs1),np.asarray(axs2),np.asarray(axs3),np.asarray(axs4)]

    # remove left tick label for right panel
    kwall = {'axis':'both','which':'both'}
    kwb = {'labelbottom':False,'bottom':False}
    kwl = {'labelleft':False,'left':False}
    
    for k in range(len(axs)):
        axs[k][0,0].tick_params(**{**kwall,**kwb})
        axs[k][0,1].tick_params(**{**kwall,**kwl,**kwb})        
        
        if mode == 'f':
            axs[k][2,1].tick_params(**{**kwall,**kwl})
            axs[k][1,0].tick_params(**{**kwall,**kwb})
            axs[k][1,1].tick_params(**{**kwall,**kwl,**kwb})

        else:
            axs[k][1,1].tick_params(**{**kwall,**kwl})
            
        for i in range(nr1):
            if labels:

                axs[k][i,0].sharey(axs[k][i,1])

                if mode == 'c':
                    axs[k][i,0].set_ylabel(labeltempc[i],labelpad=0)
                else:
                    axs[k][i,0].set_ylabel(labeltempf[i],labelpad=0)
                for j in range(nc1):
                    axs[k][i,j].margins(x=0)
                    axs[k][i,j].set_xlim(0,2*np.pi)

        for j in range(nc1):

            if labels:
                axs[k][-1,j].margins(y=0)
                axs[k][-1,j].set_xlabel(r'$\phi$',labelpad=0)
                
                axs[k][-1,j].set_xticks(np.arange(0,3,1)*np.pi)
                axs[k][-1,j].set_xticklabels(pi_label_short)
            
    return fig,axs


def draw_forcing_sols(axs,a,T,eps,del1,pmax,pmin,init,full_rhs,
                      recompute:bool=False):
    
    nr1,nc1 = axs.shape
    system1 = a.system1
    kw1 = {'a':a,'return_data':False,'pmin':pmin,'pmax':pmax}

    # draw nullclines
    for j in range(nc1):
        co1,co2 = pl_exist_2d(del1=del1[j],eps=eps[j],**kw1)
        path1 = co1.get_paths()[0]
        patch1 = matplotlib.patches.PathPatch(path1,facecolor='none',lw=1)
        axs[0,j].add_patch(patch1)

        path2 = co2.get_paths()[0]
        patch2 = matplotlib.patches.PathPatch(path2,facecolor='none',lw=1,
                                              edgecolor='gray',ls=':')
        axs[0,j].add_patch(patch2)

    # draw 1d phase lines
    for j in range(nc1):
        x = np.linspace(0,2*np.pi,200)
        y = rhs_avg_1d(0,x,a,eps[j],del1[j])
        
        axs[1,j].plot(x,y,color='k',lw=1)
        axs[1,j].axhline(0,x[0],x[-1],color='gray',lw=1,ls=':')

    # trajectory
    dt = .02;
    t = np.arange(0,T,dt)
    th_init = init

    for j in range(nc1):
        y0 = system1.lc['dat'][int((th_init/(2*np.pi))*system1.TN),:]
        args0 = [a,eps[j],del1[j]]
        
        solf = _get_sol(full_rhs,y0,t,args=args0,recompute=recompute)
        
        tp,fp = get_phase(t,solf,skipn=100,system1=system1)
        force_phase = np.mod((a._m[1]+del1[j])*tp,2*np.pi)
        
        fp2 = np.mod(fp-a.om*force_phase,2*np.pi)
        axs[2,j].scatter(fp2,tp,s=5,color='gray',alpha=.5,
                         label='Full')
        
        args1 = {'t_eval':t,'t_span':[0,t[-1]],'args':(*args0,),**kw_sim}

        solr2d = solve_ivp(rhs_avg_2d,y0=[th_init,0],**args1)
        axs[2,j].plot(np.mod(solr2d.y.T[:,0],2*np.pi),t,
                                 color='tab:blue',alpha=.75,label='2D')
        # solution on 2d phase plane
        xs = np.mod(solr2d.y.T[:,0],2*np.pi)
        ys = solr2d.y.T[:,1]
        discont_idxs = np.abs(np.gradient(xs,1)) > np.pi/2
        
        xs[discont_idxs] = np.nan
        ys[discont_idxs] = np.nan
            
        line, = axs[0,j].plot(xs,ys,color='tab:blue',alpha=.75)
        add_arrow_to_line2D(axs[0,j],line,arrow_locs=[.25,.75])
            
        axs[0,j].set_ylim(pmin,pmax)
        axs[2,j].set_ylim(T,0)

        # solution on 1d phase plane
        solr1d = solve_ivp(rhs_avg_1d,y0=[th_init],**args1)
        axs[1+1,j].plot(np.mod(solr1d.y.T[:,0],2*np.pi),t,
                        color='tab:red',alpha=.75,label='1D',
                        ls='--')
        
        xs = np.mod(solr1d.y.T[:,0],2*np.pi)
        ys = np.zeros(len(xs))
        discont_idxs = np.abs(np.gradient(xs,1)) > np.pi/2

        xs[discont_idxs] = np.nan
        ys[discont_idxs] = np.nan

        line, = axs[1,j].plot(xs,ys,color='tab:red',alpha=.75,
                              ls='--')
        add_arrow_to_line2D(axs[1,j],line,arrow_locs=[.25,.75])

    return axs

def draw_solutions(axs,a,T,eps,del1,pmax,pmin,init,full_rhs):

    mode = 'c'
    d1_idx = 0

    nr1,nc1 = axs.shape
    system1 = a.system1;system2 = a.system2

    kw1 = {'eps':eps,'a':a,'return_data':True,'pmin':pmin,'pmax':pmax}

    # 1d
    for j in range(nc1):
        x = np.linspace(0,2*np.pi,200)

        if mode == 'f':
            y = rhs_avg_1d(0,x,a,eps[j],del1[j])
        else:
            y = rhs_avg_1dc(0,x,a,eps[j])
        axs[0,j].plot(x,y,color='k',lw=1)
        axs[0,j].axhline(0,x[0],x[-1],color='gray',lw=1,ls='--')

    # trajectory
    dt = .02;
    t = np.arange(0,T,dt)
    th_init = init

    for j in range(nc1):
        y0a = list(system1.lc['dat'][int((th_init/(2*np.pi))*system1.TN),:])
        y0b = list(a.system2.lc['dat'][int((0/(2*np.pi))*a.system2.TN),:])
        y0 = np.array(y0a+y0b)
                    
        args0 = [a,eps[j],del1[j]]

        recompute = True
        solf = _get_sol(full_rhs,y0,t,args=args0,recompute=recompute)

        if mode == 'c':

            v1_peak_idxs = find_peaks(solf[:,0])[0]
            v2_peak_idxs = find_peaks(solf[:,4])[0]
            
            # match total number of peaks
            min_idx = np.amin([len(v1_peak_idxs),len(v2_peak_idxs)])
            v1_peak_idxs = v1_peak_idxs[:min_idx]
            v2_peak_idxs = v2_peak_idxs[:min_idx]
            
            fp1 = t[v1_peak_idxs]
            fp2 = t[v2_peak_idxs]

            tp1 = t[v1_peak_idxs]
            tp2 = tp1
            T = fp1[-1]-fp1[-2]
            y2 = np.mod(fp2-a.om*fp1,T)/T*2*np.pi
            print('approx per',T)

            
            #dim1 = a.system1.dim;dim2 = a.system2.dim
            #tp1,fp1 = get_phase(t,solf[:,:dim1],skipn=100,system1=system1)
            #tp2,fp2 = get_phase(t,solf[:,dim1:],skipn=100,system1=system2)
            #y2 = np.mod(fp1-a.om*fp2,2*np.pi)#/T*2*np.pi
            
            axs[d1_idx+1,j].scatter(y2,tp2,s=5,color='gray',alpha=.5,
                                    label='Full')
            
        else:
            tp,fp = get_phase(t,solf,skipn=100,system1=system1)
            force_phase = np.mod((a._m[1]+del1[j])*tp,2*np.pi)
            fp2 = np.mod(fp-a.om*force_phase,2*np.pi)
            axs[d1_idx+1,j].scatter(fp2,tp,s=5,color='gray',alpha=.5,
                                    label='Full')
        
        args1 = {'t_eval':t,'t_span':[0,t[-1]],'args':(*args0,),**kw_sim}

        # solution on 1d phase plane
        solr1d = solve_ivp(rhs_avg_1dc,y0=[th_init],**args1)
        axs[d1_idx+1,j].plot(np.mod(solr1d.y.T[:,0],2*np.pi),t,
                             color='tab:red',alpha=.75,label='1D',
                             ls='--')
        
        xs = np.mod(solr1d.y.T[:,0],2*np.pi)
        ys = np.zeros(len(xs))
        discont_idxs = np.abs(np.gradient(xs,1)) > np.pi/2

        xs[discont_idxs] = np.nan
        ys[discont_idxs] = np.nan

        line, = axs[d1_idx,j].plot(xs,ys,color='tab:red',alpha=.75,
                                   ls='--')
        add_arrow_to_line2D(axs[d1_idx,j],line,arrow_locs=[.25,.75])


    #axs[d1_idx+1,0].legend()
    #axs[d1_idx+1,0].set_ylim(axs[d1_idx+1,0].get_ylim()[::-1])

    return axs

def trajectories_cgl1():
    """
    Plot phase plane, phase line, and phases over time
    """
    
    kw_cgl = copy.deepcopy(kw_cgl_template)
    kw_cgl['rhs'] = c1.rhs_old2
    
    system1 = rsp(**{'pardict':pd_cgl_template,**kw_cgl})

    pl_list = [(1,1),(2,1),(3,1),(4,1)]
    T_list = [250,500,700,1500]
    e_list = [(.2,.2),(.1,.1),(.1,.1),(.1,.1)]
    d_list = [(.01,.08),(.01,.025),(.001,.008),(.0007,.0015)]
    pmax_list = [1.5,1.5,1.5,1.5]
    pmin_list = [-1,-1,-1,-1]
    init_list = [1,1,.5,.5]

    full_rhs = _full_cgl1
    fig,axs = _setup_trajectories_plot()

    for k in range(len(axs)):
        # run simulations and plot
        a = nmc(system1,None,recompute_list=[],
                _n=('om0',pl_list[k][0]),_m=('om1',pl_list[k][1]),
                NP=300,NH=300)
        
        draw_forcing_sols(axs[k],a,T_list[k],e_list[k],d_list[k],
                          pmax_list[k],pmin_list[k],init_list[k],
                          full_rhs)

        del a

    # set title
    ct = 0
    for k in range(len(axs)):
        axs[k][0,0].set_title(labels[ct],loc='left')
        ct += 1

        axs[k][0,1].set_title(labels[ct],loc='left')
        ct += 1

    # fix title with parameter values
    nr1,nc1 = axs[0].shape
    for k in range(len(axs)):
        for j in range(nc1):
            ti1 = axs[k][0,j].get_title()
            ti1 += str(pl_list[k][0])+':'+str(pl_list[k][1])
            #t1 += r', $\varepsilon='+str(e_list[k])+'$'
            ti1 += r', $\delta = '+str(d_list[k][j])+'$'
            
            axs[k][0,j].set_title(ti1)

    axs[2][1,0].yaxis.labelpad=-10
    
    
    return fig


def trajectories_thal1():
    """
    Plot phase plane, phase line, and phases over time
    """
    
    kw_thal = copy.deepcopy(kw_thal_template)
    kw_thal['rhs'] = t1.rhs
    
    system1 = rsp(**{'pardict':pd_thal_template,**kw_thal})

    pl_list = [(1,1),(2,1),(1,2),(2,3)]
    T_list = [800,200,1000,1000]
    e_list = [(.05,.05),(.05,.05),(.05,.05),(.05,.05)]
    d_list = [(.01,.095),(.01,.05),(.0,.035),(.0,.008)]
    pmax_list = [.5,.5,.5,.5]
    pmin_list = [-.5,-.5,-.5,-.5]
    init_list = [5,5,5.5,5]

    full_rhs = _full_thal1
    fig,axs = _setup_trajectories_plot(lo=0.085)

    for k in range(len(axs)):
        # run simulations and plot
        a = nmc(system1,None,recompute_list=[],
                _n=('om0',pl_list[k][0]),_m=('om1',pl_list[k][1]),
                NP=300,NH=300)
        
        draw_forcing_sols(axs[k],a,T_list[k],e_list[k],d_list[k],
                          pmax_list[k],pmin_list[k],init_list[k],
                          full_rhs)

        del a
            
    # set title
    ct = 0
    for k in range(len(axs)):
        axs[k][0,0].set_title(labels[ct],loc='left')
        ct += 1
                              
        axs[k][0,1].set_title(labels[ct],loc='left')
        ct += 1

    # fix title with parameter values
    nr1,nc1 = axs[0].shape
    for k in range(len(axs)):
        for j in range(nc1):
            ti1 = axs[k][0,j].get_title()
            ti1 += str(pl_list[k][0])+':'+str(pl_list[k][1])
            #t1 += r', $\varepsilon='+str(e_list[k])+'$'
            ti1 += r', $\delta = '+str(d_list[k][j])+'$'
            
            axs[k][0,j].set_title(ti1)
    
    return fig


def trajectories_gw2():
    """
    pair of goodwin oscillators
    """
    
    kw_thal = copy.deepcopy(kw_gw_template)
    kw_thal['rhs'] = t2.rhs
    kw_thal['var_names'] = ['v','h','r','w']
    kw_thal['init'] = np.array([-.64,0.71,0.25,0,6.28])
    kw_thal['forcing_fn'] = None
    kw_thal['coupling'] = t2.coupling
    kw_thal['trunc_order'] = 3
    

def trajectories_thal2():
    """
    Plot phase plane, phase line, and phases over time
    for a pair of thalamic oscillators
    """
    
    kw_thal = copy.deepcopy(kw_thal_template)
    kw_thal['rhs'] = t2.rhs
    kw_thal['var_names'] = ['v','h','r','w']
    kw_thal['init'] = np.array([-.64,0.71,0.25,0,6.28])
    kw_thal['forcing_fn'] = None
    kw_thal['coupling'] = t2.coupling
    kw_thal['trunc_order'] = 3
    #kw_thal['save_fig'] = True

    pd_thal_template2 = deepcopy(pd_thal_template)
    pd_thal_template2['ib'] = 3.5
    
    kw_thal['model_name'] = 'thal0';kw_thal['idx']=0
    system1 = rsp(**{'pardict':pd_thal_template2,**kw_thal})

    kw_thal['model_name'] = 'thal1';kw_thal['idx']=1
    system2 = rsp(**{'pardict':pd_thal_template2,**kw_thal})

    #pl_list = [(1,1),(2,1),(1,2),(2,3)]
    T_list = [3000,200,1000,2000]
    #N_list = [500,500,500,500]

    pl_list = [(1,1),(1,2),(2,1),(2,3)]
    N_list = [1100,1500,1500,1500]
    
    e_list = [(.093,.093),(.1,.1),(.1,.1),(.1,.1)]
    d_list = [(.0,.001),(.0,.017),(.0,.008),(0,.001)]
    
    skipn_list = [50,20,50,50]
    
    init_list = [.5,0,0,.5]

    full_rhs = _full_thal2
    fig,axs_list = _setup_trajectories_plot(mode='c')
    # axs is a list of four plot objects, each with 2x2 subplots.

    for k in range(len(axs_list[:3])):
        print('e',e_list[k],';','d',d_list[k])

        for j in range(2):
            # run simulations and plot
            #recompute_list = ['p_data_thal0','p_data_thal1']
            #recompute_list += ['h_data_thal0','h_data_thal1']
            #recompute_list += ['k_thal0','k_thal1']

            a = nmc(system1,system2,
                    #recompute_list=recompute_list,
                    _n=('om0',pl_list[k][0]),
                    _m=('om1',pl_list[k][1]),
                    save_fig=False,NH=N_list[k],
                    del1=d_list[k][j])
        
            draw_full_solutions(axs_list[k][-1,j],a,T_list[k],e_list[k][j],
                                init_list[k],full_rhs=full_rhs,
                                skipn=skipn_list[k])

            draw_1d_rhs(axs_list[k][0,j],a,T_list[k],e_list[k][j],init_list[k])
            
            draw_1d_solutions(axs_list[k][-1,j],a,T_list[k],e_list[k][j],
                              init_list[k])

            draw_3d_solutions(axs_list[k][-1,j],a,T_list[k],e_list[k][j],
                              init_list[k],rhs=_redu_3dc_thal)

            del a
            
    # set title
    ct = 0
    for k in range(len(axs_list)):
        axs_list[k][0,0].set_title(labels[ct],loc='left')
        ct += 1
                              
        axs_list[k][0,1].set_title(labels[ct],loc='left')
        ct += 1

    # fix title with parameter values
    nr1,nc1 = axs_list[0].shape
    for k in range(len(axs_list)):
        for j in range(nc1):
            ti1 = axs_list[k][0,j].get_title()
            ti1 += str(pl_list[k][0])+':'+str(pl_list[k][1])
            #t1 += r', $\varepsilon='+str(e_list[k])+'$'
            ti1 += r', $\delta = '+str(d_list[k][j])+'$'
            
            axs_list[k][0,j].set_title(ti1)

    axs_list[0][1,1].legend()

    return fig


def trajectories_thal2b():
    """
    Plot phase plane, phase line, and phases over time
    """
    
    kw_thal = copy.deepcopy(kw_thal_template)
    kw_thal['rhs'] = t2.rhs
    kw_thal['var_names'] = ['v','h','r','w']
    kw_thal['init'] = np.array([-.64,0.71,0.25,0,6.28])
    kw_thal['forcing_fn'] = None
    kw_thal['coupling'] = t2.coupling
    kw_thal['trunc_order'] = 3
    #kw_thal['save_fig'] = True

    pd_thal = copy.deepcopy(pd_thal_template)
    pd_thal['ib'] = 8.5
    
    kw_thal['model_name'] = 'thal0b';kw_thal['idx']=0
    system1 = rsp(**{'pardict':pd_thal,**kw_thal})

    kw_thal['model_name'] = 'thal1b';kw_thal['idx']=1
    system2 = rsp(**{'pardict':pd_thal,**kw_thal})

    pl_list = [(1,1),(2,1),(1,2),(2,3)]
    T_list = [1000,250,500,500]
    e_list = [.15,.15,.1,.1]
    d_list = [(.0,.01),(.01,.05),(.0,.03),(.0,.01)]
    pmax_list = [.5,.5,.5,.5]
    pmin_list = [-.5,-.5,-.5,-.5]
    init_list = [5,3,5,5]

    full_rhs = _full_thal2
    fig,axs = _setup_trajectories_plot(mode='c')

    for k in range(len(axs[:1])):
        # run simulations and plot
        a = nmc(system1,system2,
                #recompute_list=['h_dat_thal0','h_dat_thal1'],
                _n=('om0',pl_list[k][0]),
                _m=('om1',pl_list[k][1]),
                save_fig=True,
                NP=500,NH=500)

        #a.Tmax = T_list[k]
        #a.eps_val = e_list[k]
        #a.del1_val = d_list[k]
        #a.pmin = pmin_list[k]
        #a.pmax = pmax_list[k]
        
        draw_solutions(axs[k],a,T_list[k],e_list[k],d_list[k],
                       pmax_list[k],pmin_list[k],init_list[k],
                       full_rhs)

        del a
            
    # set title
    ct = 0
    for k in range(len(axs)):
        axs[k][0,0].set_title(labels[ct],loc='left')
        ct += 1
                              
        axs[k][0,1].set_title(labels[ct],loc='left')
        ct += 1

    # fix title with parameter values
    nr1,nc1 = axs[0].shape
    for k in range(len(axs)):
        for j in range(nc1):
            ti1 = axs[k][0,j].get_title()
            ti1 += str(pl_list[k][0])+':'+str(pl_list[k][1])
            #t1 += r', $\varepsilon='+str(e_list[k])+'$'
            ti1 += r', $\delta = '+str(d_list[k][j])+'$'
            
            axs[k][0,j].set_title(ti1)

    return fig


def trajectories_gt2():
    """
    Plot phase plane, phase line, and phases over time
    """
    
    kw_thal = copy.deepcopy(kw_thal_template)
    kw_thal['rhs'] = t2.rhs
    kw_thal['var_names'] = ['v','h','r','w']
    kw_thal['init'] = np.array([-.64,0.71,0.25,0,6.28])
    kw_thal['forcing_fn'] = None
    kw_thal['coupling'] = gt.coupling_tm
    #kw_thal['save_fig'] = True
    kw_thal['trunc_order'] = 3
    
    kw_thal['model_name'] = 'gt0';kw_thal['idx']=0
    system1 = rsp(**{'pardict':pd_thal_template,**kw_thal})

    kw_gw = copy.deepcopy(kw_gw_template)
    kw_gw['model_name'] = 'gt1';kw_gw['idx']=1
    kw_gw['trunc_order'] = 3
    
    system2 = rsp(**{'pardict':pd_gw_template,**kw_gw})

    pl_list = [(1,1),(2,1),(1,2),(2,3)]
    T_list = [500,250,500,500]
    e_list = [.048,.1,.1,.1]
    d_list = [(-.055,.03),(.01,.05),(.0,.03),(.0,.01)]
    pmax_list = [.5,.5,.5,.5]
    pmin_list = [-.5,-.5,-.5,-.5]
    init_list = [np.pi,0,5,5]

    full_rhs = _full_gt2
    fig,axs = _setup_trajectories_plot(mode='c')

    for k in range(len(axs[:1])):
        # run simulations and plot
        a = nmc(system1,system2,
                recompute_list=[],
                _n=('om0',pl_list[k][0]),
                _m=('om1',pl_list[k][1]),
                NP=300,NH=300)

        draw_solutions(axs[k],a,T_list[k],e_list[k],d_list[k],
                       pmax_list[k],pmin_list[k],init_list[k],
                       full_rhs)

        del a
            
    # set title
    ct = 0
    for k in range(len(axs)):
        axs[k][0,0].set_title(labels[ct],loc='left')
        ct += 1
                              
        axs[k][0,1].set_title(labels[ct],loc='left')
        ct += 1

    # fix title with parameter values
    nr1,nc1 = axs[0].shape
    for k in range(len(axs)):
        for j in range(nc1):
            pass
            #ti1 = axs[k][0,j].get_title()
            #ti1 += str(pl_list[k][0])+':'+str(pl_list[k][1])
            #t1 += r', $\varepsilon='+str(e_list[k])+'$'
            #ti1 += r', $\delta = '+str(d_list[k][j])+'$'
            
            #axs[k][0,j].set_title(ti1)

    return fig


def _get_fr(rhs,a,eps,del1,th_init=0,dt=.02,T=1000,
            data_dir='fr',recompute=False):
    """
    get frequency ratio given eps, delta.
    """
    t = np.arange(0,T,dt)
    args = [a,eps,del1]

    if not(os.path.isdir(data_dir)):
        os.mkdir(data_dir)
    
    rhs_name = rhs.__name__
    ratio = str(a._n[1])+str(a._m[1])
    raw = ('{}/fr_{}_{}_ratio={}_T={}_dt={}_e={}_d={}.txt')
    fpars = [data_dir,rhs_name,a.system1.model_name,
             ratio,t[-1],t[1]-t[0],eps,del1]
    args1 = {'t_eval':t,'t_span':[0,t[-1]],'args':(*args,),**kw_sim}

    fname = raw.format(*fpars)
    file_dne  = not(os.path.isfile(fname))

    if file_dne or recompute:

        timef = t*(a._m[1]+del1)
        if 'full' in rhs.__name__:
            y0 = a.system1.lc['dat'][int((th_init/(2*np.pi))*a.system1.TN),:]
            timer = 0
            
        elif '1d' in rhs.__name__:
            y0 = [th_init]
            timer = t*a._n[1]*(1+del1/a._m[1])
        elif '2d' in rhs.__name__:
            y0 = [th_init,0]
            timer = t*a._n[1]*(1+del1/a._m[1])

        sol = solve_ivp(rhs,y0=y0,**args1)
        y = sol.y.T[:,0]

        if not('full' in rhs.__name__):
            y += timer
            y = np.mod(y,2*np.pi)

        freq_force = freq_est(t,a.system1.forcing_fn(timef))

        
        freq_full = freq_est(t,y,prominence=.15)

        fr = freq_full/freq_force

        np.savetxt(fname,[fr])
        
    else:
        fr = np.loadtxt(fname)

    return fr


def fr_cgl(recompute=False):
    kw_cgl = copy.deepcopy(kw_cgl_template)
    kw_cgl['rhs'] = c1.rhs_old2
    system1 = rsp(**{'pardict':pd_cgl_template,**kw_cgl})

    pl_list = [(1,1),(2,1),(3,1),(4,1)]
    T_list = [2000,2000,2000,3000]

    e_list = [.1,.1,.1,.06]
    pmax_list = [.3,.3,.25,.25]
    pmin_list = [-.3,-.3,-.25,-.25]
    dlim_list = [(-.1,.1),(-.04,.04),(-.015,.015),(-.002,.002)]
    dN_list = [30,30,30,30]
    th_init = 0

    fig,axs = plt.subplots(1,4,figsize=(8,2.5))

    for k in range(len(axs)):
        a = nmc(system1,None,
                _n=('om0',pl_list[k][0]),
                _m=('om1',pl_list[k][1]),
                NP=300,NH=300)

        del_range = np.linspace(dlim_list[k][0],dlim_list[k][1],dN_list[k])

        fr_full = []
        fr_redu2d = []
        fr_redu1d = []

        for i in range(len(del_range)):
                
            kwa = {'a':a,'eps':e_list[k],'del1':del_range[i],'T':T_list[k],
                   'recompute':recompute}
            fr_full.append(_get_fr(_full_cgl1,**kwa))
            fr_redu2d.append(_get_fr(rhs_avg_2d,**kwa))
            fr_redu1d.append(_get_fr(rhs_avg_1d,**kwa))
    
        axs[k].plot(del_range,fr_full,color='k',label='Full')
        axs[k].plot(del_range,fr_redu2d,color='tab:blue',label='2D',alpha=.75)
        axs[k].plot(del_range,fr_redu1d,color='tab:red',label='1D',alpha=.75,
                    ls='--')

        axs[k].legend()

        axs[k].set_title(labels[k],loc='left')
        axs[k].set_xlabel(r'$\delta$')

        tt = axs[k].get_title()
        tt += '{}:{}'.format(a._n[1],a._m[1])
        axs[k].set_title(tt)

        axs[k].set_xlim(del_range[0],del_range[-1])

    axs[0].set_ylabel('Freq. Ratio $\omega_X/(\omega_Y+\delta)$')
        
    plt.tight_layout()
        
    return fig

def load_tongue(system1,pl_list,exponents,mode='1d',
                data_dir='fr',recompute=False):
    """
    load or generate solution
    """

    if not(os.path.isdir(data_dir)):
        os.mkdir(data_dir)
        
    ratio = str(pl_list[0])+str(pl_list[1])

    raw = ('{}/tongue{}_{}_ratio={}_ex0={}_exf={}.txt')
    fpars = [data_dir,mode,system1.model_name,ratio,exponents[0],exponents[-1]]
    fname = raw.format(*fpars)
    file_dne  = not(os.path.isfile(fname))

    if file_dne or recompute:
        a = nmc(system1,None,
                _n=('om0',pl_list[0]),
                _m=('om1',pl_list[1]),
                NP=300,NH=300)
        
        d1 = list(-2**exponents[::-1]);d2 = list(2**exponents)
        dvals = np.array(d1+[0]+d2)

        if mode in ['1d',1]:
            d2d,v2d = get_tongue_1d(dvals,a,deps=.01,max_eps=.5)
        elif mode in ['2d',2]:
            d2d,v2d = get_tongue_2d(dvals,a,deps=.01,max_eps=.5)
        else:
            raise ValueError('Invalid mode '+str(mode))
        
        data = np.zeros([len(d2d),2])
        data[:,0] = d2d;data[:,1] = v2d
        
        np.savetxt(fname,data)
        
    else:
        data = np.loadtxt(fname)

    return data[:,0],data[:,1]


def tongues_cgl():
    kw_cgl = copy.deepcopy(kw_cgl_template)
    kw_cgl['rhs'] = c1.rhs_old2
     
    system1 = rsp(**{'pardict':pd_cgl_template,**kw_cgl})
    pl_list = [(1,1),(2,1),(3,1),(4,1)]
    
    fig,axs = plt.subplots(1,1,figsize=(8,2))

    for k in range(len(pl_list)):

        exps = np.linspace(-15,0,50,endpoint=False)
        d1 = list(-2**exps[::-1]);d2 = list(2**exps)
        dvals = np.array(d1+[0]+d2)

        d1d,v1d = load_tongue(system1,pl_list[k],exps,mode=1)
        d2d,v2d = load_tongue(system1,pl_list[k],exps[::5],mode=2)

        x2d = pl_list[k][0]/(pl_list[k][1]+d2d)
        x1d = pl_list[k][0]/(pl_list[k][1]+d1d)
        axs.plot(x2d,v2d,color='tab:blue',alpha=.75)
        axs.plot(x1d,v1d,color='tab:red',alpha=.75,ls='--')

    # inset for 4:1
    axins = axs.inset_axes([.7,.2,.2,.7],
                           xlim=(4-.01,4+.007),
                           ylim=(0,.14))

    axins.plot(x2d,v2d,color='tab:blue',alpha=.75)
    axins.plot(x1d,v1d,color='tab:red',alpha=.75,ls='--')
    axins.tick_params(labelbottom=False,labelleft=False,
                      bottom=False,left=False)
    
    _,cc = axs.indicate_inset_zoom(axins)
    cc[0].set_visible(True)
    cc[1].set_visible(True)
    cc[2].set_visible(True)
    cc[3].set_visible(True)
    #plt.show()

    axs.set_xlabel(r'$\omega_X/(\omega_Y+\delta)$')
    axs.set_ylabel(r'$\varepsilon$')
    

    axs.set_xticks([1/1,2/1,3/1,4/1])
    axs.set_xticklabels(['$1{:}1$','$2{:}1$','$3{:}1$','$4{:}1$'])
    axs.set_xlim(.8,4.1)
    axs.set_ylim(-.01,0.3)

    plt.tight_layout()

    return fig


def fr_thal(recompute=False):
    kw_thal = copy.deepcopy(kw_thal_template)
    kw_thal['rhs'] = t1.rhs
    system1 = rsp(**{'pardict':pd_thal_template,**kw_thal})

    pl_list = [(1,1),(1,2),(2,1),(2,3)]
    T_list = [1000,1000,1000,1000]

    e_list = [.05,.05,.03,.03]
    pmax_list = [.25,.25,.25,.25]
    pmin_list = [-.25,-.25,-.25,-.25]
    dlim_list = [(-.2,.2),(-.05,.05),(-.05,.05),(-.02,.02)]
    dN_list = [30,30,30,30]
    th_init = 0

    fig,axs = plt.subplots(1,len(pl_list),figsize=(8,2.5))

    for k in range(len(pl_list)):
        a = nmc(system1,None,
                _n=('om0',pl_list[k][0]),
                _m=('om1',pl_list[k][1]),
                NP=300,NH=300)

        del_range = np.linspace(dlim_list[k][0],dlim_list[k][1],dN_list[k])

        fr_full = []
        fr_redu2d = []
        fr_redu1d = []

        for i in range(len(del_range)):

            kwa = {'a':a,'eps':e_list[k],'del1':del_range[i],'T':T_list[k],
                   'recompute':recompute}
            fr_full.append(_get_fr(_full_thal1,**kwa))
            fr_redu2d.append(_get_fr(rhs_avg_2d,**kwa))
            fr_redu1d.append(_get_fr(rhs_avg_1d,**kwa))
    
        axs[k].plot(del_range,fr_full,color='k',label='Full')
        axs[k].plot(del_range,fr_redu2d,color='tab:blue',label='2D',alpha=.75)
        axs[k].plot(del_range,fr_redu1d,color='tab:red',ls='--',
                    label='1D',alpha=.75)

        axs[k].legend()

        axs[k].set_title(labels[k],loc='left')
        axs[k].set_xlabel(r'$\delta$')

        tt = axs[k].get_title()
        tt += '{}:{}'.format(a._n[1],a._m[1])
        axs[k].set_title(tt)
        
    plt.tight_layout()
        
    return fig



def tongues_thal():
    kw_thal = copy.deepcopy(kw_thal_template)
    kw_thal['rhs'] = t1.rhs
    
    system1 = rsp(**{'pardict':pd_thal_template,**kw_thal})
    pl_list = [(1,1),(2,1),(1,2),(2,3),(3,1),(1,3),(4,1),(1,4)]
    
    fig,axs = plt.subplots(1,1,figsize=(8,2))

    for k in range(len(pl_list)):
        
        exps = np.linspace(-15,0,50,endpoint=False)
        d1 = list(-2**exps[::-1]);d2 = list(2**exps)
        dvals = np.array(d1+[0]+d2)

        d1d,v1d = load_tongue(system1,pl_list[k],exps,mode=1)
        d2d,v2d = load_tongue(system1,pl_list[k],exps[::5],mode=2)

        x2d = pl_list[k][0]/(pl_list[k][1]+d2d)
        x1d = pl_list[k][0]/(pl_list[k][1]+d1d)
        axs.plot(x2d,v2d,color='tab:blue',alpha=.75)
        axs.plot(x1d,v1d,color='tab:red',alpha=.75,ls='--')

    axs.set_xlabel(r'$\omega_X/(\omega_Y+\delta)$')
    axs.set_ylabel(r'$\varepsilon$')

    ticklist2 = []
    labellist2 = []
    for k in range(len(pl_list)):
        n = pl_list[k][0];m = pl_list[k][1]
        ticklist2.append(n/m)
        labellist2.append(r'$'+str(n)+'{:}'+str(m)+'$')

    
    axs.set_xlim(2e-1,5)
    axs.set_ylim(-.01,0.4)

    axs.set_xscale('log')

    axs.set_xticks(ticklist2)
    axs.set_xticklabels(labellist2) 
   
    plt.tight_layout()

    return fig


def bif1d_cgl1():
    
    kw_cgl = copy.deepcopy(kw_cgl_template)
    kw_cgl['rhs'] = c1.rhs_old2
    system1 = rsp(**{'pardict':pd_cgl_template,**kw_cgl})

    pl_list = [(1,1),(2,1),(3,1),(4,1)]
    e_list = [(.2,.2),(.1,.1),(.1,.1),(.1,.1)]
    d_list = [(.01,.08),(.01,.025),(.001,.008),(.0007,.0015)]
    pmax_list = [1.5,1.5,1.5,1.5]
    pmin_list = [-1,-1,-1,-1]
    init_list = [1,1,.5,.5]

    full_rhs = _full_cgl1

    fig,axs = _setup_trajectories_plot(labels=False,hspace=.07)

    k = 0
    a = nmc(system1,None,recompute_list=[],
            _n=('om0',pl_list[k][0]),_m=('om1',pl_list[k][1]),
            NP=300,NH=300)

    # add model diagrams (top left)
    add_diagram_2d(axs[k][0,0],a,.01,(.001,.5,100))
    add_diagram_1d(axs[k][1,0],a,.01,(.001,.5,200),rhs=rhs_avg_1d)
    add_diagram_full(axs[k][2,0],a,.01,(.0275,.5,100),rhs=_full_cgl1)
    #add_diagram_full(axs[0,2],a,.01,(.0275,.5,10),rhs=_full_cgl1,
    #                 phi0=np.pi/4) # adds nothing

    add_diagram_2d(axs[k][0,1],a,.08,(.001,.5,100))
    add_diagram_1d(axs[k][1,1],a,.08,(.001,.5,200),rhs=rhs_avg_1d)
    add_diagram_full(axs[k][2,1],a,.08,(.24,.5,100),rhs=_full_cgl1,
                     recompute=False)

    k = 1
    a = nmc(system1,None,recompute_list=[],
            _n=('om0',pl_list[k][0]),_m=('om1',pl_list[k][1]),            
            NP=300,NH=300)

    # add model diagrams (top right)
    add_diagram_2d(axs[k][0,0],a,.01,(.001,.5,100))
    add_diagram_1d(axs[k][1,0],a,.01,(.001,.5,200),rhs=rhs_avg_1d)
    add_diagram_full(axs[k][2,0],a,.01,(.056,.5,100),rhs=_full_cgl1,
                     recompute=False)

    add_diagram_2d(axs[k][0,1],a,.025,(.001,.5,100))
    add_diagram_1d(axs[k][1,1],a,.025,(.001,.5,200),rhs=rhs_avg_1d)
    add_diagram_full(axs[k][2,1],a,.025,(.145,.5,100),rhs=_full_cgl1,
                     recompute=False)

    k = 2
    # add model diagrams (bottom left)
    a = nmc(system1,None,recompute_list=[],
            _n=('om0',pl_list[k][0]),_m=('om1',pl_list[k][1]),            
            NP=300,NH=300)

    add_diagram_2d(axs[k][0,0],a,.001,(.001,.5,100))
    add_diagram_1d(axs[k][1,0],a,.001,(.001,.5,200),rhs=rhs_avg_1d)
    add_diagram_full(axs[k][2,0],a,.001,(.02,.5,100),rhs=_full_cgl1,
                     maxt=3500,recompute=False,scale_t_eps=False)

    add_diagram_2d(axs[k][0,1],a,.008,(.001,.5,100))
    add_diagram_1d(axs[k][1,1],a,.008,(.001,.5,200),rhs=rhs_avg_1d)
    add_diagram_full(axs[k][2,1],a,.008,(.164,.5,50),rhs=_full_cgl1,
                     maxt=2000,recompute=False,scale_t_eps=False)

    k = 3
    # add model diagrams (bottom left)
    a = nmc(system1,None,recompute_list=[],
            _n=('om0',pl_list[k][0]),_m=('om1',pl_list[k][1]),            
            NP=300,NH=300)

    add_diagram_2d(axs[k][0,0],a,.0007,(.001,.5,100))
    add_diagram_1d(axs[k][1,0],a,.0007,(.001,.5,200),rhs=rhs_avg_1d)
    add_diagram_full(axs[k][2,0],a,.0007,(.086,.5,25),rhs=_full_cgl1,
                     maxt=5000,recompute=False,scale_t_eps=False)

    add_diagram_2d(axs[k][0,1],a,.0015,(.001,.5,200))
    add_diagram_1d(axs[k][1,1],a,.0015,(.001,.5,200),rhs=rhs_avg_1d)
    add_diagram_full(axs[k][2,1],a,.0015,(.22,.5,25),rhs=_full_cgl1,
                     maxt=5000,recompute=False,scale_t_eps=False)

    # mark eps values
    for k in range(4):
        text = r'$\varepsilon='+str(e_list[k][0])+'$'
        axs[k][-1,0].text(e_list[k][0]+.02,.25,text)
        axs[k][-1,1].text(e_list[k][0]+.02,.25,text)
        
        for j in range(3):
            argt = {'ls':'--','color':'gray','lw':1,'clip_on':False}
            axs[k][j,0].axvline(e_list[k][0],-.05,1.05,**argt)
            axs[k][j,1].axvline(e_list[k][1],-.05,1.05,**argt)
    
    for i in range(len(axs)):
        for j in range(3):
            axs[i][j,0].set_ylabel(r'$\phi$',labelpad=0)
            for k in range(2):
                axs[i][j,k].set_ylim(-.1,2*np.pi+.1)
                axs[i][j,k].set_xlim(0,.5)
            
                axs[i][j,k].set_yticks(np.arange(0,3,1)*np.pi)
                axs[i][j,k].set_yticklabels(pi_label_short)

            

        for j in range(2):
            axs[i][-1,j].set_xlabel(r'$\varepsilon$',labelpad=0)

    # set title
    ct = 0
    for k in range(len(axs)):
        axs[k][0,0].set_title(labels[ct],loc='left')
        ct += 1
                              
        axs[k][0,1].set_title(labels[ct],loc='left')
        ct += 1

    # fix title with parameter values
    nr1,nc1 = axs[0].shape
    for k in range(len(axs)):
        for j in range(nc1):
            ti1 = axs[k][0,j].get_title()
            ti1 += str(pl_list[k][0])+':'+str(pl_list[k][1])
            #t1 += r', $\varepsilon='+str(e_list[k])+'$'
            ti1 += r', $\delta = '+str(d_list[k][j])+'$'
            
            axs[k][0,j].set_title(ti1)

    #axs[2][1,0].yaxis.labelpad=-10

    return fig



def bif1d_thal1():
    
    kw_thal = copy.deepcopy(kw_thal_template)
    kw_thal['rhs'] = t1.rhs

    system1 = rsp(**{'pardict':pd_thal_template,**kw_thal})

    pl_list = [(1,1),(2,1),(1,2),(2,3)]
    T_list = [800,200,1000,1000]
    e_list = [(.05,.05),(.05,.05),(.05,.05),(.05,.05)]
    d_list = [(.01,.09),(.01,.05),(.0,.035),(.0,.008)]
    pmax_list = [.5,.5,.5,.5]
    pmin_list = [-.5,-.5,-.5,-.5]
    init_list = [5,5,5.5,5]

    full_rhs = _full_thal1
    fig,axs = _setup_trajectories_plot(labels=False,hspace=.07,lo=.06)
    
    k = 0
    a = nmc(system1,None,recompute_list=[],
            _n=('om0',pl_list[k][0]),_m=('om1',pl_list[k][1]),
            NP=300,NH=300)

    # add model diagrams (top left)
    add_diagram_2d(axs[k][0,0],a,.01,(.001,.3,100))
    add_diagram_1d(axs[k][1,0],a,.01,(.001,.3,200),rhs=rhs_avg_1d)
    add_diagram_full(axs[k][2,0],a,.01,(.01,.3,25),rhs=full_rhs,
                     maxt=500,scale_t_eps=False)
    
    add_diagram_2d(axs[k][0,1],a,.09,(.001,.3,100))
    add_diagram_1d(axs[k][1,1],a,.09,(.001,.3,200),rhs=rhs_avg_1d)
    add_diagram_full(axs[k][2,1],a,.09,(.052,.3,100),rhs=full_rhs,
                     maxt=500,scale_t_eps=False,recompute=False)
    
    k = 1
    a = nmc(system1,None,recompute_list=[],
            _n=('om0',pl_list[k][0]),_m=('om1',pl_list[k][1]),            
            NP=300,NH=300)

    # add model diagrams (top right)
    add_diagram_2d(axs[k][0,0],a,.01,(.001,.3,100))
    add_diagram_1d(axs[k][1,0],a,.01,(.001,.3,200),rhs=rhs_avg_1d)
    add_diagram_full(axs[k][2,0],a,.01,(.015,.3,25),rhs=full_rhs,
                     maxt=500,scale_t_eps=False,recompute=False)

    add_diagram_2d(axs[k][0,1],a,.05,(.001,.3,100))
    add_diagram_1d(axs[k][1,1],a,.05,(.001,.3,200),rhs=rhs_avg_1d)
    add_diagram_full(axs[k][2,1],a,.05,(.065,.3,25),rhs=full_rhs,
                     maxt=300,scale_t_eps=False,recompute=False)

    k = 2
    # add model diagrams (bottom left)
    a = nmc(system1,None,recompute_list=[],
            _n=('om0',pl_list[k][0]),_m=('om1',pl_list[k][1]),            
            NP=300,NH=300)

    add_diagram_2d(axs[k][0,0],a,.0,(.001,.3,100))
    add_diagram_1d(axs[k][1,0],a,.0,(.001,.3,200),rhs=rhs_avg_1d)
    add_diagram_full(axs[k][2,0],a,.0,(.01,.3,25),rhs=full_rhs,
                     maxt=1000,scale_t_eps=False,recompute=False)

    add_diagram_2d(axs[k][0,1],a,.035,(.001,.3,100))
    add_diagram_1d(axs[k][1,1],a,.035,(.001,.3,200),rhs=rhs_avg_1d)
    add_diagram_full(axs[k][2,1],a,.035,(.065,.3,25),rhs=full_rhs,
                     maxt=1000,scale_t_eps=False,recompute=False)
    
    k = 3
    # add model diagrams (bottom right)
    a = nmc(system1,None,recompute_list=[],
            _n=('om0',pl_list[k][0]),_m=('om1',pl_list[k][1]),            
            NP=300,NH=300)

    add_diagram_2d(axs[k][0,0],a,.0,(.001,.3,100))
    add_diagram_1d(axs[k][1,0],a,.0,(.001,.3,200),rhs=rhs_avg_1d)
    add_diagram_full(axs[k][2,0],a,.0,(.005,.063,5),rhs=full_rhs,
                     maxt=10000,scale_t_eps=False,recompute=False,
                     branch_tol=.5)

    add_diagram_2d(axs[k][0,1],a,.008,(.005,.3,200))
    add_diagram_1d(axs[k][1,1],a,.008,(.005,.3,200),rhs=rhs_avg_1d)
    add_diagram_full(axs[k][2,1],a,.008,(.023,.085,40),rhs=full_rhs,
                     maxt=1000,scale_t_eps=False,recompute=False)

    
    # mark eps values
    for k in range(4):
        text = r'$\varepsilon='+str(e_list[k][0])+'$'
        axs[k][-1,0].text(e_list[k][0]+.02,5.5,text)
        axs[k][-1,1].text(e_list[k][0]+.02,5.5,text)
        
        for j in range(3):
            argt = {'ls':'--','color':'gray','lw':1,'clip_on':False}
            axs[k][j,0].axvline(e_list[k][0],-.05,1.05,**argt)
            axs[k][j,1].axvline(e_list[k][1],-.05,1.05,**argt)
    
    for i in range(len(axs)):
        for j in range(3):
            axs[i][j,0].set_ylabel(r'$\phi$',labelpad=0)
            for k in range(2):
                axs[i][j,k].set_ylim(-.1,2*np.pi+.1)
                axs[i][j,k].set_xlim(0,.29)
            
                axs[i][j,k].set_yticks(np.arange(0,3,1)*np.pi)
                axs[i][j,k].set_yticklabels(pi_label_short)

            

        for j in range(2):
            axs[i][-1,j].set_xlabel(r'$\varepsilon$',labelpad=0)

    # set title
    ct = 0
    for k in range(len(axs)):
        axs[k][0,0].set_title(labels[ct],loc='left')
        ct += 1
                              
        axs[k][0,1].set_title(labels[ct],loc='left')
        ct += 1

    # fix title with parameter values
    nr1,nc1 = axs[0].shape
    for k in range(len(axs)):
        for j in range(nc1):
            ti1 = axs[k][0,j].get_title()
            ti1 += str(pl_list[k][0])+':'+str(pl_list[k][1])
            #t1 += r', $\varepsilon='+str(e_list[k])+'$'
            ti1 += r', $\delta = '+str(d_list[k][j])+'$'
            
            axs[k][0,j].set_title(ti1)

    #axs[2][1,0].yaxis.labelpad=-10

    return fig


def bif_thal2_11():

    kw_thal = copy.deepcopy(kw_thal_template)
    kw_thal['rhs'] = t2.rhs
    kw_thal['var_names'] = ['v','h','r','w']
    kw_thal['init'] = np.array([-.64,0.71,0.25,0,6.28])
    kw_thal['forcing_fn'] = None
    kw_thal['coupling'] = t2.coupling
    kw_thal['trunc_order'] = 3
    #kw_thal['save_fig'] = True

    pd_thal_template2 = deepcopy(pd_thal_template)
    pd_thal_template2['ib'] = 3.5
    
    kw_thal['model_name'] = 'thal0';kw_thal['idx']=0
    system1 = rsp(**{'pardict':pd_thal_template2,**kw_thal})

    kw_thal['model_name'] = 'thal1';kw_thal['idx']=1
    system2 = rsp(**{'pardict':pd_thal_template2,**kw_thal})

    del_list = [0,.0005,.001,.0015]
    eps_list = np.linspace(0,.15,200)

    # (init, eps_init, eps_final, d_eps), ...
    follow_pars = [[(3,.01,.001,-.0005),(3,.01,.31,.01),
                    (.1,.2,.1,-.0005),  (.1,.2,.3,.01),
                    (5,.2,.1,-.0005),  (5,.2,.3,.01)],
                   
                   [(3,.02,.0005,-.0005),(3,.02,.3,.01),
                    (.5,.2,.27,.01),     (.5,.2,.08,-.005),
                    (6,.2,.3,.01),       (6,.2,.125,-.005)],
                   
                   [(3,.02,.0005,-.0005),(3,.02,.3,.01),
                    (.5,.2,.31,.01),     (.5,.2,.04,-.01),
                    (6,.2,.3,.01),(6,.2,.14,-.01)],
                   
                   [(3,.02,.0005,-.0005),(3,.02,.3,.01),
                    (.5,.05,.061,.001),(.5,.05,.035,-.002),
                    (6,.2,.3,.01),(6,.2,.14,-.01)]
                   ]

    data_list = []
    obj_list = []

    # get full bifurcation diagram
    for i,dd in enumerate(del_list):
        obj = nm.nmCoupling(system1,system2,_n=('om0',1),_m=('om1',1),
                            NH=1000,save_fig=False,del1=dd,
                            recompute_list=['k_thal0','k_thal1'])

        branch_data = []
        for j in range(len(follow_pars[i])):
            e0,e1,e2,e3 = follow_pars[i][j]
            a_full_branch = follow_phase_diffs(init=e0,eps_init=e1,eps_final=e2,
                                               deps=e3,a=obj,del1=dd,
                                               bifdir='bif1d_thal2/',
                                               _full_rhs=_full)
            branch_data.append(a_full_branch)

        data_list.append(branch_data)
        obj_list.append(obj)

        
    fig,axs=plt.subplots(1,4,figsize=(8,2),gridspec_kw={'wspace':.5})
    axs = axs.flatten()

    for i in range(len(del_list)):
        # plot 1d
        if i == 1:
            label1 = '1D'
        else:
            label1 = ''
        add_diagram_1d(axs[i],obj_list[i],del_list[i],(.001,.25,500),rhs=_redu_c,
                       domain=np.linspace(-.1,2*np.pi+.1,5000),label=label1)
        
        # plot full
        for j in range(len(data_list[i])):
            data = data_list[i][j]
            if i == 1 and j == 0:
                label1 = 'Full'
            else:
                label1 = ''

            if del_list[i] == 0:
                lw=2;zorder=-10
            else:
                lw=1.5;zorder=-10
                
            kwarg = dict(color='k',zorder=zorder,label=label1,lw=lw)
            axs[i].plot(data[:,0],np.mod(data[:,1],2*np.pi),**kwarg)
            
        #axs[i].set_ylim(-.1,2*np.pi+.1)
        axs[i].set_ylim(-.1,2*np.pi+.1)
        axs[i].set_yticks(np.arange(0,3,1)*np.pi)
        axs[i].set_yticklabels(pi_label_short)

        axs[i].set_xlim(0,eps_list[-1])

        axs[i].set_xlabel(r'$\varepsilon$')
        axs[i].set_title(labels[i],loc='left')

        # add parameter value
        tt = axs[i].get_title()
        tt += r'$\delta='+str(del_list[i])+'$'
        axs[i].set_title(tt)

        axs[i].set_ylabel(r'$\phi$')

    # 3d diagram
    follow_3d = [[(3,.1,.31,.01),(3,.1,0.001,-.002)],
                 [(3,.1,.31,.01),(3,.1,0.001,-.002)],
                 [(3,.1,.31,.01),(3,.1,0.001,-.002)],
                 [(3,.1,.31,.01),(3,.1,0.001,-.002)],
                 ]
    
    for i in range(len(del_list)):
        list_3d = []
        for j in range(len(follow_3d[i])):
            e0,e1,e2,e3 = follow_3d[i][j]
            _,init = phase_lock_r3d([e0,0,0],obj_list[i],e1,_redu_3dc_thal)
            dat = follow_locking_3d(init,obj_list[i],(e1,e2,e3),
                                    rhs=_redu_3dc_thal,recompute=False)

            if i == 1 and j == 0:
                label1 = '3D'
            else:
                label1 = ''

            if del_list[i] == 0.0:
                fudge = .02;lw=3.5;zorder=-5
            else:
                fudge = 0;lw=1.5;zorder=3

            kwarg = dict(color='tab:blue',lw=lw,zorder=zorder,label=label1)
            axs[i].plot(dat[:,0],np.mod(dat[:,1]+fudge,2*np.pi),**kwarg)

    plt.subplots_adjust(left=.075,right=.97,bottom=.16)
    axs[1].legend()
    
    return fig


def bif_thal2_12():

    kw_thal = copy.deepcopy(kw_thal_template)
    kw_thal['rhs'] = t2.rhs
    kw_thal['var_names'] = ['v','h','r','w']
    kw_thal['init'] = np.array([-.64,0.71,0.25,0,6.28])
    kw_thal['forcing_fn'] = None
    kw_thal['coupling'] = t2.coupling
    kw_thal['trunc_order'] = 3
    #kw_thal['save_fig'] = True

    pd_thal_template2 = deepcopy(pd_thal_template)
    pd_thal_template2['ib'] = 3.5
    
    kw_thal['model_name'] = 'thal0';kw_thal['idx']=0
    system1 = rsp(**{'pardict':pd_thal_template2,**kw_thal})

    kw_thal['model_name'] = 'thal1';kw_thal['idx']=1
    system2 = rsp(**{'pardict':pd_thal_template2,**kw_thal})

    del_list = [0,.01,.014,.018]
    eps_list = np.linspace(0,.25,200)

    # (init, eps_init, eps_final, d_eps), ...
    follow_pars = [[(1,.1,.001,-.005),(1,.1,.31,.01)],
                   
                   [(2,.02,.001,-.005),  (2,.02,.31,.01),
                    (0,.025,.001,-.005),(0,.025,.31,.01)],
                   
                   [(0,.025,.31,.01), (0,.025,.001,-.005)],

                   [(0,.25,.185,-.002),  (0,.25,.31,.01)]
                   ]

    data_list = []
    obj_list = []

    # get full bifurcation diagram
    recompute_list = ['p_data_thal0','p_data_thal1']
    recompute_list += ['h_data_thal0','h_data_thal1']
    recompute_list += ['k_thal0','k_thal1']
    
    for i,dd in enumerate(del_list):
        obj = nm.nmCoupling(system1,system2,_n=('om0',1),_m=('om1',2),
                            recompute_list=recompute_list,
                            NH=1500,save_fig=False,del1=dd,
                            )

        branch_data = []
        for j in range(len(follow_pars[i])):
            e0,e1,e2,e3 = follow_pars[i][j]
            a_full_branch = follow_phase_diffs(init=e0,eps_init=e1,eps_final=e2,
                                               deps=e3,a=obj,del1=dd,
                                               bifdir='bif1d_thal2/',
                                               _full_rhs=_full)
            branch_data.append(a_full_branch)

        data_list.append(branch_data)
        obj_list.append(obj)

        
    fig,axs=plt.subplots(1,4,figsize=(8,2),gridspec_kw={'wspace':.5})
    axs = axs.flatten()

    #inset for close lines
    axins = [axs[1].inset_axes([.1,.23,.8,.5]),
             axs[2].inset_axes([.1,.23,.8,.5]),
             axs[3].inset_axes([.1,.23,.8,.5])]

    for i in range(len(del_list)):
        # plot 1d
        if i == 0:
            label1 = '1D'
        else:
            label1 = ''
        add_diagram_1d(axs[i],obj_list[i],del_list[i],(.001,.25,500),rhs=_redu_c,
                       domain=np.linspace(-.1,2*np.pi+.1,5000),label=label1)

        if i in [1,2,3]:
            add_diagram_1d(axins[i-1],obj_list[i],del_list[i],(.001,.25,500),
                           rhs=_redu_c,domain=np.linspace(-.1,2*np.pi+.1,5000),
                           label=label1)

        # plot full
        for j in range(len(data_list[i])):
            data = data_list[i][j]
            if i == 0 and j == 0:
                label1 = 'Full'
            else:
                label1 = ''

            if del_list[i] == 0:
                lw=3;zorder=-10
            else:
                lw=3;zorder=-10
                
            kwarg = dict(color='k',zorder=zorder,label=label1,lw=lw)
            axs[i].plot(data[:,0],np.mod(data[:,1],2*np.pi),**kwarg)

            if i in [1,2,3]:
                axins[i-1].plot(data[:,0],np.mod(data[:,1],2*np.pi),**kwarg)

            
        #axs[i].set_ylim(-.1,2*np.pi+.1)
        axs[i].set_ylim(0,np.pi)
        axs[i].set_yticks(np.arange(0,3,1)/2*np.pi)
        axs[i].set_yticklabels(pi_label_half)

        axs[i].set_xlim(0,eps_list[-1])

        axs[i].set_xlabel(r'$\varepsilon$')
        axs[i].set_title(labels[i],loc='left')

        # add parameter value
        tt = axs[i].get_title()
        tt += r'$\delta='+str(del_list[i])+'$'
        axs[i].set_title(tt)

        axs[i].set_ylabel(r'$\phi$')

    # 3d diagram
    follow_3d = [[(1,.1,.31,.01),(1,.1,0.001,-.01)],
                 [(.5,.1,.31,.01),(.5,.1,0.001,-.01)],
                 [(.5,.05,.26,.01),(.5,.05,0.001,-.01)],
                 [(5,.1,.12,.01),(5,.1,.12,.01)],
                 ]
    
    for i in range(len(del_list)):
        list_3d = []
        
        for j in range(len(follow_3d[i])):
            e0,e1,e2,e3 = follow_3d[i][j]
            _,init = phase_lock_r3d([e0,0,0],obj_list[i],e1,_redu_3dc_thal,
                                    )
            dat = follow_locking_3d(init,obj_list[i],(e1,e2,e3),
                                    bifdir='bifthal2_r3d/',rhs=_redu_3dc_thal,
                                    recompute=False)

            if i == 0 and j == 0:
                label1 = '3D'
            else:
                label1 = ''

            if del_list[i] == 0.0:
                fudge = .0;lw=2.5;zorder=-5
            else:
                fudge = 0;lw=2.5;zorder=-5

            kwarg = dict(color='tab:blue',lw=lw,zorder=zorder,label=label1)
            if i < 3:
                axs[i].plot(dat[:,0],np.mod(dat[:,1]+fudge,2*np.pi),**kwarg)

            if i in [1,2,3]:
                axins[i-1].plot(dat[:,0],np.mod(dat[:,1]+fudge,2*np.pi),**kwarg)

    axins[0].set_xlim(0,.25);axins[1].set_xlim(0,.25)
    axins[2].set_xlim(.18,.25)
    
    axins[0].set_ylim(.3,.5)
    axins[1].set_ylim(.3,.4)
    axins[2].set_ylim(.15,.4)
        
    for i in range(len(axins)):
        
        axs[i+1].indicate_inset_zoom(axins[i])
        axins[i].tick_params(labelbottom=False,labelleft=False,
                             bottom=False,left=False)

    plt.subplots_adjust(left=.075,right=.97,bottom=.2)
    axs[0].legend()

        
    
    
    return fig


def bif1d_gt2():
    kw_thal = copy.deepcopy(kw_thal_template)
    kw_thal['rhs'] = t2.rhs
    kw_thal['var_names'] = ['v','h','r','w']
    kw_thal['init'] = np.array([-.64,0.71,0.25,0,6.28])
    kw_thal['forcing_fn'] = None
    kw_thal['coupling'] = gt.coupling_tm
    #kw_thal['save_fig'] = True
    kw_thal['trunc_order'] = 2
    
    kw_thal['model_name'] = 'gt0';kw_thal['idx']=0
    system1 = rsp(**{'pardict':pd_thal_template,**kw_thal})

    kw_gw = copy.deepcopy(kw_gw_template)
    kw_gw['model_name'] = 'gt1';kw_gw['idx']=1
    kw_gw['trunc_order'] = 3
    
    system2 = rsp(**{'pardict':pd_gw_template,**kw_gw})
    
    fig,axs=plt.subplots(10,1,figsize=(8,15))

    del_list = np.linspace(-.08,-.02,10)
    eps_list = np.linspace(0,.1,100)
    
    for k in range(len(del_list)):
        n = 1;m = 1
        a = nmc(system1,system2,
                recompute_list=[],
                _n=('om0',n),_m=('om1',m),
                NP=300,NH=300)

        zu_list = []
        zs_list = []

        zu = []
        zs = []

        for eps in eps_list:
            z1,z2 = bif1d(a,eps,del_list[k])

            zu.append(z1)
            zs.append(z2)

        for xe, ye in zip(eps_list, zu):
            axs[k].scatter([xe] * len(ye), ye,s=2,c='red')

        for xe, ye in zip(eps_list, zs):
            axs[k].scatter([xe] * len(ye), ye,s=2,c='k')

        """
        for xe, ye in zip(eps_list, zu1):
            axs.scatter([xe] * len(ye), ye,s=2,c='tab:orange')

        for xe, ye in zip(eps_list, zs1):
            axs.scatter([xe] * len(ye), ye,s=2,c='gray')
        """

        axs[k].set_ylim(-.1,2*np.pi+.1)

    return fig




def bif_gw2_11():
    """
    Bifurcations for 1:1 phase-locking in Goodwin.
    """

    kw_gw2 = copy.deepcopy(kw_gw_template)
    
    system1 = rsp(idx=0,model_name='gw0',**{'pardict':pd_gw_template,**kw_gw2})
    system2 = rsp(idx=1,model_name='gw1',**{'pardict':pd_gw_template,**kw_gw2})
    
    kw_bif = {'method':'LSODA','dense_output':True,'rtol':1e-9,'atol':1e-9}
    eps_list = np.linspace(0.001,.1,50)
    del_list = [0,.01,.03,.04]

    # (init, eps_init, eps_final, d_eps), ...
    follow_pars = [[(0,.04,.11,.01),(0,.05,.001,-.002)],
                   [(2,.05,.11,.01),(2,.05,.001,-.002)],
                   [(2,.05,.11,.01),(2,.05,.001,-.002)],
                   [(2,.05,.11,.01),(2,.05,.001,-.002)]
                   ]
    
    data_list = []
    obj_list = []

    # get full bifurcation diagram
    for i,dd in enumerate(del_list):
        obj = nm.nmCoupling(system1,system2,_n=('om0',1),_m=('om1',1),
                            NH=1000,save_fig=False,del1=dd)

        branch_data = []
        for j in range(len(follow_pars[i])):
            e0,e1,e2,e3 = follow_pars[i][j]
            a_full_branch = follow_phase_diffs(init=e0,eps_init=e1,eps_final=e2,
                                               deps=e3,a=obj,del1=dd,
                                               bifdir='bif1d_gw2/',
                                               _full_rhs=_full)
            branch_data.append(a_full_branch)

        data_list.append(branch_data)
        obj_list.append(obj)

        
    fig,axs = plt.subplots(1,4,figsize=(8,2),gridspec_kw={'wspace':.5})
    axs = axs.flatten()

    for i in range(len(del_list)):
        # plot 1d
        if i == 1:
            label1 = '1D'
        else:
            label1 = ''
        add_diagram_1d(axs[i],obj_list[i],del_list[i],(.001,.1,1000),rhs=_redu_c,
                       domain=np.linspace(-.1,2*np.pi+.1,5000),label=label1)

        # plot full
        for j in range(len(data_list[i])):
            data = data_list[i][j]
            if i == 1 and j == 0:
                label1 = 'Full'
            else:
                label1 = ''

            if del_list[i] == 0:
                fudge = .02;lw=5;zorder=-10
            else:
                fudge = 0;lw=1.5;zorder=5
                
            kwarg = dict(color='k',zorder=zorder,label=label1,lw=lw)
            axs[i].plot(data[:,0],np.mod(data[:,1]+fudge,2*np.pi),**kwarg)
            
        #axs[i].set_ylim(-.1,2*np.pi+.1)
        axs[i].set_ylim(-.1,2*np.pi+.1)
        axs[i].set_yticks(np.arange(0,3,1)*np.pi)
        axs[i].set_yticklabels(pi_label_short)

        axs[i].set_xlim(0,eps_list[-1])

        axs[i].set_xlabel(r'$\varepsilon$')
        axs[i].set_title(labels[i],loc='left')

        # add parameter value
        tt = axs[i].get_title()
        tt += r'$\delta='+str(del_list[i])+'$'
        axs[i].set_title(tt)

        axs[i].set_ylabel(r'$\phi$')


    # 3d diagram
    follow_3d = [[(.1,.05,.11,.01),(.1,.05,0.001,-.002)],
                 [(1,.05,.11,.01),(1,.05,0.001,-.002)],
                 [(1,.05,.092,.002),(1,.05,0.001,-.002)],
                 [(1,.03,.052,.001),(1,.03,0.001,-.002)]]
    
    for i in range(len(del_list)):
        list_3d = []
        for j in range(len(follow_3d[i])):
            e0,e1,e2,e3 = follow_3d[i][j]
            _,init = phase_lock_r3d([e0,0,0],obj_list[i],e1,_redu_3dc_gw)
            dat = follow_locking_3d(init,obj_list[i],(e1,e2,e3),
                                    rhs=_redu_3dc_gw,recompute=False)

            if i == 1 and j == 0:
                label1 = '3D'
            else:
                label1 = ''

            if del_list[i] == 0.0:
                fudge = .02;lw=3.5;zorder=-5
            else:
                fudge = 0;lw=1.5;zorder=3

            kwarg = dict(color='tab:blue',lw=lw,zorder=zorder,label=label1)
            axs[i].plot(dat[:,0],np.mod(dat[:,1]+fudge,2*np.pi),**kwarg)


    plt.subplots_adjust(left=.075,right=.97,bottom=.16)

    #plt.tight_layout()

    axs[1].legend()
    
    return fig


def bif_gw2_12():
    """
    Bifurcations for 1:2 phase-locking in Goodwin.
    """

    kw_gw2 = copy.deepcopy(kw_gw_template)
    
    system1 = rsp(idx=0,model_name='gw0',**{'pardict':pd_gw_template,**kw_gw2})
    system2 = rsp(idx=1,model_name='gw1',**{'pardict':pd_gw_template,**kw_gw2})
    
    kw_bif = {'method':'LSODA','dense_output':True,'rtol':1e-9,'atol':1e-9}
    eps_list = np.linspace(0.001,.1,50)

    # (init, eps_init, eps_final, d_eps), ...
    follow_pars = [[(0,.005,.078,.01),(2.5,0.05,0.11,.01),(2.5,.05,.035,-.002)],
                   
                   [(2,.02,.001,-.001),(2,.02,.11,.01),(0,.025,.0005,-.005),
                    (0,.025,.045,.002)],
                   
                   [(2,.02,.0005,-.005),(2,.02,.11,.01),(0,.025,.0005,-.005)],
                   [(2,.02,.0005,-.005),(2,.02,.11,.005),(1,.005,.0005,-.001)]
                   ]
    
    """
    follow_pars = [[(0,0.08,0.04,-.0005),(0,0.061,0.0005,-.0005)],
                   
                   [(2,0.02,0.0005,-.0005),(2,0.02,0.1,.0005),
                    (0,0.025,0.0005,-.0005),(0,0.025,0.1,.0005)],
                   
                   [(2,0.02,0.0005,-.0005),(2,0.02,0.1,.0005),
                    (0,0.025,0.0005,-.0005)],
                   
                   [(2,0.02,0.0005,-.0005),(2,0.02,0.1,.0005),
                    (1,0.005,0.0005,-.0005),(1,0.005,0.01,.0005)]]
    """
    del_list = [0,.0003,.0005,.0007] # [0,.0003,.0005,.0007]
    data_list = []
    obj_list = []

    # get full bifurcation diagram
    for i,dd in enumerate(del_list):
        obj = nm.nmCoupling(system1,system2,_n=('om0',1),_m=('om1',2),
                            NH=700,save_fig=False,del1=dd)

        branch_data = []
        for j in range(len(follow_pars[i])):
            e0,e1,e2,e3 = follow_pars[i][j]
            a_full_branch = follow_phase_diffs(init=e0,eps_init=e1,eps_final=e2,
                                               deps=e3,a=obj,del1=dd,
                                               bifdir='bif1d_gw2/',
                                               _full_rhs=_full)
            branch_data.append(a_full_branch)

        data_list.append(branch_data)
        obj_list.append(obj)

        
    fig,axs = plt.subplots(1,4,figsize=(8,2),gridspec_kw={'wspace':.5})
    axs = axs.flatten()

    for i in range(len(del_list)):
        # plot 1d
        if i == 0:
            label1 = '1D'
        else:
            label1 = ''
        add_diagram_1d(axs[i],obj_list[i],del_list[i],(.001,.1,1000),rhs=_redu_c,
                       domain=np.linspace(-.1,2*np.pi+.1,5000),label=label1,
                       tol=.1)

        # plot data
        for j in range(len(data_list[i])):
            data = data_list[i][j]
            if i == 0 and j == 0:
                label1 = 'Full'
            else:
                label1 = ''
                
            kwarg = dict(color='k',zorder=5)
            axs[i].plot(data[:,0],np.mod(data[:,1],2*np.pi),label=label1,
                        **kwarg)
            axs[i].plot(data[:,0],np.mod(data[:,2],2*np.pi),**kwarg)
            
        #axs[i].set_ylim(-.1,2*np.pi+.1)
        axs[i].set_ylim(-.1,np.pi+.1)
        axs[i].set_yticks(np.arange(0,3,1)/2*np.pi)
        axs[i].set_yticklabels(pi_label_half)

        axs[i].set_xlim(0,eps_list[-1])

        axs[i].set_xlabel(r'$\varepsilon$')
        axs[i].set_title(labels[i],loc='left')

        # add parameter value
        tt = axs[i].get_title()
        tt += r'$\delta='+str(del_list[i])+'$'
        axs[i].set_title(tt)

        axs[i].set_ylabel(r'$\phi$')

    # get 3d diagram
    
    follow_3d = [[(1.5,.05,.1,.002),(1.5,.05,0.001,-.002)],
                 [(1.5,.05,.1,.002),(1.5,.05,0.001,-.002)],
                 [(1.5,.05,.1,.002),(1.5,.05,0.001,-.002)],
                 [(1.5,.05,.1,.002),(1.5,.05,0.001,-.002),
                  (.8,.005,.009,.0005),(.8,.005,.001,-.001)]]
    
    for i in range(len(del_list)):
        list_3d = []
        for j in range(len(follow_3d[i])):
            e0,e1,e2,e3 = follow_3d[i][j]
            _,init = phase_lock_r3d([e0,0,0],obj_list[i],e1,_redu_3dc_gw)
            dat = follow_locking_3d(init,obj_list[i],(e1,e2,e3),
                                    rhs=_redu_3dc_gw,recompute=False)

            if i == 0 and j == 0:
                label1 = '3D'
            else:
                label1 = ''

            kwarg = dict(color='tab:blue',label=label1)
            axs[i].plot(dat[:,0],np.mod(dat[:,1],2*np.pi),**kwarg)

    plt.subplots_adjust(left=.075,right=.97,bottom=.16)

    axs[0].legend(handlelength=1,loc='lower left',borderpad=0.1,ncol=2,
                  handletextpad=0.2,columnspacing=-.1,borderaxespad=.2,
                  labelspacing=.1,frameon=False)
    
    return fig


def bif_gw2_23():
    """
    Bifurcations for 2:3 phase-locking in Goodwin.
    """
    kw_gw2 = copy.deepcopy(kw_gw_template)
    
    system1 = rsp(idx=0,model_name='gw0',**{'pardict':pd_gw_template,**kw_gw2})
    system2 = rsp(idx=1,model_name='gw1',**{'pardict':pd_gw_template,**kw_gw2})
    
    kw_bif = {'method':'LSODA','dense_output':True,'rtol':1e-9,'atol':1e-9}
    eps_list = np.linspace(0.001,.1,50)

    # (init, eps_init, eps_final, d_eps), ...
    follow_pars = [[(0,.01,.0775,.005),(0,.05,.001,-.005)],
                   [(1,.05,.11,.01),(1,.05,.001,-.005)],
                   [(0,.01,.024,.0005),(0,.01,.001,-.001)],
                   [(0,.005,.0135,.0005),(0,.005,.0005,-.0005)]
                   ]
    
    del_list = [0,.002,.0045,.0047]
    data_list = []
    obj_list = []


    # get full bifurcation diagram
    for i,dd in enumerate(del_list):
        obj = nm.nmCoupling(system1,system2,_n=('om0',2),_m=('om1',3),
                            NH=1000,save_fig=False,del1=dd)

        branch_data = []
        for j in range(len(follow_pars[i])):
            e0,e1,e2,e3 = follow_pars[i][j]
            a_full_branch = follow_phase_diffs(init=e0,eps_init=e1,eps_final=e2,
                                               deps=e3,a=obj,del1=dd,
                                               bifdir='bif1d_gw2/',
                                               _full_rhs=_full)
            branch_data.append(a_full_branch)

        data_list.append(branch_data)
        obj_list.append(obj)    

    fig,axs = plt.subplots(1,4,figsize=(8,2),gridspec_kw={'wspace':.5})
    axs = axs.flatten()

    for i in range(len(del_list)):
        # plot 1d
        if i == 1:
            label1 = '1D'
        else:
            label1 = ''
        add_diagram_1d(axs[i],obj_list[i],del_list[i],(.001,.1,1000),rhs=_redu_c,
                       domain=np.linspace(-.1,2*np.pi+.1,5000),label=label1)

        # plot full
        for j in range(len(data_list[i])):
            data = data_list[i][j]
            if i == 1 and j == 0:
                label1 = 'Full'
            else:
                label1 = ''
                
            kwarg = dict(color='k')
            axs[i].plot(data[:,0],np.mod(data[:,1],2*np.pi),label=label1,
                        **kwarg)
            axs[i].plot(data[:,0],np.mod(data[:,2],2*np.pi),**kwarg)
            axs[i].plot(data[:,0],np.mod(data[:,3],2*np.pi),**kwarg)
            
        #axs[i].set_ylim(-.1,2*np.pi+.1)
        axs[i].set_ylim(-.1,2*np.pi/3+.1)
        axs[i].set_yticks(np.arange(0,3,1)*np.pi/3)
        axs[i].set_yticklabels(pi_label_third)

        axs[i].set_xlim(0,eps_list[-1])

        axs[i].set_xlabel(r'$\varepsilon$')
        axs[i].set_title(labels[i],loc='left')

        # add parameter value
        tt = axs[i].get_title()
        tt += r'$\delta='+str(del_list[i])+'$'
        axs[i].set_title(tt)

        axs[i].set_ylabel(r'$\phi$')

    # 3d diagram
    follow_3d = [[(1,.05,.11,.01),(1,.05,.001,-.002)],
                 [(1,.05,.11,.01),(1,.05,.001,-.002)],
                 [(1,.05,.084,.002),(1,.05,.001,-.002)],
                 [(1,.03,.052,.002),(1,.03,.001,-.002)],
                   ]

    for i in range(len(del_list)):
        list_3d = []
        for j in range(len(follow_3d[i])):
            e0,e1,e2,e3 = follow_3d[i][j]
            _,init = phase_lock_r3d([e0,0,0],obj_list[i],e1,_redu_3dc_gw)
            dat = follow_locking_3d(init,obj_list[i],(e1,e2,e3),
                                    rhs=_redu_3dc_gw,recompute=False)

            if i == 1 and j == 0:
                label1 = '3D'
            else:
                label1 = ''

            if del_list[i] == 0.0:
                fudge = .02;lw=3.5;zorder=-5
            else:
                fudge = 0;lw=1.5;zorder=3

            kwarg = dict(color='tab:blue',lw=lw,zorder=zorder,label=label1)
            axs[i].plot(dat[:,0],np.mod(dat[:,1]+fudge,2*np.pi),**kwarg)


    plt.subplots_adjust(left=.075,right=.97,bottom=.16)

    axs[1].legend()
    
    return fig



def generate_figure(function, args, filenames, dpi=100):
    # workaround for python bug where forked processes use the same random 
    # filename.
    #tempfile._name_sequence = None;

    fig = function(*args)

    if type(filenames) is list:
        for name in filenames:
            fig.savefig(name,dpi=dpi)
    else:
        fig.savefig(filenames,dpi=dpi)

def main():

    #quick_plots_thalamic()

    # create figs directory if it doesn't exist
    if not(os.path.isdir('figs')):
        os.mkdir('figs')
    
    # listed in order of Figures in paper
    figures = [

        

        #(forcing_fn,[],['figs/f_forcing.pdf'],200),
        
        #(trajectories_cgl1,[],['figs/f_traj_cgl1.pdf',
        #                      'figs/f_traj_cgl1.png'],200),
        

        #(trajectories_thal1,[],['figs/f_traj_thal1.pdf',
        #                       'figs/f_traj_thal1.png'],200),

        #(trajectories_gw2,[],['figs/f_traj_gw2.pdf',
        #                      'figs/f_traj_gw2.png'],200),
        
        #(trajectories_thal2,[],['figs/f_traj_thal2.pdf',
        #                       'figs/f_traj_thal2.png'],200),

        #(trajectories_thal2b,[],['figs/f_traj_thal2b.pdf',
        #                        'figs/f_traj_thal2b.png'],200),

        #(trajectories_gt2,[],['figs/f_traj_gt2.pdf',
        #                       'figs/f_traj_gt2.png'],200),
                              
        #(fr_cgl,[],['figs/f_fr_cgl.pdf',
        #            'figs/f_fr_cgl.png'],200),

        #(fr_thal,[],['figs/f_fr_thal.pdf',
        #             'figs/f_fr_thal.png'],200),
        
        #(tongues_cgl,[],['figs/f_tongues_cgl.pdf',
        #                 'figs/f_tongues_cgl.png'],200),

        
        #(tongues_thal,[],['figs/f_tongues_thal.pdf',
        #                  'figs/f_tongues_thal.png'],200),


        #(bif1d_cgl1,[],['figs/f_bif1d_cgl1.png']),
        #(bif1d_thal1,[],['figs/f_bif1d_thal1.png','figs/f_bif1d_thal1.pdf']),
        
        #(bif1d_gt2,[],['figs/f_bif1d_gt2.png']),

        #(bif_gw2_11,[],['figs/f_bif1d_gw2_11.png']),
        #(bif_gw2_12,[],['figs/f_bif1d_gw2_12.png']),
        #(bif_gw2_23,[],['figs/f_bif1d_gw2_23.png']),

        #(bif_thal2_11,[],['figs/f_bif1d_thal2_11.png']),
        (bif_thal2_12,[],['figs/f_bif1d_thal2_12.png']),
        

    ]
    
    for fig in figures:
        generate_figure(*fig)


if __name__ == "__main__":
    
    __spec__ = None
    
    main()
