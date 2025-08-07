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

import gw1
import cgl1 as c1
#import cgl2 as c2

import thal1 as t1
import thal2 as t2

import gw2
import gwt

import gt

from lib.util import follow_phase_diffs, _get_sol, load_bif1d_f, load_phis_force
from lib.util import follow_phase_diffs_del

from lib.rhs import _full,rhs_avg_1df
from lib.functions import *
from lib.plot_util import *

from lib.rhs import _redu_c,_redu_c2

rhs_avg_1dc = _redu_c
rhs_avg_1dc2 = _redu_c2

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

labels = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P']

pd_cgl_template = {'sig':.08,'rho':.12,'mu':1,'om':1,'om_fix':1}

kw_cgl_template = {'var_names':['x','y'],
                   'init':np.array([1,0,2*np.pi]),
                   'TN':2000,
                   'idx':0,
                   'model_name':'cglf0',
                   'trunc_order':1,
                   'recompute_list':[],
                   'g_forward':False,
                   'z_forward':False,
                   'i_forward':[False,True,True,True,True,True,True],
                   'i_bad_dx':[False,True,False,False,False,False,False],
                   'max_iter':20,
                   'rtol':1e-12,
                   'atol':1e-12,
                   'rel_tol':1e-9,
                   'forcing_fn':[g1,lambda t:-20*g1(t+1)]}

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
                    'trunc_order':1,
                    'recompute_list':[],
                    'z_forward':False,
                    'i_forward':False,
                    'i_bad_dx':[False,True,False,False],
                    'max_iter':200,
                    'rtol':1e-13,
                    'atol':1e-13,
                    'rel_tol':1e-9,
                    'forcing_fn':[g1,lambda t:-20*g1(t+1)],
                    'factor':1,
                    'save_fig':False,
                    'dir_root':'jupyter/data'
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
                  'i_forward':[False,True,False,False,False,False],
                  'i_bad_dx':[False,True,False,False,False,False],
                  'max_iter':20,
                  'rtol':1e-12,
                  'atol':1e-12,
                  'rel_tol':1e-9,
                  'save_fig':False,
                  'lc_prominence':0.05,
                  'factor':0.5,
                  'dir_root':'jupyter/data'}
    
kw_sim = {'rtol':1e-7,'atol':1e-7,'method':'LSODA'}

pi_label_short = [r"$0$", r"$2\pi$"]
pi_label_half = [r"$0$", r"$\pi/2$", r"$\pi$"]
pi_label_third = [r"$0$", r"$\pi/3$", r"$2\pi/3$"]

# Marker for max frequency difference
MAX_DIFF = 0.02

# keyword for legend
kw_legend = dict(handlelength=1,borderpad=0.1,ncol=3,handletextpad=0.2,
                 columnspacing=0.1,borderaxespad=.2,labelspacing=.1,
                 frameon=True)

# line width for 3d and full
LW1D = 1
LW3D = 2.2
LWF = 2.7


# zorder for 3d and full. default matplotlib is 2 for lines
Z3D = 1
ZF = -1

# domain for 1d bifurcation diagram
DD = np.linspace(-.1,2*np.pi+.1,5000)

def _full_cgl1(t,y,a,eps=0,del1=0):
    pd1 = a.system1.pardict
    omx = a._n[1];omf = a._m[1]
    ofix = pd1['om_fix0']

    us = a.system1.forcing_fn

    u = 0
    for i in range(len(us)):
        u += eps**(i+1)*us[i](t*(del1+omf))
    
    out1 = c1.rhs_old2(t,y,pd1,'val',0) + ofix*omx*np.array([u/ofix,0])
    return np.array(list(out1))


def _cgl1_aut(t,y,a,eps=0,del1=0):
    pd1 = a.system1.pardict
    omx = a._n[1];omf = a._m[1]
    ofix = pd1['om_fix0']
    s = y[-1]

    us = a.system1.forcing_fn

    u = 0
    for i in range(len(us)):
        u += eps**(i+1)*us[i](s*(del1+omf))
    
    out1 = c1.rhs_aut(t,y,pd1,'val',0) + ofix*omx*np.array([u/ofix,0,0])
    return np.array(list(out1))


def _full_gw1(t,y,a,eps=0,del1=0):
    pd1 = a.system1.pardict
    omx = a._n[1];omf = a._m[1]
    ofix = pd1['om_fix0']
    u = a.system1.forcing_fn(t*(del1+omf))
    
    out1 = gw1.rhs(t,y,pd1,'val',0) + eps*ofix*omx*np.array([u,0,0])
    return np.array(list(out1))


def _full_thal1(t,y,a,eps=0,del1=0):
    pd1 = a.system1.pardict
    omx = a._n[1];omf = a._m[1]
    ofix = pd1['om_fix0']
    us = a.system1.forcing_fn

    u = 0
    for i in range(len(us)):
        u += eps**(i+1)*us[i](t*(del1+omf))
    
    out1 = t1.rhs(t,y,pd1,'val',0) + ofix*omx*np.array([u/ofix,0,0])
    return np.array(list(out1))

def _full_gwt(t,y,a,eps=0,del1=0):
    pd1 = a.system1.pardict;pd2 = a.system2.pardict
    y1 = y[:4];y2 = y[4:]
    
    c1 = gt.coupling_tm(y,pd1,'val',0)
    c2 = gt.coupling_gw(list(y2)+list(y1),pd2,'val',1)
    
    out1 = t2.rhs(t,y1,pd1,'val',0) + eps*(del1 + c1)
    out2 = gt.rhs(t,y2,pd2,'val',1) + eps*c2
    return np.array(list(out1)+list(out2))


def forcing_fn():

    fig,axs = plt.subplots(1,1,figsize=(4,2))

    x2 = np.linspace(0,4*np.pi,300)
    
    #g1m = lambda x:-g1(x)
    #fnlist = [gau,g1m]
    #xlist = [x1,x2]
    #title_add = ['Gaussian','Zero-Mean Periodized Gaussian']

    fnlist = [g1,lambda t:-20*g1(t+1)] # order eps and eps^2
    epslist = np.linspace(0,.1,3)

    cmap = matplotlib.colormaps['viridis']
    colors = cmap(np.linspace(0,.9,len(epslist)))

    for i,eps in enumerate(epslist):
        
        axs.plot(fnlist[0](x2)+eps*fnlist[1](x2),color=colors[i],
                 label=r'$\varepsilon={}$'.format(eps))

    axs.set_xlabel(r'$t$')
    axs.set_ylabel(r'$-p(t) + 20 \varepsilon p(t)+\bar{p}$')

    axs.margins(x=0)

    axs.legend()

    axs.set_ylim(-1,2)

    plt.tight_layout()

    return fig


kws_text = {'ha':'center','va':'center','zorder':10}
kws_ins_ticks = {'bottom':False,'left':False,'labelleft':False,'labelbottom':False}
kws_ins = {'width':'100%','height':'100%'}
kws_anno = dict(xycoords='data',textcoords='axes fraction',
                arrowprops=dict(facecolor='black',width=2,headwidth=2))

def tongues_cgl1():
    """
    2 parameter bifurcation data is from ./xpp/cgl1_f_h*.ode
    Note that these diagrams were modified after exporting from XPP,
    specifically in the order of the data, so that they could be
    more easily plotted.

    See ./bifdat_2par/reorder.ipynb for more details on what was
    included and excluded.
    """

    
    kw_cgl = copy.deepcopy(kw_cgl_template)
    kw_cgl['rhs'] = c1.rhs_old2
    
    system1 = rsp(**{'pardict':pd_cgl_template,**kw_cgl})

    pl_list = [(1,1),(2,1),(3,1),(4,1)]
    init_list = [1,1,.5,.5]
    xlims = [(-.3,.3),(-.18,.18),(-.06,.06),(-.003,.003)]
    ylims = [(0,.2),(0,.2),(0,.2),(0,.1)]
    
    d1s = [-.1,-.065,-.019,-.0013]; e1s = [.1,.1,.1,.06]
    d2s = [.05,.07,.012,.0007]; e2s = [.1,.15,.1,.06]
    d3s = [.1,.07,.02,.0015]; e3s = [.1,.1,.1,.06]
    
    Tlist1 = [200,500,1000,10000]
    Tlist2 = [2000,250,1000,5000]
    Tlist3 = [200,250,500,5000]

    axins1_bboxs = [(.1,.1,.2,.2),(.1,.1,.2,.2),(.1,.1,.2,.2),(.1,.1,.2,.2)]
    axins2_bboxs = [(.4,.7,.2,.2),(.4,.7,.2,.2),(.4,.7,.2,.2),(.4,.7,.2,.2)]
    axins3_bboxs = [(.7,.1,.2,.2),(.7,.1,.2,.2),(.7,.1,.2,.2),(.7,.1,.2,.2)]

    fig, axs = plt.subplots(2,2,figsize=(8,6))
    axs = axs.flatten()

    dir = './bifdat_2par/'
    
    for k,ax in enumerate(axs):
        n = pl_list[k][0];m = pl_list[k][1]
        T1 = Tlist1[k];T2 = Tlist2[k]; T3 = Tlist3[k]
        d1 = d1s[k]; e1 = e1s[k]
        d2 = d2s[k]; e2 = e2s[k]
        d3 = d3s[k]; e3 = e3s[k]

        a1 = nmc(system1,None,recompute_list=[],_n=('om0',n),_m=('om1',m),NH=301,del1=d1)
        a2 = nmc(system1,None,recompute_list=[],_n=('om0',n),_m=('om1',m),NH=301,del1=d2)
        a3 = nmc(system1,None,recompute_list=[],_n=('om0',n),_m=('om1',m),NH=301,del1=d3)

        axins1_bbox = axins1_bboxs[k]
        axins2_bbox = axins2_bboxs[k]
        axins3_bbox = axins3_bboxs[k]
        
        fname_o1 = dir + 'cgl1_f_h{}{}_2par_o1_fixed.dat'.format(n,m)
        fname_o2 = dir + 'cgl1_f_h{}{}_2par_fixed.dat'.format(n,m)

        dat_o1 = np.loadtxt(fname_o1)
        dat_o2 = np.loadtxt(fname_o2)

        ax.plot(dat_o1[:,0],dat_o1[:,1],color='k')
        ax.fill_between(dat_o1[:,0], dat_o1[:,1], y2=.5, facecolor='red', alpha=.25)

        ax.plot(dat_o2[:,0],dat_o2[:,1],color='k')
        ax.fill_between(dat_o2[:,0], dat_o2[:,1], y2=.5, facecolor='green', alpha=.25)

        ax.text(.8,.95,'Locking',transform=ax.transAxes,**kws_text)
        ax.text(.85,.55,'Drift',transform=ax.transAxes,**kws_text)
        ax.text(.15,.55,'Drift',transform=ax.transAxes,**kws_text)

        ax.set_xlim(*xlims[k])
        ax.set_ylim(*ylims[k])

        axins1 = inset_axes(ax,bbox_to_anchor=axins1_bbox,bbox_transform=ax.transAxes,**kws_ins)
        axins2 = inset_axes(ax,bbox_to_anchor=axins2_bbox,bbox_transform=ax.transAxes,**kws_ins)
        axins3 = inset_axes(ax,bbox_to_anchor=axins3_bbox,bbox_transform=ax.transAxes,**kws_ins)


        ################ solution 1
        args0 = [a1,e1,d1];t = np.arange(0,T1,.02);th_init = init_list[k]
        y0 = system1.lc['dat'][int((th_init/(2*np.pi))*system1.TN),:]

        solf = _get_sol(_full_cgl1,y0,t,args=args0,recompute=False)
        tp,fp = get_phase(t,solf,skipn=100,system1=system1)
        force_phase = np.mod((a1._m[1]+a1.del1)*tp,2*np.pi)
        fp2 = np.mod(fp-a1.om*force_phase,2*np.pi)

        axins1.scatter(fp2,tp,s=5,color='gray',alpha=.5,label='Full')
        axins1.set_ylim(T1,0)
        axins1.set_ylabel('$t$')

        ################ solution 2
        args0 = [a2,e2,d2];t = np.arange(0,T2,.02);th_init = init_list[k]
        y0 = system1.lc['dat'][int((th_init/(2*np.pi))*system1.TN),:]

        solf = _get_sol(_full_cgl1,y0,t,args=args0,recompute=False)
        tp,fp = get_phase(t,solf,skipn=100,system1=system1)
        force_phase = np.mod((a2._m[1]+a2.del1)*tp,2*np.pi)
        fp2 = np.mod(fp-a2.om*force_phase,2*np.pi)

        axins2.scatter(fp2,tp,s=5,color='gray',alpha=.5,label='Full')
        axins2.set_ylim(T2,0)
        axins2.set_ylabel('$t$')

        
        ################ solution 3
        args0 = [a3,e3,d3];t = np.arange(0,T3,.02);th_init = init_list[k]
        y0 = system1.lc['dat'][int((th_init/(2*np.pi))*system1.TN),:]

        solf = _get_sol(_full_cgl1,y0,t,args=args0,recompute=False)
        tp,fp = get_phase(t,solf,skipn=100,system1=system1)
        force_phase = np.mod((a3._m[1]+a3.del1)*tp,2*np.pi)
        fp2 = np.mod(fp-a3.om*force_phase,2*np.pi)

        axins3.scatter(fp2,tp,s=5,color='gray',alpha=.5,label='Full')
        axins3.set_ylim(T3,0)
        axins3.set_ylabel('$t$')


        ####### axis technicals

        ax.scatter([d1],[e1],s=100,facecolor='none',edgecolor='k')
        ax.scatter([d2],[e2],s=100,facecolor='none',edgecolor='k')
        ax.scatter([d3],[e3],s=100,facecolor='none',edgecolor='k')
        
        axins1.tick_params(**kws_ins_ticks)
        axins2.tick_params(**kws_ins_ticks)
        axins3.tick_params(**kws_ins_ticks)
        
        ax.annotate('',xy=(d1, e1),xytext=(axins1_bbox[0],axins1_bbox[1]),**kws_anno)
        ax.annotate('',xy=(d2, e2),xytext=(axins2_bbox[0],axins2_bbox[1]),**kws_anno)
        ax.annotate('',xy=(d3, e3),xytext=(axins3_bbox[0],axins3_bbox[1]),**kws_anno)

        ax.set_xlabel(r'$\delta$',labelpad=0)
        ax.set_ylabel(r'$\varepsilon$',labelpad=0)

        ax.set_title(labels[k],loc='left')
        ti1 = ax.get_title()
        ti1 += str(pl_list[k][0])+':'+str(pl_list[k][1])
        
        ax.set_title(ti1)

    plt.subplots_adjust(hspace=.3,left=.075,right=.97,bottom=.075,top=.95)
    return fig




labeltempf = [r'$\mathcal{H}$',r'$t$']
labeltempc = [r'$\mathcal{H}$',r'$t$']

def _setup_trajectories_plot(mode='f',labels=True,
                             wspace=0.1,hspace=0.075,
                             padw=0.05,padh=0.05,
                             lo=0.075,hi=0.96):
    """
    plot arranged as 0 top left, 1 top right, 2 bottom left, 3 bottom right
    each object in axs is a 2x3 set of axis objects.

    mode: 'f' is for forcing, 'c' is for coupling
    """

    if mode == 'f':
        w = 8;h = 5
    else:
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
            axs[k][1,1].tick_params(**{**kwl})

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
                axs[k][-1,j].set_xlabel(r'$\phi$',labelpad=-10)
                
                axs[k][-1,j].set_xticks([0,2*np.pi])
                axs[k][-1,j].set_xticklabels(pi_label_short)

    return fig,axs


def _setup_bif_plot(mode='f',labels=True,
                    wspace=0.1,hspace=0.07,
                    padw=0.04,padh=0.09,
                    lo=0.09,hi=0.94):
    """
    plot arranged as 0 top left, 1 top right, 2 bottom left, 3 bottom right
    each object in axs is a 2x3 set of axis objects.

    mode: 'f' is for forcing, 'c' is for coupling
    """

    if mode == 'f':
        w = 8;h = 4
    else:
        w = 8;h = 6
    nr1 = 1;nc1 = 2
    
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
        #axs[k][0,0].tick_params(**{**kwall,**kwb})
        axs[k][0,1].tick_params(**{**kwl})
            
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
                axs[k][-1,j].set_xlabel(r'$\phi$',labelpad=-10)
                
                axs[k][-1,j].set_xticks([0,2*np.pi])
                axs[k][-1,j].set_xticklabels(pi_label_short)

    return fig,axs


def draw_full_forcing_sols(axs,a,T,eps,del1,init,full_rhs,
                           label='Full',recompute:bool=False):
    
    system1 = a.system1
    kw1 = {'a':a,'return_data':False}

    # trajectory
    dt = .02;
    t = np.arange(0,T,dt)
    th_init = init

    y0 = system1.lc['dat'][int((th_init/(2*np.pi))*system1.TN),:]
    args0 = [a,eps,del1]

    solf = _get_sol(full_rhs,y0,t,args=args0,recompute=recompute)        
    tp,fp = get_phase(t,solf,skipn=200,system1=system1)
    force_phase = (a._m[1]+del1)*tp
    fp2 = np.mod(fp-a.om*force_phase,2*np.pi)
    
    # plot full sol
    axs.scatter(fp2,tp,s=5,color='k',label=label)
    axs.set_ylim(T,0)
    
    return axs

def draw_redu_rhs(axs,a,T,eps,del1,init,miter=None,color='#57acdc',
                  hline=True,label='',ls='-'):
    
    # draw 1d phase lines
    x = np.linspace(0,2*np.pi,200)
    y = rhs_avg_1df(0,x,a,eps,del1,miter)
    axs.plot(x,y,color=color,label=label,ls=ls)

    if hline:
        axs.axhline(0,x[0],x[-1],color='gray',ls=':')

    return axs

def draw_redu_forcing_phase_line(axs,a,T,eps,del1,init,miter=None):
    
    dt = .02;
    t = np.arange(0,T,dt)
    th_init = init

    args0 = [a,eps,del1,miter]
    args1 = {'t_eval':t,'t_span':[0,t[-1]],'args':(*args0,),**kw_sim}
    solr1d = solve_ivp(rhs_avg_1df,y0=[th_init],**args1)

    xs = np.mod(solr1d.y.T[:,0],2*np.pi); ys = np.zeros(len(xs))
    discont_idxs = np.abs(np.gradient(xs,1)) > np.pi/2

    xs[discont_idxs] = np.nan; ys[discont_idxs] = np.nan

    # 1d solution on phase line
    line, = axs.plot(xs,ys,color='#57acdc',alpha=.75,ls='--')
    add_arrow_to_line2D(axs,line,arrow_locs=[.5])

    return axs


def draw_redu_forcing_trajectory(axs,a,T,eps,del1,init,miter=None,color='#57acdc',
                                 label='',ls='-',arrow_locs=[.5]):

    dt = .02;
    t = np.arange(0,T,dt)
    th_init = init

    args0 = [a,eps,del1,miter]
    args1 = {'t_eval':t,'t_span':[0,t[-1]],'args':(*args0,),**kw_sim}
    solr1d = solve_ivp(rhs_avg_1df,y0=[th_init],**args1)

    # 1d solution over time
    xs = np.mod(solr1d.y.T[:,0],2*np.pi); ys = np.zeros(len(xs))
    discont_idxs = np.abs(np.gradient(xs,1)) > np.pi/2
    xs[discont_idxs] = np.nan; t[discont_idxs] = np.nan
    axs.plot(xs,t,color=color,alpha=.75,label=label,ls=ls)

    # add arrow
    line, = axs.plot(xs,t,color=color,alpha=.75,ls=ls)
    add_arrow_to_line2D(axs,line,arrow_locs=arrow_locs,arrowsize=2)
    
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
            y = rhs_avg_1df(0,x,a,eps[j],del1[j])
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

def traj_cgl1():
    """
    Plot phase plane, phase line, and phases over time
    """
    
    kw_cgl = copy.deepcopy(kw_cgl_template)
    kw_cgl['rhs'] = c1.rhs_old2
    
    system1 = rsp(**{'pardict':pd_cgl_template,**kw_cgl})

    pl_list = [(1,1),(2,1),(3,1),(4,1)]
    T_list = [500,500,1000,10000]
    e_list = [(.1,.1),(.1,.1),(.1,.1),(.06,.06)]
    d_list = [(0,.05),(.0,.025),(.0,.008),(0,7e-4)]
    init_list = [1,1,.5,.5]

    full_rhs = _full_cgl1
    fig,axs = _setup_trajectories_plot()

    for k in range(len(axs)):

        T = T_list[k]
        init = init_list[k]
        e0 = e_list[k][0]; e1 = e_list[k][1]; d0 = d_list[k][0]; d1 = d_list[k][1]
        a00 = axs[k][0,0]; a01 = axs[k][0,1]; a10 = axs[k][1,0]; a11 = axs[k][1,1]
        n = pl_list[k][0]; m = pl_list[k][1]

        # run simulations and plot
        a0 = nmc(system1,None,_n=('om0',n),_m=('om1',m),NH=502,del1=d0)
        a1 = nmc(system1,None,_n=('om0',n),_m=('om1',m),NH=502,del1=d1)

        draw_full_forcing_sols(a10,a0,T,e0,d0,init,full_rhs,recompute=False)
        draw_full_forcing_sols(a11,a1,T,e1,d1,init,full_rhs,recompute=False)

        draw_redu_rhs(a00,a0,T,e0,d0,init,label=r'$O(\varepsilon^2)$')
        draw_redu_forcing_trajectory(a10,a0,T,e0,d0,init)

        draw_redu_rhs(a00,a0,T,e0,d0,init,miter=1,color='tab:red',ls='--',
                      hline=False,label=r'$O(\varepsilon)$')
        draw_redu_forcing_trajectory(a10,a0,T,e0,d0,init,miter=1,color='tab:red',ls='--',
                                     arrow_locs=[.75])

        draw_redu_rhs(a01,a1,T,e1,d1,init)
        draw_redu_forcing_trajectory(a11,a1,T,e1,d1,init)

        draw_redu_rhs(a01,a1,T,e1,d1,init,miter=1,color='tab:red',hline=False,ls='--')
        draw_redu_forcing_trajectory(a11,a1,T,e1,d1,init,miter=1,color='tab:red',ls='--',
                                     arrow_locs=[.75])

        if k == 0:
            legend_kw = dict(fontsize=10,handletextpad=0.1,markerscale=.5,borderpad=.1,
                             handlelength=1)
            a10.legend(loc='lower left',**legend_kw)
            a00.legend(**legend_kw)

        del a0
        del a1

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


def traj_thal1():
    """
    Plot phase plane, phase line, and phases over time
    """
    
    kw_thal = copy.deepcopy(kw_thal_template)
    kw_thal['rhs'] = t1.rhs
    
    system1 = rsp(**{'pardict':pd_thal_template,**kw_thal})

    pl_list = [(1,1),(2,1),(1,2),(2,3)]
    T_list = [1000,500,5000,5000]
    e_list = [(.0225,.0225),(.026,.026),(.035,.035),(.022,.022)]
    d_list = [(.0,.04),(.0,.025),(.0,.02),(.0,.005)]
    init_list = [5,5,5,5]

    full_rhs = _full_thal1
    fig,axs = _setup_trajectories_plot(lo=0.085)

    for k in range(len(axs)):
        
        T = T_list[k]
        init = init_list[k]
        e0 = e_list[k][0]; e1 = e_list[k][1]; d0 = d_list[k][0]; d1 = d_list[k][1]
        a00 = axs[k][0,0]; a01 = axs[k][0,1]; a10 = axs[k][1,0]; a11 = axs[k][1,1]
        n = pl_list[k][0]; m = pl_list[k][1]

        # run simulations and plot
        a0 = nmc(system1,None,_n=('om0',n),_m=('om1',m),NH=1024,del1=d0)
        a1 = nmc(system1,None,_n=('om0',n),_m=('om1',m),NH=1024,del1=d1)

        draw_full_forcing_sols(a10,a0,T,e0,d0,init,full_rhs,recompute=False)
        draw_full_forcing_sols(a11,a1,T,e1,d1,init,full_rhs,recompute=False)

        draw_redu_rhs(a00,a0,T,e0,d0,init,label=r'$O(\varepsilon^2)$')
        draw_redu_forcing_trajectory(a10,a0,T,e0,d0,init)

        draw_redu_rhs(a00,a0,T,e0,d0,init,miter=1,color='tab:red',ls='--',
                      hline=False,label=r'$O(\varepsilon)$')
        draw_redu_forcing_trajectory(a10,a0,T,e0,d0,init,miter=1,color='tab:red',ls='--',
                                     arrow_locs=[.75])

        draw_redu_rhs(a01,a1,T,e1,d1,init)
        draw_redu_forcing_trajectory(a11,a1,T,e1,d1,init)

        draw_redu_rhs(a01,a1,T,e1,d1,init,miter=1,color='tab:red',hline=False,ls='--')
        draw_redu_forcing_trajectory(a11,a1,T,e1,d1,init,miter=1,color='tab:red',ls='--',
                                     arrow_locs=[.75])
        
        if k == 0:
            legend_kw = dict(fontsize=10,handletextpad=0.1,markerscale=.5,borderpad=.1,
                             handlelength=1)
            #a10.legend(loc='lower left',**legend_kw)
            a10.legend(**legend_kw)
            a00.legend(**legend_kw)

        del a0
        del a1



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
            ti1 += r', $\delta = '+str(d_list[k][j])+'$'
            
            axs[k][0,j].set_title(ti1)

    return fig


    

def traj_thal2(NH=1024):
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
    kw_thal['trunc_order'] = 1
    #kw_thal['save_fig'] = True
    kw_thal['TN'] = 20000

    pd_thal_template2 = deepcopy(pd_thal_template)
    pd_thal_template2['ib'] = 8.5
    pd_thal_template2['esyn'] = 0
    
    kw_thal['model_name'] = 'thal0_85';kw_thal['idx']=0
    system1 = rsp(**{'pardict':pd_thal_template2,**kw_thal})

    kw_thal['model_name'] = 'thal1_85';kw_thal['idx']=1
    system2 = rsp(**{'pardict':pd_thal_template2,**kw_thal})

    #pl_list = [(1,1),(2,1),(1,2),(2,3)]
    T_list = [2000,1000,1000,
              1000,1000,1000,
              1000,1000,1000]

    pl_list = [(1,1),(1,2),(2,1),
               (1,3),(3,1),(2,3),
               (3,2),(3,4),(4,3)]
    N_list = [NH]*9
    
    e_list = 9*[.1]
    b_list = [.025, -0.04, 0.027,
              0.025, 0.02, -0.005,
              0.004, 0.002, 0.0019]
    het_coeffs_list = [(1,100),(1,100),(1,100),
                       (1,100),(1,200),(1,2000),
                       (1,2000),(1,1000),(1,1000)]
    
    
    skipn_list = 9*[100]
    
    init_list = [6, 6, 6,
                 0, 2, 6,
                 6, 6, 6]
    rlist = [False,False,False,False,False,False,False,False,False]

    full_rhs = _full

    padw=0.045; padh=0.03
    lo=0.07; hi=0.96

    fig = plt.figure(figsize=(8,8))
    nr1 = 2

    kws = {'wspace':0,'hspace':0.05,'nrows':2,'ncols':1,'height_ratios':[1,2]}
    
    gs1 = fig.add_gridspec(left=lo, right=0.33-padw,bottom=0.66+padh,top=hi,**kws)
    axs1 = [fig.add_subplot(gs1[i]) for i in range(nr1)]
    gs2 = fig.add_gridspec(left=0.33+padw, right=.66-padw,bottom=0.66+padh,top=hi,**kws)
    axs2 = [fig.add_subplot(gs2[i]) for i in range(nr1)]
    gs3 = fig.add_gridspec(left=0.66+padw, right=hi,bottom=0.66+padh,top=hi,**kws)
    axs3 = [fig.add_subplot(gs3[i]) for i in range(nr1)]

    gs4 = fig.add_gridspec(left=lo, right=0.33-padw,bottom=0.33+padh,top=.66-padh,**kws)
    axs4 = [fig.add_subplot(gs4[i]) for i in range(nr1)]
    gs5 = fig.add_gridspec(left=.33+padw, right=0.66-padw,bottom=0.33+padh,top=.66-padh,**kws)
    axs5 = [fig.add_subplot(gs5[i]) for i in range(nr1)]
    gs6 = fig.add_gridspec(left=.66+padw, right=hi,bottom=0.33+padh,top=.66-padh,**kws)
    axs6 = [fig.add_subplot(gs6[i]) for i in range(nr1)]

    gs7 = fig.add_gridspec(left=lo, right=0.33-padw,bottom=lo,top=.33-padh,**kws)
    axs7 = [fig.add_subplot(gs7[i]) for i in range(nr1)]
    gs8 = fig.add_gridspec(left=.33+padw, right=0.66-padw,bottom=lo,top=.33-padh,**kws)
    axs8 = [fig.add_subplot(gs8[i]) for i in range(nr1)]
    gs9 = fig.add_gridspec(left=.66+padw, right=hi,bottom=lo,top=.33-padh,**kws)
    axs9 = [fig.add_subplot(gs9[i]) for i in range(nr1)]

    axs_list = [np.asarray(axs1),np.asarray(axs2),np.asarray(axs3),
                np.asarray(axs4),np.asarray(axs5),np.asarray(axs6),
                np.asarray(axs7),np.asarray(axs8),np.asarray(axs9)]

    kws_o1 = dict(rhs=_redu_c2,miter=1,label=r'$O(\varepsilon)$',color='tab:red',ls='--')
    kws_o2 = dict(rhs=_redu_c2,miter=2,label=r'$O(\varepsilon^2)$',lw=1.5)
    
    for k,axs in enumerate(axs_list):
        print('pl',pl_list[k],'e',e_list[k],';','d',b_list[k])
        
        n = pl_list[k][0];m = pl_list[k][1]
        NH = N_list[k]
        T = T_list[k]
        e = e_list[k]; b = b_list[k]
        init = init_list[k];skipn = skipn_list[k];
        re = rlist[k]
        het_coeffs = het_coeffs_list[k]
        
        a = nmc(system1,system2,_n=('om0',n),_m=('om1',m),NH=NH,het_coeffs=het_coeffs)

        a.system1.pardict['del0'] = b
        draw_full_solutions(axs[1],a,T,e,b,init,full_rhs=full_rhs,skipn=skipn,recompute=re)

        draw_1d_solutions(axs[1],a,T,e,b,init,arrow_locs=[0.4],**kws_o1)
        draw_1d_solutions(axs[1],a,T,e,b,init,arrow_locs=[0.6],**kws_o2)

        draw_1d_rhs(axs[0],a,e,b,**kws_o1)
        draw_1d_rhs(axs[0],a,e,b,**kws_o2)

        del a

        # axis technicals

        axs[1].set_ylim(T,0)
        
        axs[0].axhline(0,0,2*np.pi,color='gray',lw=1,ls='--')
        #a01.axhline(0,0,2*np.pi,color='gray',lw=1,ls='--')
        axs[0].set_xlim(0,2*np.pi)
        axs[1].set_xlim(0,2*np.pi)

        if k == 0:
            legend_kw = dict(fontsize=8,handletextpad=0.1,markerscale=.5,borderpad=.1,
                             handlelength=1,labelspacing=.09)
            axs[0].legend(**legend_kw)
            

        
    # set title
    ct = 0
    for k,axs in enumerate(axs_list):
        axs[0].set_title(labels[k],loc='left')
        ti1 = axs[0].get_title()
        ti1 += str(pl_list[k][0])+':'+str(pl_list[k][1])
        #ti1 += r', $\varepsilon='+str(e_list[k])+'$'
        ti1 += r', $b = '+str(b_list[k])+'$'
        
        axs[0].set_title(ti1)

        #axs[0].margins(y=0)
        axs[1].set_xlabel(r'$\phi$',labelpad=-10)
        
        axs[1].set_xticks([0,2*np.pi])
        axs[1].set_xticklabels(pi_label_short)
        
        # remove left tick label for right panel
        kwall = {'axis':'both','which':'both'}
        kwb = {'labelbottom':False,'bottom':False}
        #kwl = {'labelleft':False,'left':False}

        axs[0].tick_params(**{**kwall,**kwb})
        #axs[k][0,1].tick_params(**{**kwall,**kwl,**kwb})

        
        axs[0].set_ylabel(r'$\mathcal{H}$',labelpad=0)
        axs[1].set_ylabel(r'$t$',labelpad=0)

    return fig



def tongues_thal():
    kw_thal = copy.deepcopy(kw_thal_template)
    kw_thal['rhs'] = t1.rhs
    
    system1 = rsp(**{'pardict':pd_thal_template,**kw_thal})

    pl_list = [(1,1),(2,1),(1,2),(2,3)]
    init_list = [1,1,.5,.5]
    xlims = [(-.2,.2),(-.13,.13),(-.06,.08),(-.04,.04)]
    ylims = [(0,.075),(0,.075),(0,.075),(0,.05)]

    # inset points
    d1s = [-.05,-.05,-.025,-.01]; e1s = [.035,.035,.03,.022] # left
    d2s = [.095,.04,.018,.007]; e2s = [.05,.04,.03,.022] # middle
    d3s = [.11,.06,.03,.01]; e3s = [.05,.04,.03,.022] # right

    # note that I didn't include 1/m in front of delta, so this must be accounted for
    factors = [1,1,2,3]
    
    Tlist1 = [500,250,1000,1000]
    Tlist2 = [500,250,1000,1000]
    Tlist3 = [500,250,1000,1000]

    axins1_bboxs = [(.1,.1,.2,.2),(.1,.1,.2,.2),(.1,.1,.2,.2),(.1,.1,.2,.2)]
    axins2_bboxs = [(.4,.7,.2,.2),(.4,.7,.2,.2),(.4,.7,.2,.2),(.4,.7,.2,.2)]
    axins3_bboxs = [(.7,.1,.2,.2),(.7,.1,.2,.2),(.7,.1,.2,.2),(.7,.1,.2,.2)]

    fig, axs = plt.subplots(2,2,figsize=(8,6))
    axs = axs.flatten()

    dir = './bifdat_2par/'
    
    for k,ax in enumerate(axs):
        n = pl_list[k][0];m = pl_list[k][1]
        T1 = Tlist1[k];T2 = Tlist2[k]; T3 = Tlist3[k]
        d1 = d1s[k]; e1 = e1s[k]
        d2 = d2s[k]; e2 = e2s[k]
        d3 = d3s[k]; e3 = e3s[k]
        factor = factors[k]

        a1 = nmc(system1,None,recompute_list=[],_n=('om0',n),_m=('om1',m),NH=100,del1=d1)
        a2 = nmc(system1,None,recompute_list=[],_n=('om0',n),_m=('om1',m),NH=100,del1=d2)
        a3 = nmc(system1,None,recompute_list=[],_n=('om0',n),_m=('om1',m),NH=100,del1=d3)

        axins1_bbox = axins1_bboxs[k]
        axins2_bbox = axins2_bboxs[k]
        axins3_bbox = axins3_bboxs[k]
        
        fname_o1 = dir + 'thal1_f_h{}{}_2par_o1_fixed.dat'.format(n,m)
        fname_o2 = dir + 'thal1_f_h{}{}_2par_fixed.dat'.format(n,m)

        dat_o1 = np.loadtxt(fname_o1)
        dat_o2 = np.loadtxt(fname_o2)

        ax.plot(dat_o1[:,0],dat_o1[:,1]/factor,color='k')
        ax.fill_between(dat_o1[:,0], dat_o1[:,1]/factor, y2=.5, facecolor='red', alpha=.25)

        ax.plot(dat_o2[:,0],dat_o2[:,1]/factor,color='k')
        ax.fill_between(dat_o2[:,0], dat_o2[:,1]/factor, y2=.5, facecolor='green', alpha=.25)

        ax.text(.5,.95,'Locking',transform=ax.transAxes,**kws_text)
        ax.text(.85,.55,'Drift',transform=ax.transAxes,**kws_text)
        ax.text(.15,.55,'Drift',transform=ax.transAxes,**kws_text)

        ax.set_xlim(*xlims[k])
        ax.set_ylim(*ylims[k])

        axins1 = inset_axes(ax,bbox_to_anchor=axins1_bbox,bbox_transform=ax.transAxes,**kws_ins)
        axins2 = inset_axes(ax,bbox_to_anchor=axins2_bbox,bbox_transform=ax.transAxes,**kws_ins)
        axins3 = inset_axes(ax,bbox_to_anchor=axins3_bbox,bbox_transform=ax.transAxes,**kws_ins)

        ################ solution 1
        args0 = [a1,e1,d1];t = np.arange(0,T1,.02);th_init = init_list[k]
        y0 = system1.lc['dat'][int((th_init/(2*np.pi))*system1.TN),:]

        solf = _get_sol(_full_thal1,y0,t,args=args0,recompute=False)
        tp,fp = get_phase(t,solf,skipn=100,system1=system1)
        force_phase = np.mod((a1._m[1]+a1.del1)*tp,2*np.pi)
        fp2 = np.mod(fp-a1.om*force_phase,2*np.pi)

        axins1.scatter(fp2,tp,s=5,color='gray',alpha=.5,label='Full')
        axins1.set_ylim(T1,0)
        axins1.set_ylabel('$t$')

        ################ solution 2
        args0 = [a2,e2,d2];t = np.arange(0,T2,.02);th_init = init_list[k]
        y0 = system1.lc['dat'][int((th_init/(2*np.pi))*system1.TN),:]

        solf = _get_sol(_full_thal1,y0,t,args=args0,recompute=False)
        tp,fp = get_phase(t,solf,skipn=100,system1=system1)
        force_phase = np.mod((a2._m[1]+a2.del1)*tp,2*np.pi)
        fp2 = np.mod(fp-a2.om*force_phase,2*np.pi)

        axins2.scatter(fp2,tp,s=5,color='gray',alpha=.5,label='Full')
        axins2.set_ylim(T2,0)
        axins2.set_ylabel('$t$')
        
        ################ solution 3
        args0 = [a3,e3,d3];t = np.arange(0,T3,.02);th_init = init_list[k]
        y0 = system1.lc['dat'][int((th_init/(2*np.pi))*system1.TN),:]

        solf = _get_sol(_full_thal1,y0,t,args=args0,recompute=False)
        tp,fp = get_phase(t,solf,skipn=100,system1=system1)
        force_phase = np.mod((a3._m[1]+a3.del1)*tp,2*np.pi)
        fp2 = np.mod(fp-a3.om*force_phase,2*np.pi)

        axins3.scatter(fp2,tp,s=5,color='gray',alpha=.5,label='Full')
        axins3.set_ylim(T3,0)
        axins3.set_ylabel('$t$')

        ####### axis technicals
        axins1.set_xlim(-.1,2*np.pi+.1)
        axins2.set_xlim(-.1,2*np.pi+.1)
        axins3.set_xlim(-.1,2*np.pi+.1)

        ax.scatter([d1],[e1],s=100,facecolor='none',edgecolor='k')
        ax.scatter([d2],[e2],s=100,facecolor='none',edgecolor='k')
        ax.scatter([d3],[e3],s=100,facecolor='none',edgecolor='k')
        
        axins1.tick_params(**kws_ins_ticks)
        axins2.tick_params(**kws_ins_ticks)
        axins3.tick_params(**kws_ins_ticks)
        
        ax.annotate('',xy=(d1, e1),xytext=(axins1_bbox[0],axins1_bbox[1]),**kws_anno)
        ax.annotate('',xy=(d2, e2),xytext=(axins2_bbox[0],axins2_bbox[1]),**kws_anno)
        ax.annotate('',xy=(d3, e3),xytext=(axins3_bbox[0],axins3_bbox[1]),**kws_anno)

        ax.set_xlabel(r'$\delta$',labelpad=0)
        ax.set_ylabel(r'$\varepsilon$',labelpad=0)

        ax.set_title(labels[k],loc='left')
        ti1 = ax.get_title()
        ti1 += str(pl_list[k][0])+':'+str(pl_list[k][1])
        
        ax.set_title(ti1)

    plt.subplots_adjust(hspace=.3,left=.075,right=.97,bottom=.075,top=.95)

    return fig


def bif1d_cgl1():
    
    kw_cgl = copy.deepcopy(kw_cgl_template)
    kw_cgl['rhs'] = c1.rhs_old2
    system1 = rsp(**{'pardict':pd_cgl_template,**kw_cgl})
    
    pl_list = [(1,1),(2,1),(3,1),(4,1)]
    T_list = [500,500,1000,10000]
    e_list = [(.1,.1),(.1,.1),(.1,.1),(.06,.06)]
    d_list = [(0,.05),(0,.025),(0,.008),(0,7e-4)]
    init_list = [1,1,.5,.5]
    etup_list = [((.003,.2,100),(.0921,.2,100)),
                 ((.003,.2,100),(.0704,.2,100)),
                 ((.003,.2,100),(.0682,.2,100)),
                 ((.003,.15,100),(.047,.15,100))]
    sort_by_list = (('max','max'),('max','max'),('max','min'),('max','max'))

    # lower right-limits to prevent their labels from showing
    # (I can then use tighter margins between panels)
    xlims = [(0,.295),(0,.295),(0,.295),(0,.145)]

    full_rhs = _full_cgl1
    fig,axs = _setup_bif_plot(labels=False,hspace=.07)

    argt = {'ls':'--','color':'gray','lw':1,'clip_on':True,'zorder':-10}
    
    for k in range(len(axs)):

        e0 = e_list[k][0]; e1 = e_list[k][1]
        d0 = d_list[k][0]; d1 = d_list[k][1]
        ax0 = axs[k][0,0]; ax1 = axs[k][0,1]
        n = pl_list[k][0]; m = pl_list[k][1]
        etup0, etup1 = etup_list[k] # for full bifurcation diagram
        sort_by0 = sort_by_list[k][0];sort_by1 = sort_by_list[k][1]

        a0 = nmc(system1,None,_n=('om0',n),_m=('om1',m),NH=502,del1=d0)
        a1 = nmc(system1,None,_n=('om0',n),_m=('om1',m),NH=502,del1=d1)

        add_diagram_1d(ax0,a0,d0,etup0,rhs=rhs_avg_1df,color='#57acdc',lw=2)
        add_diagram_1d(ax0,a0,d0,etup0,rhs=rhs_avg_1df,miter=1,color='tab:red')
        
        add_diagram_1d(ax1,a1,d1,(.001,etup1[1],200),rhs=rhs_avg_1df,color='#57acdc',lw=2,
                       label=r'$O(\varepsilon^2)$')
        add_diagram_1d(ax1,a1,d1,(.001,etup1[1],200),rhs=rhs_avg_1df,miter=1,color='tab:red',
                       label=r'$O(\varepsilon)$')

        bif0 = load_diagram_full_f(a0,d0,eps_tup=etup0,dir1='bifdat2',rhs=_full_cgl1,phi0=np.pi,
                                   recompute=False,maxt=100,sort_by=sort_by0)
        bif0 = np.array(bif0)
        if k > 0: # ignore other phase diffs since they are all more or less the same.
            bif0 = bif0[:,0]
            
        ax0.plot(np.linspace(*etup0),bif0,color='k')
        
        bif1 = load_diagram_full_f(a1,d1,eps_tup=etup1,dir1='bifdat2',rhs=_full_cgl1,phi0=np.pi,
                                   recompute=False,maxt=100,sort_by=sort_by1)
        bif1 = np.array(bif1)
        if k > 0:
            bif1 = bif1[:,0]
        ax1.plot(np.linspace(*etup1),bif1,color='k',label='Full')

        # label trajectory plots
        ax0.axvline(e0,-.05,1.05,**argt)
        ax1.axvline(e1,-.05,1.05,**argt)

        if k == 0:
            legend_kw = dict(fontsize=10,handletextpad=0.2,markerscale=.5,borderpad=.2,
                             handlelength=1)
            ax1.legend(**legend_kw)
        
    
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


    # axis labels
    for i in range(len(axs)):
        for j in range(1):
            axs[i][j,0].set_ylabel(r'$\phi$',labelpad=0)
            for k in range(2):
                axs[i][j,k].set_ylim(-.1,2*np.pi+.1)
                #axs[i][j,k].set_xlim(*xlims[i])
                                
                axs[i][j,k].set_yticks([0,2*np.pi])
                axs[i][j,k].set_yticklabels(pi_label_short)

        for j in range(2):
            axs[i][-1,j].set_xlabel(r'$\varepsilon$',labelpad=0)


    return fig






def bif1d_thal1():
    
    kw_thal = copy.deepcopy(kw_thal_template)
    kw_thal['rhs'] = t1.rhs

    system1 = rsp(**{'pardict':pd_thal_template,**kw_thal})

    pl_list = [(1,1),(2,1),(1,2),(2,3)]
    T_list = [800,200,1000,1000]
    e_list = [(.0225,.0225),(.026,.026),(.035,.035),(.022,.022)]
    d_list = [(.0,.04),(.0,.025),(.0,.02),(.0,.005)]

    # for each ratio, initial conditions for each delta.
    init_list = [((0,),(0,)),
                 ((2,),(2,)),
                 ((1,2),(4,1)),
                 ((6,4,2),(6,4,2))]

    etup_list = [((.002,.075,.01),(.024,.075,.001)),
                 ((.002,.076,.001),(.024,.075,.001)),
                 ((.002,.076,.001),(.032,.076,.001)),
                 ((.002,.0501,.001),(.016,.0501,.001))]
    

    full_rhs = _full_thal1
    fig,axs = _setup_bif_plot(labels=False,hspace=.07)
    argt = {'ls':'--','color':'gray','lw':1,'clip_on':True,'zorder':-10}

    for k in range(len(axs)):

        e0 = e_list[k][0]; e1 = e_list[k][1]
        d0 = d_list[k][0]; d1 = d_list[k][1]
        ax0 = axs[k][0,0]; ax1 = axs[k][0,1]
        n = pl_list[k][0]; m = pl_list[k][1]
        etup0, etup1 = etup_list[k] # for full bifurcation diagram
        bif_init0 = init_list[k][0];bif_init1 = init_list[k][1]

        a0 = nmc(system1,None,_n=('om0',n),_m=('om1',m),NH=1024,del1=d0)
        a1 = nmc(system1,None,_n=('om0',n),_m=('om1',m),NH=1024,del1=d1)

        add_diagram_1d(ax0,a0,d0,(.001,etup0[1],500),rhs=rhs_avg_1df,color='#57acdc',lw=2)
        add_diagram_1d(ax0,a0,d0,(.001,etup0[1],500),rhs=rhs_avg_1df,miter=1,color='tab:red')
        
        add_diagram_1d(ax1,a1,d1,(.001,etup1[1],500),rhs=rhs_avg_1df,color='#57acdc',lw=2,
                       label=r'$O(\varepsilon^2)$')
        add_diagram_1d(ax1,a1,d1,(.001,etup1[1],500),rhs=rhs_avg_1df,miter=1,color='tab:red',
                       label=r'$O(\varepsilon)$')

        # full diagrams
        out0 = load_bif1d_f(_full_thal1,a0,d0,etup=etup0,phi0=bif_init0[0])
        philist0 = load_phis_force(_full_thal1,a0,d0,etup=etup0,phi0=bif_init0[0])
        ax0.plot(np.arange(*etup0),np.mod(-philist0,2*np.pi),color='k',label='Full')

        out1 = load_bif1d_f(_full_thal1,a1,d1,etup=etup1,phi0=bif_init1[0])
        philist1 = load_phis_force(_full_thal1,a1,d1,etup=etup1,phi0=bif_init1[0])
        ax1.plot(np.arange(*etup1),np.mod(-philist1,2*np.pi),color='k',label='Full')
        
        
        
        # label trajectory plots
        ax0.axvline(e0,-.05,1.05,**argt)
        ax1.axvline(e1,-.05,1.05,**argt)
        
        if k == 0:
            legend_kw = dict(fontsize=10,handletextpad=0.2,markerscale=.5,borderpad=.2,
                             handlelength=1)
            ax1.legend(**legend_kw)

    
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


    # axis labels
    for i in range(len(axs)):
        for j in range(1):
            axs[i][j,0].set_ylabel(r'$\phi$',labelpad=0)
            for k in range(2):
                axs[i][j,k].set_ylim(-.1,2*np.pi+.1)
                #axs[i][j,k].set_xlim(*xlims[i])
                                
                axs[i][j,k].set_yticks([0,2*np.pi])
                axs[i][j,k].set_yticklabels(pi_label_short)

        for j in range(2):
            axs[i][-1,j].set_xlabel(r'$\varepsilon$',labelpad=0)

    #axs[2][1,0].yaxis.labelpad=-10

    return fig


def bif_thal2a(NH=1024):

    kw_thal = copy.deepcopy(kw_thal_template)
    kw_thal['rhs'] = t2.rhs
    kw_thal['var_names'] = ['v','h','r','w']
    kw_thal['init'] = np.array([-.64,0.71,0.25,0,6.28])
    kw_thal['forcing_fn'] = None
    kw_thal['coupling'] = t2.coupling
    kw_thal['trunc_order'] = 1
    kw_thal['TN'] = 20000
    #kw_thal['save_fig'] = True

    pd_thal_template2 = deepcopy(pd_thal_template)
    pd_thal_template2['ib'] = 8.5
    
    kw_thal['model_name'] = 'thal0_85';kw_thal['idx']=0
    system1 = rsp(**{'pardict':pd_thal_template2,**kw_thal})

    kw_thal['model_name'] = 'thal1_85';kw_thal['idx']=1
    system2 = rsp(**{'pardict':pd_thal_template2,**kw_thal})

    fig,axs_all = plt.subplots(4,4,figsize=(8,7),gridspec_kw={'wspace':.5,'hspace':.6})

    # A, B, C, D
    re_list = [(False,False,False,False), #12
               (False,False,False,False), #21
               (False,False,False,False), #13
               (False,False,False,False)] #31
    pl_list = [(1,2),(2,1),(1,3),(3,1)]

    dicts_list = [
        # 1:2
        ([dict(phi0=4,eps_init=0.111,eps_final=0.001,deps=-.01),
          dict(phi0=4,eps_init=0.011,eps_final=0.001,deps=-.001)], # A
         [dict(phi0=4,eps_init=0.102,eps_final=0.001,deps=-.01)], # B
         [dict(phi0=4,del_init=0,del_final=-.035,ddel=-.001), # C
          dict(phi0=0,del_init=0,del_final=.031,ddel=.001)],
         [dict(phi0=4.,del_init=0,del_final=-.05,ddel=-.001), # D
          dict(phi0=4.,del_init=0,del_final=.035,ddel=.001)]
         ),
        
        # 2:1
        ([dict(phi0=4,eps_init=0.101,eps_final=0.01,deps=-.01),
          dict(phi0=4,eps_init=0.011,eps_final=0.001,deps=-.001)], # A
         [dict(phi0=4.5,eps_init=0.101,eps_final=0.011,deps=-.001), # B
          dict(phi0=4.5,eps_init=0.011,eps_final=0.001,deps=-.001)],
         [dict(phi0=5,del_init=0,del_final=-.042,ddel=-.003), # C
          dict(phi0=5,del_init=0,del_final=.026,ddel=.002)],
         [dict(phi0=5,del_init=0,del_final=-.055,ddel=-.005), # D
          dict(phi0=5,del_init=0,del_final=.026,ddel=.002),
          dict(phi0=5,del_init=-.01,del_final=.01,ddel=.005)]
         ),
        
        # 1:3
        ([dict(phi0=5,eps_init=0.103,eps_final=0.001,deps=-.01)], # A
         [dict(phi0=5,eps_init=0.103,eps_final=0.001,deps=-.01)], # B
         [dict(phi0=5,del_init=0,del_final=-.03,ddel=-.002), # C
          dict(phi0=5,del_init=0,del_final=.03,ddel=.002)],
         [dict(phi0=5,del_init=0,del_final=-.045,ddel=-.001), # D
          #dict(phi0=5,del_init=-.025,del_final=-.03,ddel=-.001),
          dict(phi0=5,del_init=0,del_final=.025,ddel=.001)]
         ),
        
        # 3:1
        ([dict(phi0=4,eps_init=0.101,eps_final=0.001,deps=-.01),
          dict(phi0=4,eps_init=0.021,eps_final=0.001,deps=-.002)], # A
         [dict(phi0=4,eps_init=0.101,eps_final=0.011,deps=-.01), # B
          dict(phi0=4,eps_init=0.021,eps_final=0.001,deps=-.002)],
         [dict(phi0=5,del_init=0,del_final=-.035,ddel=-.005), # C
          dict(phi0=5,del_init=0,del_final=.025,ddel=.005)],
         [dict(phi0=5,del_init=0,del_final=-.035,ddel=-.005), # D
          dict(phi0=5,del_init=0,del_final=.02,ddel=.005)]
         ),
    ]

    het_coeffs_list = [(1,100),(1,100),(1,100),(1,200)]
    
    etup_list = 4*[(0,.1,500)]
    dtup_list = [(-.05,.04,500),(-.05,.03,500),(-.045,.035,500),(-.035,.03,500)]

    del_list_all = [(-.01,-.02),(.01,.02),(.0,.02),(0,.017)]
    eps_list_all = 4*[(.05,.1)]

    sort_by = 'max'
    # for each phase-locking case, have 4 different bifurcations

    for k in range(len(pl_list)):
        n = pl_list[k][0];m = pl_list[k][1]
        het_coeffs = het_coeffs_list[k]
        in_dictsA,in_dictsB,in_dictsC,in_dictsD = dicts_list[k]
        
        del_list = del_list_all[k]
        eps_list = eps_list_all[k]
        etup = etup_list[k]
        dtup = dtup_list[k]
        re1,re2,re3,re4 = re_list[k]

        a = nmc(system1,system2,_n=('om0',n),_m=('om1',m),NH=NH,het_coeffs=het_coeffs)
        axs = axs_all[k,:]

        ################  full bifurcations
        # compute/load full diagrams for nm
        # A
        print('A')
        kwA = {'a':a,'del1':del_list[0],'recompute':re1,'bifdir':'bif1d_thal2/','_full_rhs':_full}
        dataA_list = []
        for dict1 in in_dictsA:
            dataA_list.append(follow_phase_diffs(**dict1,**kwA))

            
        # B
        print('B')
        kwB = {'a':a,'del1':del_list[1],'recompute':re2,'bifdir':'bif1d_thal2/','_full_rhs':_full}
        dataB_list = []
        for dict1 in in_dictsB:
            dataB_list.append(follow_phase_diffs(**dict1,**kwB))

        # C
        print('C')
        kwC = {'a':a,'eps':eps_list[0],'recompute':re3,'bifdir':'bif1d_thal2/','_full_rhs':_full}
        dataC_list = []
        for dict1 in in_dictsC:
            dataC_list.append(follow_phase_diffs_del(**dict1,**kwC))
            
        # D
        print('D')
        dataD_list = []
        kwD = {'a':a,'eps':eps_list[1],'recompute':re4,'bifdir':'bif1d_thal2/','_full_rhs':_full}
        for dict1 in in_dictsD:
            dataD_list.append(follow_phase_diffs_del(**dict1,**kwD))
        
        # plot full
        for data in dataA_list:
            axs[0].plot(data[:,0],data[:,-a._m[1]:],color='k',lw=3)
            
        for data in dataB_list:
            axs[1].plot(data[:,0],data[:,-a._m[1]:],color='k',lw=3)

        for data in dataC_list:
            axs[2].plot(data[:,0],data[:,-a._m[1]:],color='k',lw=3)

        for data in dataD_list:
            axs[3].plot(data[:,0],data[:,-a._m[1]:],color='k',lw=3)
        
        kws_o1 = dict(rhs=_redu_c2,miter=1,label=r'$O(\varepsilon)$',color='tab:red')
        kws_o2 = dict(rhs=_redu_c2,miter=2,label=r'$O(\varepsilon^2)$',color='#57acdc',lw=1.5)

        for j,ax in enumerate(axs):
            ax.set_ylabel(r'$\phi$')
            ax.set_ylim(-.1,2*np.pi+.1)
            ax.set_yticks(2*np.arange(0,2,1)*np.pi)
            ax.set_yticklabels(pi_label_short)
            ax.set_title(labels[4*k+j],loc='left')


        for j,ax in enumerate(axs[:2]):

            # plot 1d
            add_diagram_1d(ax,a,del_list[j],etup,**kws_o1)
            add_diagram_1d(ax,a,del_list[j],etup,**kws_o2)
            ax.set_xlabel(r'$\varepsilon$')

            # add parameter value
            tt = ax.get_title()
            tt += str(n)+':'+str(m)
            tt += r', $b='+str(del_list[j])+'$'
            ax.set_title(tt)


        for j,ax in enumerate(axs[2:]):

            add_diagram_1d_del(ax,a,eps_list[j],dtup,**kws_o1)
            add_diagram_1d_del(ax,a,eps_list[j],dtup,**kws_o2)
            ax.set_xlabel(r'$b$')

            # add parameter value
            tt = ax.get_title()
            tt += str(n)+':'+str(m)
            tt += r', $\varepsilon='+str(eps_list[j])+'$'
            ax.set_title(tt)

    plt.subplots_adjust(left=.075,right=.97,bottom=.06,top=.95)
    
    return fig


def bif_thal2b(NH=1024):

    kw_thal = copy.deepcopy(kw_thal_template)
    kw_thal['rhs'] = t2.rhs
    kw_thal['var_names'] = ['v','h','r','w']
    kw_thal['init'] = np.array([-.64,0.71,0.25,0,6.28])
    kw_thal['forcing_fn'] = None
    kw_thal['coupling'] = t2.coupling
    kw_thal['trunc_order'] = 1
    kw_thal['TN'] = 20000
    #kw_thal['save_fig'] = True

    pd_thal_template2 = deepcopy(pd_thal_template)
    pd_thal_template2['ib'] = 8.5
    
    kw_thal['model_name'] = 'thal0_85';kw_thal['idx']=0
    system1 = rsp(**{'pardict':pd_thal_template2,**kw_thal})

    kw_thal['model_name'] = 'thal1_85';kw_thal['idx']=1
    system2 = rsp(**{'pardict':pd_thal_template2,**kw_thal})

    fig,axs_all = plt.subplots(4,4,figsize=(8,7),gridspec_kw={'wspace':.5,'hspace':.6})

    # A, B, C, D
    re_list = [(False,False,False,False), #23
               (True,False,False,False), #32
               (True,False,False,False), #34
               (True,False,False,False)] #43
    pl_list = [(2,3),(3,2),(3,4),(4,3)]

    

    dicts_list = [
        # 2:3
        ([dict(phi0=5,eps_init=0.101,eps_final=0.001,deps=-.01),
          dict(phi0=5,eps_init=0.011,eps_final=0.001,deps=-.001)], # A
         [dict(phi0=5,eps_init=0.111,eps_final=0.001,deps=-.01),
          dict(phi0=5,eps_init=0.011,eps_final=0.001,deps=-.001)], # B
         [dict(phi0=5,del_init=0,del_final=-.007,ddel=-.001), # C
          dict(phi0=5,del_init=-.006,del_final=-.014,ddel=-.001),
          dict(phi0=5,del_init=0,del_final=.007,ddel=.001)],
         [dict(phi0=5,del_init=0,del_final=-.007,ddel=-.001), # D
          dict(phi0=5,del_init=-.006,del_final=-.011,ddel=-.001),
          dict(phi0=5,del_init=0,del_final=.003,ddel=.001)]
         ),
        
        # 3:2
        ([dict(phi0=5,eps_init=0.101,eps_final=0.011,deps=-.01),
          dict(phi0=5,eps_init=0.021,eps_final=0.001,deps=-.001)], # A
         [dict(phi0=5,eps_init=0.116,eps_final=0.001,deps=-.005),
          dict(phi0=5,eps_init=0.011,eps_final=0.001,deps=-.001)], # B
         [dict(phi0=5,del_init=0,del_final=-.009,ddel=-.001), # C
          dict(phi0=5,del_init=0,del_final=.006,ddel=.001)],
         [dict(phi0=5,del_init=0,del_final=-.009,ddel=-.001), # D
          dict(phi0=5,del_init=0,del_final=.0035,ddel=.0005)]),
        
        # 3:4
        ([dict(phi0=5.8,eps_init=0.103,eps_final=0.001,deps=-.002)], # A
         [dict(phi0=5.8,eps_init=0.103,eps_final=0.001,deps=-.002)], # B
         [dict(phi0=5,del_init=0,del_final=-.0025,ddel=-.0001), # C
          dict(phi0=5,del_init=0,del_final=.002,ddel=.0001)],
         [dict(phi0=3,del_init=0,del_final=-.0025,ddel=-.0001), # D
          dict(phi0=3,del_init=0,del_final=.002,ddel=.0001)]),
        
        # 4:3
        ([dict(phi0=5,eps_init=0.101,eps_final=0.069,deps=-.01),
          dict(phi0=5,eps_init=0.069,eps_final=0.001,deps=-.01)], # A
         [dict(phi0=5,eps_init=0.101,eps_final=0.001,deps=-.002)], # B
         [dict(phi0=5,del_init=0,del_final=-.0025,ddel=-.0001), # C
          dict(phi0=5,del_init=0,del_final=.002,ddel=.0001)],
         [dict(phi0=5,del_init=0,del_final=-.0025,ddel=-.0001), # D
          dict(phi0=5,del_init=0,del_final=.002,ddel=.0001)]),
    ]

    het_coeffs_list = [(1,2000),(1,2000),(1,1000),(1,1000)]

    etup_list = 4*[(0,.1,500)]
    dtup_list = [(-.015,.006,500),(-.007,.005,500),(-.003,.003,500),(-.003,.003,500)]

    del_list_all = [(0,.0025),(0,-.004),(0,.0015),(0,.0015)]
    eps_list_all = 4*[(.05,.1)]


    # for each phase-locking case, have 4 different bifurcations
    for k in range(len(pl_list)):
        n = pl_list[k][0];m = pl_list[k][1]
        het_coeffs = het_coeffs_list[k]
        in_dictsA,in_dictsB,in_dictsC,in_dictsD = dicts_list[k]
        
        del_list = del_list_all[k]
        eps_list = eps_list_all[k]
        etup = etup_list[k]
        dtup = dtup_list[k]
        re1,re2,re3,re4 = re_list[k]

        a = nmc(system1,system2,_n=('om0',n),_m=('om1',m),NH=NH,het_coeffs=het_coeffs)
        axs = axs_all[k,:]

        ################  full bifurcations
        # compute/load full diagrams for nm
        print('A')
        kwA = {'a':a,'del1':del_list[0],'recompute':re1,'bifdir':'bif1d_thal2/','_full_rhs':_full}
        dataA_list = []
        for dict1 in in_dictsA:
            dataA_list.append(follow_phase_diffs(**dict1,**kwA))

        # B
        print('B')
        kwB = {'a':a,'del1':del_list[1],'recompute':re2,'bifdir':'bif1d_thal2/','_full_rhs':_full}
        dataB_list = []
        for dict1 in in_dictsB:
            dataB_list.append(follow_phase_diffs(**dict1,**kwB))

        # C
        print('C')
        kwC = {'a':a,'eps':eps_list[0],'recompute':re3,'bifdir':'bif1d_thal2/','_full_rhs':_full}
        dataC_list = []
        for dict1 in in_dictsC:
            dataC_list.append(follow_phase_diffs_del(**dict1,**kwC))

        # D
        print('D')
        dataD_list = []
        kwD = {'a':a,'eps':eps_list[1],'recompute':re4,'bifdir':'bif1d_thal2/','_full_rhs':_full}
        for dict1 in in_dictsD:
            dataD_list.append(follow_phase_diffs_del(**dict1,**kwD))

        # plot full
        for data in dataA_list:
            axs[0].plot(data[:,0],data[:,-a._m[1]:],color='k',lw=3)
        for data in dataB_list:
            axs[1].plot(data[:,0],data[:,-a._m[1]:],color='k',lw=3)

        for data in dataC_list:
            axs[2].plot(data[:,0],data[:,-a._m[1]:],color='k',lw=3)
        for data in dataD_list:
            axs[3].plot(data[:,0],data[:,-a._m[1]:],color='k',lw=3)

        kws_o1 = dict(rhs=_redu_c2,miter=1,label=r'$O(\varepsilon)$',color='tab:red')
        kws_o2 = dict(rhs=_redu_c2,miter=2,label=r'$O(\varepsilon^2)$',color='#57acdc',lw=1.5)

        for j,ax in enumerate(axs):
            ax.set_ylabel(r'$\phi$')
            ax.set_ylim(-.1,2*np.pi+.1)
            ax.set_yticks(2*np.arange(0,2,1)*np.pi)
            ax.set_yticklabels(pi_label_short)
            ax.set_title(labels[4*k+j],loc='left')


        for j,ax in enumerate(axs[:2]):

            # plot 1d
            add_diagram_1d(ax,a,del_list[j],etup,**kws_o1)
            add_diagram_1d(ax,a,del_list[j],etup,**kws_o2)
            ax.set_xlabel(r'$\varepsilon$')

            # add parameter value and phase-locking
            tt = ax.get_title()
            tt += str(n)+':'+str(m)
            tt += r', $b='+str(del_list[j])+'$'
            ax.set_title(tt)


        for j,ax in enumerate(axs[2:]):

            add_diagram_1d_del(ax,a,eps_list[j],dtup,**kws_o1)
            add_diagram_1d_del(ax,a,eps_list[j],dtup,**kws_o2)
            ax.set_xlabel(r'$b$')

            # add parameter value
            tt = ax.get_title()
            tt += str(n)+':'+str(m)
            tt += r', $\varepsilon='+str(eps_list[j])+'$'
            ax.set_title(tt)

    plt.subplots_adjust(left=.075,right=.97,bottom=.06,top=.95)

    return fig


def bif_thal2_11(NH=1024):

    kw_thal = copy.deepcopy(kw_thal_template)
    kw_thal['rhs'] = t2.rhs
    kw_thal['var_names'] = ['v','h','r','w']
    kw_thal['init'] = np.array([-.64,0.71,0.25,0,6.28])
    kw_thal['forcing_fn'] = None
    kw_thal['coupling'] = t2.coupling
    kw_thal['trunc_order'] = 1
    kw_thal['TN'] = 20000
    #kw_thal['save_fig'] = True

    pd_thal_template2 = deepcopy(pd_thal_template)
    pd_thal_template2['ib'] = 8.5
    
    kw_thal['model_name'] = 'thal0_85';kw_thal['idx']=0
    system1 = rsp(**{'pardict':pd_thal_template2,**kw_thal})

    kw_thal['model_name'] = 'thal1_85';kw_thal['idx']=1
    system2 = rsp(**{'pardict':pd_thal_template2,**kw_thal})

    #kw1 = {'system1':system1,'system2':system2,'n':1,'m':1,'NH':NH,
    #       'del_list':del_list,'bifdir_key':'thal2'}
    #data_list, data_3d_list, obj_list = load_full_bifurcation_data(**kw1)

    
    a = nmc(system1,system2,_n=('om0',1),_m=('om1',1),NH=NH,het_coeffs=[1,100])


    ################  full bifurcations

    # A 
    kwA = {'a':a,'del1':0.01,'recompute':False,'bifdir':'bif1d_thal2/','_full_rhs':_full}
    in_dictsA = [dict(phi0=6,eps_init=0.101,eps_final=0.001,deps=-.001)]
    dataA = follow_phase_diffs(**in_dictsA[0],**kwA)

    # B
    kwB = {'a':a,'del1':0.02,'recompute':False,'bifdir':'bif1d_thal2/','_full_rhs':_full}
    in_dictsB = [dict(phi0=5.5,eps_init=0.101,eps_final=0.001,deps=-.001)]
    dataB = follow_phase_diffs(**in_dictsB[0],**kwB)

    # C
    kwC = {'a':a,'eps':0.05,'recompute':False,'bifdir':'bif1d_thal2/','_full_rhs':_full}
    in_dictsC = [dict(phi0=0.,del_init=0,del_final=-.035,ddel=-.001),
                 dict(phi0=0.,del_init=0,del_final=.025,ddel=.001)]
    dataC1 = follow_phase_diffs_del(**in_dictsC[0],**kwC)
    dataC2 = follow_phase_diffs_del(**in_dictsC[1],**kwC)
    dataC2[0][2] = 2*np.pi-.0001

    # D
    kwD = {'a':a,'eps':0.1,'recompute':False,'bifdir':'bif1d_thal2/','_full_rhs':_full}
    in_dictsD = [dict(phi0=0.,del_init=0,del_final=-.05,ddel=-.001),
                 dict(phi0=0.,del_init=0,del_final=.035,ddel=.001)]
    dataD1 = follow_phase_diffs_del(**in_dictsD[0],**kwD)
    dataD2 = follow_phase_diffs_del(**in_dictsD[1],**kwD)

    fig,axs = plt.subplots(1,4,figsize=(8,2),gridspec_kw={'wspace':.5})

            
    # plot full
    axs[0].plot(dataA[:,0],dataA[:,2],color='k',lw=3)
    axs[1].plot(dataB[:,0],dataB[:,2],color='k',lw=3)
    
    axs[2].plot(dataC1[:,0],dataC1[:,2],color='k',lw=3)
    axs[2].plot(dataC2[:,0],dataC2[:,2],color='k',lw=3)

    axs[3].plot(dataD1[:,0],dataD1[:,2],color='k',lw=3)
    axs[3].plot(dataD2[:,0],dataD2[:,2],color='k',lw=3)
    #for j in range(len(data_list[i])):
    #    data1,data2 = data_list[i][j] # data1 is periods, data2 is t_diffs

    #    for k in range(obj_list[i]._m[1]+obj_list[i]._n[1]):
    #        kwarg = dict(color='k',zorder=ZF,label='Full'*(i+j+k==0),lw=LWF)
    #        y = 2*np.pi*data2[:,k+1]/data1[:,1]
    #        axs[i].plot(data1[:,0],np.mod(y,2*np.pi),**kwarg)


    del_list = [.01,.02]
    eps_list = [.05,.1]

    etup = (0,.1,100)
    dtup = (-.045,.03,100)
         
    kws_o1 = dict(rhs=_redu_c2,miter=1,label=r'$O(\varepsilon)$',color='tab:red')
    kws_o2 = dict(rhs=_redu_c2,miter=2,label=r'$O(\varepsilon^2)$',color='#57acdc',lw=1.5)

    for k,ax in enumerate(axs):
        ax.set_ylabel(r'$\phi$')
        ax.set_ylim(-.1,2*np.pi+.1)
        ax.set_yticks(2*np.arange(0,2,1)*np.pi)
        ax.set_yticklabels(pi_label_short)
        ax.set_title(labels[k],loc='left')


    for k,ax in enumerate(axs[:2]):
            
        # plot 1d
        add_diagram_1d(ax,a,del_list[k],etup,**kws_o1)
        add_diagram_1d(ax,a,del_list[k],etup,**kws_o2)
        ax.set_xlabel(r'$\varepsilon$')
        
        # add parameter value
        tt = ax.get_title()
        tt += r'$b='+str(del_list[k])+'$'
        ax.set_title(tt)


    for k,ax in enumerate(axs[2:]):

        add_diagram_1d_del(ax,a,eps_list[k],dtup,**kws_o1)
        add_diagram_1d_del(ax,a,eps_list[k],dtup,**kws_o2)
        ax.set_xlabel(r'$b$')

        # add parameter value
        tt = ax.get_title()
        tt += r'$\varepsilon='+str(eps_list[k])+'$'
        ax.set_title(tt)
        

        """
        #axs[i].set_xlabel(r'$\varepsilon$',loc='right')
        #axs[i].xaxis.set_label_coords(.75,-.05)
        
        axs[i].set_xlabel(r'$\varepsilon$')
        axs[i].set_title(labels[i],loc='left')
        


    # add vertical lines corresponding to trajectory figure
    axs[0].axvline(0.1,ls=':',color='gray',zorder=-2)
    axs[2].axvline(0.1,ls=':',color='gray',zorder=-2)

    axs[0].legend(**kw_legend,bbox_to_anchor=(.3,.4,.4,.4),loc='center')
    """

    plt.subplots_adjust(left=.075,right=.97,bottom=.2)
    
    return fig



def bif2_thal2(NH=1024):
    

    kw_thal = copy.deepcopy(kw_thal_template)
    kw_thal['rhs'] = t2.rhs
    kw_thal['var_names'] = ['v','h','r','w']
    kw_thal['init'] = np.array([-.64,0.71,0.25,0,6.28])
    kw_thal['forcing_fn'] = None
    kw_thal['coupling'] = t2.coupling
    kw_thal['trunc_order'] = 1
    #kw_thal['save_fig'] = True
    kw_thal['TN'] = 20000

    pd_thal_template2 = deepcopy(pd_thal_template)
    pd_thal_template2['ib'] = 8.5
    pd_thal_template2['esyn'] = 0
    
    kw_thal['model_name'] = 'thal0_85';kw_thal['idx']=0
    system1 = rsp(**{'pardict':pd_thal_template2,**kw_thal})

    kw_thal['model_name'] = 'thal1_85';kw_thal['idx']=1
    system2 = rsp(**{'pardict':pd_thal_template2,**kw_thal})

    init_list = [6, 6, 6,
                 0, 2, 6,
                 6, 6, 6]

    pl_list = [(1,1),(1,2),(2,1),
               (1,3),(3,1),(2,3),
               (3,2),(3,4),(4,3)]
    N_list = [NH]*9
    
    
    skipn_list = 9*[100]
    
    init_list = [6, 6, 6, 0, 2, 6, 6, 6, 6]
    rlist = [False,False,False,False,False,False,False,False,False]

    full_rhs = _full

    padw=0.045; padh=0.03
    lo=0.07; hi=0.96

    axins1_bboxs = 9*[(.15,.3,.2,.2)]
    axins2_bboxs = 9*[(.2,.9,.2,.2)]
    axins3_bboxs = 9*[(.7,.6,.2,.2)]

    be_list_all = [((-.075,.1),(-.05,.075),(-.025,.09)), #11
                   ((-.075,.1),(-.05,.08),(-.025,.08)), #12
                   ((-.075,.08),(-.06,.06),(-.025,.08)), #21
                   
                   ((-.075,.09),(-.05,.09),(-.025,.09)), #13
                   ((-.075,.05),(-.05,.04),(-.025,.05)), #31
                   ((-.01,.05),(0,.05),(0.005,.05)), #23
                   
                   ((-.01,.05),(0,.05),(0.01,.05)), #32
                   ((-.01,.1),(-.005,.1),(0,.1)), #34
                   ((-.01,.1),(-.005,.1),(0,.1)), #43
                   
                   ]

    T_list_all = [(1000,1000,1000), (1000,1000,1000), (1000,1000,1000),
                  (1000,1000,1000), (2000,2000,2000), (1000,1000,1000),
                  (1000,1000,1000), (1000,1000,1000), (1000,1000,1000)]

    #T_list_all = [(10,10,10), (10,10,10), (10,10,10),
    #              (10,10,10), (10,10,10), (10,10,10),
    #              (10,10,10), (10,10,10), (10,10,10)]

    het_coeffs_list = [(1,100),(1,100),(1,100),
                       (1,100),(1,200),(1,2000),
                       (1,2000),(1,1000),(1,1000)]

    fig,axs = plt.subplots(3,3,figsize=(8,8))
    axs = axs.flatten()

    # created manually. twopar data from /xpp/thal2r_nm.ode
    data_dir = 'bifdat_2par/'
    kw_fill = dict(color='green',alpha=.25,edgecolor='none')

    for k in range(len(pl_list)):
        phi0 = init_list[k]
        het_coeffs = het_coeffs_list[k]
        
        n = pl_list[k][0];m = pl_list[k][1]
        T1, T2, T3 = T_list_all[k]

        a = nmc(system1,system2,_n=('om0',n),_m=('om1',m),NH=NH,het_coeffs=het_coeffs)
        
        ax = axs[k]
        axins1_bbox = axins1_bboxs[k]
        axins2_bbox = axins2_bboxs[k]
        axins3_bbox = axins3_bboxs[k]
        
        fname_raw = 'thal2r_{}{}_2par_fixed.dat'
        fname = data_dir + fname_raw.format(n,m)

        data = np.loadtxt(fname)

        split_idx = np.where(np.isnan(data[:,0]))[0][0]

        br1 = data[:split_idx]
        br2 = data[split_idx+1:]
        
        ax.plot(br1[:,0],br1[:,1],color='k')
        ax.plot(br2[:,0],br2[:,1],color='k')

        ax.fill_between(br1[:,0],br1[:,1],.2,**kw_fill)
        ax.fill_between(br2[:,0],br2[:,1],-.2,**kw_fill)

        # fill middle space
        xlo = np.max(br1[:,0]);xhi = np.min(br2[:,0])
        ax.fill_between([xlo,xhi],[-.2,-.2],.2,**kw_fill)

        ######### trajectories
        axins1 = inset_axes(ax,bbox_to_anchor=axins1_bbox,bbox_transform=ax.transAxes,**kws_ins)
        axins2 = inset_axes(ax,bbox_to_anchor=axins2_bbox,bbox_transform=ax.transAxes,**kws_ins)
        axins3 = inset_axes(ax,bbox_to_anchor=axins3_bbox,bbox_transform=ax.transAxes,**kws_ins)


        be_list = be_list_all[k]
        b1,e1 = be_list[0]
        b2,e2 = be_list[1]
        b3,e3 = be_list[2]
            
        ################ solution 1
        args0 = [a,e1,b1];t = np.arange(0,T1,.02)
        
        y0a = a.system1.lc['dat'][int((phi0/(2*np.pi)) * a.system1.TN),:]
        y0b = a.system2.lc['dat'][int((0/(2*np.pi)) * a.system2.TN),:]
        y0 = np.array([y0a,y0b]).flatten()

        solf = _get_sol(_full,y0,t,args=args0,recompute=False)
        ta, pha = get_phase(t,solf[:,:4], skipn=100,system1=a.system1)
        tb, phb = get_phase(t,solf[:,4:], skipn=100,system1=a.system2)
        
        fp2 = np.mod(pha - a.om*phb,2*np.pi)
        
        axins1.scatter(fp2,ta,s=5,color='gray',alpha=.5,label='Full')
        axins1.set_ylim(T1,0)
        axins1.set_ylabel('$t$')
        
        ################ solution 2
        args0 = [a,e2,b2];t = np.arange(0,T2,.02)
        
        y0a = a.system1.lc['dat'][int((phi0/(2*np.pi)) * a.system1.TN),:]
        y0b = a.system2.lc['dat'][int((0/(2*np.pi)) * a.system2.TN),:]
        y0 = np.array([y0a,y0b]).flatten()

        solf = _get_sol(_full,y0,t,args=args0,recompute=False)
        ta, pha = get_phase(t,solf[:,:4], skipn=100,system1=a.system1)
        tb, phb = get_phase(t,solf[:,4:], skipn=100,system1=a.system2)

        fp2 = np.mod(pha - a.om*phb,2*np.pi)

        axins2.scatter(fp2,ta,s=5,color='gray',alpha=.5,label='Full')
        axins2.set_ylim(T1,0)
        axins2.set_ylabel('$t$')



        ################ solution 3
        args0 = [a,e3,b3];t = np.arange(0,T3,.02);th_init = init_list[k]

        y0a = a.system1.lc['dat'][int((phi0/(2*np.pi)) * a.system1.TN),:]
        y0b = a.system2.lc['dat'][int((0/(2*np.pi)) * a.system2.TN),:]
        y0 = np.array([y0a,y0b]).flatten()

        solf = _get_sol(_full,y0,t,args=args0,recompute=False)
        ta, pha = get_phase(t,solf[:,:4], skipn=100,system1=a.system1)
        tb, phb = get_phase(t,solf[:,4:], skipn=100,system1=a.system2)

        fp2 = np.mod(pha - a.om*phb,2*np.pi)

        axins3.scatter(fp2,ta,s=5,color='gray',alpha=.5,label='Full')
        axins3.set_ylim(T1,0)
        axins3.set_ylabel('$t$')


        # axis technicals

        ax.scatter([b1],[e1],s=100,facecolor='none',edgecolor='k')
        ax.scatter([b2],[e2],s=100,facecolor='none',edgecolor='k')
        ax.scatter([b3],[e3],s=100,facecolor='none',edgecolor='k')

        ax.annotate('',xy=(b1, e1),xytext=(axins1_bbox[0],axins1_bbox[1]),**kws_anno)
        ax.annotate('',xy=(b2, e2),xytext=(axins2_bbox[0],axins2_bbox[1]),**kws_anno)
        ax.annotate('',xy=(b3, e3),xytext=(axins3_bbox[0],axins3_bbox[1]),**kws_anno)
                
        ax.text(.85,.05,'Locking',transform=ax.transAxes,**kws_text)
        ax.text(.9,.95,'Drift',transform=ax.transAxes,**kws_text)
        ax.text(.1,.05,'Drift',transform=ax.transAxes,**kws_text)
        
        axins1.tick_params(**kws_ins_ticks)
        axins2.tick_params(**kws_ins_ticks)
        axins3.tick_params(**kws_ins_ticks)
        
        ax.set_title(labels[k],loc='left')
        ti1 = ax.get_title()
        ti1 += str(n)+':'+str(m)
        ax.set_title(ti1)

        ax.set_xlim(-.1,.1)
        ax.set_ylim(-.2,.2)

        ax.set_ylabel(r'$\varepsilon$')
        ax.set_xlabel(r'$b$')

        axins1.set_xlim(-.1,2*np.pi+.1)
        axins2.set_xlim(-.1,2*np.pi+.1)
        axins3.set_xlim(-.1,2*np.pi+.1)


    axs[5].set_xlim(-.05,.05) # 2:3
    axs[-3].set_xlim(-.05,.05) # 3:2
    
    axs[-1].set_xlim(-.025,.025) # 4:3
    axs[-2].set_xlim(-.025,.025) # 3:4

    
        

    plt.subplots_adjust(hspace=.32,wspace=.3,left=.075,right=.97,bottom=.075,top=.95)
    return fig



def generate_figure(function, args, filenames, dpi=200):
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
        #(forcing_fn,[],['figs/f_forcing.pdf']),
        
        #(tongues_cgl1,[],['figs/f_tongues_cgl.pdf','figs/f_tongues_cgl.png']),
        #(tongues_thal,[],['figs/f_tongues_thal.pdf','figs/f_tongues_thal.png']),

        #(traj_cgl1,[],['figs/f_traj_cgl1.pdf','figs/f_traj_cgl1.png']),
        #(traj_thal1,[],['figs/f_traj_thal1.pdf','figs/f_traj_thal1.png']),
        #(traj_thal2,[1024],['figs/f_traj_thal2.pdf','figs/f_traj_thal2.png']),

        #(bif1d_cgl1,[],['figs/f_bif1d_cgl1.pdf']),
        #(bif1d_thal1,[],['figs/f_bif1d_thal1.pdf','figs/f_bif1d_thal1.png']),
        
        #(bif_thal2_11,[],['figs/f_bif1d_thal2_11.pdf']),
        #(bif_thal2a,[],['figs/f_bif1d_thal2a.pdf']),
        (bif_thal2b,[],['figs/f_bif1d_thal2b.pdf']),
        
        #(bif2_thal2,[],['figs/f_bif2_thal2.pdf']),
        
        
        
    ]
    
    for fig in figures:
        generate_figure(*fig)


if __name__ == "__main__":
    
    __spec__ = None
    
    main()
