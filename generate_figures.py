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

import thal1 as t1
import thal2 as t2
import vdp_thal

from lib.util import follow_phase_diffs, _get_sol, load_bif1d_f, load_phis_force
from lib.util import load_bif1d_f_u, load_phis_force_u
from lib.util import follow_phase_diffs_del, follow_phase_diffs_u,follow_phase_diffs_u_del


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

labels = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

pd_cgl_template = {'sig':.08,'rho':.12,'mu':1,'om':1,'om_fix':1}

def g4(x):
    return (g3(x)/2.7)**3

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
                   'forcing_fn':[lambda t: -g4(t),lambda t:20*g4(t+2)]}

pd_thal_template = {'gL':0.05,'gna':3,'gk':5,
                    'gt':5,'eL':-70,'ena':50,
                    'ek':-90,'et':0,'esyn':0,
                    'c':1,'alpha':3,'beta':2,
                    'sigmat':0.8,'vt':-20,
                    'ib':8.5,'om':1,'om_fix':1,
                    'del':0}

pd_vdp_template = {'mu':.04,'sigmat':0.1,'vt':1,
                   'alpha':3,'beta':2,'esyn':-2,
                   'del':0,'om':1,'om_fix':1}



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
                    'forcing_fn':[lambda t: -g4(t),lambda t:20*g4(t+2)],
                    'factor':1,
                    'save_fig':False,
                    'dir_root':'jupyter/data'
                    }

kw_vdp_template = {'var_names':['v','h','w'],
                   'pardict':pd_vdp_template,
                   'rhs':vdp_thal.rhs_vdp,
                   'coupling':vdp_thal.coupling_vdp,
                   'init':np.array([.32,0.128,0.54,20]),
                   'TN':10000,
                   'trunc_order':1,
                   'z_forward':False,
                   'i_forward':False,
                   'i_bad_dx':[False,True,False,False,False,False],
                   'max_iter':20,
                   'rtol':1e-10,
                   'atol':1e-10,
                   'rel_tol':1e-8,
                   'save_fig':True,
                   'lc_prominence':.05}

kw_sim = {'rtol':1e-7,'atol':1e-7,'method':'LSODA'}

pi_label_short = [r"$0$", r"$2\pi$"]
pi_label_half = [r"$0$", r"$\pi/2$", r"$\pi$"]
pi_label_third = [r"$0$", r"$\pi/3$", r"$2\pi/3$"]

# keyword for legend
kw_legend = dict(handlelength=1,borderpad=0.1,ncol=3,handletextpad=0.2,
                 columnspacing=0.1,borderaxespad=.2,labelspacing=.1,
                 frameon=True)

def _full_cgl1(t,y,a,eps=0,del1=0,c_sign=1):
    pd1 = a.system1.pardict
    omx = a._n[1];omf = a._m[1]
    ofix = pd1['om_fix0']

    us = a.system1.forcing_fn

    u = 0
    for i in range(len(us)):
        u += eps**(i+1)*us[i](t*(del1+omf))
    
    out1 = c1.rhs_old2(t,y,pd1,'val',0) + c_sign*ofix*omx*np.array([u/ofix,0])
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


def _full_thal1(t,y,a,eps=0,del1=0,c_sign=1):
    pd1 = a.system1.pardict
    omx = a._n[1];omf = a._m[1]
    ofix = pd1['om_fix0']
    us = a.system1.forcing_fn

    u = 0
    for i in range(len(us)):
        u += eps**(i+1)*us[i](t*(del1+omf))
    
    out1 = t1.rhs(t,y,pd1,'val',0) + c_sign*ofix*omx*np.array([u/ofix,0,0])
    return np.array(list(out1))


def forcing_fn():

    fig,axs = plt.subplots(1,1,figsize=(4,2))

    x2 = np.linspace(0,4*np.pi,300)
    
    #g1m = lambda x:-g1(x)
    #fnlist = [gau,g1m]
    #xlist = [x1,x2]
    #title_add = ['Gaussian','Zero-Mean Periodized Gaussian']

    fnlist = [lambda t: -g4(t),lambda t:20*g4(t+2)] # order eps and eps^2
    epslist = np.linspace(0,.1,3)

    cmap = matplotlib.colormaps['viridis']
    colors = cmap(np.linspace(0,.9,len(epslist)))

    for i,eps in enumerate(epslist):
        
        axs.plot(x2,fnlist[0](x2)+eps*fnlist[1](x2),color=colors[i],
                 label=r'$\varepsilon={}$'.format(eps))

    axs.set_xlabel(r'$t$')
    axs.set_ylabel(r'$-p(t) + 20 \varepsilon p(t+2)$')

    axs.margins(x=0)

    axs.legend(fontsize=8)

    axs.set_ylim(-1,2)

    plt.tight_layout()

    return fig


kws_text = {'ha':'center','va':'center','zorder':10}
kws_ins_ticks = {'bottom':False,'left':False,'labelleft':False,'labelbottom':False}
kws_ins = {'width':'100%','height':'100%'}
kws_anno = dict(xycoords='data',textcoords='axes fraction',
                arrowprops=dict(facecolor='black',width=2,headwidth=2))


def tongues_cgl1_v2():
    """
    * Two parameter bifurcation data of the reduced model is from ./v2_xpp/cgl1_f_h*.ode
    * cgl1_f_h*.ode contains Fourier approximations of h-functions.
    * The h-functions are computed using ./v2_juypter/bif2_cgl1_fourier.ipynb,
      which calls the nBodyCoupling code to generate the raw h functions.

    * The two parameter bifurcation data of the full model is from ./v2_xpp/cgl1f.ode
    * The full bifurcation diagram data is saved to ./v2_xpp/cgl1f_*_2par*.dat

    Note that points were reordered for easier plotting, or removed due to being numerically inaccurate.
    Only the most numerically accurate bifurcation points are shown.
    
    """

    
    kw_cgl = copy.deepcopy(kw_cgl_template)
    kw_cgl['rhs'] = c1.rhs_old2
    
    system1 = rsp(**{'pardict':pd_cgl_template,**kw_cgl})

    pl_list = [(1,1),(2,1),(3,1)]
    init_list = [1,1,.5,.5]
    xlims = [(-.15,.15),(-.075,.075),(-.002,.002),(-.0015,.0015)]
    ylims = [(0,.1),(0,.1),(0,.04),(0,.04)]
    
    d1s = [-.1,-.065,-.019,-.0013]; e1s = [.1,.1,.1,.06]
    d2s = [.05,.07,.012,.0007]; e2s = [.1,.15,.1,.06]
    d3s = [.1,.07,.02,.0015]; e3s = [.1,.1,.1,.06]
    
    Tlist1 = [200,500,1000,10000]
    Tlist2 = [2000,250,1000,5000]
    Tlist3 = [200,250,500,5000]

    axins1_bboxs = [(.1,.1,.2,.2),(.1,.1,.2,.2),(.1,.1,.2,.2),(.1,.1,.2,.2)]
    axins2_bboxs = [(.4,.7,.2,.2),(.4,.7,.2,.2),(.4,.7,.2,.2),(.4,.7,.2,.2)]
    axins3_bboxs = [(.7,.1,.2,.2),(.7,.1,.2,.2),(.7,.1,.2,.2),(.7,.1,.2,.2)]

    fig, axs = plt.subplots(1,3,figsize=(8,2))
    axs = axs.flatten()

    dir_redu = './v2_bifdat_2par/redu/'
    dir_full = './v2_bifdat_2par/full/'
    
    for k,ax in enumerate(axs):

        n = pl_list[k][0];m = pl_list[k][1]
        T1 = Tlist1[k];T2 = Tlist2[k]; T3 = Tlist3[k]
        d1 = d1s[k]; e1 = e1s[k]
        d2 = d2s[k]; e2 = e2s[k]
        d3 = d3s[k]; e3 = e3s[k]

        axins1_bbox = axins1_bboxs[k]
        axins2_bbox = axins2_bboxs[k]
        axins3_bbox = axins3_bboxs[k]

        # load h function 2 par
        fname_o1_p = dir_redu + 'cgl1f_{}{}_o1_pos.dat'.format(n,m)
        fname_o1_n = dir_redu + 'cgl1f_{}{}_o1_neg.dat'.format(n,m)

        fname_o2_p = dir_redu + 'cgl1f_{}{}_o2_pos_fixed.dat'.format(n,m)
        fname_o2_n = dir_redu + 'cgl1f_{}{}_o2_neg_fixed.dat'.format(n,m)

        dat_o1_p = np.loadtxt(fname_o1_p)
        dat_o1_n = np.loadtxt(fname_o1_n)

        dat_o2_p = np.loadtxt(fname_o2_p)
        dat_o2_n = np.loadtxt(fname_o2_n)

        ax.plot(dat_o1_p[:,1],dat_o1_p[:,0],color='tab:red',label='Order 1',ls='--')
        ax.plot(dat_o1_n[:,1],dat_o1_n[:,0],color='tab:red',ls='--')
        #ax.fill_between(dat_o1[:,0], dat_o1[:,1], y2=.5, facecolor='red', alpha=.25)

        ax.plot(dat_o2_p[:,1],dat_o2_p[:,0],color='#57acdc',label='Order 2')
        ax.plot(dat_o2_n[:,1],dat_o2_n[:,0],color='#57acdc')
        #ax.fill_between(dat_o2[:,0], dat_o2[:,1], y2=.5, facecolor='green', alpha=.25)

        # load data full
        twopar1 = np.loadtxt(dir_full+'cgl1f_{}{}_pos_fixed.dat'.format(n,m))
        twopar2 = np.loadtxt(dir_full+'cgl1f_{}{}_neg_fixed.dat'.format(n,m))

        twopar_eps1 = twopar1[:,0];twopar_del1 = twopar1[:,1]
        twopar_eps2 = twopar2[:,0];twopar_del2 = twopar2[:,1]

        ax.plot(twopar_del1,twopar_eps1,color='k',label='Full')
        ax.plot(twopar_del2,twopar_eps2,color='k')


        #ax.text(.8,.95,'Locking',transform=ax.transAxes,**kws_text)
        #ax.text(.85,.55,'Drift',transform=ax.transAxes,**kws_text)
        #ax.text(.15,.55,'Drift',transform=ax.transAxes,**kws_text)

        ax.set_xlim(*xlims[k])
        ax.set_ylim(*ylims[k])

        #axins1 = inset_axes(ax,bbox_to_anchor=axins1_bbox,bbox_transform=ax.transAxes,**kws_ins)
        #axins2 = inset_axes(ax,bbox_to_anchor=axins2_bbox,bbox_transform=ax.transAxes,**kws_ins)
        #axins3 = inset_axes(ax,bbox_to_anchor=axins3_bbox,bbox_transform=ax.transAxes,**kws_ins)

        ####### axis technicals

        #ax.scatter([d1],[e1],s=100,facecolor='none',edgecolor='k')
        #ax.scatter([d2],[e2],s=100,facecolor='none',edgecolor='k')
        #ax.scatter([d3],[e3],s=100,facecolor='none',edgecolor='k')

        #axins1.tick_params(**kws_ins_ticks)
        #axins2.tick_params(**kws_ins_ticks)
        #axins3.tick_params(**kws_ins_ticks)

        #ax.annotate('',xy=(d1, e1),xytext=(axins1_bbox[0],axins1_bbox[1]),**kws_anno)
        #ax.annotate('',xy=(d2, e2),xytext=(axins2_bbox[0],axins2_bbox[1]),**kws_anno)
        #ax.annotate('',xy=(d3, e3),xytext=(axins3_bbox[0],axins3_bbox[1]),**kws_anno)

        ax.set_xlabel(r'$\delta$',labelpad=0)
        ax.set_ylabel(r'$\varepsilon$',labelpad=0)

        ax.set_title(labels[k],loc='left')
        ti1 = ax.get_title()
        ti1 += str(pl_list[k][0])+':'+str(pl_list[k][1])

        ax.set_title(ti1)

    
    kw_legend2 = dict(handlelength=1,borderpad=0.2,handletextpad=0.2,
                      borderaxespad=.2,labelspacing=.1,frameon=True)

    axs[0].legend(**kw_legend2)


    plt.subplots_adjust(left=.075,right=.97,bottom=.18,top=.89,wspace=.3)

    return fig



def tongues_thal1_v2():
    """
    * Two parameter bifurcation data of the reduced model is from ./v2_xpp/thal1_f_h*.ode
    * thal1_f_h*.ode contains Fourier approximations of h-functions.
    * The h-functions are computed using ./v2_juypter/bif2_thal1_fourier.ipynb,
      which calls the nBodyCoupling code to generate the raw h functions.

    * The two parameter bifurcation data of the full model is from ./v2_xpp/thal1f.ode
    * The full bifurcation diagram data is saved to ./v2_xpp/thal1f_*_2par*.dat

    Note that points were reordered for easier plotting, or removed due to being numerically inaccurate.
    Only the most numerically accurate bifurcation points are shown.
    
    """

    
    kw_thal = copy.deepcopy(kw_thal_template)
    kw_thal['rhs'] = t1.rhs
    
    system1 = rsp(**{'pardict':pd_thal_template,**kw_thal})

    pl_list = [(1,1),(2,1),(1,2),(2,3)]
    init_list = [1,1,.5,.5]
    xlims = [(-.2,.2),(-.13,.13),(-.08,.08),(-.05,.02)]
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

    fig, axs = plt.subplots(1,4,figsize=(8,2))
    axs = axs.flatten()

    dir_redu = './v2_bifdat_2par/redu/'
    dir_full = './v2_bifdat_2par/full/'
    
    for k,ax in enumerate(axs[:4]):

        n = pl_list[k][0];m = pl_list[k][1]
        T1 = Tlist1[k];T2 = Tlist2[k]; T3 = Tlist3[k]
        d1 = d1s[k]; e1 = e1s[k]
        d2 = d2s[k]; e2 = e2s[k]
        d3 = d3s[k]; e3 = e3s[k]

        axins1_bbox = axins1_bboxs[k]
        axins2_bbox = axins2_bboxs[k]
        axins3_bbox = axins3_bboxs[k]

        # load h function 2 par

        if (k == 2) or (k == 3):
            suff = '_fixed'
        else:
            suff = ''

        fname_o1_p = dir_redu + ('thal1f_{}{}_o1_pos'+suff+'.dat').format(n,m)
        fname_o1_n = dir_redu + ('thal1f_{}{}_o1_neg'+suff+'.dat').format(n,m)

        dat_o1_p = np.loadtxt(fname_o1_p)
        dat_o1_n = np.loadtxt(fname_o1_n)
        
        ax.plot(dat_o1_p[:,1],dat_o1_p[:,0],color='tab:red',label='Order 1',ls='--')
        ax.plot(dat_o1_n[:,1],dat_o1_n[:,0],color='tab:red',ls='--')
        #ax.fill_between(dat_o1[:,0], dat_o1[:,1], y2=.5, facecolor='red', alpha=.25)

        
        fname_o2_p = dir_redu + 'thal1f_{}{}_o2_pos_fixed.dat'.format(n,m)
        fname_o2_n = dir_redu + 'thal1f_{}{}_o2_neg_fixed.dat'.format(n,m)

        dat_o2_p = np.loadtxt(fname_o2_p)
        dat_o2_n = np.loadtxt(fname_o2_n)

        ax.plot(dat_o2_p[:,1],dat_o2_p[:,0],color='#57acdc',label='Order 2')
        ax.plot(dat_o2_n[:,1],dat_o2_n[:,0],color='#57acdc')
        #ax.fill_between(dat_o2[:,0], dat_o2[:,1], y2=.5, facecolor='green', alpha=.25)

        # load data full

        twopar1 = np.loadtxt(dir_full+'thal1f_{}{}_pos_fixed.dat'.format(n,m))
        twopar2 = np.loadtxt(dir_full+'thal1f_{}{}_neg_fixed.dat'.format(n,m))

        twopar_eps1 = twopar1[:,0];twopar_del1 = twopar1[:,1]
        twopar_eps2 = twopar2[:,0];twopar_del2 = twopar2[:,1]

        factor = 1
        if k == 2:
            factor = 1/2
        if k == 3:
            factor = 1/3
        # included a division by _m[1] in the reduction but not in the full model.
        
        ax.plot(twopar_del1*factor,twopar_eps1,color='k',label='Full')
        ax.plot(twopar_del2*factor,twopar_eps2,color='k')
        
        #ax.text(.8,.95,'Locking',transform=ax.transAxes,**kws_text)
        #ax.text(.85,.55,'Drift',transform=ax.transAxes,**kws_text)
        #ax.text(.15,.55,'Drift',transform=ax.transAxes,**kws_text)

        ax.set_xlim(*xlims[k])
        ax.set_ylim(*ylims[k])

        #axins1 = inset_axes(ax,bbox_to_anchor=axins1_bbox,bbox_transform=ax.transAxes,**kws_ins)
        #axins2 = inset_axes(ax,bbox_to_anchor=axins2_bbox,bbox_transform=ax.transAxes,**kws_ins)
        #axins3 = inset_axes(ax,bbox_to_anchor=axins3_bbox,bbox_transform=ax.transAxes,**kws_ins)

        ####### axis technicals

        #ax.scatter([d1],[e1],s=100,facecolor='none',edgecolor='k')
        #ax.scatter([d2],[e2],s=100,facecolor='none',edgecolor='k')
        #ax.scatter([d3],[e3],s=100,facecolor='none',edgecolor='k')

        #axins1.tick_params(**kws_ins_ticks)
        #axins2.tick_params(**kws_ins_ticks)
        #axins3.tick_params(**kws_ins_ticks)

        #ax.annotate('',xy=(d1, e1),xytext=(axins1_bbox[0],axins1_bbox[1]),**kws_anno)
        #ax.annotate('',xy=(d2, e2),xytext=(axins2_bbox[0],axins2_bbox[1]),**kws_anno)
        #ax.annotate('',xy=(d3, e3),xytext=(axins3_bbox[0],axins3_bbox[1]),**kws_anno)

        ax.set_xlabel(r'$\delta$',labelpad=0)
        ax.set_ylabel(r'$\varepsilon$',labelpad=0)

        ax.set_title(labels[k],loc='left')
        ti1 = ax.get_title()
        ti1 += str(pl_list[k][0])+':'+str(pl_list[k][1])

        ax.set_title(ti1)

    
    kw_legend2 = dict(handlelength=1,borderpad=0.2,handletextpad=0.2,
                      borderaxespad=.2,labelspacing=.1,frameon=True)

    axs[0].legend(**kw_legend2)


    plt.subplots_adjust(left=.075,right=.97,bottom=.18,top=.89,wspace=.3)

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



def _setup_trajectories_plot_1x3(mode='f',labels=True,wspace=0.1,hspace=0.3,
                                padw=0.05,padh=0.1,lo=0.075,hi=0.96):
    """
    plot arranged as 0 top left, 1 top right, 2 bottom left, 3 bottom right
    each object in axs is a 2x3 set of axis objects.

    mode: 'f' is for forcing, 'c' is for coupling
    """

    if mode == 'f':
        w = 8;h = 6
    else:
        w = 8;h = 6
    nr1 = 5;nc1 = 1
    
    fig = plt.figure(figsize=(w,h))

    kws = {'wspace':wspace,'hspace':hspace,'nrows':nr1,'ncols':nc1,
           'height_ratios':[1,1,.2,1,1]}

    gs1 = fig.add_gridspec(left=lo, right=0.33-padw,bottom=0.+padh,top=hi,**kws)
    axs1 = [fig.add_subplot(gs1[i]) for i in range(nr1)]

    gs2 = fig.add_gridspec(left=0.33+padw, right=0.66-padw,bottom=0.+padh,top=hi,**kws)
    axs2 = [fig.add_subplot(gs2[i]) for i in range(nr1)]

    gs3 = fig.add_gridspec(left=0.66+padw, right=hi-padw,bottom=0+padh,top=hi,**kws)
    axs3 = [fig.add_subplot(gs3[i]) for i in range(nr1)]

    axs = [np.asarray(axs1),np.asarray(axs2),np.asarray(axs3)]

    # remove left tick label for right panel
    kwall = {'axis':'both','which':'both'}
    kwb = {'labelbottom':False,'bottom':False}
    kwl = {'labelleft':False,'left':False}
    
    for k in range(len(axs)):
        #axs[k][0].tick_params(**{**kwall,**kwb})
        #axs[k][2].tick_params(**{**kwall,**kwl,**kwb})        

        for i in range(nr1):
            if i != 2:
                axs[k][i].margins(x=0)
                axs[k][i].set_xlim(0,2*np.pi)
                axs[k][i].set_xlabel(r'$\phi$',labelpad=-10)
                axs[k][i].set_xticks([0,2*np.pi])
                axs[k][i].set_xticklabels(pi_label_short)

        # erasee invis subplot
        axs[k][2].tick_params(**{**kwall,**kwb,**kwl})
        axs[k][2].spines['top'].set_visible(False)
        axs[k][2].spines['right'].set_visible(False)
        axs[k][2].spines['bottom'].set_visible(False)
        axs[k][2].spines['left'].set_visible(False)

        axs[k][2].zorder=-1
        
        
                
        """
        for j in range(nc1):

            if labels:
                axs[k][-1,j].margins(y=0)
                axs[k][-1,j].set_xlabel(r'$\phi$',labelpad=-10)
                
                
                
        """
    return fig,axs


def _setup_bif_plot(mode='f',labels=True,wspace=0.1,hspace=0.07,
                    padw=0.04,padh=0.09,lo=0.09,hi=0.94,
                    nr1 = 1,nc1 = 2):
    """
    plot arranged as 0 top left, 1 top right, 2 bottom left, 3 bottom right
    each object in axs is a 2x3 set of axis objects.

    mode: 'f' is for forcing, 'c' is for coupling
    """

    if mode == 'f':
        w = 8;h = 4
    else:
        w = 8;h = 6
    
    
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

def _setup_bif_plot_1x3(mode='f',labels=True,wspace=0.1,hspace=0.07,
                        padw=0.04,padh=0.09,lo=0.06,hi=0.94):
    """
    plot arranged as 0 top left, 1 top right, 2 bottom left, 3 bottom right
    each object in axs is a 2x3 set of axis objects.

    mode: 'f' is for forcing, 'c' is for coupling
    """

    if mode == 'f':
        w = 8;h = 4
    else:
        w = 8;h = 6
    nr1 = 2;nc1 = 1
    
    fig = plt.figure(figsize=(w,h))

    kws = {'wspace':wspace,'hspace':hspace,'nrows':nr1,'ncols':nc1}

    gs1 = fig.add_gridspec(left=lo, right=0.33-padw,bottom=0.+padh,top=hi,**kws)
    axs1 = [fig.add_subplot(gs1[i]) for i in range(nr1)]

    gs2 = fig.add_gridspec(left=0.33+padw, right=0.66-padw,bottom=0.+padh,top=hi,**kws)
    axs2 = [fig.add_subplot(gs2[i]) for i in range(nr1)]

    gs3 = fig.add_gridspec(left=0.66+padw, right=hi,bottom=0+padh,top=hi,**kws)
    axs3 = [fig.add_subplot(gs3[i]) for i in range(nr1)]

    axs = [np.asarray(axs1),np.asarray(axs2),np.asarray(axs3)]

    # remove left tick label for right panel
    kwall = {'axis':'both','which':'both'}
    kwb = {'labelbottom':False,'bottom':False}
    kwl = {'labelleft':False,'left':False}
    
    for k in range(len(axs)):
        pass
        #axs[k][0,0].tick_params(**{**kwall,**kwb})
        #axs[k][0,1].tick_params(**{**kwl})

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

    pl_list = [(1,1),(2,1),(3,1)]
    T_list = [1000,1000,4000]
    e_list = [(.04,.04),(.04,.04),(.015,.015)]
    d_list = [(0,.02),(.0,.01),(.0,.001)]
    init_list = [1,1,.5,.5]

    full_rhs = _full_cgl1
    fig,axs = _setup_trajectories_plot_1x3()

    for k in range(len(axs)):

        T = T_list[k]
        init = init_list[k]
        e0 = e_list[k][0]; e1 = e_list[k][1]; d0 = d_list[k][0]; d1 = d_list[k][1]
        a00 = axs[k][0]; a10 = axs[k][1]; a20 = axs[k][3]; a30 = axs[k][4]
        
        n = pl_list[k][0]; m = pl_list[k][1]

        # run simulations and plot
        a0 = nmc(system1,None,_n=('om0',n),_m=('om1',m),NH=502)
        a1 = nmc(system1,None,_n=('om0',n),_m=('om1',m),NH=502)

        draw_full_forcing_sols(a10,a0,T,e0,d0,init,full_rhs,recompute=False)
        draw_full_forcing_sols(a30,a1,T,e1,d1,init,full_rhs,recompute=False)

        draw_redu_rhs(a00,a0,T,e0,d0,init,label=r'$O(\varepsilon^2)$')
        draw_redu_forcing_trajectory(a10,a0,T,e0,d0,init)

        draw_redu_rhs(a00,a0,T,e0,d0,init,miter=1,color='tab:red',ls='--',
                      hline=False,label=r'$O(\varepsilon)$')
        draw_redu_forcing_trajectory(a10,a0,T,e0,d0,init,miter=1,color='tab:red',ls='--',
                                     arrow_locs=[.75])

        draw_redu_rhs(a20,a1,T,e1,d1,init)
        draw_redu_forcing_trajectory(a30,a1,T,e1,d1,init)

        draw_redu_rhs(a20,a1,T,e1,d1,init,miter=1,color='tab:red',hline=False,ls='--')
        draw_redu_forcing_trajectory(a30,a1,T,e1,d1,init,miter=1,color='tab:red',ls='--',
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
        axs[k][0].set_title(labels[ct],loc='left')
        ct += 1

        axs[k][3].set_title(labels[ct],loc='left')
        ct += 1

    # fix title with parameter values
    nr1 = axs[0].shape
    for k in range(len(axs)):

        axs[k][0].set_ylabel(labeltempf[0],labelpad=0)
        axs[k][1].set_ylabel(labeltempf[1],labelpad=0)

        axs[k][3].set_ylabel(labeltempf[0],labelpad=0)
        axs[k][4].set_ylabel(labeltempf[1],labelpad=0)
        
        for jj,j in enumerate([0,3]):
            ti1 = axs[k][j].get_title()
            ti1 += str(pl_list[k][0])+':'+str(pl_list[k][1])
            #t1 += r', $\varepsilon='+str(e_list[k])+'$'
            ti1 += r', $\delta = '+str(d_list[k][jj])+'$'

            
            
            axs[k][j].set_title(ti1)

    
    return fig


def traj_thal1():
    """
    Plot phase plane, phase line, and phases over time
    """
    
    kw_thal = copy.deepcopy(kw_thal_template)
    kw_thal['rhs'] = t1.rhs
    
    system1 = rsp(**{'pardict':pd_thal_template,**kw_thal})

    pl_list = [(1,1),(2,1),(1,2),(2,3)]
    T_list = [500,500,500,1000]
    e_list = [(.03,.03),(.02,.02),(.04,.04),(.025,.025)]
    d_list = [(.0,.03),(-.02,.0),(.0,-.05),(-.05,-.07)]
    init_list = [5,5,5,5]

    full_rhs = _full_thal1
    fig,axs = _setup_trajectories_plot(lo=0.085)

    for k in range(len(axs)):
            
        
        T = T_list[k]
        init = init_list[k]
        e0 = e_list[k][0]; e1 = e_list[k][1]; d0 = d_list[k][0]; d1 = d_list[k][1]
        a00 = axs[k][0,0]; a01 = axs[k][0,1]; a10 = axs[k][1,0]; a11 = axs[k][1,1]
        n = pl_list[k][0]; m = pl_list[k][1]

        
        #if k == 3:
        #    d0 *= m;d1 *= m

        # run simulations and plot
        a0 = nmc(system1,None,_n=('om0',n),_m=('om1',m),NH=1024)
        a1 = nmc(system1,None,_n=('om0',n),_m=('om1',m),NH=1024)

        draw_full_forcing_sols(a10,a0,T,e0,d0,init,full_rhs,recompute=True)
        draw_full_forcing_sols(a11,a1,T,e1,d1,init,full_rhs,recompute=True)

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
            dd = d_list[k][j]
            ti1 += r', $\delta = '+str(np.round(dd/pl_list[k][1],3))+'$'


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


def bif1d_cgl1():
    
    kw_cgl = copy.deepcopy(kw_cgl_template)
    kw_cgl['rhs'] = c1.rhs_old2
    system1 = rsp(**{'pardict':pd_cgl_template,**kw_cgl})
    
    pl_list = [(1,1),(2,1),(3,1)]
    e_list = [(.04,.04),(.04,.04),(.015,.015)]
    d_list = [(0,.02),(.0,.01),(.0,.001)]
    
    etup_list = [((.003,.11,.001),(.02,.11,.001)),
                 ((.003,.11,.001),(.02,.11,.001)),
                 ((.001,.11,.001),(.017,.11,.0002)),]
    sort_by_list = (('max','max'),('max','max'),('max','min'),('max','max'))

    # lower right-limits to prevent their labels from showing
    # (I can then use tighter margins between panels)
    xlims = [(0,.295),(0,.295),(0,.295),(0,.145)]

    full_rhs = _full_cgl1
    fig,axs = _setup_bif_plot_1x3(labels=False,hspace=.4)

    for k in range(len(axs)):
        sort_by0 = sort_by_list[k][0];sort_by1 = sort_by_list[k][1]
        e0 = e_list[k][0]; e1 = e_list[k][1]; d0 = d_list[k][0]; d1 = d_list[k][1]
        ax0 = axs[k][0]; ax1 = axs[k][1]
        etup0, etup1 = etup_list[k] # for full bifurcation diagram
        n = pl_list[k][0]; m = pl_list[k][1]

        a = nmc(system1,None,_n=('om0',n),_m=('om1',m),NH=502)

        a.del1 = d0
        add_diagram_1d(ax0,a,d0,(.001,etup1[1],200),rhs=rhs_avg_1df,color='#57acdc',lw=2)
        add_diagram_1d(ax0,a,d0,(.001,etup1[1],200),rhs=rhs_avg_1df,miter=1,color='tab:red')

        a.del1 = d1
        add_diagram_1d(ax1,a,d1,(.001,etup1[1],200),rhs=rhs_avg_1df,color='#57acdc',lw=2,
                       label=r'$O(\varepsilon^2)$')
        add_diagram_1d(ax1,a,d1,(.001,etup1[1],200),rhs=rhs_avg_1df,miter=1,color='tab:red',
                       label=r'$O(\varepsilon)$')

        
        
        a.del1 = d0
        out = load_bif1d_f(_full_cgl1,a,d0,etup=etup0,max_iter=20,recompute=False)
        erange = out[:,0]
        philist = load_phis_force(_full_cgl1,a,d0,etup=etup0,period_multiple=10,recompute=False)
        y = np.mod(-philist,2*np.pi)
        x = erange
        
        
        threshold = 0.5 
        discont_indices = np.where(np.abs(np.diff(y)) >= threshold)[0] + 1
        x_discontinuous = np.insert(x, discont_indices, np.nan)
        y_discontinuous = np.insert(y, discont_indices, np.nan)
        
        ax0.plot(x_discontinuous,y_discontinuous,color='k',lw=3,zorder=-1)

        if k == 2:
            phi0 = 3
        else:
            phi0 = 0
        a.del1 = d1
        out = load_bif1d_f(_full_cgl1,a,d1,etup=etup1,max_iter=20,phi0=phi0,recompute=False)
        erange = out[:,0]
        philist = load_phis_force(_full_cgl1,a,d1,etup=etup1,phi0=phi0,period_multiple=10,recompute=False)
        
        y = np.mod(-philist,2*np.pi)
        x = erange
        
        threshold = 0.5 
        discont_indices = np.where(np.abs(np.diff(y)) >= threshold)[0] + 1
        x_discontinuous = np.insert(x, discont_indices, np.nan)
        y_discontinuous = np.insert(y, discont_indices, np.nan)
        
        ax1.plot(x_discontinuous,y_discontinuous,color='k',lw=3,zorder=-1)


    # thease are obtained from v2_jupyter/bif_cgl1.ipynb
    #eps and phi0 for d=0
    ephi_list_u1 = [
        (
            [(0.72964387, 0.23887643),(.1,0,-.001)], # 1:1 b=0
        ),
        (
            [(0.19557787, 0.92066561),(.04,.001,-.001)], # 2:1 b=0
            [(0.19557787, 0.92066561),(.0399,.101,.001)]
        ),
        (
            [(0.49970307, 0.853852),(.001,.003,.001)], # 3:1 b=0
            [(0.45737233, 0.87034769),(.003,.01,.001)],
            [(0.40150308, 0.8906003),(.01,.015,.001)],
            [(0.21534335, 0.93973195),(.014,.02,.001)],
            [(0.30610986, 0.91958616),(0.019, 0.025, 0.001)],
            [(0.21534335, 0.93973195),(0.024, 0.03, 0.001)],
            [(0.21534335, 0.93973195),(.029,.037,.001)],
            [(-0.03212262,  0.95613382),(.036,.045,.001)],
            [(-0.49580001,  0.82118073),(.044,.055,.001)],
            [(-0.96305915,  0.24245608),(.054,.07,.001)],
            [(-0.99552177, -0.26096216), (.069,.11,.001)]
        ),
    ]
    
    #eps and phi0 for d!=0
    ephi_list_u2 = [
        (
            [(0.59423526, 0.78924236),(.05,.001,-.001)], # 1:1 b=.02
            [(0.59423526, 0.78924236),(.05,.11,.001)],
        ),
        (
            [(-0.11149186, 0.93852517) ,(.05,.03,-.001)], # 2:1 b=.01
            [(-0.11149186, 0.93852517),(.0499,.101,.001)]
        ), #(-0.16936787, 0.94961894)
        (
            [(-1.01385148, -0.07309835),(.054,.058,.001)], # 3:1 b=.01
            [(-0.92385795, -0.48841295),(0.057, 0.065, 0.001)],
            [(-0.89258014, -0.56352711),(.069,.08,.001)],
            [(-0.89258014, -0.56352711),(.079,.101,.001)],
            [(-0.93976493, -0.44691321),(0.0541, 0.05, -0.001)],
            [(-0.96122647, -0.38468592),(.0501,.046,-.001)],
            [(-0.99240274, -0.26640006),(.047,.046,-.0005)],
            [(-0.99325916, -0.26221709),(0.046, 0.043, -0.0002)],
            [(-1.01338126, -0.09982719),(.043,.04,-.0002)],
            [(-1.00297001,  0.10129733),(0.04, 0.035, -0.0002)],
            [(-0.89161303,  0.43677825),(.035,.03,-.0002)],
            [(-0.74535968,  0.6455293),(.03,.025,-.0002)],
            [(-0.66604341,  0.72715829),(.025,.02,-.0002)],
            [(-0.69738279,  0.70371388),(.02,.015,-.0002)]
        ), #(-0.16936787, 0.94961894)
        
        
    ]

    
    # add unstable points
    for k in range(len(axs)):
        sort_by0 = sort_by_list[k][0];sort_by1 = sort_by_list[k][1]
        e0 = e_list[k][0]; e1 = e_list[k][1]; d0 = d_list[k][0]; d1 = d_list[k][1]
        ax0 = axs[k][0]; ax1 = axs[k][1]
        etup0, etup1 = etup_list[k] # for full bifurcation diagram
        n = pl_list[k][0]; m = pl_list[k][1]

        a = nmc(system1,None,_n=('om0',n),_m=('om1',m),NH=502)

        for kk,list1 in enumerate(ephi_list_u1[k]):
            
            init0,etup_temp = list1
            a.del1 = d0
            out = load_bif1d_f_u(_full_cgl1,a,d0,etup=etup_temp,max_iter=20,recompute=False,use_point=init0)
            erange = out[:,0]
            philist = load_phis_force_u(_full_cgl1,a,d0,etup=etup_temp,period_multiple=10,recompute=False,use_point=init0)
            ax0.plot(erange,np.mod(-philist,2*np.pi),zorder=-1,color='k',ls='--',lw=3)

        if k <= 1:
            for kk,list1 in enumerate(ephi_list_u2[k]):
                recompute=False
                init0,etup_temp = list1
                print(list1,init0,etup_temp)
                a.del1 = d1
                out = load_bif1d_f_u(_full_cgl1,a,d1,etup=etup_temp,max_iter=20,recompute=recompute,use_point=init0)
                erange = out[:,0]
                philist = load_phis_force_u(_full_cgl1,a,d1,etup=etup_temp,period_multiple=10,recompute=recompute,use_point=init0)
                ax1.plot(erange,np.mod(-philist,2*np.pi),zorder=-1,color='k',ls='--',lw=3)
        else:
            dd1 = []
            sol1 = []
            
            dd2 = []
            sol2 = []
            
            for kk,list1 in enumerate(ephi_list_u2[k]):
            
                recompute=False
                init0,etup_temp = list1
                
                a.del1 = d1
                
                out = load_bif1d_f_u(_full_cgl1,a,d1,etup=etup_temp,max_iter=20,recompute=recompute,use_point=init0)
                erange = out[:,0]
                philist = load_phis_force_u(_full_cgl1,a,d1,etup=etup_temp,period_multiple=10,recompute=recompute,use_point=init0)
                
                if kk <= 3:
                    sol1 += list(philist)
                    dd1 += list(erange)
                else:
                    sol2 += list(philist)
                    dd2 += list(erange)
                    
            sol = np.array(sol1[::-1] + sol2)
            erange = dd1[::-1] + dd2
            
            ax1.plot(erange,np.mod(-sol,2*np.pi),zorder=-1,color='k',ls='--',lw=3)

        
        if k == 0:
            legend_kw = dict(fontsize=10,handletextpad=0.2,markerscale=.5,borderpad=.2,
                             handlelength=1)
            ax1.legend(**legend_kw)
        
    # decorations
    argt = {'ls':'--','color':'gray','lw':1,'clip_on':True,'zorder':-10}
    
    for k in range(len(axs)):
        ax0 = axs[k][0]; ax1 = axs[k][1]
        
        ax0.set_xlim(0,0.1)
        ax1.set_xlim(0,0.1)
        
        ax0.set_xlabel(r'$\varepsilon$')
        ax0.xaxis.set_label_coords(1.05, 0.02) 
        
        ax1.set_xlabel(r'$\varepsilon$')
        ax1.xaxis.set_label_coords(1.05, 0.02)
        
        ax0.set_title(labels[k],loc='left')
        ax1.set_title(labels[k+3],loc='left')
        
        ax0.set_ylabel(r'$\phi$',labelpad=-10)
        ax0.set_ylim(0,2*np.pi)
        ax0.set_yticks([0,2*np.pi])
        ax0.set_yticklabels(pi_label_short)
        ax0.set_title(labels[k],loc='left')
        
        ax1.set_ylabel(r'$\phi$',labelpad=-10)
        ax1.set_ylim(0,2*np.pi)
        ax1.set_yticks([0,2*np.pi])
        ax1.set_yticklabels(pi_label_short)
        ax1.set_title(labels[k],loc='left')
        
        # label trajectory plots
        #ax0.axvline(e0,-.05,1.05,**argt)
        #ax1.axvline(e1,-.05,1.05,**argt)
        
        ti1 = ax0.get_title()
        ti1 += str(pl_list[k][0])+':'+str(pl_list[k][1])
        #t1 += r', $\varepsilon='+str(e_list[k])+'$'
        ti1 += r', $\delta = '+str(d_list[k][0])+'$'
        ax0.set_title(ti1)
        
        ti1 = ax1.get_title()
        ti1 += str(pl_list[k][0])+':'+str(pl_list[k][1])
        #t1 += r', $\varepsilon='+str(e_list[k])+'$'
        ti1 += r', $\delta = '+str(d_list[k][1])+'$'
        ax1.set_title(ti1)
        
        

    
    return fig






def bif1d_thal1():
    
    kw_thal = copy.deepcopy(kw_thal_template)
    kw_thal['rhs'] = t1.rhs

    system1 = rsp(**{'pardict':pd_thal_template,**kw_thal})

    pl_list = [(1,1),(2,1),(1,2),(2,3)]
    d_list = [(.0,.02),(-.02,.0),(.0,-.025),(-.017,-.023)]

    etup_list = [((.002,.075,.01),(.018,.075,.001)),
                 ((.009,.075,.0005),(.014,.075,.001)),
                 ((.036,.06,.001),(.008,.055,.001)),
                 ((.039,.0436,.0001),(.037,.043,.001))]
    
    full_rhs = _full_thal1
    fig,axs = _setup_bif_plot(labels=False,hspace=.07,wspace=.2,nr1=1,nc1=2)
    argt = {'ls':'--','color':'gray','lw':1,'clip_on':True,'zorder':-10}
    
    #e_list = [(.04,.04),(.04,.04),(.015,.015)]
    #d_list = [(0,.02),(-.02,.0),(.0,.001)]
    
    #etup_list = [((.001,.101,.001),(.02,.11,.001)),
    #             ((.007,.11,.001),(.01,.11,.001)),
    #             ((.001,.11,.001),(.017,.11,.0002)),]
    #sort_by_list = (('max','max'),('max','max'),('max','min'),('max','max'))

    # lower right-limits to prevent their labels from showing
    # (I can then use tighter margins between panels)
    xlims = [(0,.295),(0,.295),(0,.295),(0,.145)]

    _full = _full_thal1
    
    for k in range(len(axs)):
        #sort_by0 = sort_by_list[k][0];sort_by1 = sort_by_list[k][1]
        #e0 = e_list[k][0]; e1 = e_list[k][1]
        d0 = d_list[k][0]; d1 = d_list[k][1]
        
        axx = axs[k].flatten()
        ax0 = axx[0]; ax1 = axx[1]
        etup0, etup1 = etup_list[k] # for full bifurcation diagram
        n = pl_list[k][0]; m = pl_list[k][1]

        a = nmc(system1,None,_n=('om0',n),_m=('om1',m),NH=502)

        if k in [3]:
            lower = .02
            upper = .05
        
        else:
            lower = .001
            upper = etup1[1]
        a.del1 = d0
        add_diagram_1d(ax0,a,d0,(lower,upper,200),rhs=rhs_avg_1df,color='#57acdc',lw=2)
        add_diagram_1d(ax0,a,d0,(lower,upper,200),rhs=rhs_avg_1df,miter=1,color='tab:red')

        a.del1 = d1
        add_diagram_1d(ax1,a,d1,(lower,upper,200),rhs=rhs_avg_1df,color='#57acdc',lw=2,
                       label=r'$O(\varepsilon^2)$')
        add_diagram_1d(ax1,a,d1,(lower,upper,200),rhs=rhs_avg_1df,miter=1,color='tab:red',
                       label=r'$O(\varepsilon)$')

        
        if k in [0]:
            phi0 = 1
        elif k in [1,3]:
            phi0 = 2
        elif k in [2]:
            phi0 = 1.5
        else:
            phi0 = 0
        a.del1 = d0
        out = load_bif1d_f(_full,a,d0,etup=etup0,phi0=phi0)
        erange = out[:,0]
        philist = load_phis_force(_full,a,d0,etup=etup0,period_multiple=10,phi0=phi0)
        y = np.mod(-philist,2*np.pi)
        x = erange
        
        
        threshold = 0.5 
        discont_indices = np.where(np.abs(np.diff(y)) >= threshold)[0] + 1
        x_discontinuous = np.insert(x, discont_indices, np.nan)
        y_discontinuous = np.insert(y, discont_indices, np.nan)
        
        if k == 2:
            ax0.plot(x_discontinuous,y_discontinuous,color='k',lw=3,zorder=-1)
            ax0.plot(x_discontinuous,y_discontinuous+np.pi,color='k',lw=3,zorder=-1)
        elif k == 3:
            ax0.plot(x_discontinuous,y_discontinuous+2*np.pi/3,color='k',lw=3,zorder=-1)
            ax0.plot(x_discontinuous,y_discontinuous,color='k',lw=3,zorder=-1)
            ax0.plot(x_discontinuous,y_discontinuous-2*np.pi/3,color='k',lw=3,zorder=-1)
        else:
            ax0.plot(x_discontinuous,y_discontinuous,color='k',lw=3,zorder=-1)

        a.del1 = d1
        out = load_bif1d_f(_full,a,d1,etup=etup1,max_iter=20,phi0=phi0,recompute=False)
        erange = out[:,0]
        philist = load_phis_force(_full,a,d1,etup=etup1,phi0=phi0,period_multiple=10,recompute=False)
        
        y = np.mod(-philist,2*np.pi)
        x = erange
        
        threshold = 0.5 
        discont_indices = np.where(np.abs(np.diff(y)) >= threshold)[0] + 1
        x_discontinuous = np.insert(x, discont_indices, np.nan)
        y_discontinuous = np.insert(y, discont_indices, np.nan)
        
        if k == 2:
            ax1.plot(x_discontinuous,y_discontinuous,color='k',lw=3,zorder=-1)
            ax1.plot(x_discontinuous,y_discontinuous-np.pi,color='k',lw=3,zorder=-1)
        elif k == 3:
            ax1.plot(x_discontinuous,y_discontinuous+2*np.pi/3,color='k',lw=3,zorder=-1)
            ax1.plot(x_discontinuous,y_discontinuous,color='k',lw=3,zorder=-1)
            ax1.plot(x_discontinuous,y_discontinuous-2*np.pi/3,color='k',lw=3,zorder=-1)
        else:
            ax1.plot(x_discontinuous,y_discontinuous,color='k',lw=3,zorder=-1)
        
        


    # thease are obtained from v2_jupyter/bif_thal1.ipynb
    #eps and phi0 for d=0
    ephi_list_u1 = [
        ( # 1:1 b=0
            [[-0.18945403623819065, 0.09591438239943723, 0.09301948731414543],(.001,.01,.0004)],
            [[-0.20444447801075152, 0.4909821100889541, 0.09259549216606044],(0.01, 0.02, 0.001)],
            [[-0.36642158515763196, 0.6326878938044321, 0.08952204239871353] , (0.02, 0.04, 0.001)],
            [[-0.4277186486464589, 0.6220908718635058, 0.07457083836889453] , (0.04, 0.07, 0.001)],
            [[-0.42630436741844424, 0.5198888162560811, 0.03370344281380619] , (0.069, 0.073, 0.0001)],
        ),
        (# 2:1 b=-.02
            [[-0.31346033,  0.55701472,  0.07300908],(.03,.0,-.001)], 
            [[-0.31346033,  0.55701472,  0.07300908],(.03,.04,.001)],
            [[-0.37272670156723875, 0.59792594127064, 0.07179809619626129] , (0.039, 0.06, 0.001)]
        ),
        (# 1:2 b = 0
            [[-0.39837031062642664, 0.6526824453936779, 0.09586944336882908] , (0.045, 0.033, -0.001)],
            [[-0.39837031062642664, 0.6526824453936779, 0.09586944336882908] , (0.045, 0.056, 0.001)]
        ),
        (# 23 b=-0.017
        [[-0.4171081581971554, 0.6570404518999735, 0.1050308387725617] , (0.04, 0.0385, -0.0005)],
        [[-0.4171081581971554, 0.6570404518999735, 0.1050308387725617] , (0.04, 0.0435, 0.0005)]
        ),
    ]
    
    #eps and phi0 for other d
    ephi_list_u2 = [
        (# 1:1 b=.02
            [[-0.12398772, 0.18925807, 0.09165775] , (0.023, 0.03, 0.001)], 
            [[-0.39413122601742295, 0.612823043800616, 0.07738862167682786] , (0.029, 0.04, 0.001)],
            [[-0.4186246669805977, 0.5923473357739142, 0.06430485021839023] , (0.039, 0.05, 0.001)],
            [[-0.4188531371309764, 0.5187601347862301, 0.03531771041111262] , (0.049, 0.07, 0.0003)],
        ),
        ( # 2:1 b=0
            [[-0.1244944736165739, 0.18448098884264694, 0.09138371824400734] , (0.03, 0.0,-0.001)],
            [[-0.1244944736165739, 0.18448098884264694, 0.09138371824400734] , (0.03, 0.051, 0.001)],
        ), 
        (# 1:2 b=-.025
            [[-0.36608117,  0.64792313,  0.09628009],(.03,.007,-.001)],
            [[-0.36608117,  0.64792313,  0.09628009],(.03,.052,.001)]
        ), #(-0.16936787, 0.94961894)
        
        ( #23
            [[-0.44085744574529734, 0.652423876533637, 0.10174957244774928],(.04,.036,-.0005)],
            [[-0.44085744574529734, 0.652423876533637, 0.10174957244774928],(.04,.06,.0005)]
        ),
        
        
    ]

    
    # add unstable points
    for k in range(len(axs)):
        #sort_by0 = sort_by_list[k][0];sort_by1 = sort_by_list[k][1]
        #e0 = e_list[k][0]; e1 = e_list[k][1]
        d0 = d_list[k][0]; d1 = d_list[k][1]
        axx = axs[k].flatten()
        ax0 = axx[0]; ax1 = axx[1]
        etup0, etup1 = etup_list[k] # for full bifurcation diagram
        n = pl_list[k][0]; m = pl_list[k][1]

        a = nmc(system1,None,_n=('om0',n),_m=('om1',m),NH=502)

        for kk,list1 in enumerate(ephi_list_u1[k]):
            
            init0,etup_temp = list1
            a.del1 = d0
            out = load_bif1d_f_u(_full,a,d0,etup=etup_temp,max_iter=20,recompute=False,use_point=init0)
            erange = out[:,0]
            philist = load_phis_force_u(_full,a,d0,etup=etup_temp,period_multiple=10,recompute=False,use_point=init0)
            
            x = erange
            if k == 2:
                x = erange[:-2]
                y = np.mod(-philist,2*np.pi)[:-2]
                y1 = np.mod(-philist,2*np.pi)[:-2] - np.pi
                print(x,y,y1)
                ax0.plot(x,y1,color='k',lw=3,ls='--',zorder=-1)
                
            elif k == 3:
                y1 = np.mod(-philist,2*np.pi)
                y = np.mod(-philist,2*np.pi) - 2*np.pi/3
                y3 = np.mod(-philist,2*np.pi) - 2*2*np.pi/3
                ax0.plot(x,y1,color='k',lw=3,ls='--',zorder=-1)
                ax0.plot(x,y3,color='k',lw=3,ls='--',zorder=-1)
            else:
                y = np.mod(-philist,2*np.pi)
            
            
            threshold = 0.5 
            discont_indices = np.where(np.abs(np.diff(y)) >= threshold)[0] + 1
            x_discontinuous = np.insert(x, discont_indices, np.nan)
            y_discontinuous = np.insert(y, discont_indices, np.nan)
            
            ax0.plot(x_discontinuous,y_discontinuous,color='k',lw=3,ls='--',zorder=-1)
            
            
            #ax0.plot(erange,np.mod(-philist,2*np.pi),zorder=-1,color='k',ls='--',lw=3)

        for kk,list1 in enumerate(ephi_list_u2[k]):
            recompute=False
            init0,etup_temp = list1
            print(list1,init0,etup_temp)
            a.del1 = d1
            out = load_bif1d_f_u(_full,a,d1,etup=etup_temp,max_iter=20,recompute=recompute,use_point=init0)
            erange = out[:,0]
            philist = load_phis_force_u(_full,a,d1,etup=etup_temp,period_multiple=10,recompute=recompute,use_point=init0)
            
            x = erange
            if k == 2:
                y1 = np.mod(-philist,2*np.pi)
                y = np.mod(-philist,2*np.pi) - np.pi
                ax1.plot(x,y1,color='k',lw=3,ls='--',zorder=-1)
                ax1.plot(x,y,color='k',lw=3,ls='--',zorder=-1)
            elif k == 3:
                y1 = np.mod(-philist,2*np.pi)
                y = np.mod(-philist,2*np.pi) - 2*np.pi/3
                y3 = np.mod(-philist,2*np.pi) - 2*2*np.pi/3
                ax1.plot(x,y1,color='k',lw=3,ls='--',zorder=-1)
                ax1.plot(x,y3,color='k',lw=3,ls='--',zorder=-1)
            else:
                y = np.mod(-philist,2*np.pi)
            #for j in range(a._m[1]):
            #    y = np.mod(np.mod(-philist,2*np.pi) - j*2*np.pi/a._m[1],2*np.pi)
            
            
            
            
            threshold = 0.5 
            discont_indices = np.where(np.abs(np.diff(y)) >= threshold)[0] + 1
            x_discontinuous = np.insert(x, discont_indices, np.nan)
            y_discontinuous = np.insert(y, discont_indices, np.nan)
            
            ax1.plot(x_discontinuous,y_discontinuous,color='k',lw=3,ls='--',zorder=-1)
            

        
        if k == 0:
            legend_kw = dict(fontsize=10,handletextpad=0.2,markerscale=.5,borderpad=.2,
                             handlelength=1)
            ax1.legend(**legend_kw)
        
    # decorations
    argt = {'ls':'--','color':'gray','lw':1,'clip_on':True,'zorder':-10}
    
    for k in range(len(axs)):
        axx = axs[k].flatten()
        ax0 = axx[0]; ax1 = axx[1]
        
        if k == 3:
            ax0.set_xlim(.03,0.055)
            ax1.set_xlim(.03,0.055)
        else:
            ax0.set_xlim(0,0.05)
            ax1.set_xlim(0,0.05)
        
        ax0.set_xlabel(r'$\varepsilon$')
        ax0.xaxis.set_label_coords(1.05, 0.02) 
        
        ax1.set_xlabel(r'$\varepsilon$')
        ax1.xaxis.set_label_coords(1.05, 0.02)
        
        ax0.set_title(labels[k],loc='left')
        ax1.set_title(labels[k+3],loc='left')
        
        ax0.set_ylabel(r'$\phi$',labelpad=-10)
        ax0.set_ylim(0,2*np.pi)
        ax0.set_yticks([0,2*np.pi])
        ax0.set_yticklabels(pi_label_short)
        ax0.set_title(labels[k],loc='left')
        
        #ax1.set_ylabel(r'$\phi$',labelpad=0)
        ax1.set_ylim(0,2*np.pi)
        ax1.set_yticks([0,2*np.pi])
        ax1.set_yticklabels(pi_label_short)
        ax1.set_title(labels[k],loc='left')
        
        # label trajectory plots
        #ax0.axvline(e0,-.05,1.05,**argt)
        #ax1.axvline(e1,-.05,1.05,**argt)
        
        ti1 = ax0.get_title()
        ti1 += str(pl_list[k][0])+':'+str(pl_list[k][1])
        #t1 += r', $\varepsilon='+str(e_list[k])+'$'
        dd = d_list[k][0]
        ti1 += r', $\delta = '+str(np.round(dd/pl_list[k][1],3))+'$'
        ax0.set_title(ti1)
        
        ti1 = ax1.get_title()
        ti1 += str(pl_list[k][0])+':'+str(pl_list[k][1])
        #t1 += r', $\varepsilon='+str(e_list[k])+'$'
        ti1 += r', $\delta = '+str(d_list[k][1])+'$'
        ax1.set_title(ti1)
        
        

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
    kwA = {'a':a,'b':0.01,'recompute':False,'bifdir':'bif1d_thal2/','_full_rhs':_full}
    in_dictsA = [dict(phi0=6,eps_init=0.101,eps_final=0.001,deps=-.001)]
    dataA = follow_phase_diffs(**in_dictsA[0],**kwA)
    print(dataA[-1,:])

    # A unstable
    kw_b = {'a':a,'b':0,'recompute':False,'bifdir':'bif1d_thal2/','_full_rhs':_full}
    da,Yinit = follow_phase_diffs_u(**dict(phi0=3.5,eps_init=.01,eps_final=0,deps=-2),max_iter=50,return_point=True,**kw_b)
    #print(da)
    kw_b['b'] = 0.01
    da,Yinit = follow_phase_diffs_u(**dict(phi0=3.5,eps_init=.01,eps_final=0,deps=-2),max_iter=50,return_point=True,use_point=Yinit,**kw_b)
    #print(da)
    da,Yinit = follow_phase_diffs_u(**dict(phi0=3.5,eps_init=.05,eps_final=0,deps=-2),max_iter=50,return_point=True,use_point=Yinit,**kw_b)
    #print(da)
    da,Yinit = follow_phase_diffs_u(**dict(phi0=3.5,eps_init=.1,eps_final=0,deps=-2),max_iter=50,return_point=True,use_point=Yinit,**kw_b)
    #print(da)

    in_dictsAu = [dict(phi0=0,eps_init=0.101,eps_final=0,deps=-.002),]
    dataAu = follow_phase_diffs_u(**in_dictsAu[0],use_point=Yinit,max_iter=50,**kw_b)
    

    # B
    kwB = {'a':a,'b':0.02,'recompute':False,'bifdir':'bif1d_thal2/','_full_rhs':_full}
    in_dictsB = [dict(phi0=5.5,eps_init=0.101,eps_final=0.001,deps=-.001)]
    dataB = follow_phase_diffs(**in_dictsB[0],**kwB)

    print('*****B UNSTABLE********')
    # B unstable
    #kw_b = {'a':a,'b':0,'recompute':False,'bifdir':'bif1d_thal2/','_full_rhs':_full}
    da,Yinit = follow_phase_diffs_u(**dict(phi0=3.5,eps_init=.1,eps_final=0,deps=-2),max_iter=50,return_point=True,use_point=Yinit,**kw_b)
    #print(da)
    kw_b['b'] = 0.015
    da,Yinit = follow_phase_diffs_u(**dict(phi0=3.5,eps_init=.1,eps_final=0,deps=-2),max_iter=50,return_point=True,use_point=Yinit,**kw_b)
    #print(da)

    kw_b['b'] = 0.02
    da,Yinit = follow_phase_diffs_u(**dict(phi0=3.5,eps_init=.1,eps_final=0,deps=-2),max_iter=50,return_point=True,use_point=Yinit,**kw_b)
    #print(da)
    
    in_dictsBu = [dict(phi0=0,eps_init=0.101,eps_final=0,deps=-.002),]
    dataBu = follow_phase_diffs_u(**in_dictsBu[0],use_point=Yinit,max_iter=50,**kw_b)


    
    # C
    kwC = {'a':a,'eps':0.05,'recompute':False,'bifdir':'bif1d_thal2/','_full_rhs':_full}
    in_dictsC = [dict(phi0=0.,del_init=0,del_final=-.035,ddel=-.001),
                 dict(phi0=0.,del_init=0,del_final=.025,ddel=.001)]
    dataC1 = follow_phase_diffs_del(**in_dictsC[0],**kwC)
    dataC2 = follow_phase_diffs_del(**in_dictsC[1],**kwC)
    dataC2[0][2] = 2*np.pi-.0001

    print('*****C UNSTABLE********')
    kw_b = {'a':a,'eps':0.01,'recompute':False,'bifdir':'bif1d_thal2/','_full_rhs':_full}
    da,Yinit = follow_phase_diffs_u_del(**dict(phi0=3.141,del_init=0,del_final=-.01,ddel=-1),max_iter=50,return_point=True,**kw_b)
    kw_b['eps'] = 0.03
    da,Yinit = follow_phase_diffs_u_del(**dict(phi0=3.141,del_init=0,del_final=-.01,ddel=-1),max_iter=50,return_point=True,use_point=Yinit,**kw_b)
    kw_b['eps'] = 0.05
    da,Yinit = follow_phase_diffs_u_del(**dict(phi0=3.141,del_init=0,del_final=-.01,ddel=-1),max_iter=50,return_point=True,use_point=Yinit,**kw_b)
    #print(da)
    
    in_dictsCu = [dict(phi0=0.,del_init=0,del_final=-.035,ddel=-.001),
                  dict(phi0=0.,del_init=0,del_final=.025,ddel=.001)]
    dataCu1 = follow_phase_diffs_u_del(**in_dictsCu[0],use_point=Yinit,max_iter=50,**kw_b)
    dataCu2 = follow_phase_diffs_u_del(**in_dictsCu[1],use_point=Yinit,max_iter=50,**kw_b)
    
    # D
    kwD = {'a':a,'eps':0.1,'recompute':False,'bifdir':'bif1d_thal2/','_full_rhs':_full}
    in_dictsD = [dict(phi0=0.,del_init=0,del_final=-.05,ddel=-.001),
                 dict(phi0=0.,del_init=0,del_final=.035,ddel=.001)]
    dataD1 = follow_phase_diffs_del(**in_dictsD[0],**kwD)
    dataD2 = follow_phase_diffs_del(**in_dictsD[1],**kwD)

    print('*****D UNSTABLE********')
    kw_b = {'a':a,'eps':0.05,'recompute':False,'bifdir':'bif1d_thal2/','_full_rhs':_full}
    da,Yinit = follow_phase_diffs_u_del(**dict(phi0=3.141,del_init=0,del_final=-.01,ddel=-1),max_iter=50,return_point=True,**kw_b)
    kw_b['eps'] = 0.075
    da,Yinit = follow_phase_diffs_u_del(**dict(phi0=3.141,del_init=0,del_final=-.01,ddel=-1),max_iter=50,return_point=True,use_point=Yinit,**kw_b)
    kw_b['eps'] = 0.1
    da,Yinit = follow_phase_diffs_u_del(**dict(phi0=3.141,del_init=0,del_final=-.01,ddel=-1),max_iter=50,return_point=True,use_point=Yinit,**kw_b)
    
    in_dictsDu = [dict(phi0=0.,del_init=0,del_final=-.05,ddel=-.001),
                  dict(phi0=0.,del_init=0,del_final=.025,ddel=.001)]
    dataDu1 = follow_phase_diffs_u_del(**in_dictsDu[0],use_point=Yinit,max_iter=50,**kw_b)
    dataDu2 = follow_phase_diffs_u_del(**in_dictsDu[1],use_point=Yinit,max_iter=50,**kw_b)

    

    fig,axs = plt.subplots(1,4,figsize=(8,2),gridspec_kw={'wspace':.5})

            
    # plot full
    axs[0].plot(dataA[:,0],dataA[:,2],color='k',lw=3,label='Full')
    axs[0].plot(dataAu[:,0],dataAu[:,2],color='k',lw=3,ls='--')
    
    axs[1].plot(dataB[:,0],dataB[:,2],color='k',lw=3)
    axs[1].plot(dataBu[:,0],dataBu[:,2],color='k',lw=3,ls='--')
    
    axs[2].plot(dataC1[:,0],dataC1[:,2],color='k',lw=3)
    axs[2].plot(dataC2[:,0],dataC2[:,2],color='k',lw=3)

    axs[2].plot(dataCu1[:,0],dataCu1[:,2],color='k',lw=3,ls='--')
    axs[2].plot(dataCu2[:,0],dataCu2[:,2],color='k',lw=3,ls='--')

    axs[3].plot(dataD1[:,0],dataD1[:,2],color='k',lw=3)
    axs[3].plot(dataD2[:,0],dataD2[:,2],color='k',lw=3)

    axs[3].plot(dataDu1[:,0],dataDu1[:,2],color='k',lw=3,ls='--')
    axs[3].plot(dataDu2[:,0],dataDu2[:,2],color='k',lw=3,ls='--')
    #for j in range(len(data_list[i])):
    #    data1,data2 = data_list[i][j] # data1 is periods, data2 is t_diffs


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
    axs[0].legend()

    plt.subplots_adjust(left=.075,right=.97,bottom=.2)
    
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

    
    dicts_list_u = [
        # 1:2 u
        ([dict(phi0=5,eps_init=0.101,eps_final=0,deps=-.01),], # A
         [dict(phi0=5.2,eps_init=0.101,eps_final=0,deps=-.01)], # B
         [dict(phi0=5.3,del_init=0,del_final=-.035,ddel=-.005), # C
          dict(phi0=5.3,del_init=0,del_final=.02,ddel=.002),
          dict(phi0=6,del_init=0.02,del_final=.031,ddel=.002)],
         [dict(phi0=5.4,del_init=0,del_final=-.05,ddel=-.005), # D
          dict(phi0=5.4,del_init=0,del_final=.02,ddel=.002),
          dict(phi0=6,del_init=0.02,del_final=.035,ddel=.002)]
         ),
        
        # 2:1 u
        ([dict(phi0=1.9,eps_init=0.101,eps_final=0.0,deps=-.01),], # A
         [dict(phi0=2.5,eps_init=0.101,eps_final=0.0,deps=-.01),], # B
         [dict(phi0=1.5,del_init=0,del_final=-.042,ddel=-.005), # C
          dict(phi0=1.5,del_init=0,del_final=.026,ddel=.005)],
         [dict(phi0=1.5,del_init=0,del_final=-.055,ddel=-.005), # D
          dict(phi0=1.5,del_init=0,del_final=.026,ddel=.002)]
         ),
        
        # 1:3 u
        ([dict(phi0=5.9,eps_init=0.101,eps_final=0,deps=-.01),
          dict(phi0=5.9,eps_init=0.012,eps_final=0,deps=-.005)], # A
         [dict(phi0=6.1,eps_init=0.021,eps_final=0,deps=-.01),
          dict(phi0=6.1,eps_init=0.02,eps_final=.11,deps=.01)], # B
         [dict(phi0=5.9,del_init=0,del_final=-.03,ddel=-.002), # C
          dict(phi0=5.9,del_init=0,del_final=.03,ddel=.002)],
         [dict(phi0=5.9,del_init=0,del_final=-.045,ddel=-.003), # D
          #dict(phi0=5,del_init=-.025,del_final=-.03,ddel=-.001),
          dict(phi0=5.9,del_init=0,del_final=.025,ddel=.003)]
         ),
        
        # 3:1 u
        ([dict(phi0=1,eps_init=0.101,eps_final=0.001,deps=-.01),], # A
         [dict(phi0=2,eps_init=0.101,eps_final=0,deps=-.01)], # B
         [dict(phi0=1.4,del_init=0,del_final=-.035,ddel=-.005), # C
          dict(phi0=1.4,del_init=0,del_final=.025,ddel=.005)],
         [dict(phi0=1.4,del_init=0,del_final=-.035,ddel=-.005), # D
          dict(phi0=1.4,del_init=0,del_final=.02,ddel=.005)]
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
        in_dictsAu,in_dictsBu,in_dictsCu,in_dictsDu = dicts_list_u[k]
        
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
        kwA = {'a':a,'b':del_list[0],'recompute':re1,'bifdir':'bif1d_thal2/','_full_rhs':_full}
        dataA_list = []
        for dict1 in in_dictsA:
            dataA_list.append(follow_phase_diffs(**dict1,**kwA))

        # B
        print('B')
        kwB = {'a':a,'b':del_list[1],'recompute':re2,'bifdir':'bif1d_thal2/','_full_rhs':_full}
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



        ########## UNSTABLE
        # A unstable
        print('A u')
        kwA = {'a':a,'b':del_list[0],'recompute':re1,'bifdir':'bif1d_thal2/','_full_rhs':_full}
        dataAu_list = []
        for dict1 in in_dictsAu:
            dataAu_list.append(follow_phase_diffs_u(**dict1,**kwA))



        # B unstable
        print('B u')
        kwB = {'a':a,'b':del_list[1],'recompute':re1,'bifdir':'bif1d_thal2/','_full_rhs':_full}
        dataBu_list = []
        for dict1 in in_dictsBu:
            dataBu_list.append(follow_phase_diffs_u(**dict1,**kwB))

                        
            

        # C unstable
        print('C u')
        kwC = {'a':a,'eps':eps_list[0],'recompute':re1,'bifdir':'bif1d_thal2/','_full_rhs':_full}
        dataCu_list = []
        for dict1 in in_dictsCu:
            dataCu_list.append(follow_phase_diffs_u_del(**dict1,**kwC))

        print('datlist',dataC_list)



        
        # D unstable
        print('D u')
        kwD = {'a':a,'eps':eps_list[1],'recompute':re1,'bifdir':'bif1d_thal2/','_full_rhs':_full}
        dataDu_list = []
        for dict1 in in_dictsDu:
            dataDu_list.append(follow_phase_diffs_u_del(**dict1,**kwD))



            
            

        for data in dataAu_list:
            axs[0].plot(data[:,0],data[:,-a._m[1]:],color='k',ls='--',lw=3)
            
        for data in dataBu_list:
            axs[1].plot(data[:,0],data[:,-a._m[1]:],color='k',ls='--',lw=3)

        for data in dataCu_list:
            axs[2].plot(data[:,0],data[:,-a._m[1]:],color='k',ls='--',lw=3)

        for j,data in enumerate(dataDu_list):
            
            if k == 0 and j == 2:
                x = data[:3,0]
                y = data[:3,-a._m[1]:]
            else:
                x = data[:,0]
                y = data[:,-a._m[1]:]

            axs[3].plot(x,y,color='k',ls='--',lw=3)

        
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
               (False,False,False,False), #32
               (False,False,False,False), #34
               (False,False,False,False)] #43
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

    init32b = [-0.5885665741006565, 0.5281379516882104, 0.09912819298330922, 0.0018987803887340637, -0.10767322448879131, 0.22438126348515572, 0.09348376685586182, 0.40780078310724055, 6.2753649553681825]
    
    init32c = [-0.5731653663110687, 0.5534051706502733, 0.09610160013365669, 0.0011995560895133406, -0.1096986882133124, 0.22606647172748065, 0.09001673183008534, 0.4036401816222591, 6.212039378814412]
    
    init32d = [-0.5819915908088896, 0.5304883307563432, 0.09268973711558165, 0.0017712955130989797, -0.11171197363122576, 0.2226974375078165, 0.08682723907533853, 0.4055089975606703, 6.1451516883353445]
    
    
    
    
    init34a = [-0.37623278021069134, 0.077832972884727, 0.08847590287218318, 0.28992456221204227, -0.4484828702765177, 0.09059861222072976, 0.08878168766203551, 0.21268497999123703, 6.22388167936219]
    
    init34b = [-0.5747203834667813, 0.5543224034354748, 0.0969123437049565, 0.0011996588966108144, -0.11340926455685398, 0.22591163107730342, 0.08993600320180876, 0.3997114189509189, 6.226013618171563]
    
    init34c =  [-0.37623278021069134, 0.077832972884727, 0.08847590287218318, 0.28992456221204227, -0.4484828702765177, 0.09059861222072976, 0.08878168766203551, 0.21268497999123703, 6.22388167936219]
    
    init34d = [-0.3698703773926263, 0.07788790152776129, 0.0840031704214259, 0.29307488362839546, -0.45155614785649734, 0.09123409054421489, 0.08469919930294688, 0.2070064479295194, 6.1435558031482405]
    
    
    
    
    init43a =  [-0.12069692633123634, 0.31315819770146963, 0.09062706474369595, 0.29715875379281254, -0.11437757941848589, 0.2640757845423882, 0.08960594103486547, 0.35457579753434243, 6.2099420820670455]
    
    init43b = [-0.19923515665213754, 0.0914416190128596, 0.09251768831582251, 0.5375081532140491, -0.10858139361078517, 0.21144052907897343, 0.09310120643771533, 0.4223487556755435, 6.2727169414600255]
    
    init43c =  [-0.12069692633123634, 0.31315819770146963, 0.09062706474369595, 0.29715875379281254, -0.11437757941848589, 0.2640757845423882, 0.08960594103486547, 0.35457579753434243, 6.2099420820670455]
    
    init43d = [-0.11805836025532761, 0.30312968502934584, 0.08713313946437636, 0.309981794350864, -0.12038710876580193, 0.2765807460807493, 0.08586304439240909, 0.332194642585823, 6.143555803945192]
    
    dicts_list_u = [
        # 2:3
        ([dict(phi0=3.9,eps_init=0.011,eps_final=0,deps=-.01), # A
          dict(phi0=3.9,eps_init=0.01,eps_final=.11,deps=.01)],
         [dict(phi0=0.05,eps_init=0.101,eps_final=0,deps=-.01)], # B
         [dict(phi0=3.3,del_init=0,del_final=-.02,ddel=-.002), # C
          dict(phi0=3.3,del_init=0,del_final=.01,ddel=.002)],
         [dict(phi0=3.3,del_init=0,del_final=-.007,ddel=-.002), # D
          dict(phi0=3.3,del_init=0,del_final=.007,ddel=.002)]
         ),
        
        # 3:2
        ([dict(phi0=3.5,eps_init=0.011,eps_final=0,deps=-.005),
          dict(phi0=3.5,eps_init=0.01,eps_final=0.11,deps=.01)], # A
          
         # B
         [dict(phi0=3.14,eps_init=0.011,eps_final=0,deps=-.002,use_point=init32b),
         dict(phi0=3.14,eps_init=0.01,eps_final=.11,deps=.01,use_point=init32b)], 
         
         # C
         [dict(phi0=0.2,del_init=0,del_final=-.0091,ddel=-.001,use_point=init32c),
          dict(phi0=0.2,del_init=0,del_final=.00611,ddel=.001,use_point=init32c)],
          
         # D
         [dict(phi0=1.5,del_init=0,del_final=-.0091,ddel=-.001,use_point=init32d),
          dict(phi0=1,del_init=0,del_final=.0036,ddel=.0005,use_point=init32d)]),
        
        
        
        # 3:4
        ([dict(phi0=3.1,eps_init=0.041,eps_final=0,deps=-.01,use_point=init34a),
          dict(phi0=3.1,eps_init=0.04,eps_final=.11,deps=.01,use_point=init34a)], # A
          
         [dict(phi0=3.1,eps_init=0.051,eps_final=0,deps=-.01,use_point=init34b),
          dict(phi0=3.1,eps_init=0.05,eps_final=.11,deps=.01,use_point=init34b)], # B
          
          # C
         [dict(phi0=5,del_init=0,del_final=-.0025,ddel=-.0003,use_point=init34c),
          dict(phi0=5,del_init=0,del_final=.0021,ddel=.0001,use_point=init34c)],
          
           # D
         [dict(phi0=3,del_init=0,del_final=-.00251,ddel=-.0003,use_point=init34d),
          dict(phi0=3,del_init=0,del_final=.002,ddel=.0003,use_point=init34d)]),
        
        # 4:3
        # A
        ([dict(phi0=5,eps_init=0.051,eps_final=0,deps=-.01,use_point=init43a),
          dict(phi0=5,eps_init=0.05,eps_final=.11,deps=.01,use_point=init43a)], 
          
         # B
         [dict(phi0=5,eps_init=0.007,eps_final=0,deps=-.002,use_point=init43b),
          dict(phi0=5,eps_init=0.0069,eps_final=.11,deps=.003,use_point=init43b)], 
         # C
         [dict(phi0=5,del_init=0,del_final=-.0025,ddel=-.0005,use_point=init43c), 
          dict(phi0=5,del_init=0,del_final=.002,ddel=.0005,use_point=init43c)],
          # D
         [dict(phi0=5,del_init=0,del_final=-.0025,ddel=-.0003,use_point=init43d), 
          dict(phi0=5,del_init=0,del_final=.002,ddel=.0003,use_point=init43d)]),
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
        in_dictsAu,in_dictsBu,in_dictsCu,in_dictsDu = dicts_list_u[k]
        
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
        kwA = {'a':a,'b':del_list[0],'recompute':re1,'bifdir':'bif1d_thal2/','_full_rhs':_full}
        dataA_list = []
        for dict1 in in_dictsA:
            dataA_list.append(follow_phase_diffs(**dict1,**kwA))

        # B
        print('B')
        kwB = {'a':a,'b':del_list[1],'recompute':re2,'bifdir':'bif1d_thal2/','_full_rhs':_full}
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
        for j,data in enumerate(dataA_list):
            if k == 0:
                axs[0].plot(data[1:,0],data[1:,-a._m[1]:],color='k',lw=3)
            else:
                axs[0].plot(data[:,0],data[:,-a._m[1]:],color='k',lw=3)
                
        for data in dataB_list:
            axs[1].plot(data[:,0],data[:,-a._m[1]:],color='k',lw=3)

        #print(dataC_list)
        for data in dataC_list:
            axs[2].plot(data[:,0],data[:,-a._m[1]:],color='k',lw=3)
        for data in dataD_list:
            axs[3].plot(data[:,0],data[:,-a._m[1]:],color='k',lw=3)


        
        
        print('datlist',dataA_list)
        ########## UNSTABLE
        # A unstable
        print('A u',eps_list[0],n,m)
        kwA = {'a':a,'b':del_list[0],'recompute':re1,'bifdir':'bif1d_thal2/','_full_rhs':_full,'tol':1e-6}
        dataAu_list = []
        for dict1 in in_dictsAu:
            dataAu_list.append(follow_phase_diffs_u(**dict1,**kwA))
        
        
        print('datlist',dataB_list)
        # B unstable
        print('B u',eps_list[1],n,m)
        kwB = {'a':a,'b':del_list[1],'recompute':re1,'bifdir':'bif1d_thal2/','_full_rhs':_full,'tol':1e-6}
        dataBu_list = []
        for dict1 in in_dictsBu:
            dataBu_list.append(follow_phase_diffs_u(**dict1,**kwB))
            
        
        print('datlist',dataC_list)
        # C unstable
        print('C u','eps=',eps_list[0],n,m)
        kwC = {'a':a,'eps':eps_list[0],'recompute':re1,'bifdir':'bif1d_thal2/','_full_rhs':_full,'tol':1e-6}
        dataCu_list = []
        for dict1 in in_dictsCu:
            dataCu_list.append(follow_phase_diffs_u_del(**dict1,**kwC))

        
        print('datlist',dataD_list)
        # D unstable
        print('D u','eps=',eps_list[1],n,m)
        kwD = {'a':a,'eps':eps_list[1],'recompute':re1,'bifdir':'bif1d_thal2/','_full_rhs':_full,'tol':1e-6}
        dataDu_list = []
        for dict1 in in_dictsDu:
            dataDu_list.append(follow_phase_diffs_u_del(**dict1,**kwD))

        

            
        
        if k == 1:
            dataBu_list = fix_flips(dataBu_list,a)
        
        
        for j,data in enumerate(dataAu_list):
            
            x = data[:,0]
            for i in range(a._m[1]):
                y = data[:,-(i+1)]
                threshold = 0.5 
                discont_indices = np.where(np.abs(np.diff(y)) >= threshold)[0] + 1
                x_disc = np.insert(x, discont_indices, np.nan)
                y_disc = np.insert(y, discont_indices, np.nan)
                axs[0].plot(x_disc,y_disc,color='k',ls='--',lw=3)
            
            
        
            
        for j,data in enumerate(dataBu_list):
            #x = data[:,0]
            #y = 
            if k == 2 and j== 0:
                data = data[:-1,:]
                
            x = data[:,0]
            for i in range(a._m[1]):
                y = data[:,-(i+1)]#y[:,-(i+1)]
                print(k,j,x,y)
                threshold = 0.5 
                discont_indices = np.where(np.abs(np.diff(y)) >= threshold)[0] + 1
                x_disc = np.insert(x, discont_indices, np.nan)
                y_disc = np.insert(y, discont_indices, np.nan)
                axs[1].plot(x_disc,y_disc,color='k',ls='--',lw=3)
            #axs[1].plot(x,y,color='k',ls='--',lw=3)

        #print(dataCu_list)
        for j,data in enumerate(dataCu_list):
            x = data[:,0]
            for i in range(a._m[1]):
                y = data[:,-(i+1)]
                threshold = 0.5 
                discont_indices = np.where(np.abs(np.diff(y)) >= threshold)[0] + 1
                x_disc = np.insert(x, discont_indices, np.nan)
                y_disc = np.insert(y, discont_indices, np.nan)
                axs[2].plot(x_disc,y_disc,color='k',ls='--',lw=3)
            #axs[2].plot(data[:,0],data[:,-a._m[1]:],color='k',ls='--',lw=3)

        if k == 0:
            dataDu_list = fix_flips(dataDu_list,a)
        for j,data in enumerate(dataDu_list):
            x = data[:,0]
            for i in range(a._m[1]):
                y = data[:,-(i+1)]
                threshold = 0.5 
                discont_indices = np.where(np.abs(np.diff(y)) >= threshold)[0] + 1
                x_disc = np.insert(x, discont_indices, np.nan)
                y_disc = np.insert(y, discont_indices, np.nan)
                axs[3].plot(x_disc,y_disc,color='k',ls='--',lw=3)
            #axs[3].plot(data[:,0],data[:,-a._m[1]:],color='k',ls='--',lw=3)
        

            

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



def bif1_vdp_thal():

    kws_o1 = dict(rhs=_redu_c2,miter=1,label=r'$O(\varepsilon)$',color='tab:red')
    kws_o2 = dict(rhs=_redu_c2,miter=2,label=r'$O(\varepsilon^2)$',color='#57acdc',lw=1.5)
    
    kw_thal = copy.deepcopy(kw_thal_template)
    kw_thal['rhs'] = vdp_thal.rhs_thal
    kw_thal['var_names'] = ['v','h','r','w']
    kw_thal['init'] = np.array([-.64,0.71,0.25,0,6.28])
    kw_thal['forcing_fn'] = None
    kw_thal['coupling'] = vdp_thal.coupling_thal
    kw_thal['trunc_order'] = 1
    kw_thal['TN'] = 10000
    kw_thal['dir_root'] = 'v2_jupyter/data/'

    kw_vdp = copy.deepcopy(kw_vdp_template)
    kw_vdp['dir_root'] = 'v2_jupyter/data/'

    pd_thal_template2 = deepcopy(pd_thal_template)
    pd_thal_template2['ib'] = 8.2
    
    kw_thal['model_name'] = 'vdp_thal0';kw_thal['idx']=0
    system1 = rsp(**{'pardict':pd_thal_template2,**kw_thal})

    kw_vdp['model_name'] = 'vdp_thal1';kw_vdp['idx']=1
    system2 = rsp(**kw_vdp)


    fig,axs = plt.subplots(5,4,figsize=(8,6),gridspec_kw={'wspace':.4,'hspace':.6})


    het_coeffs_list = [(1,20),(1,20),(1,20),(1,20),(1,20)]
    pl_list = [(1,1),(1,2),(2,1),(2,3),(3,2)]
    del_list = [(-.1,-.2,-.3,-.8), # 11
                (-.086,-.1,-.14,-.2), # 12
                (-.1,-.145,-.15,-.4), # 21
                (-.106,-.109,-.12,-.14), # 23
                (-.108,-.112,-.12,-.13)] # 32
    etup = (.005,.1,100)

    

    in_dicts_list_s = [
        ( # 1:1
            [dict(phi0=2,eps_init=0.11,eps_final=0,deps=-.002)], # 1:1 b=-0.1
            [dict(phi0=2,eps_init=0.11,eps_final=0,deps=-.002)], # 1:1 b=-0.2
            [dict(phi0=3,eps_init=0.11,eps_final=0,deps=-.002)], # 1:1 b=-0.3
            [dict(phi0=2,eps_init=0.055,eps_final=0,deps=-.001),# 1:1 b=-0.8
             dict(phi0=2,eps_init=0.054,eps_final=0.1,deps=.001,max_iter=50)]
        ), 
                       
        (# 1:2
            [dict(phi0=4,eps_init=0.05,eps_final=0.001,deps=-.002), #1:2 b=-0.086
             dict(phi0=4,eps_init=0.049,eps_final=0.2,deps=.002)],
            [dict(phi0=4,eps_init=0.05,eps_final=0.001,deps=-.002), #1:2 b=-.1
             dict(phi0=4,eps_init=0.049,eps_final=0.102,deps=.002)],
            [dict(phi0=0,eps_init=0.2,eps_final=0.001,deps=-.01)], #1:2 b=-.14
            [dict(phi0=0,eps_init=0.125,eps_final=0.0,deps=-.002),
             dict(phi0=0,eps_init=0.124,eps_final=0.2,deps=.002)],# 1:2 b=-.2
        ),
        
        ( # 2:1
            [dict(phi0=2,eps_init=0.05,eps_final=0,deps=-.001), # 21 b=-.1
             dict(phi0=2,eps_init=0.049,eps_final=0.102,deps=.002),
             dict(phi0=2,eps_init=0.102,eps_final=0.15,deps=.01)],
            [dict(phi0=2,eps_init=0.05,eps_final=0.001,deps=-.001), # 21 b=-.145
             dict(phi0=4.4,eps_init=0.049,eps_final=0.102,deps=.002),
             dict(phi0=4.4,eps_init=0.1,eps_final=0.15,deps=.01)],
            [dict(phi0=2.6,eps_init=0.048,eps_final=0,deps=-.001), # 21 b=-.15
             dict(phi0=2,eps_init=0.047,eps_final=0.102,deps=.002),
             dict(phi0=2,eps_init=0.1,eps_final=0.15,deps=.01)],
            [dict(phi0=2,eps_init=0.09,eps_final=0.001,deps=-.001), # 21 b=-.4
             dict(phi0=2,eps_init=0.089,eps_final=0.15,deps=.001)]
        ),
        
        ( # 2:3
            [dict(phi0=0,eps_init=0.005,eps_final=0.0,deps=-.001), # 2:3 b=-.106
             dict(phi0=0,eps_init=0.004,eps_final=0.1,deps=.001)],
            [dict(phi0=0,eps_init=0.02,eps_final=0.001,deps=-.002), # 23 b=-.109
             dict(phi0=0,eps_init=0.019,eps_final=0.05,deps=.002)],
            [dict(phi0=0,eps_init=0.055,eps_final=0.001,deps=-.002), # 23 b=-.12
             dict(phi0=0,eps_init=0.054,eps_final=0.1,deps=.002)],
            [dict(phi0=0.5,eps_init=0.095,eps_final=0.001,deps=-.001), # 23 b=-.14
             dict(phi0=0.5,eps_init=0.094,eps_final=0.1,deps=.001)],
        ),

        ( # 3:2
             [dict(phi0=5,eps_init=0.018,eps_final=0.001,deps=-.001,max_iter=50), # 32 b=-.108
              dict(phi0=4.6,eps_init=0.017,eps_final=0.1,deps=.001,max_iter=50)],
             [dict(phi0=5.1,eps_init=0.02,eps_final=0.001,deps=-.001,max_iter=50), # 32 b=-.112
              dict(phi0=5.1,eps_init=0.0199,eps_final=0.1,deps=.001,max_iter=50)],
             [dict(phi0=5.1,eps_init=0.045,eps_final=0.001,deps=-.001), # 32 b=-.12
              dict(phi0=1.5,eps_init=0.044,eps_final=0.15,deps=.001)],
             [dict(phi0=4,eps_init=0.075,eps_final=0.001,deps=-.001), # 32 b=-.13
              dict(phi0=4,eps_init=0.074,eps_final=0.1,deps=.001)]
   
        ),

    ]

    Yinit_list = [
        [ # 1:1
            dict(phi0=5.5,eps_init=0.03,eps_final=0.01,deps=-1), #1:1,b=-.1
            dict(phi0=5.5,eps_init=0.05,eps_final=0.0,deps=-1), #1:1,b=-.2
            dict(phi0=5.5,eps_init=0.05,eps_final=0.0,deps=-1), #1:1,b=-.3
            dict(phi0=5,eps_init=0.055,eps_final=0.0,deps=-1)#1:1, b=-.8
        ],
        
        [# 1:2
            dict(phi0=2.5,eps_init=0.05,eps_final=0.045,deps=-.01), # 1:2 b=-0.086
            dict(phi0=2.5,eps_init=0.05,eps_final=0.045,deps=-1), # 1:2 b=-.1
            dict(phi0=2.5,eps_init=0.1,eps_final=0,deps=-1), # 1:2 b=-.14
            dict(phi0=5.5,eps_init=0.125,eps_final=0,deps=-1) # 1:2 b=-.2
        ],
        
        [ # 2:1
            dict(phi0=5,eps_init=0.2,eps_final=0.19,deps=-1), # 21 b=-.1
            dict(phi0=5,eps_init=0.049,eps_final=0,deps=-1), # 21 b=-.145
            dict(phi0=5.1,eps_init=0.049,eps_final=0,deps=-1), # 21 b=-.15
            dict(phi0=4.4,eps_init=0.09,eps_final=0.2,deps=1), # 21 b=-.4
         ],
        
        [ # 2:3
            dict(phi0=1.2,eps_init=0.005,eps_final=0,deps=-1), # 23 b=-.106
            dict(phi0=3,eps_init=0.02,eps_final=0.0,deps=-1), # 23 b=-.109
            dict(phi0=5.5,eps_init=0.055,eps_final=0,deps=-1), # 23 b=-.12
            dict(phi0=5.5,eps_init=0.095,eps_final=0,deps=-1), # 23 b=-.14
         ],
        
        [ # 3:2
            dict(phi0=3.5,eps_init=0.017,eps_final=0.0,deps=-1), # 32 b=-.108
            dict(phi0=3.5,eps_init=0.03,eps_final=0.01,deps=-1), # 32 b=-.112
            dict(phi0=3.5,eps_init=0.045,eps_final=0.01,deps=-1), # 32 b=-.12
            dict(phi0=3.5,eps_init=0.075,eps_final=0,deps=-1), # 32 b=-.13
         ],
    ]
    
    in_dicts_list_u = [
        ( # 1:1
            [dict(phi0=5.5,eps_init=0.03,eps_final=0,deps=-.002),
             dict(phi0=5.5,eps_init=0.029,eps_final=0.1,deps=.002)],# 1:1 b=-.1
            [dict(phi0=5.5,eps_init=0.05,eps_final=0,deps=-.002),
             dict(phi0=5.5,eps_init=0.049,eps_final=0.1,deps=.002)],  # 1:1 b=-.2
            [dict(phi0=5.5,eps_init=0.05,eps_final=0,deps=-.002),
             dict(phi0=5.5,eps_init=0.049,eps_final=0.1,deps=.002)], # 1:1 b=-.3
            [dict(phi0=5.5,eps_init=0.055,eps_final=0,deps=-.001),# 1:1 b=-.8
             dict(phi0=5.5,eps_init=0.054,eps_final=0.1,deps=.001)]
        ),
        
        ( # 1:2
            [dict(phi0=2.5,eps_init=0.05,eps_final=0.001,deps=-.002), 
             dict(phi0=2.5,eps_init=0.05,eps_final=0.2,deps=.002)], # 1:2 b=-.086
            [dict(phi0=2.5,eps_init=0.05,eps_final=0.001,deps=-.002),
             dict(phi0=2.5,eps_init=0.05,eps_final=0.2,deps=.002)], # 1:2 b=-.1
            [dict(phi0=2.5,eps_init=0.1,eps_final=0.0,deps=-.005),
             dict(phi0=2.5,eps_init=0.099,eps_final=0.21,deps=.005)], # 1:2 b=-.14
            [dict(phi0=5.5,eps_init=0.125,eps_final=0.0,deps=-.002),
             dict(phi0=5.5,eps_init=0.124,eps_final=0.14,deps=.002),] # 1:2 b=-.2
             
        ),
        
        ( # 2:1
            [dict(phi0=5,eps_init=0.05,eps_final=0,deps=-.001), # 21 b=-.1
             dict(phi0=5,eps_init=0.049,eps_final=0.102,deps=.002),
             dict(phi0=5,eps_init=0.1,eps_final=0.15,deps=.01)],
            [dict(phi0=5,eps_init=0.05,eps_final=0,deps=-.001), # 21 b=-.145
             dict(phi0=4.8,eps_init=0.049,eps_final=0.102,deps=.002),
             dict(phi0=4.8,eps_init=0.1,eps_final=0.15,deps=.01)],
            [dict(phi0=5,eps_init=0.05,eps_final=0.001,deps=-.001), # 21 b=-.15
             dict(phi0=5.1,eps_init=0.049,eps_final=0.102,deps=.002),
             dict(phi0=5.1,eps_init=0.1,eps_final=0.15,deps=.01)],
            [dict(phi0=4.4,eps_init=0.09,eps_final=0.001,deps=-.001), # 21 b=-.4
             dict(phi0=4.4,eps_init=0.089,eps_final=0.15,deps=.001)]
        ),
        
        ( # 2:3
            [dict(phi0=1.2,eps_init=0.005,eps_final=0,deps=-.0002), # 23 b=-.106
             dict(phi0=5.3,eps_init=0.004,eps_final=0.015,deps=.0002)],
            [dict(phi0=3,eps_init=0.02,eps_final=0.0,deps=-.002), # 23 b=-.109
             dict(phi0=3,eps_init=0.019,eps_final=0.05,deps=.002)],
            [dict(phi0=5.5,eps_init=0.055,eps_final=0.001,deps=-.001), # 23 b=-.12
             dict(phi0=5.5,eps_init=0.054,eps_final=0.1,deps=.001)],
            [dict(phi0=5.5,eps_init=0.095,eps_final=0.001,deps=-.001), # 23 b=-.14
             dict(phi0=5.5,eps_init=0.094,eps_final=0.15,deps=.001)]
        ),
        
        ( # 3:2
            [dict(phi0=3.5,eps_init=0.017,eps_final=0.001,deps=-.001), # 32 b=-.108
             dict(phi0=3.5,eps_init=0.016,eps_final=0.1,deps=.001)],
            [dict(phi0=3.5,eps_init=0.03,eps_final=0.001,deps=-.001), # 32 b=-.112
             dict(phi0=3.5,eps_init=0.0299,eps_final=0.1,deps=.001)],
            [dict(phi0=3.5,eps_init=0.045,eps_final=0.001,deps=-.001), # 32 b=-.12
             dict(phi0=3.,eps_init=0.044,eps_final=0.1,deps=.001)],
            [dict(phi0=3.5,eps_init=0.075,eps_final=0.001,deps=-.001), # 32 b=-.13
             dict(phi0=3.5,eps_init=0.074,eps_final=0.1,deps=.001)]
        ),
        
    ]

    ########### loop n:m
    for k in range(5):
        nn,mm = pl_list[k]
        nm_tup = pl_list[k]
        het_coeffs = het_coeffs_list[k]
        nm_val = str(nm_tup[0])+str(nm_tup[1])
        a = nm.nmCoupling(system1,system2,_n=('om0',nn),_m=('om1',mm),het_coeffs=het_coeffs,NH=1024)
        kw_b = {'a':a,'b':0,'recompute':False,'bifdir':'./v2_jupyter/bif1d_vdp_thal/','_full_rhs':_full}


        ########### loop b
        for j in range(4):


            kw_b['b'] = del_list[k][j]
            in_dicts = in_dicts_list_s[k][j]

            data_bs_list = []
            for i in range(len(in_dicts)):
                in1 = in_dicts[i]
                dat = follow_phase_diffs(**in1,**kw_b)
                data_bs_list.append(dat)

            dat,Yinit = follow_phase_diffs_u(**Yinit_list[k][j],return_point=True,**kw_b)

            in_dicts = in_dicts_list_u[k][j]
            print(in_dicts)

            data_bu_list = []
            for i in range(len(in_dicts)):
                in1 = in_dicts[i]
                dat = follow_phase_diffs_u(use_point=Yinit,**in1,**kw_b)
                data_bu_list.append(dat)

            # plot all stable
            draw_quick_plot_f(axs[k,j],data_bs_list,a,color='k')
            draw_quick_plot_f(axs[k,j],data_bu_list,a,ls='--',color='gray')


            add_diagram_1d(axs[k,j],a,del_list[k][j],etup,**kws_o1)
            add_diagram_1d(axs[k,j],a,del_list[k][j],etup,**kws_o2)


    axs_flat = np.asarray(axs).flatten()
    for k,ax in enumerate(axs_flat):
        ax.set_xlabel(r'$\varepsilon$')
        ax.xaxis.set_label_coords(1.05, 0.1) 
        
        ax.set_ylabel(r'$\phi$',labelpad=-10)
        ax.set_ylim(0,2*np.pi)
        ax.set_yticks([0,2*np.pi])
        ax.set_yticklabels(pi_label_short)
        ax.set_title(labels[k],loc='left')

    nr,nc = axs.shape
    
    for i in range(nr):
        for j in range(nc):
            ti1 = axs[i,j].get_title()
            ti1 += str(pl_list[i][0])+':'+str(pl_list[i][1])
            ti1 += r', $b = '+str(del_list[i][j])+'$'

            axs[i,j].set_title(ti1)

        


    plt.subplots_adjust(left=.04,right=.96,bottom=.04,top=.96)
    
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



def thal2_bif2_full_data(pl_list):

    dats_lo = []
    dats_hi = []

    for k in range(len(pl_list)):
        n = pl_list[k][0];m = pl_list[k][1]
        
        fname_lo = './v2_bifdat_2par/full/thal2_2par_lo_{}{}.dat'.format(n,m)
        dats_lo.append(np.loadtxt(fname_lo))
        
        fname_hi = './v2_bifdat_2par/full/thal2_2par_hi_{}{}.dat'.format(n,m)
        dats_hi.append(np.loadtxt(fname_hi))

    return dats_lo,dats_hi



def thal2_bif2_redu_data():

    datas = []
    twopar_neg = np.loadtxt('./v2_bifdat_2par/redu/thal2_11_o2_neg_fixed.dat')
    twopar_neg2 = np.loadtxt('./v2_bifdat_2par/redu/thal2_11_o2_neg2_fixed.dat')
    twopar_pos = np.loadtxt('./v2_bifdat_2par/redu/thal2_11_o2_pos_fixed.dat')
    
    d = copy.deepcopy(twopar_pos[:,0])
    twopar_pos[:,0] = twopar_pos[:,1]
    twopar_pos[:,1] = d    
    datas.append([twopar_neg,twopar_neg2,twopar_pos])
    
    twopar_neg = np.loadtxt('./v2_bifdat_2par/redu/thal2_12_o2_neg_fixed.dat')
    twopar_neg2 = np.loadtxt('./v2_bifdat_2par/redu/thal2_12_o2_neg2_fixed.dat')
    twopar_pos = np.loadtxt('./v2_bifdat_2par/redu/thal2_12_o2_pos_fixed.dat')
    datas.append([twopar_neg,twopar_neg2,twopar_pos])


    twopar_neg = np.loadtxt('./v2_bifdat_2par/redu/thal2_21_o2_neg_fixed.dat')
    twopar_neg2 = np.loadtxt('./v2_bifdat_2par/redu/thal2_21_o2_neg2_fixed.dat')
    twopar_pos = np.loadtxt('./v2_bifdat_2par/redu/thal2_21_o2_pos_fixed.dat')
    datas.append([twopar_neg,twopar_neg2,twopar_pos])

    
    twopar_neg = np.loadtxt('./v2_bifdat_2par/redu/thal2_13_o2_neg_fixed.dat')
    twopar_neg2 = np.loadtxt('./v2_bifdat_2par/redu/thal2_13_o2_neg2_fixed.dat')
    twopar_pos = np.loadtxt('./v2_bifdat_2par/redu/thal2_13_o2_pos_fixed.dat')
    datas.append([twopar_neg,twopar_neg2,twopar_pos])

    twopar_neg = np.loadtxt('./v2_bifdat_2par/redu/thal2_31_o2_neg_fixed.dat')
    twopar_neg2 = np.loadtxt('./v2_bifdat_2par/redu/thal2_31_o2_neg2_fixed.dat')
    twopar_pos = np.loadtxt('./v2_bifdat_2par/redu/thal2_31_o2_pos_fixed.dat')
    datas.append([twopar_neg,twopar_neg2,twopar_pos])

    
    twopar_neg = np.loadtxt('./v2_bifdat_2par/redu/thal2_23_2k_o2_neg_fixed.dat')
    twopar_neg2 = np.loadtxt('./v2_bifdat_2par/redu/thal2_23_2k_o2_neg2_fixed.dat')
    twopar_pos = np.loadtxt('./v2_bifdat_2par/redu/thal2_23_2k_o2_pos_fixed.dat')
    datas.append([twopar_neg,twopar_neg2,twopar_pos])

    twopar_neg = np.loadtxt('./v2_bifdat_2par/redu/thal2_32_2k_o2_neg_fixed.dat')
    twopar_neg2 = np.loadtxt('./v2_bifdat_2par/redu/thal2_32_2k_o2_neg2_fixed.dat')
    twopar_pos = np.loadtxt('./v2_bifdat_2par/redu/thal2_32_2k_o2_pos_fixed.dat')
    datas.append([twopar_neg,twopar_neg2,twopar_pos])

    
    twopar_neg = np.loadtxt('./v2_bifdat_2par/redu/thal2_34_1k_o2_neg_fixed.dat')
    twopar_neg2 = np.loadtxt('./v2_bifdat_2par/redu/thal2_34_1k_o2_neg2_fixed.dat')
    twopar_pos = np.loadtxt('./v2_bifdat_2par/redu/thal2_34_1k_o2_pos_fixed.dat')
    datas.append([twopar_neg,twopar_neg2,twopar_pos])

    
    twopar_neg = np.loadtxt('./v2_bifdat_2par/redu/thal2_43_1k_o2_neg_fixed.dat')
    twopar_neg2 = np.loadtxt('./v2_bifdat_2par/redu/thal2_43_1k_o2_neg2_fixed.dat')
    twopar_pos = np.loadtxt('./v2_bifdat_2par/redu/thal2_43_1k_o2_pos_fixed.dat')
    datas.append([twopar_neg,twopar_neg2,twopar_pos])

    
    return datas


def bif2_thal2_v2(NH=1024):

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

    pl_list = [(1,1),(1,2),(2,1),
               (1,3),(3,1),(2,3),
               (3,2),(3,4),(4,3)]

    N_list = [NH]*9

    xlims = [(-.1,.05),(-.1,.05),(-.1,.05),
             (-.1,.05),(-.1,.05),(-.02,.01),
             (-.04,.01),(-.02,.01),(-.015,.01),
             ]
    
    ylims = [(0,.18),(0,.18),(0,.18),
             (0,.18),(0,.18),(0,.15),
             (0,.15),(0,.15),(0,.15),
             ]

    padw=0.045; padh=0.05
    lo=0.07; hi=0.96

    fig,axs = plt.subplots(3,3,figsize=(8,7))
    axs = axs.flatten()

    nmax = 9
    datas_redu = thal2_bif2_redu_data()
    datas_full_lo,datas_full_hi = thal2_bif2_full_data(pl_list[:nmax])
    
    for k,ax in enumerate(axs):
        n = pl_list[k][0];m = pl_list[k][1]

        ax.set_title(labels[k],loc='left')
        ti1 = ax.get_title()
        ti1 += str(n)+':'+str(m)

        ax.set_title(ti1)

        #### plot redu 
        for dat in datas_redu[k]:
            ax.plot(dat[:,1],dat[:,0],color='#57acdc')

        #### plot full
        if k <= 8:
            b_vals_full = datas_full_lo[k][:,0]
            e_vals_full = datas_full_lo[k][:,1]
            e_hi_full = datas_full_hi[k][:,1]
        
            ax.fill_between(b_vals_full,e_vals_full,e_hi_full,color='gray',alpha=.25)
            ax.plot(b_vals_full,e_vals_full,marker='.',color='k',markersize=10,alpha=.5)
            ax.plot(b_vals_full,e_hi_full,marker='.',color='k',markersize=10,alpha=.5)

        
        ax.set_xlim(*xlims[k])
        ax.set_ylim(*ylims[k])

        ax.set_xlabel(r'$b$',labelpad=-2)
        ax.set_ylabel(r'$\varepsilon$')
        

    plt.subplots_adjust(hspace=.32,wspace=.3,left=.075,right=.97,bottom=.075,top=.95)
    return fig


def vdp_thal_bif2_redu_data():

    datas = []
    twopar_neg = np.loadtxt('./v2_bifdat_2par/redu/vdp_thal_11_o2_neg_fixed.dat')
    twopar_neg2 = np.loadtxt('./v2_bifdat_2par/redu/vdp_thal_11_o2_neg2_fixed.dat')
    twopar_pos = np.loadtxt('./v2_bifdat_2par/redu/vdp_thal_11_o2_pos_fixed.dat')
    datas.append([twopar_neg,twopar_neg2,twopar_pos])

    twopar_neg = np.loadtxt('./v2_bifdat_2par/redu/vdp_thal_12_o2_neg_fixed.dat')
    twopar_neg2 = np.loadtxt('./v2_bifdat_2par/redu/vdp_thal_12_o2_neg2_fixed.dat')
    datas.append([twopar_neg,twopar_neg2])

    twopar_neg = np.loadtxt('./v2_bifdat_2par/redu/vdp_thal_21_o2_neg_fixed.dat')
    twopar_neg2 = np.loadtxt('./v2_bifdat_2par/redu/vdp_thal_21_o2_neg2_fixed.dat')
    twopar_pos = np.loadtxt('./v2_bifdat_2par/redu/vdp_thal_21_o2_pos_fixed.dat')
    datas.append([twopar_neg,twopar_neg2,twopar_pos])

    twopar_neg = np.loadtxt('./v2_bifdat_2par/redu/vdp_thal_23_o2_neg_fixed.dat')
    twopar_neg2 = np.loadtxt('./v2_bifdat_2par/redu/vdp_thal_23_o2_neg2_fixed.dat')
    datas.append([twopar_neg,twopar_neg2])

    twopar_neg = np.loadtxt('./v2_bifdat_2par/redu/vdp_thal_32_o2_neg_fixed.dat')
    twopar_neg2 = np.loadtxt('./v2_bifdat_2par/redu/vdp_thal_32_o2_neg2_fixed.dat')
    datas.append([twopar_neg,twopar_neg2])

    return datas


def vdp_thal_bif2_full_data():

    dats_lo = []
    dats_hi = []

    pl_list = [(1,1),(1,2),(2,1),(2,3),(3,2)]

    for k in range(len(pl_list)):
        n = pl_list[k][0];m = pl_list[k][1]
        
        fname_lo = './v2_bifdat_2par/full/vdp_thal_2par_lo_{}{}.dat'.format(n,m)
        dats_lo.append(np.loadtxt(fname_lo))
        
        fname_hi = './v2_bifdat_2par/full/vdp_thal_2par_hi_{}{}.dat'.format(n,m)
        dats_hi.append(np.loadtxt(fname_hi))

    return dats_lo,dats_hi

def bif2_vdp_thal():

    pl_list = [(1,1),(1,2),(2,1),(2,3),(3,2)]
    xlims = [(-1.01,.05),(-.4,-.05),(-.6,.006),(-.25,-.1),(-.31,-.09)]
    ylims = [(0,.1),(0,.19),(0,.13),(0,.15),(0,.15)]

    fig = plt.figure(figsize=(8,5))

    ax1 = plt.subplot2grid(shape=(2,6), loc=(0,0), colspan=2)
    ax2 = plt.subplot2grid((2,6), (0,2), colspan=2)
    ax3 = plt.subplot2grid((2,6), (0,4), colspan=2)
    ax4 = plt.subplot2grid((2,6), (1,1), colspan=2)
    ax5 = plt.subplot2grid((2,6), (1,3), colspan=2)

    axs = np.asarray([ax1,ax2,ax3,ax4,ax5])

    datas_redu = vdp_thal_bif2_redu_data()
    datas_full_lo,datas_full_hi = vdp_thal_bif2_full_data()
    
    for k,ax in enumerate(axs):
        n = pl_list[k][0];m = pl_list[k][1]

        ax.set_title(labels[k],loc='left')
        ti1 = ax.get_title()
        ti1 += str(n)+':'+str(m)

        ax.set_title(ti1)

        #### plot redu 
        for dat in datas_redu[k]:
            ax.plot(dat[:,1],dat[:,0],color='#57acdc')

        #### plot full
        b_vals_full = datas_full_lo[k][:,0]
        e_vals_full = datas_full_lo[k][:,1]
        e_hi_full = datas_full_hi[k][:,1]
        
        ax.fill_between(b_vals_full,e_vals_full,e_hi_full,color='gray',alpha=.25)
        ax.plot(b_vals_full,e_vals_full,marker='.',color='k',markersize=10,alpha=.5)
        ax.plot(b_vals_full,e_hi_full,marker='.',color='k',markersize=10,alpha=.5)

        ax.set_xlim(*xlims[k])
        ax.set_ylim(*ylims[k])

        
    
    plt.subplots_adjust(hspace=.32,wspace=.7,left=.075,right=.97,bottom=.075,top=.95)

    
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
        (forcing_fn,[],['figs/f_forcing.pdf']),
        
        (tongues_thal1_v2,[],['figs/f_tongues_thal_v2.pdf']),
        (tongues_cgl1_v2,[],['figs/f_tongues_cgl_v2.pdf']),

        (traj_cgl1,[],['figs/f_traj_cgl1.pdf']),
        (traj_thal1,[],['figs/f_traj_thal1.pdf']),
        (traj_thal2,[1024],['figs/f_traj_thal2.pdf']),
        
        (bif1d_cgl1,[],['figs/f_bif1d_cgl1.pdf']),
        (bif1d_thal1,[],['figs/f_bif1d_thal1.pdf']),
        
        (bif_thal2_11,[],['figs/f_bif1d_thal2_11.pdf']),
        #(bif_thal2a,[],['figs/f_bif1d_thal2a.pdf']),
        (bif_thal2b,[],['figs/f_bif1d_thal2b.pdf']),

        (bif1_vdp_thal,[],['figs/f_bif1d_vdp_thal.pdf']),
        
        (bif2_thal2_v2,[],['figs/f_bif2_thal2_v2.pdf']),
        (bif2_vdp_thal,[],['figs/f_bif2_vdp_thal.pdf']),
        
    ]
    
    for fig in figures:
        generate_figure(*fig)


if __name__ == "__main__":
    
    __spec__ = None
    
    main()
