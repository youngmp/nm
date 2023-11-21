"""
Generate figures for strong coupling paper
"""


# user-defined
import nmCoupling as nm
import response as rp
#import cgl_thalamic as ct
import cgl1 as c1
from cgl_thalamic import rhs_cgl

nmc = nm.nmCoupling;rsp = rp.Response

import os
import matplotlib.gridspec as gridspec
import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt
import string
import sympy as sym
import scipy as sp

from matplotlib import cm
from scipy.signal import find_peaks
from scipy.integrate import solve_ivp
#from matplotlib.legend_handler import HandlerBase
from scipy.optimize import brentq, root, bisect

#from mpl_toolkits.axes_grid1 import make_axes_locatable

import copy


matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['pgf.texsystem'] = 'pdflatex'
preamble = (r'\usepackage{amsmath}'
            r'\usepackage{siunitx}'
            r'\usepackage{bm}'
            r'\newcommand{\ve}{\varepsilon}')

matplotlib.rcParams['text.latex.preamble'] = preamble

fontsize = 12

def ct_redu(t,y,eps,system1,del1):
    h = 0
    for i in range(system1.miter):
        h += eps**(i+1)*(system1.h['lam'][i](y))
    h += del1
    return h

def ct_full(t,y,eps,pd1,pd2,del1):
    vector_field = rhs_cgl(t,y,pd1,'val',0)
    coupling = c1.coupling_cgl([*y,system2(t,del1)],pd2,'val',1)
    return np.array(list(vector_field + eps*couling))

def ff(x,a,om,del1=0):
    return a*np.sin((om+del1)*x)+.2


def pl_exist(eps,del1,system1):
    xtemp = np.linspace(0,2*np.pi,1000)
    h = ct_redu(0,xtemp,eps,system1,del1)
    
    z1 = xtemp[1:][(h[1:]>0)*(h[:-1]<=0)]
    z2 = xtemp[1:][(h[1:]<0)*(h[:-1]>=0)]
    if eps == 0:
        return -1
        
    if len(z1)+len(z2) > 0:
        return 1
    else:
        return -1

def get_tongue(del_list,system1,deps=.002,max_eps=.3,min_eps=0):

    ve_exist = np.zeros(len(del_list))
    
    for i in range(len(del_list)):
        print(np.round((i+1)/len(del_list),2),'    ',end='\r')
        eps = min_eps
        while not(pl_exist(eps,del_list[i],system1)+1)\
        and eps <= max_eps:
            eps += deps
        if eps >= max_eps:
            ve_exist[i] = np.nan
        else:
            out = bisect(pl_exist,0,eps+deps,args=(del_list[i],system1))
            ve_exist[i] = out
    print('')
    return del_list,ve_exist

def tongues_cgl():
    pd1 = {'q':1,'d':.9,'sig':.1,'rho':.15,'mu':.15,
       'om':1,'om_fix':1,'alc':3,'bec':2,'esyn':0}
    pd2 = {'om':1,'amp':1,'om_fix':1}

    kws1 = {'var_names':['x','y','w'],
            'rhs':rhs_cgl,
            'init':np.array([.333,0,0,2*np.pi]),
            'TN':2000,
            'idx':0,
            'model_name':'cglf0',
            'trunc_order':5,
            'recompute_list':[],
            'z_forward':False,
            'i_forward':False,
            'i_bad_dx':False,
            'coupling':c1.coupling_cgl}

    # default period must be 2*np.pi
    kws2 = {'var_names':[],
            'rhs':None,
            'init':None,
            'coupling':None,
            'model_name':'f1',
            'forcing_fn':ff,
            'idx':1,
            'TN':0}


    data_dir = 'tongue_data'
    if not(os.path.isdir(data_dir)):
        os.mkdir(data_dir)

    system1 = rsp(pardict=copy.deepcopy(pd1),**copy.deepcopy(kws1))
    system2 = rsp(pardict=copy.deepcopy(pd2),**copy.deepcopy(kws2))

    ############## 1:1
    dtemp = np.linspace(-10,0,100)
    data11a = load_forcing(1,1,system1,system2,dtemp,dir1=data_dir,
                          log=True,sign=1)
    data11b = load_forcing(1,1,system1,system2,dtemp,dir1=data_dir,
                          log=True,sign=-1)

    ############## 1:3
    dtemp = np.linspace(-10,0,100)
    data13a = load_forcing(1,3,system1,system2,dtemp,dir1=data_dir,
                           log=True,sign=1)
    data13b = load_forcing(1,3,system1,system2,dtemp,dir1=data_dir,
                           log=True,sign=-1,recompute=False)

    ############## 1:2
    data12a = load_forcing(1,2,system1,system2,dtemp,dir1=data_dir,
                          log=True,sign=1)
    data12b = load_forcing(1,2,system1,system2,dtemp,dir1=data_dir,
                           log=True,sign=-1,recompute=False)

    ############## 2:3
    data23a = load_forcing(2,3,system1,system2,dtemp,dir1=data_dir,
                          log=True,sign=1)
    data23b = load_forcing(2,3,system1,system2,dtemp,dir1=data_dir,
                           log=True,sign=-1,recompute=False)
    
    cmap = matplotlib.colormaps['viridis']
    

    fig,axs = plt.subplots(figsize=(8,3))

    axs.plot(1/(3+data13a[:,0]),data13a[:,1],color=cmap(0),label='$1{:}3$')
    axs.plot(1/(3+data13b[:,0]),data13b[:,1],color=cmap(0))

    axs.plot(1/(2+data12a[:,0]),data12a[:,1],color=cmap(.3),ls='--',
             label='$1{:}2$')
    axs.plot(1/(2+data12b[:,0]),data12b[:,1],color=cmap(.3),ls='--')

    axs.plot(2/(3+data23a[:,0]),data23a[:,1],color=cmap(.6),ls=':',
             label='$2{:}3$')
    axs.plot(2/(3+data23b[:,0]),data23b[:,1],color=cmap(.6),ls=':')
    
    axs.plot(1/(1+data11a[:,0]),data11a[:,1],color=cmap(.9),ls='-.',
             label='$1{:}1$')
    axs.plot(1/(1+data11b[:,0]),data11b[:,1],color=cmap(.9),ls='-.')

    axs.set_xlim(0,1.2)
    axs.set_ylim(0,.5)

    #axs.set_yscale('log')
    axs.set_xlabel(r'$\omega_X/(\omega_f+\delta)$',fontsize=fontsize)
    axs.set_ylabel(r'$\varepsilon$',fontsize=fontsize)

    axs.set_xticks([1/3,1/2,2/3,1])
    axs.set_xticklabels(['$1/3$','$1/2$','$2/3$','$1$'],
                        fontsize=fontsize)

    axs.legend()

    return fig
    #plt.savefig('../')
    
    
    

def load_forcing(n,m,system1,system2,dtemp,dir1='',recompute=False,
                 log=False,sign=1):

    ratio = str(n)+str(m)
    rawname = '{}/cgl1f0_ratio={}_sign={}_d0={}_de={}_dN={}_log={}.txt'
    fname = rawname.format(dir1,ratio,sign,dtemp[0],dtemp[-1],len(dtemp),log)
    
    file_dne = not(os.path.isfile(fname))

    if file_dne or recompute:
        a = nmc(system1,system2,
                _n=('om0',n),_m=('om1',m),
                NP=300,NH=300)

        if log:
            in1 = sign*10**dtemp
        else:
            in1 = dtemp

        del_list,ve_list = get_tongue(in1,system1,deps=.002,max_eps=1)

        data = np.zeros([len(del_list),2])
        data[:,0] = del_list
        data[:,1] = ve_list

        np.savetxt(fname,data)
        
    else:
        data = np.loadtxt(fname)

    return data

def generate_figure(function, args, filenames, dpi=100):
    # workaround for python bug where forked processes use the same random 
    # filename.
    #tempfile._name_sequence = None;

    fig = function(*args)

    if type(filenames) == list:
        for name in filenames:
            if name.split('.')[-1] == 'ps':
                fig.savefig(name, orientation='landscape',dpi=dpi)
            else:
                fig.savefig(name,dpi=dpi,bbox_inches='tight')
    else:
        if name.split('.')[-1] == 'ps':
            fig.savefig(filenames,orientation='landscape',dpi=dpi)
        else:
            fig.savefig(filenames,dpi=dpi)

def main():

    #quick_plots_thalamic()

    # create figs directory if it doesn't exist
    if not(os.path.isdir('figs')):
        os.mkdir('figs')
    
    # listed in order of Figures in paper
    figures = [

        (tongues_cgl,[],['figs/f_tongues_cgl.pdf',
                          'figs/f_tongues_cgl.png'],200),

    ]
    
    for fig in figures:
        generate_figure(*fig)


if __name__ == "__main__":
    
    __spec__ = None
    
    main()
