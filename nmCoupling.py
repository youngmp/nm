"""
nmCoupling.py computes the higher-order interaction functions
for $N=2$ heterogeneous models

For a given oscillator,
* Solve for $\phi$ in terms of $\\theta_i$, (13), (14) (Park and Wilson 2020)
* Compute the higher-order interaction functions (15) (Park and Wilson 2020)

Assumptions:
-N=2, two models
-No self coupling
-One nontrivial Floquet multiplier for each model
"""


import copy
import lib.lib_sym as slib


#import lib.lib as lib
from lib import lib
from lib.interp_basic import interp_basic as interpb
from lib.lambdifyn import lambdifyn as ldn
#from lib.fast_splines import interp2d
from lib.fast_interp import interp2d
from lib import fnames
#from lam_vec import lam_vec

#import inspect
import time
import os

import math
import sys
#import multiprocessing as multip
import tqdm
from pathos.pools import ProcessPool
from pathos.pools import _ProcessPool
import multiprocessing as mp
from multiprocessing import shared_memory

import scipy.interpolate as si
import numpy as np
from numpy.fft import fft,ifft,fftfreq,fft2

#import scipy as sp
import sympy as sym
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import dill
dill.settings['recurse'] = True

from sympy import Matrix, symbols, Sum, Indexed, collect, expand
from sympy import sympify
from sympy.physics.quantum import TensorProduct as kp
from sympy.utilities.lambdify import lambdify, implemented_function
from sympy.matrices.dense import matrix_multiply_elementwise as me
from sympy.tensor.array import derive_by_array

from scipy.signal import fftconvolve
#import pdoc

from itertools import chain
import numexpr as ne

imp_fn = implemented_function

from scipy.interpolate import RectBivariateSpline as interp2rbso
from scipy.interpolate import interpn, RegularGridInterpolator
from scipy.integrate import solve_ivp, trapezoid

from collections.abc import Iterable

import logging # for debugging
import warnings

logging.getLogger('matplotlib.font_manager').disabled = True

np.set_printoptions(linewidth=np.inf)

import warnings

warnings.filterwarnings('ignore',category=dill.PicklingWarning)
warnings.filterwarnings('ignore',category=dill.UnpicklingWarning)

import pyfftw

fftw = pyfftw.interfaces.numpy_fft.fft
ifftw = pyfftw.interfaces.numpy_fft.ifft
fftw2 = pyfftw.interfaces.numpy_fft.fft2
ifft2 = pyfftw.interfaces.numpy_fft.ifft2
fftwn = pyfftw.interfaces.numpy_fft.fftn
ifftn = pyfftw.interfaces.numpy_fft.ifftn

exp = np.exp

#try:
#    import cupy as cp
#    import cupyx.scipy.fft as cufft
#    import scipy.fft
#    scipy.fft.set_global_backend(cufft)
#except ImportError:
#    warnings.warn('Warning: cupy not available. This is ok'
#                  'if you are not calculating G and H functions')

def flatten(xs):
    for x in xs:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from flatten(x)
        else:
            yield x

def module_exists(module_name):
    try:
        __import__(module_name)
    except ImportError:
        return False
    else:
        return True


class nmCoupling(object):    
    def __init__(self,system1,system2,
                 NP=100,NH=100,
                 trunc_order=3,trunc_deriv=3,
                 TN=20000,

                 _n=(None,1),_m=(None,1),
                 
                 max_iter=100,
                 rtol=1e-7,atol=1e-7,rel_tol=1e-6,                 
                 method='LSODA', # good shit
                 
                 max_n=-1, # Fourier terms. -1 = no truncation
                 
                 gij_parallel=True,processes=6,
                 chunksize=1,chunk_g=1,process_g=5,
                 log_level='CRITICAL',log_file='log_nm.log',
                 
                 recompute_list=[]):

        """
        N=2 only.
        
        Reserved names: ...        
        """
        
        assert(type(_n) is tuple);assert(type(_m) is tuple)
        assert(_n[0] in system1.pardict);assert(_m[0] in system2.pardict)
        
        self.system1 = system1;self.system2 = system2
        self.recompute_list = recompute_list

        self.trunc_deriv = trunc_deriv
        self._n = _n;self._m = _m
        self.Tx = system1.T/self._n[1]
        self.Ty = system2.T/self._m[1]

        self.om = self._n[1]/self._m[1]
        self.T = self._m[1]*2*np.pi
        
        ths_str = ' '.join(['th'+str(i) for i in range(2)])
        self.ths = symbols(ths_str)

                # discretization for p_X,p_Y
        self.NP = NP
        self.an,self.dan = np.linspace(0,2*np.pi,NP,
                                       retstep=True,endpoint=False)

        # discretization for H_X,H_Y
        self.NH = NH
        self.bn,self.dbn = np.linspace(0,2*np.pi*self._m[1],NH,
                                       retstep=True,endpoint=False)

        

        self.log_level = log_level;self.log_file = log_file
        
        if self.log_level == 'DEBUG':
            self.log_level = logging.DEBUG
        elif self.log_level == 'INFO':
            self.log_level = logging.INFO
        elif self.log_level == 'WARNING':
            self.log_level = logging.WARNING
        elif self.log_level == 'ERROR':
            self.log_level = logging.ERROR
        elif self.log_level == 'CRITICAL':
            self.log_level = logging.CRITICAL

        FORMAT = '%(asctime)s %(message)s'
        logging.basicConfig(filename=self.log_file,level=self.log_level,
                            format=FORMAT)

        system1.h['dat']={};system1.h['imp']={};system1.h['lam']={}
        system1.h['sym']={};system1.p['dat']={};system1.p['imp']={}
        system1.p['lam']={};system1.p['sym']={}
        

        # global replacement rules based on system index
        system1.rule_lc={};system1.rule_g={};system1.rule_z={}
        system1.rule_i={};system1.rule_p={}
        
        system2.rule_lc={};system2.rule_g={};system2.rule_z={}
        system2.rule_i={};system2.rule_p={}

        for j,key in enumerate(system1.var_names):
            in1 = self.ths[system1.idx]
            system1.rule_lc[system1.syms[j]] = system1.lc['imp_'+key](in1)
            
        for key in system1.var_names:
            for k in range(system1.miter):
                in1 = self.ths[system1.idx]

                kg = sym.Indexed('g_'+system1.model_name+'_'+key,k)
                system1.rule_g[kg] = system1.g['imp_'+key][k](in1)

                kz = sym.Indexed('z_'+system1.model_name+'_'+key,k)
                system1.rule_z[kz] = system1.z['imp_'+key][k](in1)
                
                ki = sym.Indexed('i_'+system1.model_name+'_'+key,k)
                system1.rule_i[ki] = system1.i['imp_'+key][k](in1)

        

        self.forcing = system2.forcing
        if not(system2.forcing):
            self._init_noforce(system1,system2)
            
        else:
            self._init_force(system1,system2)

    def _init_noforce(self,system1,system2):
        self.t,self.dt = np.linspace(0,2*np.pi,100,retstep=True)
                
        pfactor1 = int((np.log(.02)/system1.kappa_val))        
        pfactor2 = int((np.log(.02)/system2.kappa_val))
        
        self.pfactor = max([pfactor1,pfactor2])
        
        # just keep these here.
        system2.h['dat']={};system2.h['imp']={};system2.h['lam']={}
        system2.h['sym']={};system2.p['dat']={};system2.p['imp']={}
        system2.p['lam']={};system2.p['sym']={}
        
        
        
        
        for j,key in enumerate(system2.var_names):
            in1 = self.ths[system2.idx]
            system2.rule_lc[system2.syms[j]] = system2.lc['imp_'+key](in1)
        
        for key in system2.var_names:
            for k in range(system2.miter):
                in1 = self.ths[system2.idx]
                
                kg = sym.Indexed('g_'+system2.model_name+'_'+key,k)
                system2.rule_g[kg] = system2.g['imp_'+key][k](in1)

                kz = sym.Indexed('z_'+system2.model_name+'_'+key,k)
                system2.rule_z[kz] = system2.z['imp_'+key][k](in1)
                
                ki = sym.Indexed('i_'+system2.model_name+'_'+key,k)
                system2.rule_i[ki] = system2.i['imp_'+key][k](in1)

        
        
        fnames.load_fnames_nm(system1,self)
        fnames.load_fnames_nm(system2,self)
        slib.load_coupling_expansions(system1)
        slib.load_coupling_expansions(system2)

        # mix up terms. G and K
        self.load_k_sym(system1,system2)
        self.load_k_sym(system2,system1)

        # load odes for p_i
        self.load_p_sym(system1)
        self.load_p_sym(system2)

        self.load_p(system1,system2,0)
        self.load_p(system2,system1,0)

        self.load_p(system1,system2,1)
        self.load_p(system2,system1,1)

        self.load_p(system1,system2,2)
        self.load_p(system2,system1,2)

        self.load_h_sym(system1)
        self.load_h_sym(system2)

        
        self.load_h(system1,system2,0)
        self.load_h(system2,system1,0)

        self.load_h(system1,system2,1)
        self.load_h(system2,system1,1)

        self.load_h(system1,system2,2)
        self.load_h(system2,system1,2)

    def _init_force(self,system1,system2):
        # system 2 should be forcing function

        

        self.pfactor = int((np.log(.02)/system1.kappa_val)/self._m[1])

        system2.pardict['om'+str(system2.idx)] = self._m[1]

        fnames.load_fnames_nm(system1,self)
        slib.load_coupling_expansions(system1)

        self.load_k_sym(system1,system2)
        self.load_p_sym(system1)

        for k in range(system1.miter):
            self.load_p(system1,system2,k)
            
        self.load_h_sym(system1)

        for k in range(system1.miter):
            self.load_h(system1,system2,k)

    def load_k_sym(self,system1,system2):
        
        """
        G is the full coupling function expanded.
        K is the same, but indexed by powers of eps.

        """
            
        # check that files exist
        val = not(os.path.isfile(system1.G['fname']))
        val += not(os.path.isfile(system1.K['fname']))

        files_dne = val

        system1.G['sym'] = {}
        system1.K['sym'] = {}
        if 'k_'+system1.model_name in self.recompute_list or files_dne:
            logging.info('* Computing G symbolic...')

            rule_trunc = {}
            for k in range(system1.miter,system1.miter+500):
                rule_trunc.update({system1.eps**k:0})

            self.generate_k_sym(system1,system2)

            # now collect terms
            # G contains full summation
            # K contains terms of summation indexed by power
            
            for key in system1.var_names:

                tmp = expand(system1.G['sym'][key],basic=True,
                             deep=True,power_base=False,
                             power_exp=False,
                             mul=True,log=False,
                             multinomial=True)
                system1.G['sym'][key] = tmp.subs(rule_trunc)
                
                system1.K['sym'][key] = []
                for k in range(system1.miter):
                    e_term = tmp.coeff(system1.eps,k)
                    system1.K['sym'][key].append(e_term)

            dill.dump(system1.G['sym'],open(system1.G['fname'],'wb'),
                      recurse=True)
            dill.dump(system1.K['sym'],open(system1.K['fname'],'wb'),
                      recurse=True)
                
        else:
            logging.info('* Loading G symbolic...')
            system1.G['sym'] = dill.load(open(system1.G['fname'],'rb'))
            system1.K['sym'] = dill.load(open(system1.K['fname'],'rb'))

        system1.G['vec'] = sym.zeros(system1.dim,1)
    
        for i,key in enumerate(system1.var_names):
            system1.G['vec'][i] = [system1.G['sym'][key]]

    def generate_k_sym(self,system1,system2):
        """
        generate terms involving the coupling term (see K in paper).
        Equations (8), (9) in Park, Wilson 2021
        """

        psym = system1.pardict_sym # shorten for readability
        fn = system1.coupling # coupling function

        # all variables and d_variables
        var_names_all = system1.var_names + system2.var_names
        z = sym.Matrix(system1.syms + system2.syms)
        dz = sym.Matrix(system1.dsyms + system2.dsyms)

        # get full coupling term for given oscillator
        system1.G['sym_fn'] = sym.flatten(fn(z,psym,option='sym',
                                             idx=system1.idx))

        if self.forcing:
            assert(z[-1,0] == system2.syms[0])
            z = z[:-1,:]
            dz = dz[:-1,:]
            
        
        # 0 and 1st derivative
        c_temp = {}
        for key_idx,key in enumerate(system1.var_names):
            c_temp[key] = system1.G['sym_fn'][key_idx]
            
            d = lib.df(system1.G['sym_fn'][key_idx],z,1).dot(dz)
            c_temp[key] += d
            
        # 2nd + derivative        
        for key_idx,key in enumerate(system1.var_names):
            for k in range(2,self.trunc_deriv+1):
                kp = lib.kProd(k,dz)
                da = lib.vec(lib.df(system1.G['sym_fn'][key_idx],z,k))
                c_temp[key] += (1/math.factorial(k))*kp.dot(da)

        # save 
        for key in system1.var_names:
            system1.G['sym'][key] = c_temp[key]

                        
        # replace dx with (g expansion in eps).
        rule = {}

        for key_idx,key in enumerate(system1.var_names):
            rule.update({system1.dsyms[key_idx]:system1.g[key+'_eps']})
        for key_idx,key in enumerate(system2.var_names):
            rule.update({system2.dsyms[key_idx]:system2.g[key+'_eps']})

        for key in system1.var_names:
            system1.G['sym'][key] = system1.G['sym'][key].subs(rule)

        
    def load_p_sym(self,system1):
        """
        generate/load the het. terms for psi ODEs.
            
        to be solved using integrating factor meothod.        
        system*.p[k] is the forcing function for oscillator * of order k
        """

        file_dne = not(os.path.isfile(system1.p['fname']))

        eps = system1.eps

        if 'p_'+system1.model_name in self.recompute_list or file_dne:
            logging.info('* Computing p symbolic...')
            print('* Computing p symbolic...')
            
            rule_trunc = {}
            for k in range(system1.miter,system1.miter+500):
                rule_trunc.update({eps**k:0})

            v1 = system1.i['vec']; v2 = system1.G['vec']
            
            tmp = eps*v1.dot(v2)
            tmp = expand(tmp,basic=True,deep=True,
                         power_base=False,power_exp=False,
                         mul=True,log=False,multinomial=True)
            
            tmp = tmp.subs(rule_trunc)
            tmp = collect(tmp,system1.eps)
            tmp = collect(expand(tmp),system1.eps)

            for k in range(system1.miter):
                system1.p['sym'][k] = tmp.coeff(system1.eps,k)

            dill.dump(system1.p['sym'],open(system1.p['fname'],'wb'),
                      recurse=True)

        else:
            logging.info('* Loading p symbolic...')
            print('* Loading p symbolic...')
            system1.p['sym'] = dill.load(open(system1.p['fname'],'rb'))
            

    def load_p(self,system1,system2,k):

        fname = system1.p['fnames_data'][k]
        file_dne = not(os.path.isfile(fname))

        if 'p_data_'+system1.model_name in self.recompute_list\
           or file_dne:
            print('* Computing p...')
            
            p_data = self.generate_p(system1,system2,k)

            if not(self.forcing):
                p_interp0 = interp2d([0,0],[2*np.pi,2*np.pi],
                                 [self.dan,self.dan],
                                 p_data,k=3,p=[True,True])

                # fix indexing (see check_isostable.ipynb)
                X,Y = np.meshgrid(self.an,self.an,indexing='ij')
                p_data = p_interp0(X+Y*self.om,Y)
                
            np.savetxt(system1.p['fnames_data'][k],p_data)

        else:
            print('* Loading p...')
            p_data = np.loadtxt(fname)

        if self.forcing:
            p_interp = interp2d([0,0],[2*np.pi,2*np.pi*self._m[1]],
                                [self.dan,self.dan],
                                p_data,k=3,p=[True,True])
        else:
            p_interp = interp2d([0,0],[2*np.pi,2*np.pi],
                                [self.dan,self.dan],
                                p_data,k=3,p=[True,True])

        ta = self.ths[0];tb = self.ths[1]
        name = 'p_'+system1.model_name+'_'+str(k)
        p_imp = imp_fn(name,self.fast_interp_lam(p_interp))
        lamtemp = lambdify(self.ths,p_imp(ta,tb))
        
        system1.p['dat'][k] = p_data

        if k == 0:
            imp = imp_fn('p_'+system1.model_name+'_0', lambda x: 0*x)
            system1.p['imp'][k] = imp
            system1.p['lam'][k] = 0
            
        else:
            system1.p['imp'][k] = p_imp
            system1.p['lam'][k] = lamtemp#p_interp

        if True:
            
            fig,axs = plt.subplots()
            axs.imshow(p_data)
            plt.savefig('figs_temp/p_'+system1.model_name+str(k)+'.png')
            plt.close()

        # put these implemented functions into the expansion
        system1.rule_p.update({sym.Indexed('p_'+system1.model_name,k):
                               system1.p['imp'][k](ta,tb)})

    def generate_p(self,system1,system2,k):

        NP = self.NP
        if k == 0:
            #pA0 is 0 (no forcing function)
            return np.zeros((NP,NP))

        kappa1 = system1.kappa_val

        rule = {**system1.rule_p,**system2.rule_p,
                **system1.rule_par,**system2.rule_par,
                **system1.rule_lc,**system2.rule_lc,
                **system1.rule_i,**system2.rule_i,
                **system1.rule_g,**system2.rule_g}
        
        if self.forcing:
            imp0 = imp_fn('forcing',system2)
            lam0 = lambdify(self.ths[1],imp0(self.ths[1]))
            rule.update({system2.syms[0]:imp0(self.ths[1])})

        
        ph_imp1 = system1.p['sym'][k].subs(rule)
        lam1 = lambdify(self.ths,ph_imp1)
        het1 = np.zeros((NP,int(self._m[1]*NP)))
                   
        an=self.an;dan=self.dan;pfactor=self.pfactor
        NP=self.NP;s=np.arange(0,self.T*pfactor*self._m[1],dan)
            
        fac = self.om*(1-system1.idx) + system1.idx
        exp1 = exp(fac*s*system1.kappa_val)

        g_in = np.fft.fft(exp1)
        a_i = np.arange(NP,dtype=int)
        
        for ll in range(len(an)):

            f_in = np.fft.fft(lam1(an[ll]+self.om*s,+s))
            conv = np.fft.ifft(f_in*g_in)
            
            if self.forcing:                
                het1[ll,:] = conv[-int(self._m[1]*NP):].real
            else:
                het1[(a_i+ll)%NP,a_i] = conv[-NP:].real

        return fac*het1*dan
    
    
    def load_h_sym(self,system):
        """
        also compute h lam
        """

        fname = system.h['fname']
        file_dne = not(os.path.isfile(fname))

        # symbolic h terms
        system.h['sym'] = {}
        
        if 'h_'+system.model_name in self.recompute_list or file_dne:
            logging.info('* Computing H symbolic...')
            print('* Computing H symbolic...')
            
            # simplify expansion for speed
            rule_trunc = {}
            for k in range(system.miter,system.miter+200):
                rule_trunc.update({system.eps**k:0})

            logging.info('Oscillator '+system.model_name)
            v1 = system.z['vec']; v2 = system.G['vec']
                    
            tmp = v1.dot(v2)
            
            tmp = sym.expand(tmp,basic=True,deep=True,
                             power_base=False,power_exp=False,
                             mul=True,log=False,
                             multinomial=True)
            
            tmp = tmp.subs(rule_trunc)
            tmp = collect(tmp,system.eps)
            tmp = collect(expand(tmp),system.eps)

            for k in range(system.miter):
                system.h['sym'][k] = tmp.coeff(system.eps,k)

            dill.dump(system.h['sym'],open(fname,'wb'),recurse=True)
                
        else:
            logging.info('* Loading H symbolic...')
            print('* Loading H symbolic...')
            system.h['sym'] = dill.load(open(fname,'rb'))
            
    def load_h(self,system1,system2,k):

        fname = system1.h['fnames_data'][k]
        file_dne = not(os.path.isfile(fname))

        if 'h_data_'+system1.model_name in self.recompute_list\
           or file_dne:
            str1 = '* Computing H {}, order={}...'
            logging.info(str1.format(system1.model_name,k))
            print(str1.format(system1.model_name,k))

            h_data = self.generate_h(system1,system2,k)
            np.savetxt(system1.h['fnames_data'][k],h_data)

        else:
            str1 = '* Loading H {}, order={}...'
            logging.info(str1.format(system1.model_name,k))
            print(str1.format(system1.model_name,k))
            
            h_data = np.loadtxt(fname)

        system1.h['dat'][k] = h_data
        hodd = h_data[::-1] - h_data

        #system1.H0[k] = np.mean(h_data)

        system1.h['lam'][k] = interpb(self.bn,h_data,2*np.pi)

        if True:    
            fig,axs = plt.subplots(1,2,figsize=(6,2))
            axs[0].plot(h_data)
            axs[1].plot(hodd)

            axs[0].set_title('H '+system1.model_name+' k='+str(k))
            axs[1].set_title('-2*Hodd '+system1.model_name+' k='+str(k))
            
            plt.savefig('figs_temp/h_'+system1.model_name+str(k)+'.png')
            plt.close()

                
    def generate_h(self,system1,system2,k):
        """
        use this if nyquist frequency is too high 
        """
        
        h = np.zeros(self.NH)

        rule = {**system1.rule_p,**system2.rule_p,
                **system1.rule_par,**system2.rule_par,
                **system1.rule_lc,**system2.rule_lc,
                **system1.rule_g,**system2.rule_g,
                **system1.rule_z,**system2.rule_z,
                **system1.rule_i,**system2.rule_i}
        
        if self.forcing:
            imp0 = imp_fn('forcing',system2)
            rule.update({system2.syms[0]:imp0(self.ths[1])})
            
        h_lam = lambdify(self.ths,system1.h['sym'][k].subs(rule))
        
        bn=self.bn;dbn=self.dbn
        n=self._n[1];m=self._m[1]

        # calculate mean
        X,Y = np.meshgrid(self.an,self.an,indexing='ij')
        system1._H0[k] = np.sum(h_lam(X,Y))*self.dan**2/(2*np.pi)**2

        # calculate coupling function
        for j in range(self.NH):
            h[j] = np.sum(h_lam(bn[j] + self.om*bn,bn))

        return h/(2*np.pi*self._m[1])*dbn
        
    def fast_interp_lam(self,fn):
        """
        interp2db object (from fast_interp)
        """
        return lambda t0=self.ths[0],t1=self.ths[0]: fn(t0,t1)

    
    def save_temp_figure(self,data,k,fn='plot',path_loc='figs_temp/'):
        """
        data should be (TN,dim)
        """

        if (not os.path.exists(path_loc)):
            os.makedirs(path_loc)
        
        fig, axs = plt.subplots(nrows=self.dim,ncols=1)
        
        for j,ax in enumerate(axs):
            key = self.var_names[j]
            ax.plot(self.tLC,data[:,j],label=key)
            ax.legend()
            
        logging.info(fn+str(k)+' ini'+str(data[0,:]))
        logging.info(fn+str(k)+' fin'+str(data[-1,:]))
        axs[0].set_title(fn+str(k))
        plt.tight_layout()
        plt.savefig(path_loc+fn+str(k)+'.png')
        plt.close()
