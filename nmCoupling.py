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
from lib.fast_interp import interp1d,interp2d
from lib.interp_basic import interp_basic as interpb
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

from scipy.integrate import quad
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
                 trunc_deriv=3,

                 _n=(None,1),_m=(None,1),
                 
                 method='LSODA', # good shit                 
                 max_n=-1, # Fourier terms. -1 = no truncation
                 
                 gij_parallel=True,processes=6,
                 chunksize=1,chunk_g=1,process_g=5,
                 log_level='CRITICAL',log_file='log_nm.log',
                 lowdim=False,
                 save_fig=False,
                 pfactor=None,
                 
                 recompute_list=[]):

        """
        N=2 only.
        Reserved names: ...        
        """
        self.save_fig = save_fig
        self.pfactor = pfactor

        self._expand_kws = {'basic':True,'deep':True,'power_base':False,
                            'power_exp':False,'mul':True,'log':False,
                            'multinomial':True}

        system1 = copy.deepcopy(system1)
        system2 = copy.deepcopy(system2)
        
        assert(type(_n) is tuple);assert(type(_m) is tuple)
        #assert(_n[0] in system1.pardict);assert(_m[0] in system2.pardict)
        
        self.system1 = system1;self.system2 = system2
        self.recompute_list = recompute_list
        self.lowdim = lowdim

        self.trunc_deriv = trunc_deriv
        self._n = _n;self._m = _m
        #self.Tx = self.system1.T/self._n[1]
        #self.Ty = self.system2.T/self._m[1]

        self.om = self._n[1]/self._m[1]
        self.T = self._m[1]*2*np.pi

        self.system1.pardict['om'+str(system1.idx)] = self._n[1]

        if not(system1.forcing):
            self.system2.pardict['om'+str(system2.idx)] = self._m[1]
        
        ths_str = ' '.join(['th'+str(i) for i in range(2)])
        self.ths = symbols(ths_str)

        # discretization for p_X,p_Y
        self.NP = NP
        self.x,self.dx = np.linspace(0,2*np.pi,NP,retstep=True,endpoint=False)
        
        self.an,self.dan = np.linspace(0,2*np.pi*self._n[1],NP*self._n[1],
                                       retstep=True,endpoint=False)

        # discretization for H_X,H_Y
        self.NH = NH
        self.bn,self.dbn = np.linspace(0,2*np.pi*self._m[1],NH*self._m[1],
                                       retstep=True,endpoint=False)

        self.be,self.dbe = np.linspace(0,2*np.pi*self._m[1],NH*self._m[1],
                                       retstep=True,endpoint=True)

        self.bne,self.dbne = np.linspace(0,2*np.pi*self._n[1],NH*self._n[1],
                                         retstep=True,endpoint=True)        

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

        if not(system1.forcing):
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
        

        self.forcing = system1.forcing
        if not(self.forcing):
            self._init_noforce(system1,system2)
            
        else:
            self._init_force(system1)

    def _init_noforce(self,system1,system2):
        self.t,self.dt = np.linspace(0,2*np.pi,100,retstep=True)

        if self.pfactor is None:
            pfactor1 = int((np.log(.05)/system1.kappa_val))
            pfactor2 = int((np.log(.05)/system2.kappa_val))
            
            self.pfactor = max([pfactor1,pfactor2])
        print('pfactor?',self.pfactor)
        
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

        for i in range(system1.miter):
            self.load_p(system1,system2,i)
            self.load_p(system2,system1,i)

        self.load_h_sym(system1)
        self.load_h_sym(system2)

        for i in range(system1.miter):
            self.load_h(system1,system2,i)
            self.load_h(system2,system1,i)

            
    def _init_force(self,system1):
        # system 2 should be forcing function
        

        self.pfactor = int((np.log(.02)/system1.kappa_val)/self._m[1])
        

        fnames.load_fnames_nm(system1,self)
        slib.load_coupling_expansions(system1)

        self.load_force(system1)
        if not(self.lowdim):

            self.generate_p_force_sym()
            self.generate_h_force_sym()

            print('* Loading p...')
            for k in range(self.system1.miter):
                self.load_p(system1,system1,k)
                self.load_h(system1,system1,k)

    def load_force(self,system1):

        in1 = [system1.t,system1.psi]
        e1 = system1.z['vec_psi'][0].subs(system1.rule_z_local)
        e2 = system1.i['vec_psi'][0].subs(system1.rule_i_local)


        # \mathcal{Z} and \mathcal{I} expansions in psi
        self.th_lam = lambdify(in1,e1)
        self.ps_lam = lambdify(in1,e2)

        u = self.system1.forcing_fn
        
        f1 = sym.symbols('f1')
        u_imp = imp_fn('uf',u)

        # symbolic h functions
        hz0 = sym.Poly(system1.z['expand_'+system1.var_names[0]]*f1,system1.psi)
        hi0 = sym.Poly(system1.i['expand_'+system1.var_names[0]]*f1,system1.psi)

        hz2 = sym.Poly(system1.z['expand_'+system1.var_names[0]],system1.psi)
        hi2 = sym.Poly(system1.i['expand_'+system1.var_names[0]],system1.psi)

        ths = self.ths

        # apply subs for each coefficient in powers of psi
        hz2_cs = hz2.coeffs()[::-1]
        hi2_cs = hi2.coeffs()[::-1]
        
        self.z_exp = []
        self.i_exp = []
        for i in range(len(hz2_cs)):
            imp0 = hz2_cs[i].subs(system1.rule_z_local).subs(system1.t,ths[0])
            lam0 = lambdify(ths[0],imp0)

            self.z_exp.append(lam0)

            imp0 = hi2_cs[i].subs(system1.rule_i_local).subs(system1.t,ths[0])
            lam0 = lambdify(ths[0],imp0)
            self.i_exp.append(lam0)
        
        hz0_cs = hz0.coeffs()[::-1]
        self.hz0_lams = []

        oms0 = system1.pardict_sym['om0']
        oms1 = system1.pardict_sym['omf']
        oms = oms0/oms1
        
        for i in range(len(hz0_cs)):
            hz0_imp = hz0_cs[i].subs(system1.rule_z_local)
            
            in1 = ths[0]+oms*system1.t
            hz0_imp = hz0_imp.subs(system1.t,in1)
            hz0_imp = hz0_imp.subs(f1,u_imp(system1.t))
            
            hz0_imp = hz0_imp.subs(oms0,self._n[1])
            hz0_imp = hz0_imp.subs(oms1,self._m[1])
            
            hz0_lam = lambdify([ths[0],system1.t],hz0_imp)
            self.hz0_lams.append(hz0_lam)

            if self.save_fig:
                tt = np.linspace(0,2*np.pi)
                X,Y = np.meshgrid(tt,tt,indexing='ij')
                fig,axs = plt.subplots()
                axs.imshow(hz0_lam(X,Y))
                plt.savefig('figs_temp/hz0_lam_'+str(i)+'.png')

        hi0_cs = hi0.coeffs()[::-1]
        self.hi0_lams = []
        
        for i in range(len(hi0_cs)):
            
            hi0_imp = hi0_cs[i].subs(system1.rule_i_local)
            hi0_imp = hi0_imp.subs(system1.t,ths[0]+oms*system1.t)
            hi0_imp = hi0_imp.subs(f1,u_imp(system1.t))

            hi0_imp = hi0_imp.subs(oms0,self._n[1])
            hi0_imp = hi0_imp.subs(oms1,self._m[1])

            hi0_lam = lambdify([ths[0],system1.t],hi0_imp)
            self.hi0_lams.append(hi0_lam)

            if self.save_fig:
                tt = np.linspace(0,2*np.pi)
                X,Y = np.meshgrid(tt,tt,indexing='ij')
                fig,axs = plt.subplots()
                axs.imshow(hi0_lam(X,Y))
                nn = self._n[1]
                mm = self._m[1]
                axs.set_title('hi0_lam_'+str(i)\
                            +'_'+str(nn)+str(mm))
                
                plt.savefig('figs_temp/hi0_lam_'+str(i)\
                            +'_'+str(nn)+str(mm)+'.png')
                plt.close()

        s1,ds1 = np.linspace(0,2*np.pi,self.NH,retstep=True)
        s2,ds2 = np.linspace(0,2*np.pi*self._m[1],self.NH,retstep=True)

        self.hz_lam = []
        self.hi_lam = []

        for i in range(system1.miter):
            solz = np.zeros(len(s1))
            soli = np.zeros(len(s1))

            for j in range(len(s2)):
                solz[j] = np.sum(self.hz0_lams[i](s1[j],s2))
                soli[j] = np.sum(self.hi0_lams[i](s1[j],s2))

            solz *= ds2/(2*np.pi*self._m[1])
            soli *= ds2/(2*np.pi*self._m[1])

            fnz = interp1d(s1[0],s1[-2],ds1,solz[:-1],p=True,k=5)
            fni = interp1d(s1[0],s1[-2],ds1,soli[:-1],p=True,k=5)

            self.hz_lam.append(fnz)
            self.hi_lam.append(fni)

        #self.generate_p_force_sym()

        
        #data = self.generate_p_force(0)
        
        #self.generate_p_force(1)


    def generate_p_force_sym(self):
        # get expansion of $I$ in psi
        # substitute expansion of psi in eps
        # collect in powers of $\ve$.

        f1 = sym.symbols('f1')
        first_var = self.system1.var_names[0]
        
        tmp = self.system1.i[first_var+'_eps']*self.system1.eps
        tmp = collect(tmp,self.system1.eps)
        tmp = collect(expand(tmp),self.system1.eps)
        
        u = self.system1.forcing_fn
        u_imp = imp_fn('uf',u)

        rule = {**self.system1.rule_i,
                **{f1:u_imp(self.system1.t)}}

        for k in range(self.system1.miter):
            self.system1.p['sym'][k] = tmp.coeff(self.system1.eps,k)
            self.system1.p['sym'][k] *= u_imp(self.ths[1])

    def generate_h_force_sym(self):
        f1 = sym.symbols('f1')

        first_var = self.system1.var_names[0]
        tmp = self.system1.z[first_var+'_eps']
        tmp = collect(tmp,self.system1.eps)
        tmp = collect(expand(tmp),self.system1.eps)
        
        u = self.system1.forcing_fn
        u_imp = imp_fn('uf',u)

        rule = {**self.system1.rule_z,
                **{f1:u_imp(self.system1.t)}}

        for k in range(self.system1.miter):
            self.system1.h['sym'][k] = tmp.coeff(self.system1.eps,k)
            self.system1.h['sym'][k] *= u_imp(self.ths[1])


            
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

                tmp = expand(system1.G['sym'][key],**self._expand_kws)
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
            tmp = expand(tmp,**self._expand_kws)
            
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

        print('fname',fname)
        
        if 'p_data_'+system1.model_name in self.recompute_list\
           or file_dne:
            
            
            p_data = self.generate_p(system1,system2,k)


            if self.save_fig:
            
                fig,axs = plt.subplots()
                axs.imshow(p_data)
                plt.savefig('figs_temp/p_untransformed_'\
                            +system1.model_name+str(k)+'.png')
                plt.close()

            #if not(self.forcing):
            #p_interp0 = interp2d([0,0],[2*np.pi*self._n[1],
            #                            2*np.pi*self._m[1]],
            #                     [self.dan,self.dan],
            #                     p_data,k=5,p=[True,True])

            # fix indexing (see check_isostable.ipynb)
            #X,Y = np.meshgrid(self.an*self._n[1],
            #                  self.an*self._m[1],
            #                  indexing='ij')
            #p_data = p_interp0(X-self.om*Y,Y)
                
            np.savetxt(system1.p['fnames_data'][k],p_data)

        else:
            
            p_data = np.loadtxt(fname)

        #if self.forcing:
        #    p_interp = interp2d([0,0],[2*np.pi,2*np.pi*self._m[1]],
        #                        [self.dan,self.dan],
        #                        p_data,k=3,p=[True,True])
        #else:
        #    p_interp = interp2d([0,0],[2*np.pi,2*np.pi],
        #                        [self.dan,self.dan],
        #                        p_data,k=3,p=[True,True])

        n=self._n[1];m=self._m[1]
        x=self.x;dx=self.dx
        
        p_interp = interp2d([0,0],[x[-1]*n,x[-1]*m],[dx*n,dx*m],p_data,
                            k=9,p=[True,True])

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

        if self.save_fig:
            
            fig,axs = plt.subplots()
            axs.imshow(p_data)
            plt.savefig('figs_temp/p_'+system1.model_name+str(k)+'.png')
            plt.close()

        # put these implemented functions into the expansion
        system1.rule_p.update({sym.Indexed('p_'+system1.model_name,k):
                               system1.p['imp'][k](ta,tb)})

        
    def generate_p(self,system1,system2,k):
        print('p order='+str(k))

        n = self._n[1];m = self._m[1]
        NP = self.NP
        
        if k == 0:
            #pA0 is 0 (no forcing function)
            return np.zeros((n*NP,m*NP))

        kappa1 = system1.kappa_val

        rule = {**system1.rule_p,**system2.rule_p,**system1.rule_par,
                **system2.rule_par,**system1.rule_lc,**system2.rule_lc,
                **system1.rule_i,**system2.rule_i,**system1.rule_g,
                **system2.rule_g}
        
        if self.forcing:
            imp0 = imp_fn('forcing',system2)
            lam0 = lambdify(self.ths[1],imp0(self.ths[1]))
            rule.update({system2.syms[0]:imp0(self.ths[1])})

        ph_imp1 = system1.p['sym'][k].subs(rule)

        if ph_imp1 == 0:
            return np.zeros((n*NP,m*NP))
        
        lam1 = lambdify(self.ths,ph_imp1)
        data = np.zeros((int(n*NP),int(m*NP)))
        
        #an=self.an;dan=self.dan;pfactor=self.pfactor
        #NP=self.NP;s=np.arange(0,self.T*pfactor*m,dan)
        
        x=self.x*n;dx=self.dx*n;pfactor=self.pfactor
        s=np.arange(0,self.T*pfactor*m,dx)
        
        fac = self.om#*(1-system1.idx) + system1.idx
        exp1 = exp(fac*s*system1.kappa_val)

        g_in = np.fft.fft(exp1)
        a_i = np.arange(NP*m,dtype=int)
        
        #for ll in range(len(an)):
        for ll in range(len(x)):

            f_in = np.fft.fft(lam1(x[ll]+fac*s,+s))
            conv = np.fft.ifft(f_in*g_in)
            data[(a_i+ll)%int(n*NP),a_i] = conv[-int(m*NP):].real
            #data[ll,:] = conv[-self._m[1]*NP:].real

        return data*dx
    
    
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
            tmp = sym.expand(tmp,**self._expand_kws)
            
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

            print('h sym',system.h['sym'])
            
    def load_h(self,system1,system2,k):

        

        fname = system1.h['fnames_data'][k]
        file_dne = not(os.path.isfile(fname))

        print('h_data_'+system1.model_name,self.recompute_list)
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

        n=self._n[1];m=self._m[1]
        #system1.h['lam'][k] = interpb(self.be,h_data,2*np.pi)
        system1.h['lam'][k] = interp1d(0,self.x[-1]*n,self.dx*n,
                                       h_data,p=True,k=9)

        if self.save_fig:    
            fig,axs = plt.subplots(1,2,figsize=(6,2))
            axs[0].plot(h_data)
            axs[1].plot(hodd)

            axs[0].set_title('H '+system1.model_name+' k='+str(k))
            axs[1].set_title('-2*Hodd '+system1.model_name+' k='+str(k))
            
            plt.savefig('figs_temp/h_'+system1.model_name+str(k)+'.png')
            plt.close()

                
    def generate_h(self,system1,system2,k):

        rule = {**system1.rule_p,**system2.rule_p,**system1.rule_par,
                **system2.rule_par,**system1.rule_lc,**system2.rule_lc,
                **system1.rule_g,**system2.rule_g,**system1.rule_z,
                **system2.rule_z,**system1.rule_i,**system2.rule_i}
        
        if self.forcing:
            imp0 = imp_fn('forcing',system2)
            rule.update({system2.syms[0]:imp0(self.ths[1])})
            
        h_lam = lambdify(self.ths,system1.h['sym'][k].subs(rule))
        
        #bn=self.be;dbn=self.dbe
        x=self.x;dx=self.dx
        n=self._n[1];m=self._m[1]

        """
        fig5,axs5 = plt.subplots()
        x = np.linspace(0,2*np.pi,m*100)
        X,Y = np.meshgrid(x*n,x*m)

        axs5.contourf(X,Y,h_lam(X,Y))
        axs5.plot((1+self.om*x),x)
        plt.savefig('figs_temp/hlam'+system1.model_name+str(k)+'.png')
        """

        def integrand(s,x):
            return h_lam(x+self.om*s,s)
                
        # calculate coupling function
        h = np.zeros(self.NH*n)
        for j in range(self.NH*n):
            #val = np.sum(h_lam(bn[j] + self.om*bn,bn))*dbn
            val = quad(integrand,x[0],x[-1]*m,
                       args=((x*n)[j],),limit=1000)[0]

            h[j] = val

        return h/(2*np.pi*m)
        
        
    def fast_interp_lam(self,fn):
        """
        interp2db object (from fast_interp)
        """
        return lambda t0=self.ths[0],t1=self.ths[1]: fn(t0,t1)

    
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
