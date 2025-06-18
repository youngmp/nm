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
#from numpy.fft import fft,ifft,fftfreq,fft2
from scipy.fft import fft,ifft,fft2,ifft2

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
import multiprocessing

pyfftw.interfaces.cache.enable()
pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()-1
pyfftw.config.PLANNER_EFFORT = 'FFTW_MEASURE'

fftw = pyfftw.interfaces.scipy_fft.fft
ifftw = pyfftw.interfaces.scipy_fft.ifft
fftw2 = pyfftw.interfaces.scipy_fft.fft2
ifft2 = pyfftw.interfaces.scipy_fft.ifft2
fftwn = pyfftw.interfaces.scipy_fft.fftn
ifftn = pyfftw.interfaces.scipy_fft.ifftn

exp = np.exp

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
                 NH=100,
                 trunc_deriv=3,

                 _n=(None,1),_m=(None,1),
                 
                 method='LSODA', # good shit                 
                 max_n=-1, # Fourier terms. -1 = no truncation
                 
                 log_level='CRITICAL',log_file='log_nm.log',
                 save_fig=False,
                 pfactor=None,
                 
                 recompute_list=[],
                 load_only_expansions=False,
                 del1=0,
                 iso_mode=False,
                 p_del=False):

        """
        iso_mode: True loads only isostables, no p_i or H functions
        N=2 only.
        Reserved names: ...        
        """

        self.p_del = p_del
        self.iso_mode = iso_mode
        self.load_only_expansions = load_only_expansions
        self.save_fig = save_fig
        self.pfactor = pfactor

        self._expand_kws = {'basic':True,'deep':True,'power_base':False,
                            'power_exp':False,'mul':True,'log':False,
                            'multinomial':True}

        system1 = copy.deepcopy(system1)
        system2 = copy.deepcopy(system2)

        # ensures het. term carries through isostable calc.
        # constant het. only.
        self.del1 = del1
        system1.pardict['del'+str(system1.idx)] = del1
        system1.rule_par['del'+str(system1.idx)] = del1
        
        assert(type(_n) is tuple);assert(type(_m) is tuple)
        #assert(_n[0] in system1.pardict);assert(_m[0] in system2.pardict)
        
        self.system1 = system1;self.system2 = system2
        self.recompute_list = recompute_list

        self.trunc_deriv = trunc_deriv
        self._n = _n;self._m = _m
        #self.Tx = self.system1.T/self._n[1]
        #self.Ty = self.system2.T/self._m[1]

        self.om = self._n[1]/self._m[1]
        self.T = 2*np.pi

        self.system1.pardict['om'+str(system1.idx)] = self._n[1]
        #self.system1.kappa_val *= self._n[1]
        
        if not(system1.forcing):
            self.system2.pardict['om'+str(system2.idx)] = self._m[1]
            #self.system2.kappa_val *= self._m[1]
        
        ths_str = ' '.join(['th'+str(i) for i in range(2)])
        psis_str = ' '.join(['ps'+str(i) for i in range(2)])
        
        self.ths = symbols(ths_str,real=True)
        self.psis = symbols(psis_str,real=True)

        # discretization for p_X,p_Y
        self.NP = NH
        
        self.x,self.dx = np.linspace(0,2*np.pi,self.NP,retstep=True,
                                     endpoint=False)
        
        self.an,self.dan = np.linspace(0,2*np.pi*self._n[1],self.NP*self._n[1],
                                       retstep=True,endpoint=False)

        # discretization for H_X,H_Y
        self.NH = NH
        self.bn,self.dbn = np.linspace(0,2*np.pi*self._m[1],NH,
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

        system1.p['dat_del']={};system1.p['imp_del']={}
        system1.p['lam_del']={};system1.h['dat_del']={};
        system1.h['imp_del']={};system1.h['lam_del']={}

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
            pfactor1 = int((np.log(.01)/system1.kappa_val)/self.T)
            pfactor2 = int((np.log(.01)/system2.kappa_val)/self.T)

            self.pfactor = max([pfactor1,pfactor2])

            print('pfactor',self.pfactor)
        
        # just keep these here.
        system2.h['dat']={};system2.h['imp']={};system2.h['lam']={}
        system2.h['sym']={};system2.p['dat']={};system2.p['imp']={}
        system2.p['lam']={};system2.p['sym']={}

        system2.p['dat_del']={};system2.p['imp_del']={}
        system2.p['lam_del']={};system2.h['dat_del']={}
        system2.h['imp_del']={};system2.h['lam_del']={}
        
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

        # need to unify these terms with forcing
        # define lambdified z and i in psi
        # can't change power, but good enough for now
        # also assumes symmetry...

        k1 = 'expand_'+str(system1.var_names[0])
        k2 = 'expand_'+str(system1.var_names[3])
        in1 = [system1.t,system1.psi]
        e1 = system1.z[k1].subs(system1.rule_z_local)
        e2 = system1.i[k1].subs(system1.rule_i_local)
        
        e3 = system1.g[k2].subs(system1.rule_g_local)
        e4 = system1.g[k1].subs(system1.rule_g_local)

        # \mathcal{Z} and \mathcal{I} expansions in psi
        self.th_lam = lambdify(in1,e1)
        self.ps_lam = lambdify(in1,e2)
        self.g_lam = lambdify(in1,e3)

        self.th_lam_v = self.th_lam

        self.g_lam_v = lambdify(in1,e4)
        self.g_lam_w = self.g_lam

        # load G and K
        self.load_k_sym(system1,system2)
        self.load_k_sym(system2,system1)

        self.load_coupling_response(system1)
        self.load_coupling_response(system2)

        self.load_3d(system1,system2)
        self.load_3d(system2,system1)

        # try loading h first. if fail, load everything else.
        file_dne = 0
        for k in range(system1.miter):
            file_dne += self.check_h_dne(system1,system2,k)
            file_dne += self.check_h_dne(system2,system1,k)

        if file_dne or 'h_'+system1.model_name in self.recompute_list\
           or 'h_'+system2.model_name in self.recompute_list\
           or 'h_data_'+system1.model_name in self.recompute_list\
           or 'h_data_'+system2.model_name in self.recompute_list\
           or 'p_data_'+system1.model_name in self.recompute_list\
           or 'p_data_'+system2.model_name in self.recompute_list\
           or 'p_'+system1.model_name in self.recompute_list\
           or 'p_'+system2.model_name in self.recompute_list:

            #self.load_nicks(system1,system2)

            if not(self.iso_mode):

                # load odes for p_i
                self.load_p_sym(system1)
                self.load_p_sym(system2)

                for i in range(system1.miter):

                    self.load_p(system1,system2,i)
                    self.load_p(system2,system1,i)


                    if self.p_del and i <= 2:
                        self.load_p_del(system1,system2,i)

                # load het. terms (mean of z and i expansions in $\ve$).
                #self.load_response_eps(system1)
                #self.load_response_eps(system2)

                self.load_h_sym(system1)
                self.load_h_sym(system2)

                for i in range(system1.miter):
                    self.load_h(system1,system2,i)
                    self.load_h(system2,system1,i)

        else:
            for i in range(system1.miter):
                self.load_h(system1,system2,i)
                self.load_h(system2,system1,i)
            
    def _init_force(self,system1):
        # system 2 should be forcing function

        self.pfactor = int((np.log(.02)/system1.kappa_val)/self._m[1])

        fnames.load_fnames_nm(system1,self)
        slib.load_coupling_expansions(system1)

        self.load_force(system1)

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
        val += not(os.path.isfile(system1.G['fname_psi']))
        val += not(os.path.isfile(system1.K['fname_psi']))

        files_dne = val

        system1.G['sym'] = {}
        system1.K['sym'] = {}
        
        system1.G['sym_psi'] = {}
        system1.K['sym_psi'] = {}
        if 'k_'+system1.model_name in self.recompute_list or files_dne:
            logging.info('* Computing G symbolic...')

            rule_trunc = {}
            for k in range(system1.miter,system1.miter+500):
                rule_trunc.update({system1.eps**k:0})
                rule_trunc.update({system1.psi**k:0})

            self.generate_k_sym(system1,system2)

            # now collect terms
            # G contains full summation
            # K contains terms of summation indexed by power
            
            for key in system1.var_names:
                tmp = expand(system1.G['sym'][key],**self._expand_kws)
                tmp_p = expand(system1.G['sym_psi'][key],**self._expand_kws)
                system1.G['sym'][key] = tmp.subs(rule_trunc)
                system1.G['sym_psi'][key] = tmp_p.subs(rule_trunc)
                
                system1.K['sym'][key] = []
                system1.K['sym_psi'][key] = []
                for k in range(system1.miter):
                    e_term = tmp.coeff(system1.eps,k)
                    e_term_p = tmp_p.coeff(system1.psi,k)
                    system1.K['sym'][key].append(e_term)
                    system1.K['sym_psi'][key].append(e_term_p)

            dill.dump(system1.G['sym'],open(system1.G['fname'],'wb'),
                      recurse=True)
            dill.dump(system1.K['sym'],open(system1.K['fname'],'wb'),
                      recurse=True)
            
            dill.dump(system1.G['sym_psi'],open(system1.G['fname_psi'],'wb'),
                      recurse=True)
            dill.dump(system1.K['sym_psi'],open(system1.K['fname_psi'],'wb'),
                      recurse=True)
                
        else:
            logging.info('* Loading G symbolic...')
            system1.G['sym'] = dill.load(open(system1.G['fname'],'rb'))
            system1.K['sym'] = dill.load(open(system1.K['fname'],'rb'))
            
            system1.G['sym_psi'] = dill.load(open(system1.G['fname_psi'],'rb'))
            system1.K['sym_psi'] = dill.load(open(system1.K['fname_psi'],'rb'))

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
            system1.G['sym_psi'][key] = c_temp[key]

                        
        # replace dx with (g expansion in eps).
        rule = {}

        for key_idx,key in enumerate(system1.var_names):
            rule.update({system1.dsyms[key_idx]:system1.g[key+'_eps']})
        for key_idx,key in enumerate(system2.var_names):
            rule.update({system2.dsyms[key_idx]:system2.g[key+'_eps']})

        for key in system1.var_names:
            system1.G['sym'][key] = system1.G['sym'][key].subs(rule)

        # do the same for psi expansion
        rule_psi = {}
        
        for key_idx,key in enumerate(system1.var_names):
            rule_psi.update({system1.dsyms[key_idx]:system1.g['expand_'+key]})
            
        # replace psi with psi_0        
        for key in system1.var_names:
            d1 = system1.G['sym_psi'][key].subs(rule_psi)
            d1 = d1.subs({system1.psi:self.psis[system1.idx]})
            system1.G['sym_psi'][key] = d1
            
        for key_idx,key in enumerate(system2.var_names):
            rule_psi.update({system2.dsyms[key_idx]:system2.g['expand_'+key]})

        # replace psi with psi_1    
        for key in system1.var_names:
            d1 = system1.G['sym_psi'][key].subs(rule_psi)
            d1 = d1 = d1.subs({system2.psi:self.psis[system2.idx]})            
            system1.G['sym_psi'][key] = d1

    def load_coupling_response(self,system):
        """
        Construct the truncated phase and isostable response functions
        needed in the averaging step.
        Assume coupling function only nontrivial in first coordinate.
        """

        file_dne = not(os.path.isfile(system.G['fname_gz']))
        file_dne += not(os.path.isfile(system.G['fname_gi']))

        if file_dne or 'gz_'+system.model_name in self.recompute_list:

            # multiply G['sym_psi'] with the z and i psi expansions
            z = system.z['expand_'+str(system.var_names[0])]
            i = system.i['expand_'+str(system.var_names[0])]
            g = system.G['sym_psi'][system.var_names[0]]

            # specify psi_index
            z = z.subs({system.psi:self.psis[system.idx]})
            i = i.subs({system.psi:self.psis[system.idx]})

            gz = sym.expand(z*g)
            gi = sym.expand(i*g)

            # truncate
            rule_trunc = {}
            for k in range(system.miter,system.miter**2):
                for ll in range(k+1):
                    rule_trunc.update({self.psis[0]**ll*self.psis[1]**(k-ll):0})

            gz = gz.subs(rule_trunc)
            gi = gi.subs(rule_trunc)

            system.gz = gz
            system.gi = gi

            dill.dump(system.gz,open(system.G['fname_gz'],'wb'),recurse=True)
            dill.dump(system.gi,open(system.G['fname_gi'],'wb'),recurse=True)

        else:
            system.gz = dill.load(open(system.G['fname_gz'],'rb'))
            system.gi = dill.load(open(system.G['fname_gi'],'rb'))

    def load_3d(self,system1,system2):
        """
        get functions for use in 3d reduction
        """
        
        s1 = system1;s2 = system2
        rule_temp1 = {**s1.rule_g_local,**s1.rule_lc_local,
                      **s1.rule_z_local,**s1.rule_i_local,
                      **s1.rule_par}
        rule_temp2 = {**s2.rule_g_local,**s2.rule_lc_local,
                      **s2.rule_z_local,**s2.rule_i_local,
                      **s2.rule_par}

        d1 = s1.gz.subs(rule_temp1)
        d1 = d1.subs({s1.t:self.ths[system1.idx]})
        d1 = d1.subs(rule_temp2)
        d1 = d1.subs({s2.t:self.ths[system2.idx]})

        d2 = s1.gi.subs(rule_temp1)
        d2 = d2.subs({s1.t:self.ths[system1.idx]})
        d2 = d2.subs(rule_temp2)
        d2 = d2.subs({s2.t:self.ths[system2.idx]})

        s1.gz_lam = lambdify([*self.ths,*self.psis],d1)
        s1.gi_lam = lambdify([*self.ths,*self.psis],d2)

    def load_nicks(self,system1,system2):
        """
        compute the H functions form Nicks et al 2024
        just up to psi^2
        """
        s1 = system1;s2 = system2
        rule_temp1 = {**s1.rule_g_local,**s1.rule_lc_local,
                      **s1.rule_z_local,**s1.rule_i_local,
                      **s1.rule_par}
        rule_temp2 = {**s2.rule_g_local,**s2.rule_lc_local,
                      **s2.rule_z_local,**s2.rule_i_local,
                      **s2.rule_par}
        
        # lowest order
        d1 = s1.gz_poly[0].subs(rule_temp1)
        d1 = d1.subs({s1.t:self.ths[system1.idx]})
        d1 = d1.subs(rule_temp2)
        d1 = d1.subs({s2.t:self.ths[system2.idx]})

        d2 = s1.gi_poly[0].subs(rule_temp1)
        d2 = d2.subs({s1.t:self.ths[system1.idx]})
        d2 = d2.subs(rule_temp2)
        d2 = d2.subs({s2.t:self.ths[system2.idx]})

        h1a_lam = lambdify(self.ths,d1)
        
        x = np.linspace(0,2*np.pi,100)
        
        
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
        """
        These p functions are in terms of the fast phases, not slow phases.
        """

        fname = system1.p['fnames_data'][k]
        file_dne = not(os.path.isfile(fname))
        
        if 'p_data_'+system1.model_name in self.recompute_list or file_dne:
            p_data = self.generate_p(system1,system2,k)
            np.savetxt(system1.p['fnames_data'][k],p_data)

        else:
            p_data = np.loadtxt(fname)

        system1.p['dat'][k] = p_data

        n=self._n[1];m=self._m[1]
        x=self.x;dx=self.dx

        p_interp = interp2d([0,0],[self.T*n,self.T*m],[dx*n,dx*m],p_data,
                            k=5,p=[True,True])

        ta = self.ths[0];tb = self.ths[1]
        name = 'p_'+system1.model_name+'_'+str(k)
        p_imp = imp_fn(name,self.fast_interp_lam(p_interp))
        lamtemp = lambdify(self.ths,p_imp(ta,tb))
        
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
            pname = 'figs_temp/p_'
            pname += system1.model_name+str(k)
            pname += '_'+str(n)+str(m)
            pname += '.png'
            plt.savefig(pname)
            plt.close()

        # put these implemented functions into the expansion
        system1.rule_p.update({sym.Indexed('p_'+system1.model_name,k):
                               system1.p['imp'][k](ta,tb)})

    def generate_p(self,system1,system2,k):
        print('p order='+str(k))

        n = self._n[1];m = self._m[1]
        NP = self.NP
        data = np.zeros([n*NP,m*NP])
        
        if k == 0:
            #pA0 is 0 (no forcing function)
            return data

        kappa1 = system1.kappa_val

        rule = {**system1.rule_p,**system2.rule_p,**system1.rule_par,
                **system2.rule_par,**system1.rule_lc,**system2.rule_lc,
                **system1.rule_i,**system2.rule_i,**system1.rule_g,
                **system2.rule_g}
        
        if self.forcing:
            # if forcing, set all delta to zero
            rule['del0'] = 0
            rule['del1'] = 0
            imp0 = imp_fn('forcing',system2)
            lam0 = lambdify(self.ths[1],imp0(self.ths[1]))
            rule.update({system2.syms[0]:imp0(self.ths[1])})

        #rule['del0'] *= 20
        #rule['del1'] *= 20
        #print('del vals',rule['del0'],rule['del1'])

        ph_imp1 = system1.p['sym'][k].subs(rule)

        if ph_imp1 == 0: # keep just in case
            return data

        # forcing function
        lam1 = lambdify(self.ths,ph_imp1)        

        #x = self.an;dx = self.dan
        x=self.x;dx=self.dx;
        pfactor=self.pfactor
        sm=np.arange(0,self.T*pfactor,dx)*m

        if self.forcing:
            fac = 1
        else:
            fac = self.om*(1-system1.idx) + system1.idx
        exp1 = exp(fac*sm*system1.kappa_val)

        g_in = np.fft.fft(exp1)
        a_i = np.arange(m*NP,dtype=int)
        
        for ll in range(n*NP):

            f1 = lam1(x[ll%NP]*n+self.om*sm,sm)
            f_in = fftw(f1)
            conv = ifftw(f_in*g_in).real
            data[(a_i+ll)%(n*NP),a_i] = conv[-m*NP:]

            if ll == 0 and False:
                fig,axs = plt.subplots(3,1)
                axs[0].plot(f1[-m*NP:])

                sm2 = np.arange(0,4*np.pi,dx)
                axs[0].scatter(np.arange(len(sm2)),lam1(0+sm2/2,sm2),s=1)
                axs[1].plot(exp1)
                axs[2].plot(conv[-m*NP:])
                
                axs[0].set_title('integrand 1')
                axs[1].set_title('integrand 2')
                axs[2].set_title('conv')
                
                plt.show()

        return fac*data*dx*m
    

    def load_response_eps(self,system):
        """
        compute mean value of each term in powers of epsilon.
        save as list where the first element corresponds to eps^1.

        assumes het. in first element of vector of variables
        """

        rule = {**system.rule_p,**system.rule_par,
                **system.rule_z,**system.rule_i}

        key = system.var_names[0]+'_eps'

        terms_z = sym.Poly(system.z[key],system.eps).coeffs()[::-1]
        terms_i = sym.Poly(system.i[key],system.eps).coeffs()[::-1]

        self.z['het'] = []
        self.i['het'] = []

        x=a11.x;dx=a11.dx
        n=a11._n[1];m=a11._m[1]
        X,Y = np.meshgrid(x*n,x*m,indexing='ij')

        for term in terms_z:
            fz = sym.lambdify(self.ths,term.subs(rule))
            z_mean = np.mean(fz(X,Y))
            self.z['het'].append(z_mean)

        for term in terms_i:
            fi = sym.lambdify(self.ths,term.subs(rule))
            i_mean = np.mean(fi(X,Y))
            self.i['het'].append(i_mean)
        
        
    
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


    def check_h_dne(self,system1,system2,k):
        """
        check if all require h files exist.
        if not, fail.
        """
        fname = system1.h['fnames_data'][k]
        file_dne = not(os.path.isfile(fname))

        return file_dne
        
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

        n=self._n[1];m=self._m[1]
        #system1.h['lam'][k] = interpb(self.be,h_data,2*np.pi)
        system1.h['lam'][k] = interp1d(0,self.T*n,self.dx,
                                       h_data,p=True,k=5)

        #system1.h['lam'][k] = interp1d(self.bn[0],self.bn[-1],self.dbn,
        #                               h_data,p=True,k=5)

        if self.save_fig:    
            fig,axs = plt.subplots(figsize=(6,2))

            axs.plot(h_data)

            ptitle = 'h '+system1.model_name
            ptitle += 'o='+str(k)
            ptitle += ', nm='+str(n)+str(m)
            axs.set_title(ptitle)
            
            pname = 'figs_temp/h_'
            pname += system1.model_name+str(k)
            pname += '_'+str(n)+str(m)
            pname += '.png'
            plt.savefig(pname)            
            plt.close()

                
    def generate_h(self,system1,system2,k):

        rule = {**system1.rule_p,**system2.rule_p,**system1.rule_par,
                **system2.rule_par,**system1.rule_lc,**system2.rule_lc,
                **system1.rule_g,**system2.rule_g,**system1.rule_z,
                **system2.rule_z,**system1.rule_i,**system2.rule_i}

        print('system1.rule_p',system1.rule_p)
        print('system2.rule_p',system2.rule_p)
        
        if self.forcing:
            rule['del0'] = 0
            rule['del1'] = 0
            imp0 = imp_fn('forcing',system2)
            rule.update({system2.syms[0]:imp0(self.ths[1])})

        h_lam = lambdify(self.ths,system1.h['sym'][k].subs(rule))

        #rule['del0'] *= 20
        #rule['del1'] *= 20
        #print('h del vals',rule['del0'],rule['del1'])

        x=self.x;dx=self.dx
        n=self._n[1];m=self._m[1]
        
        X,Y = np.meshgrid(x*n,x*m,indexing='ij')
        h_integrand = h_lam(X,Y)
        ft2 = fft2(h_integrand)
        # get only those in n:m indices
        # n*i,-m*i
        ft2_new = list([ft2[0,0]])
        ft2_new += list(np.diag(np.flipud(ft2[1:,1:]))[::-1])
        ft2_new = np.asarray(ft2_new)
        ft2_new /= self.NH

        out = ifft(ft2_new).real

        return out

        """
        bn=self.be;dbn=self.dbe
        #bn=self.be;dbn=self.dbe
        n=self._n[1];m=self._m[1]

        # calculate mean
        #X,Y = np.meshgrid(self.an,self.an*self._m[1],indexing='ij')
        #system1._H0[k] = np.sum(h_lam(X,Y))*self.dan**2/(2*np.pi*self._m[1])**2

        def integrand(x,y):
            return h_lam(y+self.om*x,x)
                
        # calculate coupling function
        h = np.zeros(self.NH)
        for j in range(self.NH):
            #val = np.sum(h_lam(bn[j] + self.om*bn,bn))*dbn
            val = quad(integrand,bn[0],bn[-1],args=(bn[j],),limit=10000)[0]

            h[j] = val

        out = h/(2*np.pi*self._m[1])

        return out
        """
        
    def fast_interp_lam(self,fn):
        """
        interp2d object (from fast_interp)
        this is needed to use interp2d objects in implemented functions
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
