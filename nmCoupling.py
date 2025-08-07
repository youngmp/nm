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
                 
                 het_coeffs = [],
                 ):

        """
        N=2 only.
        Reserved names: ...
        het_coeffs: for heterogeneous coefficients. index 0 is order eps, then eps^2 etc.
        """

        print('Initializing {}{} Coupling...'.format(_n[1],_m[1]))

        self.het_coeffs = het_coeffs
        self.save_fig = save_fig
        self.pfactor = pfactor

        self._expand_kws = {'basic':True,'deep':True,'power_base':False,
                            'power_exp':False,'mul':True,'log':False,
                            'multinomial':True}

        system1 = copy.deepcopy(system1)
        system2 = copy.deepcopy(system2)

        # ensures het. term carries through isostable calc.
        # constant het. only.
        #self.del1 = del1
        #system1.pardict['del'+str(system1.idx)] = del1
        #system1.rule_par['del'+str(system1.idx)] = del1

        system1.pardict['del1'] = 0
        system1.rule_par['del1'] = 0
        system1.pardict_sym['del1'] = 0
        
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
        self.b = symbols('b',real=True)

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
        system1.p['lam']={};system1.p['sym']={};

        system1.p['dat_hom']={};system1.p['dat_het']={}
        system1.p['sym_hom']={};system1.p['sym_het']={}
        system1.p['imp_hom']={};system1.p['imp_het']={}
        system1.p['lam_hom']={};system1.p['lam_het']={}

        system1.h['dat_hom']={};system1.h['dat_het']={}
        system1.h['sym_hom']={};system1.h['sym_het']={}
        system1.h['lam_hom']={};system1.h['lam_het']={}

        system2.p['dat_hom']={};system2.p['dat_het']={}
        system2.p['sym_hom']={};system2.p['sym_het']={}
        system2.p['imp_hom']={};system2.p['imp_het']={}
        system2.p['lam_hom']={};system2.p['lam_het']={}

        system2.h['dat_hom']={};system2.h['dat_het']={}
        system2.h['sym_hom']={};system2.h['sym_het']={}
        system2.h['lam_hom']={};system2.h['lam_het']={}

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
        slib.load_coupling_expansions(system1,True)
        slib.load_coupling_expansions(system2,True)

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

            # load odes for p_i
            self.load_p_sym(system1)
            self.load_p_sym(system2)

            for i in range(system1.miter):
                self.load_p_hom_het(system1,system2,i)
                self.load_p_hom_het(system2,system1,i)

                #self.load_p(system1,system2,i)
                #self.load_p(system2,system1,i)

            self.load_h_sym(system1)
            self.load_h_sym(system2)

            for i in range(system1.miter):
                self.load_h_hom_het(system1,system2,i)
                self.load_h_hom_het(system2,system1,i)
                
                #self.load_h(system1,system2,i)
                #self.load_h(system2,system1,i)

        else:
            self.load_p_sym(system1)
            self.load_p_sym(system2)
            
            for i in range(system1.miter):
                self.load_p_hom_het(system1,system2,i)
                self.load_p_hom_het(system2,system1,i)

                self.load_h_sym(system1)
                self.load_h_sym(system2)
                
                #self.load_h(system1,system2,i)
                #self.load_h(system2,system1,i)

                self.load_h_hom_het(system1,system2,i)
                self.load_h_hom_het(system2,system1,i)
            
    def _init_force(self,system1):
        print('Warning: parameter del1 will be ignored')
        # system 2 should be forcing function

        self.pfactor = int((np.log(.02)/system1.kappa_val)/self._m[1])

        fnames.load_fnames_nm(system1,self)
        slib.load_coupling_expansions(system1,True)

        self.generate_p_force_sym()
        self.generate_h_force_sym()

        print('* Loading p...')
        for k in range(self.system1.miter):
            self.load_p(system1,system1,k)
            self.load_h(system1,system1,k)

    def generate_p_force_sym(self):
        # get expansion of $I$ in psi
        # substitute expansion of psi in eps
        # collect in powers of $\ve$.

        us = self.system1.forcing_fn
        f1s = [sym.symbols('f'+str(i)) for i in range(1,len(us)+1)]
        first_var = self.system1.var_names[0]

        u_expr = 0
        eps = self.system1.eps
        
        for i in range(len(f1s)):
            u_expr += eps**(i+1)*f1s[i]

        # collect i terms in eps
        tmp = self.system1.i[first_var+'_eps']*u_expr
        tmp = collect(tmp,self.system1.eps)
        tmp = collect(expand(tmp),self.system1.eps)
        
        
        u_imps = []
        for i in range(1,len(us)+1):
            u_imp = imp_fn('f_imp'+str(i),us[i-1])
            u_imps.append(u_imp)

        u_rule = {f1s[i]:u_imps[i](self.ths[1]) for i in range(len(us))}

        rule = {**self.system1.rule_i,**u_rule}

        for k in range(self.system1.miter):
            self.system1.p['sym'][k] = tmp.coeff(self.system1.eps,k).subs(rule)

    def generate_h_force_sym(self):

        us = self.system1.forcing_fn
        f1s = [sym.symbols('f'+str(i)) for i in range(len(us))]
        first_var = self.system1.var_names[0]

        u_expr = 0
        eps = self.system1.eps
        
        for i in range(len(f1s)):
            u_expr += eps**i*f1s[i]

        # collect z terms in eps
        tmp = self.system1.z[first_var+'_eps']*u_expr
        tmp = collect(tmp,self.system1.eps)
        tmp = collect(expand(tmp),self.system1.eps)

        u_imps = []
        for i in range(1,len(us)+1):
            u_imp = imp_fn('f_imp'+str(i),us[i-1])
            u_imps.append(u_imp)

        u_rule = {f1s[i]:u_imps[i](self.ths[1]) for i in range(len(us))}

        rule = {**self.system1.rule_z,**u_rule}

        for k in range(self.system1.miter):
            self.system1.h['sym'][k] = tmp.coeff(self.system1.eps,k).subs(rule)

            
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
            #logging.info('* Computing G symbolic...')
            print('* Computing G symbolic...')

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

        # substitute explicit het terms.
        # take del0 and expand it according to het_coeffs
        # del0 -> del0*het_coeffs[0] + eps*del0**2*het_coeffs[1] + ...
        tot = 0
        b = self.system1.pardict_sym['del0'] # alias
        e = self.system1.eps
        for i in range(len(self.het_coeffs)):
            tot += e**i*b**(i+1)*self.het_coeffs[i]
            
        rule_del0 = {self.system1.pardict_sym['del0']:tot}
        
        system1.G['vec'][0] = system1.G['vec'][0].subs(rule_del0)

        

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
        system1.G['sym_fn'] = sym.flatten(fn(z,psym,option='sym',idx=system1.idx))

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
            d1 = d1.subs({system2.psi:self.psis[system2.idx]})            
            system1.G['sym_psi'][key] = d1

    def load_coupling_response(self,system):
        """
        Construct the truncated phase and isostable response functions
        needed in the averaging step.
        Assume coupling function nonzero only in first coordinate.
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

        
        
    def load_p_sym(self,system1):
        """
        generate/load the het. terms for psi ODEs.
            
        to be solved using integrating factor meothod.        
        system*.p[k] is the forcing function for oscillator * of order k

        note p is defined as Indexed('p_'+obj.model_name,k) in /lib/lib_sym.py
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

        # hom het symbols
        mname = system1.model_name
        
        # construct sym hom
        system1.p['sym_hom'] = {0:0}
        for k in range(1,system1.miter):
            system1.p['sym_hom'][k] = sym.IndexedBase('p_'+mname+'_hom')[k]

        # construct sym het
        system1.p['sym_het'] = {0:0}

        # expansions will always be in del0, never in del1
        b = self.system1.pardict_sym['del0']
        for k in range(1,system1.miter):
            tot = 0
            for j in range(k):
                tot += b**(j+1)*sym.IndexedBase('p_'+mname+'_het')[k,j+1]
            system1.p['sym_het'][k] = tot

        # create substitution rule for p_thal0_test[k] into explicit hom and het parts
        system1.rule_p2het = {}

        # note that it's possible for system2 to have het. terms even if del1 = 0
        for k in range(1,system1.miter):
            system1.rule_p2het[sym.Indexed('p_'+mname,k)] = system1.p['sym_hom'][k] + system1.p['sym_het'][k]

    def load_p(self,system1,system2,k):
        """
        These p functions are in terms of the fast phases, not slow phases.
        
        """

        fname = system1.p['fnames_data'][k]
        file_dne = not(os.path.isfile(fname))
        
        if 'p_data_'+system1.model_name in self.recompute_list or file_dne:
            expr = system1.p['sym'][k]
            p_data = self.generate_p(system1,system2,expr,k)
            np.savetxt(system1.p['fnames_data'][k],p_data)

        else:
            p_data = np.loadtxt(fname)

        system1.p['dat'][k] = p_data

        n=self._n[1];m=self._m[1]
        x=self.x;dx=self.dx

        p_interp = interp2d([0,0],[self.T*n,self.T*m],[dx*n,dx*m],p_data,k=1,p=[True,True])

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



    def load_p_hom_het(self,system1,system2,k):
        """
        These p functions are in terms of the fast phases, not slow phases.

        explicitly save homogeneous and heterogeneous parts of p functions.
        
        """

        fname_hom = system1.p['fnames_data_hom'][k]
        fname_hets = system1.p['fnames_data_het'][k]
        
        file_dne = not(os.path.isfile(fname_hom))
        for j in range(len(fname_hets)):
            file_dne += not(os.path.isfile(fname_hets[j]))

        # rule to break up het. terms
        rule = {**system1.rule_p2het,**system2.rule_p2het}

        #print('rule in load p hom het',rule)

        # alias
        b = self.system1.pardict_sym['del0']
        
        if 'p_data_'+system1.model_name in self.recompute_list or file_dne:

            # plug in explicit homogenous and heterogenous expansions

            # hom.
            expr = sym.expand(system1.p['sym'][k].subs(rule))
            expr = sym.collect(expr,b)
            expr_hom = expr.coeff(b,0)

            # het terms
            expr_hets = []
            for j in range(k):
                expr_het = expr.coeff(b,j+1)
                expr_hets.append(expr_het)

            #print('generating hom')
            dat_hom = self.generate_p(system1,system2,expr_hom,k)

            #print('expr_hets',expr_hets)

            dat_hets = []
            for j in range(k):
                #print('generating het',j)
                dat_het = self.generate_p(system1,system2,expr_hets[j],k)
                dat_hets.append(dat_het)
            
            np.savetxt(system1.p['fnames_data_hom'][k],dat_hom)

            for j in range(k):
                np.savetxt(system1.p['fnames_data_het'][k][j],dat_hets[j])

        else:
            dat_hom = np.loadtxt(fname_hom)
            dat_hets = []
            for j in range(k):
                dat_het = np.loadtxt(fname_hets[j])
                dat_hets.append(dat_het)

        system1.p['dat_hom'][k] = dat_hom
        system1.p['dat_het'][k] = dat_hets

        n=self._n[1];m=self._m[1]
        x=self.x;dx=self.dx

        interp_hom = interp2d([0,0],[self.T*n,self.T*m],[dx*n,dx*m],dat_hom,k=1,p=[True,True])

        interp_hets = []
        for j in range(k):
            interp_het = interp2d([0,0],[self.T*n,self.T*m],[dx*n,dx*m],dat_hets[j],k=1,p=[True,True])
            interp_hets.append(interp_het)

        ta = self.ths[0];tb = self.ths[1]
        name_hom = 'p_'+system1.model_name+'_hom'+str(k)
        imp_hom = imp_fn(name_hom,self.fast_interp_lam(interp_hom))
        lam_hom = lambdify(self.ths,imp_hom(ta,tb))

        imp_hets = []
        lam_hets = []

        for j in range(k):
            name_het = 'p_'+system1.model_name+'_het'+str(k)+str(j+1)
            imp_het = imp_fn(name_het,self.fast_interp_lam(interp_hets[j]))
            lam_het = lambdify(self.ths,imp_het(ta,tb))
            imp_hets.append(imp_het)
            lam_hets.append(lam_het)
            
        
        if k == 0:
            imp_hom = imp_fn('p_'+system1.model_name+'_hom0', lambda x: 0*x)
            imp_hets = [imp_fn('p_'+system1.model_name+'_het00', lambda x: 0*x)]
            system1.p['imp_hom'][k] = imp_hom
            system1.p['lam_hom'][k] = 0

            system1.p['imp_het'][k] = imp_hets
            system1.p['lam_het'][k] = [0]
            
        else:
            system1.p['imp_hom'][k] = imp_hom
            system1.p['lam_hom'][k] = lam_hom

            system1.p['imp_het'][k] = imp_hets
            system1.p['lam_het'][k] = lam_hets

        if self.save_fig:
            
            fig,axs = plt.subplots()
            axs.imshow(p_data)
            pname = 'figs_temp/p_'
            pname += system1.model_name+str(k)
            pname += '_'+str(n)+str(m)
            pname += '.png'
            plt.savefig(pname)
            plt.close()

        mname = system1.model_name
        system1.rule_p.update({sym.Indexed('p_'+mname+'_hom',k):system1.p['imp_hom'][k](ta,tb)})
        for j in range(len(system1.p['lam_het'][k])):
            system1.rule_p.update({sym.IndexedBase('p_'+mname+'_het')[k,j+1]:system1.p['imp_het'][k][j](ta,tb)})

        
    def generate_p(self,system1,system2,expr,k):
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

        rule.pop('del0',None)
        #rule.pop('del1',None)

        ph_imp1 = expr.subs(rule)

        #print('ph_imp1',ph_imp1)

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
            data[(a_i+ll)%(n*NP),a_i] = conv[-m*(NP):]

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

            #print('v1,v2',v1,v2)
                    
            tmp = v1.dot(v2)
            tmp = sym.expand(tmp,**self._expand_kws)
            
            tmp = tmp.subs(rule_trunc)
            tmp = collect(tmp,system.eps)
            tmp = collect(expand(tmp),system.eps)

            for k in range(system.miter):
                system.h['sym'][k] = tmp.coeff(system.eps,k)

                #print('\tidx,k,h sym',system.idx,k,system.h['sym'][k])

            dill.dump(system.h['sym'],open(fname,'wb'),recurse=True)
                
        else:
            #logging.info('* Loading H symbolic...')
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

            expr = system1.h['sym'][k]
            h_data = self.generate_h(system1,system2,expr,k)
            np.savetxt(system1.h['fnames_data'][k],h_data)

        else:
            str1 = '* Loading H {}, order={}...'
            logging.info(str1.format(system1.model_name,k))
            print(str1.format(system1.model_name,k))
            
            h_data = np.loadtxt(fname)

        system1.h['dat'][k] = h_data
        hodd = h_data[::-1] - h_data

        n=self._n[1];m=self._m[1]

        system1.h['lam'][k] = interp1d(0,self.T*n,self.dx,h_data,p=True,k=3)

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



    def load_h_hom_het(self,system1,system2,k):
        """
        explicitly save homogeneous and heterogeneous parts of p functions.
        """

        fname_hom = system1.h['fnames_data_hom'][k]
        fname_hets = system1.h['fnames_data_het'][k]

        file_dne = not(os.path.isfile(fname_hom))
        for j in range(len(fname_hets)):
            file_dne += not(os.path.isfile(fname_hets[j]))

        # rule to break up het. terms
        rule = {**system1.rule_p2het,**system2.rule_p2het}

        # alias
        b = self.system1.pardict_sym['del0']

        if 'h_data_'+system1.model_name in self.recompute_list or file_dne:
            str1 = '* Computing H {}, order={}...'
            print(str1.format(system1.model_name,k))

            # plug in explicit homogenous and heterogenous expansions
            expr = sym.expand(system1.h['sym'][k].subs(rule))
            expr = sym.collect(expr,b)

            #print('expr,h',system1.idx,expr)
            expr_hom = expr.coeff(b,0)

            expr_hets = []
            for j in range(k+1):
                
                expr_het = expr.coeff(b,j+1)
                expr_hets.append(expr_het)
                #print('expr_hets',expr_hets[j])

            #print('dat hom')
            dat_hom = self.generate_h(system1,system2,expr_hom,k)

            dat_hets = []
            for j in range(k+1):
                #print('dat het',j)
                
                dat_het = self.generate_h(system1,system2,expr_hets[j],k)
                dat_hets.append(dat_het)

            np.savetxt(system1.h['fnames_data_hom'][k],dat_hom)

            np.savetxt(system1.h['fnames_data_het'][k][j],dat_hets[j])
            for j in range(k+1):
                np.savetxt(system1.h['fnames_data_het'][k][j],dat_hets[j])


        else:
            str1 = '* Loading H {}, order={}...'
            print(str1.format(system1.model_name,k))

            # note that h filenmaes need to start with k+1 het. terms
            dat_hom = np.loadtxt(fname_hom)

            dat_hets = []
            for j in range(k+1):
                dat_het = np.loadtxt(fname_hets[j])
                dat_hets.append(dat_het)

        system1.h['dat_hom'][k] = dat_hom
        system1.h['dat_het'][k] = dat_hets

        n=self._n[1];m=self._m[1]

        system1.h['lam_hom'][k] = interp1d(0,self.T*n,self.dx,dat_hom,p=True,k=3)

        interp_hets = []
        for j in range(k+1):
            interp_het = interp1d(0,self.T*n,self.dx,dat_hets[j],p=True,k=3)
            interp_hets.append(interp_het)

        

        system1.h['lam_het'][k] = interp_hets

                
    def generate_h(self,system1,system2,expr,k):

        rule = {**system1.rule_p,**system2.rule_p,**system1.rule_par,
                **system2.rule_par,**system1.rule_lc,**system2.rule_lc,
                **system1.rule_g,**system2.rule_g,**system1.rule_z,
                **system2.rule_z,**system1.rule_i,**system2.rule_i}

        #print('k,expr',k,expr)

        if expr == 0:
            return np.zeros(len(self.x))
        
        h_lam = lambdify(self.ths,expr.subs(rule))

        x=self.x;dx=self.dx
        n=self._n[1];m=self._m[1]

        #print('idx, expr in h:',system1.idx,expr)
        #print('idx, imp in h:',system1.idx,expr.subs(rule))
        
        X,Y = np.meshgrid(x*n,x*m,indexing='ij')
        h_integrand = h_lam(X,Y)
        ft2 = fft2(h_integrand)
        # get only those in n:m indices
        # n*i,-m*i
        ft2_new = list([ft2[0,0]])
        ft2_new += list(np.diag(np.fliplr(ft2[1:,1:])))
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
