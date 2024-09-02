from .rhs import _redu_c
from . import util
from .util import (bif1d, _get_sol, get_phase, add_arrow_to_line2D,
                   _get_sol_3d)

from scipy.spatial.distance import cdist
from scipy.signal import find_peaks
from scipy.integrate import solve_ivp

import matplotlib.pyplot as plt

import scipy as sp
import numpy as np

import os


def construct_branch(eps_list,zu,tol:float=.25):
    """
    tol: tolerance to cut discontinuous points.
    zu must be list for this code to work.
    """
    
    x = [];y = []
    idx2 = 0
    zu = list(zu)
    for i in range(len(zu)):
        if len(zu[i]) > 0:
            
            # take first point
            if len(x) == 0:
                y.append(zu[i][0])
                x.append(eps_list[i])
                zu[i] = np.delete(zu[i],0)

            # save subsequent points
            else:
                min_idx = np.argmin(np.abs(y[idx2]-zu[i]))
                if np.abs(y[idx2]-zu[i][min_idx]) < tol and\
                   np.abs(eps_list[idx2]-eps_list[min_idx]) < tol:
                #if np.abs(y[idx2]-zu[i][min_idx]) < tol or\
                #   np.abs(np.tan(y[idx2])-np.tan(zu[i][min_idx])) < tol:
                    y.append(zu[i][min_idx])
                    x.append(eps_list[i])
                    zu[i] = np.delete(zu[i],min_idx)
                    idx2 += 1
    return zu, x, y


def add_diagram_1d(axs,a,del1,eps_tup=(0.001,.5,200),
                   rhs=_redu_c,
                   domain=np.linspace(-.1,2*np.pi+.1,2000),
                   include_limits=True,label='',tol=.25):
    """
    for non-scatter diagrams (a little slower than scatter)
    """
    
    zu = [];zs = []

    eps_list = np.linspace(*eps_tup)
    for eps in eps_list:
        z1,z2 = bif1d(a,eps,del1,domain,rhs=rhs)

        zu.append(np.mod(z1,2*np.pi))
        zs.append(np.mod(z2,2*np.pi))

    # separate branches
    x = [1]
    while len(x)>0:
        zu,x,y = construct_branch(eps_list,zu,tol=tol)
        axs.plot(x,y,color='tab:red',ls='--')

    x = [1]
    count = 0
    while len(x)>0:
        zs,x,y = construct_branch(eps_list,zs,tol=tol)
        if count == 0:
            label1 = label
        else:
            label1 = ''
        axs.plot(x,y,color='tab:red',label=label1)
        count += 1

    if include_limits:
        axs.set_ylim(-.1,2*np.pi+.1)
        axs.set_xlim(eps_list[0],eps_list[-1])

    return axs

def add_diagram_1d_scatter(axs,a,del1,eps_tup=(0.001,.5,200),
                   rhs=_redu_c,
                   domain=np.linspace(-.1,2*np.pi+.1,2000),
                   include_limits=True,label=''):

    eps_list = np.linspace(*eps_tup)
    zu = [];zs = []

    for eps in eps_list:
        z1,z2 = bif1d(a,eps,del1,domain=domain,rhs=rhs)

        zu.append(np.mod(z1,2*np.pi))
        zs.append(np.mod(z2,2*np.pi))

    for xe, ye in zip(eps_list, zu):
        axs.scatter([xe] * len(ye), ye,s=2,c='red')

    for xe, ye in zip(eps_list, zs):
        axs.scatter([xe] * len(ye), ye,s=2,c='k')

    if include_limits:
        axs.set_ylim(-.1,2*np.pi+.1)
        axs.set_xlim(eps_list[0],eps_list[-1])

    return axs


def load_diagram_2d(a,del1,eps_tup=[0.01,0.5,100],dir1='bifdat'):
    
    model_name = a.system1.model_name
    ratio = str(a._n[1])+str(a._m[1])
    elo, ehi, N = eps_tup
    rawname_s = '{}/{}_2d_s_de={}_ratio={}_elo={}_ehi={}_N={}.npy'
    rawname_u = '{}/{}_2d_u_de={}_ratio={}_elo={}_ehi={}_N={}.npy'
    fname_s = rawname_s.format(dir1,model_name,del1,ratio,elo,ehi,N)
    fname_u = rawname_u.format(dir1,model_name,del1,ratio,elo,ehi,N)
    
    if not(os.path.isdir(dir1)):
        os.mkdir(dir1)

    file_dne = not(os.path.isfile(fname_s))
    file_dne += not(os.path.isfile(fname_u))

    eps_list = np.linspace(*eps_tup)

    if file_dne:
        zs = [];zu = []
        for eps in eps_list:
            co1,co2,pts = util.pl_exist_2d(eps,del1,a,return_data=True,
                                           pmin=-2,pmax=2)

            path1 = co1.get_paths()[0]
            path2 = co2.get_paths()[0]

            # brute force distance for intersections
            # based on finding local minima in distances
            d = cdist(path1.vertices,path2.vertices)
            p1_min_idxs = sp.signal.find_peaks(-np.amin(d,axis=1))[0]
            p2_min_idxs = np.argmin(d,axis=1)[p1_min_idxs]

            sub_pts_s = []
            sub_pts_u = []

            for i in range(len(p1_min_idxs)):
                idx1 = p1_min_idxs[i]
                idx2 = p2_min_idxs[i]
                v1 = np.array([path1.vertices[idx1,0],path1.vertices[idx1,1]])
                v2 = np.array([path2.vertices[idx2,0],path2.vertices[idx2,1]])

                if np.linalg.norm(v1-v2) < 2e-3:
                    J = util.jac_2d(v1,a,eps,del1)
                    if util.is_stable(J):
                        sub_pts_s.append(path1.vertices[idx1,0])
                    else:
                        sub_pts_u.append(path1.vertices[idx1,0])

            zs.append(sub_pts_s)
            zu.append(sub_pts_u)

        np.save(fname_s,np.array(zs,dtype=object))
        np.save(fname_u,np.array(zu,dtype=object))

        
    else:
         zs = np.load(fname_s,allow_pickle=True)
         zu = np.load(fname_u,allow_pickle=True)

    return zs,zu

def add_diagram_2d(axs,a,del1,eps_tup):

    zs,zu = load_diagram_2d(a,del1,eps_tup=eps_tup,
                            dir1='bifdat')

    eps_list = np.linspace(*eps_tup)
    
    # separate branches
    x = [1]
    while len(x)>0:
        zu,x,y = construct_branch(eps_list,zu)
        axs.plot(x,y,color='tab:blue',ls='--')

    x = [1]
    while len(x)>0:
        zs,x,y = construct_branch(eps_list,zs)
        axs.plot(x,y,color='tab:blue')

    axs.set_ylim(-.1,2*np.pi+.1)
    axs.set_xlim(eps_list[0],eps_list[-1])

    return axs


def load_diagram_full(a,del1,eps_tup=[0.01,0.011,2],dir1='bifdat',
                      rhs=None,phi0=np.pi,recompute=False,
                      maxt=100,scale_t_eps=True):
    
    """
    winow shift: needed to search for other phase-locked states
    besides the peaks that are closest to the peak of the oscillator.
    important for, e.g., 1:2 forcing.
    """
    model_name = a.system1.model_name
    ratio = str(a._n[1])+str(a._m[1])
    elo, ehi, N = eps_tup
    rawname_s = '{}/{}_full_s_p0={}_ratio={}_elo={}_ehi={}_N={}.npy'
    aa = (dir1,model_name,phi0,ratio,elo,ehi,N,)
    fname_s = rawname_s.format(*aa)
    
    if not(os.path.isdir(dir1)):
        os.mkdir(dir1)

    file_dne = not(os.path.isfile(fname_s))
    #file_dne += not(os.path.isfile(fname_u))

    eps_list = np.linspace(*eps_tup)

    if file_dne or recompute:
        eps_list = np.linspace(*eps_tup)
        phi_list = []
        for eps_init in eps_list:
            if scale_t_eps:
                tot = maxt/eps_init
            else:
                tot = maxt
            
            phi = util.get_phase_diff_f(rhs=rhs,
                                        phi0=phi0,
                                        a=a,
                                        eps=eps_init,
                                        del1=del1,
                                        max_time=tot)
            
            phi_list.append(phi)

        np.savetxt(fname_s,np.array(phi_list))
        #np.save(fname_u,np.array(zu,dtype=object))

        
    else:
         phi_list = list(np.loadtxt(fname_s))
         #zu = np.load(fname_u,allow_pickle=True)

    return phi_list#,zu

def add_diagram_full(axs,a,del1,eps_tup,rhs=None,phi0=np.pi,
                     recompute=False,maxt=100,scale_t_eps=True,
                     branch_tol=6e-1,similarity_tol=5e-1):

    zs = load_diagram_full(a,del1,eps_tup=eps_tup,dir1='bifdat',
                           rhs=rhs,phi0=phi0,recompute=recompute,
                           maxt=maxt,scale_t_eps=scale_t_eps)

    zs = np.mod(zs,2*np.pi)
    eps_list = np.linspace(*eps_tup)


    print('zs original',zs)
    if a._m[1] > 1:
        for i in range(a._m[1]):
            zs,x,y = construct_branch(eps_list,zs,tol=branch_tol)
            axs.plot(x,y,color='k')
        print('zs remaining',zs)
        # now that zs has a bunch of empty elements, remove them until
        # a nonempty element is found.
        zs_remainder = []
        eps_remainder = []
        for i in range(len(zs)):
            temp_list = []
            if len(zs[i]) != 0:
                
                # check if point is close to existing points on diagram
                for pt_y in zs[i]:
                    pt_x = eps_list[i]
                    print('')
                    print('check point',pt_x,pt_y)
                    diff_prev = 10
                    for line in axs.lines:
                        pt = np.array([pt_x,pt_y])
                        diff = np.linalg.norm(pt-line.get_xydata(),axis=1)
                        
                        #print('diff min',np.amin(diff),diff_prev)
                        if np.amin(diff) < diff_prev:
                            diff_prev = np.amin(diff)
                    print(diff_prev,similarity_tol)
                    if diff_prev > similarity_tol:
                        temp_list.append(pt_y)

                if len(temp_list) == 0:
                    pass
                else:
                    print('zs[i]',i,zs[i],len(zs[i]),temp_list)
                    zs_remainder.append(temp_list)
                    eps_remainder.append(eps_list[i])

        print('zs collected',zs_remainder)
        if len(zs_remainder) == 0:
            pass
        else:
                           
            for i in range(len(zs_remainder[0])):
                
                zs_remainder,x,y = construct_branch(eps_remainder,zs_remainder,
                                                    tol=branch_tol)
                axs.plot(x,y,color='k')
        print('zs_remainder',zs_remainder)
    else:
        axs.plot(eps_list,zs,color='k')
    

    return axs


def draw_full_solutions(axs,a,T,eps,init,full_rhs,recompute=False,
                        label='Full',skipn=50):
    """
    draw full solution give axis object
    init is initial phase.
    """
    
    system1 = a.system1;system2 = a.system2
    kw1 = {'eps':eps,'a':a,'return_data':True}
    dt = .02;t = np.arange(0,T,dt)

    # estimate initial phase
    y0a = list(system1.lc['dat'][int((init/(2*np.pi))*system1.TN),:])
    y0b = list(system2.lc['dat'][int((0/(2*np.pi))*system2.TN),:])
    y0 = np.array(y0a+y0b)

    solf = _get_sol(full_rhs,y0,t,args=(a,eps),recompute=recompute)

    tpa, phasea = get_phase(t,solf[:,:4],skipn=skipn,system1=system1)
    tpb, phaseb = get_phase(t,solf[:,4:],skipn=skipn,system1=system2)

    y = np.mod(phasea-a.om*phaseb,2*np.pi)
    axs.scatter(y,tpa,s=5,color='gray',alpha=.25,label=label)

    return axs


def draw_1d_rhs(axs,a,T,eps,init,rhs=_redu_c):
    x = np.linspace(0,2*np.pi,200)
    y = rhs(0,x,a,eps)
    axs.plot(x,y,color='k',lw=1)
    axs.axhline(0,x[0],x[-1],color='gray',lw=1,ls='--')

    # trajectory
    dt = .02;
    t = np.arange(0,T,dt)
    th_init = init

    args1 = {'args':[a,eps],'rtol':1e-7,'atol':1e-7,'method':'LSODA',
             't_span':[0,t[-1]],'t_eval':t}

    th_init = init
    # solution on 1d phase plane
    solr1d = solve_ivp(rhs,y0=[th_init],**args1)
        
    xs = np.mod(solr1d.y.T[:,0],2*np.pi)
    ys = np.zeros(len(xs))
    discont_idxs = np.abs(np.gradient(xs,1)) > np.pi/2

    xs[discont_idxs] = np.nan
    ys[discont_idxs] = np.nan

    line, = axs.plot(xs,ys,color='tab:red',alpha=.75,ls='--')
    add_arrow_to_line2D(axs,line,arrow_locs=[.25,.75])


    return axs

def draw_1d_solutions(axs,a,T,eps,init,rhs=_redu_c):

    system1 = a.system1;system2 = a.system2
    kw1 = {'eps':eps,'a':a,'return_data':True}
    dt = .02;t = np.arange(0,T,dt)

    kw1 = {'eps':eps,'a':a,'return_data':True}

    # trajectory
    dt = .02;
    t = np.arange(0,T,dt)
    th_init = init

    args1 = {'args':[a,eps],'rtol':1e-7,'atol':1e-7,'method':'LSODA',
             't_span':[0,t[-1]],'t_eval':t}

    # solution on 1d phase plane
    solr1d = solve_ivp(rhs,y0=[th_init],**args1)
    axs.plot(np.mod(solr1d.y.T[:,0],2*np.pi),t,color='tab:red',label='1D',
             ls='--',zorder=5)
        
    return axs


def draw_3d_solutions(axs,a,T,eps,init,rhs):

    system1 = a.system1;system2 = a.system2
    kw1 = {'eps':eps,'a':a,'return_data':True}
    dt = .02;t = np.arange(0,T,dt)
    th_init = init
    
    # load or recompute solution
    y = _get_sol_3d(rhs,y0=[th_init,0,0],t=t,args=(a,eps,a.del1))

    axs.plot(np.mod(y[:,0],2*np.pi),t,color='tab:blue',alpha=.75,label='3D',
             zorder=1,lw=2)

    return axs


def draw_quick_plot_f(axs,data_list,a,norm=True):
    
    for i in range(len(data_list)):
        data1,data2 = data_list[i] # data1 is periods, data2 is t_diffs

        for j in range(a._m[1]+a._n[1]):
            if norm: # normalize by n period
                y = 2*np.pi*data2[:,j+1]/data1[:,1]
            else:
                y = data2[:,j+1]
            axs.plot(data1[:,0],np.mod(y,2*np.pi),color='gray',lw=2)
        
    axs.set_title('Full del={}'.format(a.del1))

    return axs

def draw_quick_plot_r3d(axs,data_list,a):
    for i in range(len(data_list)):
        data = data_list[i]
        axs.plot(data[:,0],np.mod(data[:,1],2*np.pi),ls=':',
                 color='tab:red',lw=2)
    
    axs.set_title('r3d del={}'.format(a.del1))
    
    return axs

def quick_plot_f(data_list,a,ylim=[-.1,2*np.pi+.1]):
    kwargs = locals()
    
    fig,axs = plt.subplots(figsize=(3,2))

    axs = draw_quick_plot_f(axs,**kwargs)
    axs.set_ylim(*ylim)
    axs.set_xlabel('eps')
    axs.set_ylabel('phi')
    
    plt.tight_layout()

def quick_plot_r(a,etup):
    fig,axs = plt.subplots(figsize=(3,2))
    axs = add_diagram_1d_scatter(axs,a,a.del1,etup)
    axs.set_xlabel('eps')
    axs.set_ylabel('phi')
    axs.set_title('r1d delta={}, k={}'.format(a.del1,a.system1.trunc_order))
    plt.tight_layout()

def quick_plot_r3d(data_list,a,ylim=[-.1,2*np.pi+.1]):
    kwargs = locals()
    
    fig,axs = plt.subplots(figsize=(3,2))
    

    axs = draw_quick_plot_r3d(axs,**kwargs)
    axs.set_xlabel('eps')
    axs.set_ylabel('phi')

    axs.set_ylim(*ylim)
    plt.tight_layout()


def draw_isi(a,data_list):
    for i in range(len(data_list)):
        data1,data2 = data_list

    
    if i == 0:
        data_t = data1
        data_diff = data2
    else:
        data_t = np.append(data_t,data1,axis=0)
        data_diff = np.append(data_diff,data2,axis=0)
                
    axs[0].plot(data1[:,0],data1[:,1:1+a._n[1]],color='gray')
    axs[1].plot(data1[:,0],data1[:,1+a._n[1]:],color='k')


        
    
def quick_plot_combined(a,kw_f=None,kw_r=None,kw_r3d=None,
                        ylim=[-.1,2*np.pi+.1]):
    
    fig,axs = plt.subplots(figsize=(3,2))
    
    if not(kw_f) is None:
        axs = draw_quick_plot_f(axs,kw_f['data_list'],a)

    if not(kw_r) is None:
        axs = add_diagram_1d_scatter(axs,a,a.del1,kw_r['etup'])

    if not(kw_r3d) is None:
        axs = draw_quick_plot_r3d(axs,kw_r3d['data_list'],a)

    if not(kw_f) is None:
        fig,axs = plt.subplots(2,1,figsize=(3,4))

        max_diff = .02
        eps_min1 = 1
        eps_min2 = 1

        for i in range(len(kw_f['data_list'])):
            data1,data2 = kw_f['data_list'][i]

            if i == 0:
                data_t = data1
                data_diff = data2
            else:
                data_t = np.append(data_t,data1,axis=0)
                data_diff = np.append(data_diff,data2,axis=0)
                
            axs[0].plot(data1[:,0],data1[:,1:1+a._n[1]],color='gray')
            axs[1].plot(data1[:,0],data1[:,1+a._n[1]:],color='k')

        if a._n[1]>1:
            y1 = data_t[:,1]
            for j in range(1,a._n[1]):
                idxs = (np.abs(y1-data_t[:,1+j])>max_diff)
                
                if sum(idxs) == 0:
                    pass
                elif eps_min1 > np.amin(data_t[:,0][idxs]):
                    eps_min1 = np.amin(data_t[:,0][idxs])

        if a._m[1]>1:
            y2 = data_t[:,1+a._n[1]]
            for j in range(1,a._m[1]):
                idxs = (np.abs(y2-data_t[:,1+a._n[1]+j])>max_diff)
                
                if sum(idxs) == 0:
                    pass
                elif eps_min2 > np.amin(data_t[:,0][idxs]):
                    eps_min2 = np.amin(data_t[:,0][idxs])

        if eps_min1 != 1 or eps_min2 != 1:
            eps_min = min(eps_min1,eps_min2)
            print('eps_min',eps_min)

            for j in range(2):
                axs[j].axvline(eps_min,ls='--',color='gray')
                axs[j].axvspan(0,eps_min,color='tab:green',alpha=.15)
                axs[j].axvspan(kw_r['etup'][1],eps_min,color='tab:red',alpha=.15)
        

        axs[0].set_title('Oscillator 1')
        axs[1].set_title('Oscillator 2')

        plt.tight_layout()

