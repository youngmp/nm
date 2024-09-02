
"""
library functions for right-hand sides
"""

import numpy as np
import matplotlib.pyplot as plt

def rhs_avg_2d(t,y,a,eps=0,del1=0,miter=None):
    """ for forcing only. 2d."""

    if miter is None:
        nn = a.system1.miter
    else:
        nn = miter
    
    th,ps = y
    dth = 0
    dps = a.system1.kappa_val*ps
    
    for i in range(nn):
        dth += eps*ps**i*a.hz_lam[i](th)
        dps += eps*ps**i*a.hi_lam[i](th)
    dth -= del1/a._m[1]
    return np.array([dth*a._n[1],dps*a._m[1]])

def rhs_avg_1d(t,th,a,eps=0,del1=0,miter=None):
    """ for forcing only. 1d"""

    if miter is None:
        nn = a.system1.miter
    else:
        nn = miter
    
    dth = 0
    
    for i in range(nn):
        dth += eps**(i+1)*a.system1.h['lam'][i](th)
    dth -= del1/a._m[1]
    return dth*a._n[1]


def _redu_c(t,y,a,eps=0,del1=None,miter=None):
    """ for coupling only. 1d"""
    system1 = a.system1; system2 = a.system2
    
    if miter is None:
        nn = a.system1.miter
    else:
        nn = miter
    
    h = 0
    for i in range(nn):
        h += eps**(i+1)*(system1.h['lam'][i](y) - system2.h['lam'][i](y))
    return h


def _redu(t,y,a,eps=.01,del1=0):
    th,ps,tf = y
    
    u = a.system1.forcing_fn(tf)
    
    dth = 1 + eps*a.th_lam(th,ps)*u
    dps = a.system1.kappa_val*ps + eps*a.ps_lam(th,ps)*u
    dtf = 1 + del1/a._m[1]
    return np.array([dth*a._n[1],dps*a._n[1],dtf*a._m[1]])


def _redu_moving(t,y,a,eps=.01,del1=0):
    th,ps,tf = y

    tf_m = tf+a._m[1]*t
    th_m = th+a._n[1]*t
    
    u = a.system1(tf_m)
    
    dth = eps*a.th_lam(th_m,ps)*u
    dps = a.system1.kappa_val*ps + eps*a.ps_lam(th_m,ps)*u
    dtf = a.om*del1/a._m[1]
    return np.array([dth*a._n[1],dps*a._n[1],dtf*a._m[1]])


def _redu_moving_avg(t,y,a,eps=.01,del1=0):

    s,ds = np.linspace(0,2*np.pi*a._m[1],a.NH,retstep=True)
    th,ps,tf = y

    om = a._n[1]/a._m[1]
    in1 = th-om*tf+om*s;
    in2 = s

    z = a.th_lam(in1,ps)
    i = a.ps_lam(in1,ps)
    u = a.system2(in2)
    
    dth = eps*np.sum(z*u)*ds/(2*np.pi*a._m[1])
    dps = a.system1.kappa_val*ps + eps*np.sum(i*u)*ds/(2*np.pi*a._m[1])
    dtf = del1
    return np.array([dth*a._n[1],dps*a._n[1],dtf])


def _full(t,y,a,eps,del1=None):
    """del1 in a pardict"""
    pd1 = a.system1.pardict;pd2 = a.system2.pardict
    y1 = y[:4];y2 = y[4:]

    c1 = a.system1.coupling(y,pd1,'val',0)
    c2 = a.system2.coupling(list(y2)+list(y1),pd2,'val',1)
    
    out1 = a.system1.rhs(t,y1,pd1,'val',0) + eps*c1
    out2 = a.system2.rhs(t,y2,pd2,'val',1) + eps*c2
    return np.array(list(out1)+list(out2))



def _redu_full(t,y,a,eps:float=.01,del1=None):
    """
    Original reduction without averaging
    """
    s1 = a.system1;s2 = a.system2
    thA,psA,thB,psB = y    
    k1 = s1.kappa_val;k2 = s2.kappa_val

    in1 = thA-a.om*thB+t;in2 = t

    dthA = eps*s1.gz_lam(in1,in2,psA,psB)
    dthB = eps*s2.gz_lam(in1,in2,psA,psB)
    
    dpsA = k1*psA + eps*s1.gi_lam(in1,in2,psA,psB)
    dpsB = k2*psB + eps*s2.gi_lam(in1,in2,psA,psB)

    return np.array([dthA,dpsA,dthB,dpsB])


def _redu_3dc_gw(t,y,a,eps:float=.01,del1=None):
    """
    Goodwin-specific 3d (1 phase diff, 2 isostables)
    """
    system1 = a.system1;system2 = a.system2
    pdA = a.system1.pardict;pdB = a.system2.pardict

    s,ds = np.linspace(0,2*np.pi*a._m[1],2000,retstep=True)
    phi,psA,psB = y

    k1 = system1.kappa_val;k2 = system2.kappa_val
    Tm = (2*np.pi*a._m[1])

    om = a._n[1]/a._m[1]
    in1 = phi+om*s;in2 = s

    zA = a.th_lam(in1,psA);iA = a.ps_lam(in1,psA)
    zB = a.th_lam(in2,psB);iB = a.ps_lam(in2,psB)

    # coupling terms
    v1 = system1.lc['lam_v0'](in1) + a.g_lam(in1,psA)
    v2 = system2.lc['lam_v1'](in2) + a.g_lam(in2,psB)

    d1 = a.system1.pardict['del0']

    GA = (pdA['K0']*(v1+v2)/2/pdA['kc0']+d1)*pdA['om_fix0']
    GB = (pdB['K1']*(v2+v1)/2/pdB['kc1'])*pdB['om_fix1']

    dthA = eps*np.sum(zA*GA)*ds/Tm
    dthB = eps*np.sum(zB*GB)*ds/Tm
    dphi = dthA - dthB
    
    dpsA = k1*psA + eps*np.sum(iA*GA)*ds/Tm
    dpsB = k2*psB + eps*np.sum(iB*GB)*ds/Tm
    
    return np.array([dphi,dpsA,dpsB])


def _redu_3dc_thal(t,y,a,eps:float=.01,del1=None):
    """
    thal-specific 3d (1 phase diff, 2 isostables)
    """
    
    s1 = a.system1;s2 = a.system2
    phi,psA,psB = y
    k1 = s1.kappa_val;k2 = s2.kappa_val

    s,ds = np.linspace(0,2*np.pi*a._m[1],3000,retstep=True,endpoint=False)
    Tm = s[-1]
    om = a._n[1]/a._m[1]
    in1 = phi+om*s;in2 = s

    dthA = np.sum(s1.gz_lam(in1,in2,psA,psB))*ds/Tm
    dthB = np.sum(s2.gz_lam(in1,in2,psA,psB))*ds/Tm
    dphi = eps*(dthA - dthB)
    
    dpsA = k1*psA + eps*np.sum(s1.gi_lam(in1,in2,psA,psB))*ds/Tm
    dpsB = k2*psB + eps*np.sum(s2.gi_lam(in1,in2,psA,psB))*ds/Tm
    
    return np.array([dphi,dpsA,dpsB])


def _redu_4dc_thal(t,y,a,eps=.01,del1=None):
    """
    thal-specific 3d (1 phase diff, 2 isostables)
    """
    
    s1 = a.system1;s2 = a.system2
    thA,psA,thB,psB = y
    k1 = s1.kappa_val;k2 = s2.kappa_val

    #dper = eps*np.mean(a.system1.z['dat'][0][:,0])
    #dper += eps**2*np.mean(a.system1.z['dat'][1][:,0])

    s,ds = np.linspace(0,2*np.pi*a._m[1],3000,retstep=True,endpoint=False)
    
    Tm = s[-1]
    om = a._n[1]/a._m[1]
    in1 = thA-om*thB + om*s;in2 = s
    
    dthA = eps*np.sum(s1.gz_lam(in1,in2,psA,psB))*ds/Tm
    dthB = eps*np.sum(s2.gz_lam(in1,in2,psA,psB))*ds/Tm
    
    dpsA = k1*psA + eps*np.sum(s1.gi_lam(in1,in2,psA,psB))*ds/Tm
    dpsB = k2*psB + eps*np.sum(s2.gi_lam(in1,in2,psA,psB))*ds/Tm
    
    return np.array([dthA,dpsA,dthB,dpsB])
