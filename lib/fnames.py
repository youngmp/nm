

def load_fnames_response(obj,model_pars=''):
        
    # coupling parameters
    sim_pars = '_TN='+str(obj.TN)

    obj.lc_fname = obj.dir1+'/lc_TN={}_'+model_pars+'.txt'
    obj.m_fname = obj.dir1+'/m_TN={}_'+model_pars+'.txt'

    obj.lc_fname = obj.lc_fname.format(obj.TN)
    obj.m_fname = obj.m_fname.format(obj.TN)

    obj.A_fname = '{}A.d'.format(obj.dir1)
    
    for key in obj.var_names:
        v = '{}g{}_sym_{}.d'
        obj.g['sym_fnames_'+key] = [v.format(obj.dir1,key,k)
                                    for k in range(obj.miter)]

        v = '{}z{}_sym_{}.d'
        obj.z['sym_fnames_'+key] = [v.format(obj.dir1,key,k)
                                    for k in range(obj.miter)]
        
        v = '{}i{}_sym_{}.d'
        obj.i['sym_fnames_'+key] = [v.format(obj.dir1,key,k)
                                    for k in range(obj.miter)]

    v = '{}g_data_{}{}{}.txt'
    obj.g['dat_fnames'] = [v.format(obj.dir1,k,model_pars,sim_pars)
                           for k in range(obj.miter)]

    v = '{}z_data_{}{}{}.txt'
    obj.z['dat_fnames'] = [v.format(obj.dir1,k,model_pars,sim_pars)
                           for k in range(obj.miter)]
    
    v = '{}i_data_{}{}{}.txt'
    obj.i['dat_fnames'] = [v.format(obj.dir1,k,model_pars,sim_pars)
                           for k in range(obj.miter)]

        
    for key in obj.var_names:
        obj.g[key+'_eps_fname'] = (obj.dir1+'g_'+key+'_miter='
                                      +str(obj.miter)+'.d')

        obj.z[key+'_eps_fname'] = (obj.dir1+'z_'+key+'_miter='
                                      +str(obj.miter)+'.d')

        obj.i[key+'_eps_fname'] = (obj.dir1+'i_'+key+'_miter='
                                      +str(obj.miter)+'.d')


def load_fnames_nm(system,obj,model_pars='',coupling_pars=''):

    pars = (system.dir1,system.miter,obj.forcing)
    system.G['fname'] = '{}G_{}_f={}.d'.format(*pars)
    system.K['fname'] = '{}K_{}_f={}.d'.format(*pars)
    
    system.G['fname_psi'] = '{}G_psi_{}_f={}.d'.format(*pars)
    system.K['fname_psi'] = '{}K_psi_{}_f={}.d'.format(*pars)
    
    system.p['fname'] = '{}p_{}_f={}.d'.format(*pars)
    system.h['fname'] = '{}h_{}_f={}.d'.format(*pars)

    system.G['fname_gz'] = '{}gz_{}_f={}.d'.format(*pars)
    system.G['fname_gi'] = '{}gi_{}_f={}.d'.format(*pars)

    fname_pars = (obj.NP,obj.NH,obj.pfactor,obj._n[1],obj._m[1],
                  obj.forcing,obj.del1)
    fname_pars2 = (obj.NP,obj.NH,obj.pfactor,obj._n[1],obj._m[1],
                  obj.forcing)
    
    val = '{}p_data_ord={}_NP={}_NH={}_piter={}_n={}_m={}_f={}.txt'
    system.p['fnames_data'] = [val.format(system.dir1,k,*fname_pars2)
                               for k in range(system.miter)]
    
    val = '{}h_data_ord={}_NP={}_NH={}_piter={}_n={}_m={}_f={}_de={}.txt'
    system.h['fnames_data'] = [val.format(system.dir1,k,*fname_pars)
                               for k in range(system.miter)]
