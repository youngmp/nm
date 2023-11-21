

def load_fnames_response(obj,model_pars=''):
        
    # coupling parameters
    sim_pars = '_TN='+str(obj.TN)

    obj.lc_fname = obj.dir1+'/lc_'+model_pars+'.txt'
    obj.m_fname = obj.dir1+'/m_'+model_pars+'.txt'

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
    obj.g['dat_fnames'] = [v.format(obj.dir1,k,
                                    model_pars,
                                    sim_pars)
                           for k in range(obj.miter)]

    v = '{}z_data_{}{}{}.txt'
    obj.z['dat_fnames'] = [v.format(obj.dir1,k,
                                    model_pars,
                                    sim_pars)
                           for k in range(obj.miter)]
    
    v = '{}i_data_{}{}{}.txt'
    obj.i['dat_fnames'] = [v.format(obj.dir1,k,
                                    model_pars,
                                    sim_pars)
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
    system.p['fname'] = '{}p_{}_f={}.d'.format(*pars)
    system.h['fname'] = '{}h_{}_f={}.d'.format(*pars)
    
    val = '{}p_data_ord={}_NP={}_NH={}_piter={}_n={}_m={}_f={}.txt'
    system.p['fnames_data'] = [val.format(system.dir1,k,obj.NP,obj.NH,
                                          obj.pfactor,obj._n[1],obj._m[1],
                                          obj.forcing)
                               for k in range(system.miter)]
    
    val = '{}h_data_ord={}_NP={}_NH={}_piter={}_n={}_m={}_f={}.txt'
    system.h['fnames_data'] = [val.format(system.dir1,k,obj.NP,obj.NH,
                                          obj.pfactor,obj._n[1],obj._m[1],
                                          obj.forcing)
                               for k in range(system.miter)]
