import numpy as np
def build_D_vec2(x, tau1, tau2=False, linear=False):
    """Build the delta vector depending on the taus. If linear means no growth stop (i.e. linear model) and if tau2=False means only 1 growth arrest time (lag model) else the model is growht, arrest, growth """
    if linear: 
        return np.arange(x.shape[0])[:,None]
    elif tau2 == False :
        D =  np.vstack((np.zeros(tau1+1)[:,None],np.arange(1,x.shape[0]-tau1)[:,None]))
        return D
    else:
        D1 =  np.vstack((np.arange(tau1+1)[:,None],(tau1)*np.ones(x.shape[0]-tau1-1)[:,None]))
        D2 =  np.vstack((np.zeros(tau2+1)[:,None],np.arange(1,x.shape[0]-tau2)[:,None]))
        return D1, D2
def log_lik_taus2(x,tau1,tau2,linear):
    """The loglik depending on taus only"""
    assert x.shape[1]==1
    if linear:
        D = build_D_vec2(x,tau1,tau2,linear)
        Dc= D-np.mean(D)
        xc = x-np.mean(x)
        d0 = np.dot(xc.T,Dc)/np.dot(Dc.T,Dc)
        f = np.dot(xc.T-d0*Dc.T,xc-d0*Dc)
        F = np.log(f/2)
        Fs = -np.dot(Dc.T,Dc)/f*2
        return {'LL':(1-x.shape[0])/2*F-0.5*np.log((1-x.shape[0])/2*Fs),'d0':d0,'tau1':0,'tau2':0}
    elif tau2 == False:
        D = build_D_vec2(x,tau1,tau2,linear)
        Dc= D-np.mean(D)
        xc = x-np.mean(x)
        d0 = np.dot(xc.T,Dc)/np.dot(Dc.T,Dc)
        f = np.dot(xc.T-d0*Dc.T,xc-d0*Dc)
        F = np.log(f/2)
        Fs = -np.dot(Dc.T,Dc)/f*2
        return {'LL':(1-x.shape[0])/2*F-0.5*np.log((1-x.shape[0])/2*Fs),'d0':d0,'tau1':tau1,'tau2':0}
    else: 
        D1,D2 = build_D_vec2(x,tau1,tau2,linear)
        D1c= D1-np.mean(D1)
        D2c= D2-np.mean(D2)
        xc = x-np.mean(x)
        d1 = (np.dot(xc.T,D1c)*np.dot(D2c.T,D2c)-np.dot(xc.T,D2c)*np.dot(D1c.T,D2c))/(-np.dot(D1c.T,D2c)**2+np.dot(D1c.T,D1c)*np.dot(D2c.T,D2c))
        d2 = (np.dot(xc.T,D2c)-d1*np.dot(D1c.T,D2c))/np.dot(D2c.T,D2c)
        f = np.dot(xc.T-d1*D1c.T-d2*D2c.T,xc-d1*D1c-d2*D2c)
        F = np.log(f/2)
        detH = -(np.dot(D1c.T,D1c)*np.dot(D2c.T,D2c)-np.dot(D2c.T,D1c)**2)/f*2
        return {'LL':(1-x.shape[0])/2*F-0.5*np.log((1-x.shape[0])/2*detH),'d1':d1,'d2':d2,'tau1':tau1,'tau2':tau2}
def best_param2(x,tau1,tau2,d1,d2,linear):
    """return best param knowing the best tau1,tau2 and so best d0"""
    if linear:
        assert d1==d2
        d0=d1
        D = build_D_vec2(x,tau1,tau2,linear)
        c0 = np.mean(x)-d0*np.mean(D)
        sigmas = np.var(x-np.ones_like(x)*c0-d0*D)
        param={'c0':c0,'sigmas':sigmas,'d0':d0,'tau1':tau1,'tau2':tau2}
        vec = {'D':D}
        return {'param':param,'vec':vec}
    elif tau2 is False:
        assert d1==d2
        d0=d1
        D = build_D_vec2(x,tau1,tau2,linear)
        c0 = np.mean(x)-d0*np.mean(D)
        sigmas = np.var(x-np.ones_like(x)*c0-d0*D)
        param={'c0':c0,'sigmas':sigmas,'d0':d0,'tau1':tau1,'tau2':tau2}
        vec = {'D':D}
        return {'param':param,'vec':vec}
    else:
        D1,D2 = build_D_vec2(x,tau1,tau2,linear)
        c0 = np.mean(x)-d1*np.mean(D1)-d2*np.mean(D2)
        sigmas = np.var(x-np.ones_like(x)*c0-d1*D1-d2*D2)
        param={'c0':c0,'sigmas':sigmas,'d1':d1,'d2':d2,'tau1':tau1,'tau2':tau2}
        vec = {'D1':D1,'D2':D2}
        return {'param':param,'vec':vec}
def predict_param(x,linear,lag_linear,linear_lag_linear,bilinear,diff=1,start=1):
    """Find best parameters setting with respective log_likelihood. Diff is minimal difference between tau1 and tau2 in the case of linear_lag_linear and start is the fisrt starting point to compute the lag (non for "linear"case)"""
    tmp = []; LL = [] 
    if linear:
        foo = log_lik_taus2(x,None,None,True)
        model_param = best_param2(x,None,None,foo['d0'],foo['d0'],True)
        max_LL = foo
    elif lag_linear:
        for tau1 in range(start,x.shape[0]-1):
            foo = log_lik_taus2(x,tau1,False,False)
            tmp.append(foo)
            LL.append(foo['LL'])
        max_LL = tmp[np.argmax(LL)]
        model_param = best_param2(x,max_LL['tau1'],False,max_LL['d0'],max_LL['d0'],False)
    elif bilinear:
        for tau1 in range(start,x.shape[0]-2):
           foo = log_lik_taus2(x,tau1,tau1,False)
           tmp.append(foo)
           LL.append(foo['LL'])
        max_LL = tmp[np.argmax(LL)]
        model_param = best_param2(x,max_LL['tau1'],max_LL['tau2'],max_LL['d1'],max_LL['d2'],False)
    elif linear_lag_linear:
        for tau1 in range(start,x.shape[0]-diff+1):
            for tau2 in range(tau1+diff,x.shape[0]-1):
                if x.shape[0] <= diff: continue
                foo = log_lik_taus2(x,tau1,tau2,False) #we have to correct in lambda
                tmp.append(foo)
                LL.append(foo['LL'])
        max_LL = tmp[np.argmax(LL)]
        model_param = best_param2(x,max_LL['tau1'],max_LL['tau2'],max_LL['d1'],max_LL['d2'],False)
    #return linear_model, stop_model
    #return {'with_stop':{'param':stop_model['param'],'log_lik':log_lik(x,stop_model['vec'],stop_model['param'])},'linear':{'param':linear_model['param'],'log_lik':log_lik(x,linear_model['vec'],linear_model['param'])}}
    return {'param':model_param['param'],'log_lik':max_LL['LL']}
def predict(x,linear,lag_linear,linear_lag_linear,bilinear):
    if linear:
        d0=log_lik_taus2(x,0,0,True)['d0']
        linear_model = best_param2(x,0,0,d0,d0,True)            
        return linear_model['vec']['D']*d0+linear_model['param']['c0']      
    elif lag_linear:
        alg = predict_param(x,False,True,False,False,diff=None,start=1)['param']
        D= build_D_vec2(x, alg['tau1'], False, linear=False)
        return D*alg['d0']+alg['c0']
    elif bilinear:
        alg = predict_param(x,False,False,False,True,diff=0,start=1)['param']
        D1,D2 = build_D_vec2(x, alg['tau1'], alg['tau2'], linear=False)
        return D1*alg['d1']+D2*alg['d2']+alg['c0'] 
    elif linear_lag_linear:
        alg = predict_param(x,False,False,True,False,diff=0,start=1)['param']
        D1,D2 = build_D_vec2(x, alg['tau1'], alg['tau2'], linear=False)
        return D1*alg['d1']+D2*alg['d2']+alg['c0']
