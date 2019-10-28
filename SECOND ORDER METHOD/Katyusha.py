import torch
import numpy as np
import NewtonMethod as nm
import display as dp

dtype = torch.float64



def miniBatchKSVRG(loss,X,C,y,yC,kernel,la,Nepochs,mratio=2,tau = None,Kmax = 1,option = 1,om1 = None,memToUse = None,useGPU = None,cobj = dp.cobjK()):
    """ 
    IMPLEMENTATION of Katyusha accelerated SVRG as defined in the paper by Allen Zhu
    lambd : the regularizer
    Nepochs :the number of passes in the external loop
    m : number of passes in the internal loop
    tau : batch size. automatically set to M if nothing specified
    m_ratio : we automatically set m = m_ratio * n // tau
    eta : step size 
    omega : momentum parameter. Automatically set to 0.9 if nothing specified
    eta_ratio : we automatically set eta = eta_ratio * (1/Lmax)
    kmax : largest value of the kernel (gaussian : kmax = 1); could be computed
    ratio: we want to memorize the alpha every ratio*epoch
    option : as in the article 
    """
    cobj.start()
    
    ################################################################################################
    #Creating Kernel matrices and functions
    ################################################################################################
        
    n = X.size(0)
    m = C.size(0)
    d = X.size(1)
    if isinstance(useGPU,type(None)):
        useGPU = torch.cuda.is_available()
        if useGPU :
            torch.cuda.empty_cache()
    if isinstance(memToUse,type(None)):
        memToUse = 0.9*psutil.virtual_memory().available
        print("no memory limit specified. At most {} GB of \
                RAM will be used".format(memToUse/10**9))
        
    factKnmP, kern, freeDoubles, freeGPU = nm.computeMemory(memToUse, kernel, d, n, m, useGPU)
    
    print("there is {} GiB free on the GPU ".format(freeGPU*8/1024**3))
    
    T = nm.createT(kern, C, freeGPU)
    cholT,cholTt = lambda x : nm.tr_solve(x,T,freeGPU),\
                   lambda x: nm.tr_solve(x,T,freeGPU,transpose = True)
    
    KnmP = factKnmP(X,C)
    
    l_fun,l_grad = l_fg(loss,n)
    KnmP_fun,KnmP_grad =  lambda u,lobj : KnmP(u,l_fun,lobj), \
                            lambda u,lobj : KnmP(u,l_grad,lobj), \
                                    
    
    
    ################################################################################################
    #Setting parameters of the method 
    ################################################################################################
    
    #batch size
    if isinstance(tau,type(None)):
        tau = m
    
    #number of iterations (divide by batch size)
    niterBatch = (mratio*n)//tau + 1
    print("--- m = {}, tau = {} ---".format(m,tau))
    
    #Smoothness constant
    if isinstance(loss.Lmax,type(None)):
        Lmax = Kmax
    else:
        Lmax = loss.Lmax*Kmax
           
    #om1 and om2, parameters of Katyusha acceleration
    om2 = 1/(2*tau)
    if isinstance(om1,type(None)):
        if m >= tau:
            om1 = float(min(np.sqrt((8*la*m*tau)/(3*Lmax)),1)*om2)
        else:
            om1 = float(min(np.sqrt((2*la)/(3*Lmax)),1/(2*m)))
            
    #Stepsize 
    eta = 1/(3*om1*Lmax)
    
    #Theta
    theta = 1 + min(eta*la,1/(4*m))
    
    cobj.keepInfo(loss,X,C,y,yC,kernel,la,freeDoubles,freeGPU,cholT,KnmP_fun,mratio,om1,om2,niterBatch)
    
    beta_prev = torch.zeros(m,1, dtype = dtype)
    x = torch.zeros(m,1,dtype = dtype)
    z = torch.zeros(m,1, dtype = dtype)
    yy = torch.zeros(m,1, dtype = dtype)
    
    for epoch in range(Nepochs):
        
        cobj.cbIterates(beta_prev,yy)
        
        #Computing big gradient 
        lobj = [y,torch.zeros(n,1,dtype = dtype)]
        grad = cholTt(KnmP_grad(cholT(beta_prev),lobj))
        d_stock = lobj[1]
        
        beta = torch.zeros(m,1,dtype = dtype)
        for t in range(niterBatch):
            S = np.random.choice(n,tau,replace = True)
            x = om1*z + om2*beta_prev + (1-om1-om2)*yy
            
            KtaumP = factKnmP(X[S,:],C)
            l_grad_tau = lgtau(loss,tau)
            KtaumP_grad =  lambda u,lobj : KtaumP(u,l_grad_tau,lobj) 
            
            
            lobjS = [y[S,:],d_stock[S,:]]
            grad_proxy = cholTt(KtaumP_grad(cholT(x),lobjS)) + grad

            dz = (1/(1 + la*eta))*(z - eta*grad_proxy) - z
            if option == 1:
                yy = (1/(1+la/(3*Lmax)))*(x - (1/(3*Lmax))*grad_proxy)
            if option == 2:
                yy = x + om1*dz
            z = z+dz
            
            beta = (theta - 1)*((theta**t)/(theta**(t+1) - 1))*yy + (theta**t - 1)/(theta**(t+1) - 1) * beta
            
        beta_prev = beta
    
    cobj.cbIterates(beta_prev,yy)
    
    alpha = makeFinal(om1,om2,niterBatch,beta_prev,yy)
    return cholT(alpha)



def l_fg(loss,n):
    #in this case, lobj = [y,funval]
    def aux_fun(lobj,a,b,u,Kr):
        p = Kr@u
        lobj[1][a:b,:] = p
        return (((loss.f(lobj[0][a:b,:],p)).view(b-a)).sum()/n)
    def aux_grad(lobj,a,b,u,Kr):
        # here, lobj = [y,derivative]
        p = Kr@u
        d = loss.df(lobj[0][a:b,:],p)
        lobj[1][a:b,:] = d
        return ((d.t()@Kr).t()/n)
    return aux_fun,aux_grad

def lgtau(loss,tau):
    def aux_grad(lobj,a,b,u,Kr):
        #here, lobj = [y[S],d[S]]
        p = Kr@u
        d = loss.df(lobj[0][a:b,:],p)
        d -= lobj[1][a:b,:] 
        return ((d.t()@Kr).t()/tau)
    return aux_grad

def makeFinal(om1,om2,niterBatch,beta_prev,yy):
    t1 = (om2*niterBatch)/(om2*niterBatch +(1-om1-om2))
    alpha = t1*beta_prev.clone() + (1-t1)*yy.clone()
    return alpha
    
    
