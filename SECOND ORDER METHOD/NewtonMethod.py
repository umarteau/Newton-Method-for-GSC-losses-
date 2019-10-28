import torch
import numpy as np
import psutil
import time 
import display as dp
import blockKernelProducts as bp
import getMemInfo as gm







dtype = torch.float64

def NewtonMethod(loss,X,C,y,yC,kernel,la_list,t_list,memToUse = None,useGPU = None,cobj = dp.cobj()):
    cobj.start()
    n = X.size(0)
    m = C.size(0)
    d = X.size(1)
    if isinstance(useGPU,type(None)):
        useGPU = torch.cuda.is_available()
    if isinstance(memToUse,type(None)):
        memToUse = 0.9*psutil.virtual_memory().available
        print("no memory limit specified. At most {} GB of \
                RAM will be used".format(memToUse/10**9))
    
    
    factKnmP, kern, freeDoubles, freeGPU = computeMemory(memToUse, kernel, d, n, m, useGPU)
    
    print("there is {} GiB free on the GPU ".format(freeGPU*8/1024**3))
    
    T = createT(kern, C, freeGPU)
    cholT,cholTt = lambda x : tr_solve(x,T,freeGPU),\
                   lambda x: tr_solve(x,T,freeGPU,transpose = True)
    
    KnmP = factKnmP(X,C)
    l_fun,l_grad,l_hess = l_fgh(loss,n)
    KnmP_fun,KnmP_grad,KnmP_hess =  lambda u,lobj : KnmP(u,l_fun,lobj), \
                                    lambda u,lobj : KnmP(u,l_grad,lobj), \
                                    lambda u,lobj : KnmP(u,l_hess,lobj)
    
    
    alpha = torch.zeros(m,1,dtype = dtype)
    
    
    cobj.keepInfo(loss,X,C,y,yC,kernel,la_list[-1],freeDoubles,freeGPU,cholT,KnmP_fun)
    
    for i in range(len(la_list)):
        t = t_list[i]
        la = la_list[i]
        
        #Compute preconditioner
        A = createA(loss,alpha,yC,T,la,n,freeGPU)
        cholA,cholAt = lambda x : tr_solve(x,A,freeGPU),\
                    lambda x: tr_solve(x,A,freeGPU,transpose = True)
        #Compute gradient

        lobj = [y,torch.zeros(n,1,dtype = dtype)]
        grad_p = cholAt(cholTt(KnmP_grad(cholT(alpha),lobj)) + la*alpha)
        
        #Compute Hessian
            
        hess = lambda x : cholTt(KnmP_hess(cholT(x),lobj)) + la*x
        hess_p = lambda x : cholAt(hess(cholA(x)))
        
        #use the callback for iterations
        cobj.cbIterates(alpha,cholA = cholA,funval =lobj[1])
        
        #Perform conjugate gradient with t iterations to obtain approximate newton step
        alpha -= cholA(conjgrad(hess_p,grad_p,t,cobj))
    
    cobj.cbIterates(alpha)
    
    return cholT(alpha)
        
        
def computeMemory(memToUse,kernel,d,n,m,useGPU):
    # check this 
    freeDoubles = memToUse/8. - 3*m*m - 4*n - 4*m
    if useGPU:
        y = torch.zeros(m)
        y = y.to('cuda')
        del y
        freeGPU = 0.95*gm.freeGPU()/8
    else:
        freeGPU = None
    
    kern =  lambda X1,X2 : bp.blockKernComp(X1, X2, kernel, freeDoubles, freeGPU)
    
        
    def factorySimpleKnmP(X, C, kern, ff, fgpu):
        Knm = bp.blockKernComp(X, C, kern, ff, fgpu)
        KnmP = lambda u,l,lobj : KtKprod(Knm, u, l,lobj,0,n)
        return KnmP

    def factoryKnmP(X, C,kern,ff,fgpu):
        n = X.size(0)
        KnmP = lambda u,l,lobj :  KnmProd(X,C,u,kern,l,lobj,0,n,ff,fgpu) 
        return KnmP

    if 0.985*freeDoubles >= n*m:
        fKnmP = lambda X, C :\
        factorySimpleKnmP(X, C, kernel, freeDoubles - n*m, freeGPU)
    else:
        fKnmP = lambda X, C :\
        factoryKnmP(X, C, kernel, freeDoubles, freeGPU)
    
    return fKnmP,kern,freeDoubles,freeGPU    

        


#################################################################################
# Matrix vector products
#################################################################################


def KtKprod(Kr,u,l,lobj,a,b):
    return l(lobj,a,b,u,Kr)
    
    

def l_fgh(loss,n):
    #in this case, lobj = [y,funval]
    def aux_fun(lobj,a,b,u,Kr):
        p = Kr@u
        lobj[1][a:b,:] = p
        return (((loss.f(lobj[0][a:b,:],p)).view(b-a)).sum()/n)
    def aux_grad(lobj,a,b,u,Kr):
        p = Kr@u
        lobj[1][a:b,:] = p
        return ((loss.df(lobj[0][a:b,:],p).t()@Kr).t()/n)
    def aux_hess(lobj,a,b,u,Kr):
        p = Kr@u
        #in this case, lobj is the pair [y,funval]
        return (((loss.ddf(lobj[0][a:b,:],lobj[1][a:b,:])*p).t()@Kr).t()/n)
    return aux_fun,aux_grad,aux_hess
        
        
    
    
def KnmProd(X,C,u,kern,l,lobj,a,b,freeDoubles,freeGPU):
    n = X.size(0)
    m = C.size(0)
    d = C.size(1)
    DMIN = 64
    useGPU = not(isinstance(freeGPU,type(None)))
    if useGPU:

        nmin_gpu = int(np.floor((freeGPU - 2*m*d - m)/(m + 2*d + 2)))
        nmin_cpu = freeDoubles/m
        
        if nmin_gpu > DMIN:
            blk = int(np.ceil(n/min(nmin_gpu,nmin_cpu)))
            useGPUhere = True
        else:
            blk = int(np.ceil(n/nmin_cpu))
            useGPUhere = False
        
    else:
        blk = int(np.ceil(n*m/freeDoubles))
        useGPUhere = False
    
    download,upload = bp.produceDU(useGPUhere)
    Cg = upload(C)
    ug = upload(u)
    lobjg = [upload(v) for v in lobj]
    Xs  = np.ceil(np.linspace(0, n, blk + 1)).astype(int)
    
    
    
    for i in range(blk):
        try:
            del Kr
        except:
            None
        X1 = upload(X[Xs[i]:Xs[i+1],:])
        Kr = kern(X1,Cg)
        pp = KtKprod(Kr,ug,l,lobjg,Xs[i],Xs[i+1])
        try:
            p += pp
        except:
            p = pp
        del Kr
        del X1
        del pp
    
    for i in range(len(lobj)):
        lobj[i] = download(lobjg[i])
    del Cg
    del ug
    del lobjg
    del X
    pfin = download(p)
    del p
    return pfin


 
        
        

############################################################################################
#Cholesky decompositions
############################################################################################


def chol(M,freeGPU):
    m = M.size(0)
    useGPU = not(isinstance(freeGPU,type(None)))
    if useGPU:
        if 2*m**2 <= 0.98*freeGPU:
            Mg = M.to('cuda')
            Tg = Mg.cholesky(upper = True)
            T = Tg.to('cpu')
            del Mg
            del Tg
            return T
        else:
            T = M.cholesky(upper = True)
            return T
    else:
        T = M.cholesky(upper = True)
        return T    
    
    
    
def createT(kern,C,freeGPU):
    K = kern(C,None)
    m = K.size(0)
    eps = 1e-15*m
    K[range(m),range(m)]+= eps
    T = chol(K,freeGPU)
    del K
    return T

def createA(loss,alp,Yc,T,la,n,freeGPU):
    m = T.size(0)
    W = loss.ddf(Yc,(alp.t()@T).t())
    useGPU = not(isinstance(freeGPU,type(None)))
    if useGPU:
        A = torch.zeros(m,m,dtype = dtype,device = 'cpu')
        download,upload = bp.produceDU(useGPU)
        nmax = int((np.sqrt(4*(freeGPU - m) + 4*m**2)-2*m)/2)
        blk = int(np.ceil(m/nmax))
        bA = np.linspace(0,m,blk+1).astype(int)
        Wg = upload(W)
        for i in range(blk):
            for j in range(i,blk):
                C1 = upload(T[bA[i]:bA[i+1],:])
                C2 = upload(T[bA[j]:bA[j+1],:].t())
                C2 *= Wg
                A[bA[i]:bA[i+1],bA[j]:bA[j+1]]= download(C1@C2)
                del C1
                del C2
        del Wg
        for i in range(blk):
            for j in range(i):
                A[bA[i]:bA[i+1],bA[j]:bA[j+1]]= A[bA[j]:bA[j+1],bA[i]:bA[i+1]].t()
    else:
        A = T.t()
        A *= W
        A = T@A
    A /= n
    A[range(m),range(m)] += la
    res = chol(A,freeGPU)
    del A
    del W
    return res
    
############################################################################################
#Solving triangular systems
############################################################################################

def tr_solve(x,T,freeGPU,transpose = False):
    m = T.size(0)
    useGPU = not(isinstance(freeGPU,type(None)))
    if useGPU:
        if m*(2*m+2) < 0.98*freeGPU:
            download,upload = bp.produceDU(useGPU)
            xg = upload(x)
            Tg = upload(T)
            resg = torch.triangular_solve(xg,Tg,upper \
                                                   = True,transpose = transpose)
            res = download(resg[0])
            del xg
            del Tg
            del resg
            torch.cuda.empty_cache()
            return res
        else:
            return torch.triangular_solve(x,T,upper \
                                                   = True,transpose = transpose)[0]
    else:
        return torch.triangular_solve(x,T,upper \
                                                   = True,transpose = transpose)[0]
        

############################################################################################
#Conjugate gradient
############################################################################################
                
                

    

def conjgrad(funA,r,t,cobj):
    # initialize parameter
    r0 = r
    p = r
    rsold = torch.sum(r.pow(2))
    n = r.size(0)
    beta = torch.zeros(n,1,dtype = dtype)
    for i in range(t):
        Ap = funA(p)
        a = rsold/( torch.sum(p*Ap))
        aux = a*p
        beta = torch.add(beta,aux)
        aux = - a*Ap
        r = torch.add(r,aux)
        rsnew = torch.sum(r.pow(2))
        p = r + (rsnew/rsold)*p
        rsold = rsnew
        cobj.cbConj(beta)
    return beta




        
        