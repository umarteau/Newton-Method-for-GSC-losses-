import torch
import numpy as np
import psutil
import time 
import matplotlib.pyplot as plt
import blockKernelProducts as bp
import getMemInfo as gm
import pandas as pd


########################################################################################################################
#cobj general class
########################################################################################################################

class cobj:
    def __init__(self):
        self.dt = None
        self.numNS = 0
        self.numCGS = 0
    
    def start(self):
        self.dt = time.time()
    
    def keepInfo(self,loss,X,C,y,yC,kernel,la,freeDoubles,freeGPU,cholT,KnmP_fun):
        return None
        
        
    def cbIterates(self,x,cholA = None,funval = None):
        print("number of approximate newton steps performed : {} \
        in {} seconds".format(self.numNS,time.time()-self.dt))
        self.numNS +=1
    
    def cbConj(self,x):
        self.numCGS +=1
        print("Performed {} steps of the conjugate gradient method".format(self.numCGS))

########################################################################################################################
# cobj to check optimization : sanity checks along every ANS and sanityCG checks on every conjugate gradient step
########################################################################################################################

class cobjSanity(cobj):
    def __init__(self):
        self.dt = None
        self.numNS = 0
        self.numCGS = 0
        self.lTime = []
        self.lCGS = []
        self.lRegLoss =[]
        
    def keepInfo(self,loss,X,C,y,yC,kernel,la,freeDoubles,freeGPU,cholT,KnmP_fun):
        st = time.time()
        n = y.size(0)
        dtype = X.dtype
        def aux(x,funval):
            n = y.size(0)
            if not(isinstance(funval,type(None))):
                return (torch.mean(loss.f(y,funval).view(n))\
                        + la/2  * (x*x).sum())
            else:
                return (KnmP_fun(cholT(x),[y,torch.zeros(n,1,dtype = dtype)]) + la/2  * (x*x).sum())
        self.regLoss = aux
        self.dt += time.time()-st
                
            
    def cbIterates(self,x,cholA = None,funval = None):
        st = time.time()
        print("number of approximate newton steps performed : {} \
        in {} seconds".format(self.numNS,time.time()-self.dt))
        self.lTime.append(time.time()-self.dt)
        self.lCGS.append(self.numCGS)
        self.lRegLoss.append(self.regLoss(x,funval))
        self.numNS +=1
        self.dt += time.time()-st
    
    def show(self,g = 1e-16, m = None):
        if isinstance(m,type(None)):
            m = min(self.lRegLoss)
        else:
            g=0
        plt.figure()
        plt.semilogy(self.lCGS,[ s - m + g for s in self.lRegLoss])
        plt.show()

        
class cobjSanityCG(cobjSanity):               
            
    def cbIterates(self,x,cholA = None,funval = None):
        st = time.time()
        print("number of approximate newton steps performed : {} \
        in {} seconds".format(self.numNS,time.time()-self.dt))
        self.regLossA = lambda y : self.regLoss(x-cholA(y),None) 
        self.numNS +=1
        self.dt += time.time()-st
        
    def cbConj(self,x):
        st = time.time()
        self.numCGS +=1
        print("Performed {} steps of the conjugate gradient method".format(self.numCGS))
        self.lCGS.append(self.numCGS)
        self.lTime.append(time.time()-self.dt)
        self.lRegLoss.append(self.regLossA(x))
        self.dt+= time.time()-st
        

########################################################################################################################
#cobj with a test set
########################################################################################################################


        
class cobjTest(cobj):
    
    def __init__(self,Xts,Yts,saveFolder = ""):
        self.dt = None
        self.numNS = 0
        self.numCGS = 0
        self.lTime = []
        self.lCGS = []
        self.lRegLoss =[]
        self.lTestError = []
        self.lTestLoss = []
        self.Xts = Xts
        self.Yts = Yts
        self.savePath = saveFolder
    
    def save(self):
        Dico = {}
        Dico['time'] = np.array(self.lTime)
        Dico['conjugate gradient steps'] = np.array(self.lCGS)
        Dico['optimization loss'] = np.array(self.lRegLoss)
        Dico['test error'] = np.array(self.lTestError)
        Dico['test loss'] = np.array(self.lTestLoss)
        DF = pd.DataFrame(Dico)
        DF.to_csv(self.savePath,index = False)
        
        
        
    
    def keepInfo(self,loss,X,C,y,yC,kernel,la,freeDoubles,freeGPU,cholT,KnmP_fun):
        st = time.time()
        
        n = y.size(0)
        dtype = X.dtype
        nts = self.Xts.size(0)
        m = C.size(0)
        d = C.size(1)
        
        self.savePath += "m = {}, la = {}.csv".format(m,la)
        
        
        def aux1(x,funval=None):
            n = y.size(0)
            if not(isinstance(funval,type(None))):
                res = torch.mean(loss.f(y,funval).view(n))\
                        + la/2  * (x*x).sum()
            else:
                res = KnmP_fun(cholT(x),[y,torch.zeros(n,1,dtype = dtype)]) + la/2  * (x*x).sum()
            return res
        self.regLoss = aux1
        
        
        factKnmPTs =  computeMemorySimple(kernel,d,nts,m,0.3*freeDoubles,freeGPU)
        KnmPTs = factKnmPTs(self.Xts,C)
        
        def aux2(x):
            funval = KnmPTs(cholT(x))
            testLoss = torch.mean(loss.f(self.Yts,funval).view(nts))
            testError = 100*torch.sum((self.Yts*funval <=0).view(nts)).type(torch.float64)/nts
            return testLoss,testError
        
        self.TestLossError = aux2
            
        self.dt += time.time()-st
    
    def cbIterates(self,x,cholA = None,funval = None):
        st = time.time()
        print("number of approximate newton steps performed : {} \
        in {} seconds".format(self.numNS,time.time()-self.dt))
        self.lTime.append(time.time()-self.dt)
        self.lCGS.append(self.numCGS)
        testLoss,testError = self.TestLossError(x)
        regLoss = self.regLoss(x,funval)
        self.lTestError.append(testError)
        self.lTestLoss.append(testLoss)
        self.lRegLoss.append(regLoss)
        self.numNS +=1
        self.save()
        self.dt += time.time()-st
        
    def show(self,g,m=None):
        if isinstance(m,type(None)):
            m = min(self.lRegLoss)
        else:
            g=0
        plt.figure()
        plt.semilogy(self.lCGS,[ s - m + g for s in self.lRegLoss])
        plt.title("training loss")
        plt.figure()
        plt.plot(self.lCGS, self.lTestLoss)
        plt.title("Test loss")
        plt.figure()
        plt.plot(self.lCGS,self.lTestError)
        plt.title("Test error")
        plt.show()
        
        
    
    
    
    
#############################################################################################
#Keeping track of the optimization in lambda 
#############################################################################################


class cobjTestBestLambda(cobj):
    
    def __init__(self,Xts,Yts,lLambda,saveFolder = ""):
        self.dt = None
        self.numNS = 0
        self.numCGS = 0
        self.lTime = []
        self.lCGS = []
        self.lRegLoss =[]
        self.lTestError = []
        self.lTestLoss = []
        self.Xts = Xts
        self.Yts = Yts
        self.lLambda = [-1] + lLambda
        self.lLambdaIndex = 0
        self.savePath = saveFolder
    
    def save(self):
        Dico = {}
        Dico['lambda'] = np.array(self.lLambda[:self.lLambdaIndex])
        Dico['time'] = np.array(self.lTime)
        Dico['conjugate gradient steps'] = np.array(self.lCGS)
        Dico['optimization loss'] = np.array(self.lRegLoss)
        Dico['test error'] = np.array(self.lTestError)
        Dico['test loss'] = np.array(self.lTestLoss)
        DF = pd.DataFrame(Dico)
        DF.to_csv(self.savePath,index = False)
        
        
        
    
    def keepInfo(self,loss,X,C,y,yC,kernel,la,freeDoubles,freeGPU,cholT,KnmP_fun):
        st = time.time()
        
        n = y.size(0)
        dtype = X.dtype
        nts = self.Xts.size(0)
        m = C.size(0)
        d = C.size(1)
        
        self.savePath += "following_lambda: m = {}, la = {}.csv".format(m,la)
        
        
        def aux1(x,t,funval=None):
            n = y.size(0)
            if not(isinstance(funval,type(None))):
                res = torch.mean(loss.f(y,funval).view(n))\
                
            else:
                res = KnmP_fun(cholT(x),[y,torch.zeros(n,1,dtype = dtype)])
                    
            if t < 0:
                return res
            else:
                return (res + t/2  * (x*x).sum())
                
        self.regLoss = aux1
        
        
        factKnmPTs =  computeMemorySimple(kernel,d,nts,m,0.3*freeDoubles,freeGPU)
        KnmPTs = factKnmPTs(self.Xts,C)
        
        def aux2(x):
            funval = KnmPTs(cholT(x))
            testLoss = torch.mean(loss.f(self.Yts,funval).view(nts))
            testError = 100*torch.sum((self.Yts*funval <=0).view(nts)).type(torch.float64)/nts
            return testLoss,testError
        
        self.TestLossError = aux2
            
        self.dt += time.time()-st
    
    def cbIterates(self,x,cholA = None,funval = None):
        st = time.time()
        print("number of approximate newton steps performed : {} \
        in {} seconds".format(self.numNS,time.time()-self.dt))
        self.lTime.append(time.time()-self.dt)
        self.lCGS.append(self.numCGS)
        testLoss,testError = self.TestLossError(x)
        regLoss = self.regLoss(x,self.lLambda[self.lLambdaIndex],funval)
        self.lLambdaIndex+=1
        self.lTestError.append(testError)
        self.lTestLoss.append(testLoss)
        self.lRegLoss.append(regLoss)
        self.numNS +=1
        self.save()
        self.dt += time.time()-st
        
    def show(self,g,m=None):
        if isinstance(m,type(None)):
            m = min(self.lRegLoss)
        else:
            g=0
        plt.figure()
        plt.semilogy(self.lCGS,[ s - m + g for s in self.lRegLoss])
        plt.title("training loss")
        plt.figure()
        plt.plot(self.lCGS, self.lTestLoss)
        plt.title("Test loss")
        plt.figure()
        plt.plot(self.lCGS,self.lTestError)
        plt.title("Test error")
        plt.show()
        
        
############################################################################################
# Finding good lambda 
############################################################################################

class cobjTestGrid(cobj):
    
    
    def __init__(self,Xts,Yts,lLambda,lSigma,m,saveFolder = ""):
        self.Xts = Xts
        self.Yts = Yts
        self.lLambda = [-1] + lLambda
        self.savePath = saveFolder+"grid search : m = {} .csv".format(m)
        self.lSigma = lSigma
        self.Dico={}
        self.Dico['lambda'] = np.array(self.lLambda)
        self.current_sigma_index = -1
        
        
    def start(self):
        self.dt = time.time()
        self.numNS = 0
        self.numCGS = 0
        self.lTime = []
        self.lCGS = []
        self.lRegLoss =[]
        self.lTestError = []
        self.lTestLoss = []
        self.lLambdaIndex = 0
        self.current_sigma_index +=1
        self.sigma_str = "sigma = {}, ".format(self.lSigma[self.current_sigma_index])
        self.Dico[self.sigma_str + 'optimization loss'] = np.zeros(len(self.lLambda))
        self.Dico[self.sigma_str + 'test error'] =  np.zeros(len(self.lLambda))
        self.Dico[self.sigma_str + 'test loss'] =  np.zeros(len(self.lLambda))
        
        
    
    def save(self):
        self.Dico[self.sigma_str + 'optimization loss'][:self.lLambdaIndex] = np.array(self.lRegLoss)
        self.Dico[self.sigma_str + 'test error'][:self.lLambdaIndex] = np.array(self.lTestError)
        self.Dico[self.sigma_str + 'test loss'][:self.lLambdaIndex] = np.array(self.lTestLoss)
        DF = pd.DataFrame(self.Dico)
        DF.to_csv(self.savePath,index = False)
        
        
        
    
    def keepInfo(self,loss,X,C,y,yC,kernel,la,freeDoubles,freeGPU,cholT,KnmP_fun):
        st = time.time()
        
        n = y.size(0)
        dtype = X.dtype
        nts = self.Xts.size(0)
        m = C.size(0)
        d = C.size(1)
        
        
        
        def aux1(x,t,funval=None):
            n = y.size(0)
            if not(isinstance(funval,type(None))):
                res = torch.mean(loss.f(y,funval).view(n))\
                
            else:
                res = KnmP_fun(cholT(x),[y,torch.zeros(n,1,dtype = dtype)])
                    
            if t < 0:
                return res
            else:
                return (res + t/2  * (x*x).sum())
                
        self.regLoss = aux1
        
        
        factKnmPTs =  computeMemorySimple(kernel,d,nts,m,0.3*freeDoubles,freeGPU)
        KnmPTs = factKnmPTs(self.Xts,C)
        
        def aux2(x):
            funval = KnmPTs(cholT(x))
            testLoss = torch.mean(loss.f(self.Yts,funval).view(nts))
            testError = 100*torch.sum((self.Yts*funval <=0).view(nts)).type(torch.float64)/nts
            return testLoss,testError
        
        self.TestLossError = aux2
            
        self.dt += time.time()-st
    
    def cbIterates(self,x,cholA = None,funval = None):
        st = time.time()
        print("number of approximate newton steps performed : {} \
        in {} seconds".format(self.numNS,time.time()-self.dt))
        self.lTime.append(time.time()-self.dt)
        self.lCGS.append(self.numCGS)
        testLoss,testError = self.TestLossError(x)
        regLoss = self.regLoss(x,self.lLambda[self.lLambdaIndex],funval)
        self.lLambdaIndex+=1
        self.lTestError.append(testError)
        self.lTestLoss.append(testLoss)
        self.lRegLoss.append(regLoss)
        self.numNS +=1
        self.save()
        self.dt += time.time()-st
        
    def makeTables(self):
        #Finding indices to select the lambdas 
        lLambda = self.lLambda
        l_index = []
        for i in range(len(lLambda)-1):
            if lLambda[i+1] != lLambda[i]:
                l_index.append(i)
        l_index.append(len(lLambda)-1)
        
        #indices to see test loss
        l_test_loss = ['lambda']+["sigma = {}, ".format(sigma) + 'test loss' for sigma in self.lSigma]
        l_test_error = ['lambda']+["sigma = {}, ".format(sigma) + 'test error' for sigma in self.lSigma]
        columns = ['lambda'] + ["sigma = {}".format(sigma) for sigma in self.lSigma]
        
        df = pd.read_csv(self.savePath)
        
        df1 = df.loc[l_index,l_test_loss]
        df1.columns = columns
        df1 = df1.set_index('lambda')
        df1.to_csv(self.savePath[:-4]+"test loss table.csv")
        
        df2 = df.loc[l_index,l_test_error]
        df2.columns = columns
        df2 = df2.set_index('lambda')
        df2.to_csv(self.savePath[:-4]+"test error table.csv")
        
        
        
        


            
            




#################################################################################
# Simple matrix vector product (no function)
#################################################################################    


def KnmProdSimple(X,C,u,kern,freeDoubles,freeGPU):
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
    dev = Cg.device
    
    p = torch.zeros(n,1,dtype = torch.float64)
    Xs  = np.ceil(np.linspace(0, n, blk + 1)).astype(int)
    for i in range(blk):
        try:
            del Kr
        except:
            None
        X1 = X[Xs[i]:Xs[i+1],:].to(dev)
        Kr = kern(X1,Cg)
        p[Xs[i]:Xs[i+1],:] = download(Kr@ug)

    del Kr
    del X1
    del Cg
    del ug
    del X
    return p


        
def computeMemorySimple(kernel,d,n,m,freeGPU,freeDoubles):
    
    
        
    def factorySimpleKnmP(X, C, kern, ff, fgpu):
        Knm = bp.blockKernComp(X, C, kern, ff, fgpu)
        KnmP = lambda u : Knm@u
        return KnmP

    def factoryKnmP(X, C,kern,ff,fgpu):
        n = X.size(0)
        KnmP = lambda u:  KnmProdSimple(X,C,u,kern,ff,fgpu)
        return KnmP

    if 0.985*freeDoubles >= n*m:
        fKnmP = lambda X, C :\
        factorySimpleKnmP(X, C, kernel, freeDoubles - n*m, freeGPU)
    else:
        fKnmP = lambda X, C :\
        factoryKnmP(X, C, kernel, freeDoubles, freeGPU)
    
    return fKnmP













##########################
########################################################### 
#Different callback objects for Katysha
############################################################

#1 : empty callbak boject


class cobjK:
    def __init__(self):
        self.dt = None
        self.numEpochs = 0

    
    def start(self):
        self.dt = time.time()
    
    def keepInfo(loss,X,C,y,yC,kernel,la,freeDoubles,freeGPU,cholT,KnmP_fun,mratio,om1,om2,niterBatch):
        self.mratio = mratio
        return None
        
        
    def cbIterates(self,x,cholA = None,funval = None):
        print("number of epochs performed : {} -- number of passes over the data performed : {} \
        in {} seconds".format(self.numEpochs,(1+self.mration)*self.numEpochs,time.time()-self.dt))
        self.numEpochs +=1
        

        
        
class cobjSanityK(cobjK):
    def __init__(self):
        self.dt = None
        self.numEpochs = 0
        self.numPasses = 0
        self.lTime = []
        self.lEpochs = []
        self.lPasses = []
        self.lRegLoss =[]
        
    def keepInfo(self,loss,X,C,y,yC,kernel,la,freeDoubles,freeGPU,cholT,KnmP_fun,mratio,om1,om2,niterBatch):
        st = time.time()
        
        self.mratio = mratio
        n = y.size(0)
        dtype = X.dtype
        t1 = (om2*niterBatch)/(om2*niterBatch +(1-om1-om2))
        def aux(beta_prev,yy):
            alpha = t1*beta_prev.clone() + (1-t1)*yy.clone()
            return (KnmP_fun(cholT(alpha),[y,torch.zeros(n,1,dtype = dtype)]) + la/2  * (alpha*alpha).sum())
        self.regLoss = aux
        self.dt += time.time()-st
                
            
    def cbIterates(self,beta_prev,yy):
        st = time.time()
        print("number of epochs performed : {} -- number of passes over the data performed : {} \
        in {} seconds".format(self.numEpochs,self.numPasses,time.time()-self.dt))
        self.lTime.append(time.time()-self.dt)
        self.lEpochs.append(self.numEpochs)
        self.lPasses.append(self.numPasses)
        self.lRegLoss.append(self.regLoss(beta_prev,yy))
        
        self.numEpochs +=1
        self.numPasses += 1+self.mratio
        self.dt += time.time()-st
    
    def show(self,g = 1e-4, m = None):
        if isinstance(m,type(None)):
            m = min(self.lRegLoss)
        else:
            g=0
        plt.figure()
        plt.semilogy(self.lEpochs,[ s - m + g for s in self.lRegLoss])
        plt.show()
    


#########################

class cobjTestK(cobjK):
    
    def __init__(self,Xts,Yts,saveFolder = ""):
        self.dt = None
        self.numEpochs = 0
        self.numPasses = 0
        self.lTime = []
        self.lEpochs = []
        self.lPasses = []
        self.lRegLoss =[]
        self.lTestError = []
        self.lTestLoss = []
        self.Xts = Xts
        self.Yts = Yts
        self.savePath = saveFolder
    
    def save(self):
        Dico = {}
        Dico['time'] = np.array(self.lTime)
        Dico['epochs'] = np.array(self.lEpochs)
        Dico['number of passes'] = np.array(self.lPasses)
        Dico['optimization loss'] = np.array(self.lRegLoss)
        Dico['test error'] = np.array(self.lTestError)
        Dico['test loss'] = np.array(self.lTestLoss)
        DF = pd.DataFrame(Dico)
        DF.to_csv(self.savePath,index = False)
        
        
        
    
    def keepInfo(self,loss,X,C,y,yC,kernel,la,freeDoubles,freeGPU,cholT,KnmP_fun,mratio,om1,om2,niterBatch):
        st = time.time()
        
        n = y.size(0)
        dtype = X.dtype
        nts = self.Xts.size(0)
        m = C.size(0)
        d  =C.size(1)
        
        self.savePath += "Katyusha m = {}, la = {}.csv".format(m,la)
        self.mratio = mratio

        t1 = (om2*niterBatch)/(om2*niterBatch +(1-om1-om2))
        def aux1(beta_prev,yy):
            alpha = t1*beta_prev.clone() + (1-t1)*yy.clone()
            return (KnmP_fun(cholT(alpha),[y,torch.zeros(n,1,dtype = dtype)]) + la/2  * (alpha*alpha).sum())
        self.regLoss = aux1
        

        
        
        factKnmPTs =  computeMemorySimple(kernel,d,nts,m,0.3*freeDoubles,freeGPU)
        KnmPTs = factKnmPTs(self.Xts,C)
        
        def aux2(beta_prev,yy):
            x  = t1*beta_prev.clone() + (1-t1)*yy.clone()
            funval = KnmPTs(cholT(x))
            testLoss = torch.mean(loss.f(self.Yts,funval).view(nts))
            testError = 100*torch.sum((self.Yts*funval <=0).view(nts)).type(torch.float64)/nts
            return testLoss,testError
        
        self.TestLossError = aux2
            
        self.dt += time.time()-st
    
    def cbIterates(self,beta_prev,yy):
        st = time.time()
        print("number of epochs performed : {} -- number of passes over the data performed : {} \
        in {} seconds".format(self.numEpochs,self.numPasses,time.time()-self.dt))
        self.lTime.append(time.time()-self.dt)
        self.lEpochs.append(self.numEpochs)
        self.lPasses.append(self.numPasses)
        testLoss,testError = self.TestLossError(beta_prev,yy)
        regLoss = self.regLoss(beta_prev,yy)
        self.lTestError.append(testError)
        self.lTestLoss.append(testLoss)
        self.lRegLoss.append(regLoss)
        self.numEpochs +=1
        self.numPasses += 1+self.mratio
        self.save()
        self.dt += time.time()-st
        
    def show(self,g,m=None):
        if isinstance(m,type(None)):
            m = min(self.lRegLoss)
        else:
            g=0
        plt.figure()
        plt.semilogy(self.lEpochs,[ s - m + g for s in self.lRegLoss])
        plt.title("training loss")
        plt.figure()
        plt.plot(self.lEpochs, self.lTestLoss)
        plt.title("Test loss")
        plt.figure()
        plt.plot(self.lEpochs,self.lTestError)
        plt.title("Test error")
        plt.show()
        
        