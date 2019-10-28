import torch
import numpy as np

class dloss(object):
    """ two times differentiable function f(y,t). 
    self.f = f,  self.df = df/dt, self.ddf = d^2f/dt^2 """
    def __init__(self,f,df,ddf,Lmax):
        self.f = f
        self.df= df
        self.ddf = ddf
        self.Lmax = Lmax
        
# square loss

f = lambda y,t: 0.5*(t-y)*(t-y)
df = lambda y,t: t-y
ddf = lambda y,t: 1 + 0*t


squareloss = dloss(f,df,ddf,1)

#logistic loss
f = lambda y,x: torch.log(1 + torch.exp(-y *x))
sigma = lambda y,x: 1/(1 + torch.exp(-y*x))
df = lambda y,x: -y*sigma(-y,x)
ddf = lambda y,x: y**2 *sigma(-y,x)*sigma(y,x)


logloss = dloss(f,df,ddf,0.25)
        
