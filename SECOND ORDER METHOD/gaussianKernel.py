import torch

def gaussianKernel(sigma):
    return (lambda  x1,x2 : aux(sigma,x1,x2))


def aux(sigma,x1,x2):
    x1_norm =  (x1*x1).sum(1)
    x2_norm = (x2*x2).sum(1)
    try:
        dist = x1 @ x2.t()
    except RuntimeError:
        torch.cuda.empty_cache()
        dist = x1 @ x2.t()
        
    del x2
    del x1
    dist *= -2
    dist += x1_norm.unsqueeze_(1).expand_as(dist)
    del x1_norm
    dist += x2_norm.unsqueeze_(0).expand_as(dist)
    del x2_norm
    dist *= -1/(2*sigma**2)
    dist.clamp_(min = -30,max = 0)
    dist.exp_()
    return dist

