import torch
import numpy as np
import psutil
import getMemInfo as gm

    



############################################################################################
#Computing Kernel Matrices
############################################################################################

        
def produceDU(useGPU):
    if useGPU:
        return (lambda x : x.to('cpu'),lambda x : x.to('cuda')) 
    else:
        return (lambda x : x,lambda x : x)


    
        
def blockKernComp(A, B, kern, freeDoubles, freeGPU):
    
    useGPU = not(isinstance(freeGPU,type(None)))
    
    if isinstance(B,type(None)):
        return(blockKernCompGPUSymmetric(A, kern, freeDoubles, freeGPU))

    elif useGPU:
        d = A.size(1)
        nmax_gpu = int(np.floor((np.sqrt((1.5*d+0.5)**2 + freeGPU)-(1.5*d+0.5))))
        print(freeDoubles)
        nmax_ram = int(np.sqrt(freeDoubles))
        nmax = min(nmax_gpu, nmax_ram)
    
        if nmax > A.size(0) and A.size(0) <= B.size(0):
            nmaxA = A.size(0);
            nmaxB = int(min(np.floor((freeGPU - 2*nmaxA* d)/(nmaxA + 2*d)), np.floor(freeDoubles/nmaxA)))
        elif nmax > B.size(0):
            nmaxB = B.size(0)
            nmaxA = int(min(np.floor((freeGPU - 2*nmaxB* d)/(nmaxB + 2*d)), np.floor(freeDoubles/nmaxB)))
        else:
            nmaxA = nmax
            nmaxB = nmax

    else:
        nmax = int(np.floor(np.sqrt(freeDoubles)))

        if nmax > A.size(0) and A.size(0) <= B.size(0):
            nmaxA = A.size(0)
            nmaxB = int(freeDoubles/nmaxA)
        elif nmax > B.size(0):
            nmaxB = B.size(0)
            nmaxA = int(freeDoubles/nmaxB)
        else:
            nmaxA = nmax
            nmaxB = nmax


    download, upload = produceDU(useGPU)

    blkA = int(np.ceil(A.size(0)/nmaxA))
    As = np.ceil(np.linspace(0, A.size(0), blkA + 1)).astype(int)
    
    blkB = int(np.ceil(B.size(0)/nmaxB))
    Bs = np.ceil(np.linspace(0, B.size(0), blkB + 1)).astype(int)

    if blkA == 1 and blkB == 1:
        M = download(kern(upload(A), upload(B)))
    else:

        M = torch.zeros(A.size(0), B.size(0),dtype = torch.float64)

        for i in range(blkA):
            C1 = upload(A[As[i]:As[i+1],:])
            for j in range(blkB):
                C2 = upload(B[Bs[j]:Bs[j+1], :])
                M[As[i]:As[i+1], Bs[j]:Bs[j+1]] = download(kern(C1,C2))
                del C2
            del C1
    return M


def blockKernCompGPUSymmetric(A, kern, freeDoubles, freeGPU):
    
    useGPU = not(isinstance(freeGPU,type(None)))
    
    if useGPU:
        d = A.size(1)
        nmax_gpu = int(np.floor((np.sqrt((1.5*d+0.5)**2 + freeGPU)-(1.5*d+0.5))))
        nmax_ram = int(np.sqrt(freeDoubles))
        nmax = min(nmax_gpu, nmax_ram)
    else:
        nmax = int(np.floor(np.sqrt(freeDoubles)))
    
    download, upload = produceDU(useGPU)
    
    blkA = int(np.ceil(A.size(0)/nmax))
    As = np.ceil(np.linspace(0, A.size(0), blkA + 1)).astype(int)
    
    if blkA == 1:
        uA = upload(A)
        Mg = kern(uA, uA)
        M = download(Mg)
        del Mg
    else:

        M = torch.zeros(A.size(0), A.size(0),dtype = torch.float64);
        for i in range(blkA):
            C1 = upload(A[As[i]:As[i+1],:])
            M[As[i]:As[i+1], As[i]:As[i+1]] = download(kern(C1,C1))
            for j in range(i+1,blkA):
                C2 = upload(A[As[j]:As[j+1], :])
                Kr = kern(C1,C2)
                M[As[i]:As[i+1], As[j]:As[j+1]] = download(Kr)
                del Kr
        del C1
        del C2
        for i in range(blkA):
            for j in range(i+1,blkA):
                M[As[j]:As[j+1], As[i]:As[i+1]] = M[As[i]:As[i+1], As[j]:As[j+1]].t()
    return M
        