# Newton-method-for-GSC-losses-

## Intro

This repository provides the code used to run the experiments of the paper "Globally Convergent Newton Methods for Ill-conditioned Generalized Self-concordant functions" (https://arxiv.org/abs/1907.01771). In particular, the folder [SECOND ORDER METHOD](https://github.com/umarteau/Newton-Method-for-GSC-losses-/tree/master/SECOND%20ORDER%20METHOD) contains a preliminary python implementation of the algorithm, that uses CPU and one GPU.

## Installation on Linux/mac

The file [environment.yml](https://github.com/umarteau/Newton-Method-for-GSC-losses-/blob/master/environment.yml) contains all dependencies needed to run the code in the python. One just needs to load the scripts in [SECOND ORDER METHOD](https://github.com/umarteau/Newton-Method-for-GSC-losses-/tree/master/SECOND%20ORDER%20METHOD)  to use the algorithm.



## Algorithm

The necessary scripts to run the second order method are located in the folder [SECOND ORDER METHOD](https://github.com/umarteau/Newton-Method-for-GSC-losses-/tree/master/SECOND%20ORDER%20METHOD). The alorithm itself, which we present in the paper, is implemented in the script [NewtonMethod.py](https://github.com/umarteau/Newton-Method-for-GSC-losses-/blob/master/SECOND%20ORDER%20METHOD/NewtonMethod.py), as the function `NewtonMethod`.

```python
def NewtonMethod(loss,X,C,y,yC,kernel,la_list,t_list,memToUse,useGPU,cobj):
    return(alpha)
```

Input : 

* `loss` an instance of the class `dloss` defined in [losses.py](https://github.com/umarteau/Newton-Method-for-GSC-losses-/blob/master/SECOND%20ORDER%20METHOD/losses.py). It represent the loss function $\ell(y,y^{\prime})$. An instance of `dloss` has the following main attributes : 
    + `loss.f` a function of two variables corresponding to $y,y^{\prime} \mapsto \ell(y,y^{\prime})$
    + `loss.df` a function of two variables corresponding to $y,y^{\prime} \mapsto \partial_{y^{\prime}}\ell(y,y^{\prime})$
    + `loss.ddf` a function of two variables corresponding to $y,y^{\prime} \mapsto \partial^2_{y^{\prime}y^{\prime}}\ell(y,y^{\prime})$
    The following losses are pre-defined in the script [losses.py](https://github.com/umarteau/Newton-Method-for-GSC-losses-/blob/master/SECOND%20ORDER%20METHOD/losses.py):
     - `squareloss` corresponding to $\ell(y,y^{\prime}) = \frac{1}{2}|y-y^{\prime}|^2$;
     - `logloss` corresponding to $\ell(y,y^{\prime}) = \log(1+e^{-yy^{\prime}})$.
* `X` a $n \times d$ double tensor containing the training points ;
* `y` a $n \times 1$ double tensor containing the labels;
* `C` a $m \times d$ double tensor containing the Nystrom centers;
* `yC` a $m \times 1$ double tensor containing the labels associated to the Nystrom centers;
* `kernel` the kernel to use. `kernel` is assumed to be a function which for any two double tensors `A,B` of sizes `(n1,d),(n2,d)` respectively, outputs `kernel(A,B)` of size `(n1,n2)` the Gram kernel matrix.
    For example, the kernel 
     ```python
    def kernel(A,B):
    return (1+A@B.t())**2
    ```
    defines the polynomial kernel of degree two.
    Gaussian kernels are pre-implemented in [gaussianKernel.py](https://github.com/umarteau/Newton-Method-for-GSC-losses-/blob/master/SECOND%20ORDER%20METHOD/gaussianKernel.py); given a standard deviation `s`, `gaussianKernel(s)` returns the associated gaussian kernel. 
* `la_list,t_list` lists of positive doubles of same length.
 The general algorithm works as follows: $K$ iterations are performed (see decreasing $\lambda$ scheme). At each iteration k, $t_k$ steps of a Newton method applied to the regularized problem with $\lambda_k$ are performed. `la_list,t_list` correspond to the two lists $(\lambda_k)_{1\leq k \leq K},~(t_k)_{1 \leq k \leq K}$. 
* `memToUse`  positive double, the maximum amount of RAM memory to use, in GB; if `memToUse = None`, it will be automatically computed.
* `useGPU` a binary flag to specify if to use the GPU. If the GPU flag is set at `True` the first GPU of the machine will be used.
* `cobj` an instance of the class `cobj` defined in [display.py](https://github.com/umarteau/Newton-Method-for-GSC-losses-/blob/master/SECOND%20ORDER%20METHOD/display.py). A `cobj` object represents both the callback object and callback function and has the following attributes:
    - `cobj.start` to start measurements
    - `cobj.keepInfo` is a function to keep the informations of interest computed before the Newton iterations
    - `cobj.cbIterates` is the callback function we want to apply at each iteration
    - `cobj.cbConj` is the callback function we want to apply during the steps of conjugate gradient descent when computing approximate Newton steps.

    `cobj()` returns the empty callback object. 

Output : 

* `alpha` a `(m,1)` double tensor corresponding to the coefficients of the model.

### Example

In this example we assume to have already loaded and preprocessed `X,y` `Xtst,Ytst` (test set). The following script executes our second order method, for 14 iterations, with a Gaussian kernel of width 5, a regularization parameter `lambda` going down to  `1e-9`, 25,000 Nystrom centers. Note that in the following code 1) we are not using any callback, 2) the GPU will be used for the computations and 3) the function will use all the free memory available on the machine (depending on the dimensionality of the problem).

```python

import numpy as np
import torch
import psutil
import sys
sys.path.append("../")
import NewtonMethod
import gaussianKernel
import losses
import display

# loss
loss = losses.logloss

# Selecting Nystrom centers
m = 25000
l = np.array(range(ntr))
np.random.shuffle(l)
l = l[:m]
C = Xtr[l,:]
yC = Ytr[l,:]


#UseGPU
useGPU = True
memToUse = 0.9*psutil.virtual_memory().available

#Gaussian kernel with sigma = 5
sigma = 5
kern = gaussianKernel.gaussianKernel(sigma)

#Parameters for Newton Method
la_list = [1e-3,1e-6,1e-9,1e-9,1e-9,1e-9,1e-9,1e-9,1e-9,1e-9,1e-9,1e-9,1e-9,1e-9]
t_list = [3,3,3,8,8,8,8,8,8,8,8,8,8,8]

#Empty callback object
cobj = display.cobj()

#Perform Newton method
alpha = NewtonMethod.NewtonMethod(loss,Xtr,C,Ytr,yC,kern,la_list,t_list,memToUse = memToUse,cobj = cobj)
```

To test the predictor learned above on the test set `Xtst`, `Ytst`, we compute loss and classification error with the help of the function `KnmPtest` in from the script [display.py](https://github.com/umarteau/Newton-Method-for-GSC-losses-/blob/master/SECOND%20ORDER%20METHOD/display.py) (that computes `kernel(Xtst, C)` in blocks), as follows


``` python

#Computing the prediction with our model 

KnmTs = display.KnmPtest(kern,Xtst,C,memToUse,useGPU)
Ypred = KnmPTs(alpha)

#Test Loss
testLoss = torch.mean(loss.f(Ytst,Ypred).view(nts))

#Test Error
testError = 100*torch.sum((Ytst*Ypred <=0).view(nts)).type(torch.float64)/nts

```


## Experiments


See the [EXPERIMENTS](https://github.com/umarteau/Newton-Method-for-GSC-losses-/tree/master/EXPERIMENTS) folder, and the associated [readme](https://github.com/umarteau/Newton-Method-for-GSC-losses-/blob/master/EXPERIMENTS/readme.md) to understand and reproduce the experiments for the paper. 
