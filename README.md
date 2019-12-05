# Newton-method-for-GSC-losses-

## Intro

This repository provides the code used to run the experiments of the paper "Globally Convergent Newton Methods for Ill-conditioned Generalized Self-concordant functions" (https://arxiv.org/abs/1907.01771). In particular, the folder SECOND ORDER METHOD contains a preliminary python implementation of the algorithm, that uses CPU and one GPU.

## Installation on Linux/mac

The file env contains all dependencies needed to run the code in the python



## Algorithm

The necessary scripts to run the second order method are located in the folder [SECOND ORDER METHOD](https://github.com/umarteau/Newton-Method-for-GSC-losses-/tree/master/SECOND%20ORDER%20METHOD). The alorithm itself, which we present in the paper, is implemented in the script [SECOND ORDER METHOD/NewtonMethod.py](https://github.com/umarteau/Newton-Method-for-GSC-losses-/blob/master/SECOND%20ORDER%20METHOD/NewtonMethod.py), as the function `NewtonMethod`.

```python
def NewtonMethod(loss,X,C,y,yC,kernel,la_list,t_list,memToUse,useGPU,cobj):
    return(alpha)
```

Input : 

* `loss` The loss function (square, logistic...), as an instance of the class `class dloss(object)` defined in [SECOND ORDER METHOD/losses.py](https://github.com/umarteau/Newton-Method-for-GSC-losses-/blob/master/SECOND%20ORDER%20METHOD/losses.py). The logistic and square losses are pre-defined as `logloss` and `squareloss`;
* `X` a $n \times d$ matrix containing the training points;
* `y` a $n \times 1$ vector containing the labels;
* `C` a $m \times d$ matrix containing the Nystrom points;
* `yC` a $m \times 1$ vector containing the labels associated to the Nystrom points
* `kernel` 
* `la_list,t_list`
* `memToUse`
* `useGPU`
* `cobj`

Output : 

* `alpha`

### Example



## Experiments
