
# Experiments


For all these experiments, one needs to download the data set and place it in the [DATASETS](https://github.com/umarteau/Newton-Method-for-GSC-losses-/tree/master/DATASETS) folder. 

- For HIGGS, the data set can be foud [here](https://archive.ics.uci.edu/ml/datasets/HIGGS) in `.csv` format.
- For SUSY, the data set can be foud [here](https://archive.ics.uci.edu/ml/datasets/SUSY) in `.csv` format.



Moreover, to run the scripts as such, one needs to convert them to the `.pt` format. For instance : 

```python
import csv
import pandas as pd
import numpy as np
import torch

df= pd.read_csv('Higgs.csv')
tensor = torch.tensor(df)
torch.save(tensor, 'Higgs.pt')

```

### Comparison with one competitor : Katysusha accelerated SVRG

Throughout the experiments, we relate to one main competitor, which is the accelerated version of SVRG taylored to the Kernel regression problem. In order to make it as effective as possible, we have implemented it using taylored batches, in the script [Katyusha.py](https://github.com/umarteau/Newton-Method-for-GSC-losses-/blob/master/SECOND%20ORDER%20METHOD/Katyusha.py). We also created a callback object associated to this algorithm in [display.py](https://github.com/umarteau/Newton-Method-for-GSC-losses-/blob/master/SECOND%20ORDER%20METHOD/display.py).

The experiments are divided in four jupyter notebooks:

- [Higgs.ipynb](https://github.com/umarteau/Newton-Method-for-GSC-losses-/blob/master/EXPERIMENTS/Higgs.ipynb) This notebook implements both Katyusha and our second order method to perform kernel logistic regression on Higgs. It also plots all the figures comparing the two methods, and compares logistic with square loss. Note that the pre-processing is included in the file.
- [HiggsGridSearch.ipynb](https://github.com/umarteau/Newton-Method-for-GSC-losses-/blob/master/EXPERIMENTS/HiggsGridSearch.ipynb) This notebook performs a grid search in order to find the best pair of parameters $\sigma,\lambda$ adapted to Higgs, thus justifying the fact that we need methods which perform well for very small $\lambda$. 
- [Susy.ipynb](https://github.com/umarteau/Newton-Method-for-GSC-losses-/blob/master/EXPERIMENTS/Susy.ipynb) This notebook implements both Katyusha and our second order method to perform kernel logistic regression on Susy. It also plots all the figures comparing the two methods, and compares logistic with square loss. Note that the pre-processing is included in the file.
- [SusyGridSearch.ipynb](https://github.com/umarteau/Newton-Method-for-GSC-losses-/blob/master/EXPERIMENTS/SusyGridSearch.ipynb) This notebook performs a grid search in order to find the best pair of parameters $\sigma,\lambda$ adapted to Susy, thus justifying the fact that we need methods which perform well for very small $\lambda$. 

