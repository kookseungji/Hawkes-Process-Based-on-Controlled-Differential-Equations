# Hawkes Process Based on Controlled Differential Equations

# Hawkes-Process-Based-on-Controlled-Differential-Equations
### Dependencies
* Python 3.7.
* [Anaconda](https://www.anaconda.com/) contains all the required packages.
* [PyTorch](https://pytorch.org/) version 1.4.0.

### Instructions 
0.   Run ``` conda env create --file environment.yaml ``` before starting the experiment.
1.   We release code for Mimic datasets in data folder. 
2. ```bash run_hpcde.sh``` to run the code.

​
[ parser ]      
* data         : The path where data exists
* scale        : coefficient of time prediction, alpha1 in the paper
* llscale      : coefficient of likelihood, alpha2 in the paper
* d_model/hh_dim      : Size of hidden vector of Embedding
* layers       : Number of Neural CDE layer
* d_ncde       : Size of hidden vector of Neural CDE for HPcde

[code]
    
* Main_HPCDE.py           : Code for training and testing.
* transformer/Models.py   : Our model 
* Utils.py                : Modules required during training
