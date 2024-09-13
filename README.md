# Hawkes Process Based on Controlled Differential Equations

![GitHub Repo stars](https://img.shields.io/github/stars/kookseungji/Hawkes-Process-Based-on-Controlled-Differential-Equations)
 [![arXiv](https://img.shields.io/badge/arXiv-2305.07031-b31b1b.svg)](https://arxiv.org/abs/2305.07031) [![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fkookseungji%2FHawkes-Process-Based-on-Controlled-Differential-Equations&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)
 [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hawkes-process-based-on-controlled/point-processes-on-memetracker)](https://paperswithcode.com/sota/point-processes-on-memetracker?p=hawkes-process-based-on-controlled)[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hawkes-process-based-on-controlled/point-processes-on-mimic-ii)](https://paperswithcode.com/sota/point-processes-on-mimic-ii?p=hawkes-process-based-on-controlled)[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hawkes-process-based-on-controlled/point-processes-on-retweet)](https://paperswithcode.com/sota/point-processes-on-retweet?p=hawkes-process-based-on-controlled)[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hawkes-process-based-on-controlled/point-processes-on-stackoverflow)](https://paperswithcode.com/sota/point-processes-on-stackoverflow?p=hawkes-process-based-on-controlled)

This repository contains the official implementation of "Hawkes Process Based on Controlled Differential Equations" as presented in IJCAI 2023.

## Introduction

HP-CDE is a novel approach to modeling temporal point processes using neural controlled differential equations (neural CDEs). It addresses two key limitations of existing neural Hawkes process models:
1. Accurately capturing irregular event dynamics 
2. Exactly calculating the log-likelihood without resorting to approximations

By using the continuous nature of neural CDEs, HP-CDE can properly handle irregular time series data and compute exact log-likelihoods, leading to improved performance on event prediction tasks.

## Requirements

- Python 3.7+
- PyTorch 1.4.0+
- CUDA-enabled GPU (recommended for faster training)

We recommend using Anaconda to manage the Python environment and dependencies.

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/kookseungji/Hawkes-Process-Based-on-Controlled-Differential-Equations.git
   cd Hawkes-Process-Based-on-Controlled-Differential-Equations
   ```

2. Create and activate the Conda environment:
   ```
   conda env create --file environment.yaml
   conda activate hpcde
   ```

## Data

The MIMIC dataset is provided in the `data` folder. For other datasets (MemeTracker, Retweet, StackOverflow), please refer to the paper for information on how to obtain and preprocess them.

## Usage

To train and evaluate the HP-CDE model, run:

```
bash run_hpcde.sh
```

This script sets up the necessary parameters and runs the main training/evaluation loop.

## Code Structure

- `Main_HPCDE.py`: Contains the main training and testing logic
- `transformer/Models.py`: Implements the HP-CDE model architecture
- `Utils.py`: Utility functions for data processing, evaluation metrics, etc.

## Configuration

The `run_hpcde.sh` script accepts several command-line arguments to configure the model and training process:

- `--data`: Path to the dataset
- `--scale`: Coefficient for time prediction loss (α₁ in the paper)
- `--llscale`: Coefficient for log-likelihood loss (α₂ in the paper)
- `--d_model` / `--hh_dim`: Size of the embedding hidden vectors
- `--layers`: Number of Neural CDE layers
- `--d_ncde`: Size of the Neural CDE hidden vectors

Refer to the script for additional hyperparameters and their default values.

## Citation

If you find this work useful in your research, please consider citing our paper:

```
@inproceedings{jo2023hawkes,
title={Hawkes process based on controlled differential equations},
author={Jo, Minju and Kook, Seungji and Park, Noseong},
booktitle={Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence (IJCAI)},
pages={2151--2159},
year={2023}
}
```