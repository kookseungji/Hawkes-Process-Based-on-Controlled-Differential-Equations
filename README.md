# Hawkes Process Based on Controlled Differential Equations

![GitHub Repo stars](https://img.shields.io/github/stars/kookseungji/Hawkes-Process-Based-on-Controlled-Differential-Equations)
 [![arXiv](https://img.shields.io/badge/arXiv-2305.07031-b31b1b.svg)](https://arxiv.org/abs/2305.07031) [![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fkookseungji%2FHawkes-Process-Based-on-Controlled-Differential-Equations&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

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

