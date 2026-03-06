# A2GBD：Attack-Agnostic Graph Backdoor Defense
implementation of WWW 2026 paper: A$^2$GBD: Attack-Agnostic Graph Backdoor Defense
<img src="https://github.com/Dcx-swjtu/A2GBD/blob/main/framework.png" width=80%>


## Overview

A2GBD is a graph learning framework for defending against backdoor attacks in graph neural networks. This implementation supports multiple datasets and GNN architectures with comprehensive evaluation metrics.

## Quick Start

### 1. Environment Setup

Create and activate a Conda environment with Python 3.9:

```bash
conda create -n GBD python=3.9 -y
conda activate GBD
pip install -r requirements.txt
```
### 2.Run Training
```bash
python train.py --dataset Cora --model_type GCN --device cuda --seed 42
```
## Arguments

| Argument       | Type     | Description                     | Choices / Example         | Default |
|----------------|----------|---------------------------------|---------------------------|---------|
| `--dataset`    | string   | Dataset name                     | `Cora`, `CiteSeer`, `PubMed` | `Cora`  |
| `--model_type` | string   | GNN backbone model               | `GCN`, `GAT`             | `GCN`   |
| `--device`     | string   | Computation device               | `cuda`, `cpu`, or GPU id (e.g. `cuda:0`) | `cuda` |
| `--seed`       | int      | Random seed for reproducibility  | Any integer   | `42`    |


## Requirements

- Python >= 3.9
- PyTorch >= 2.4
- torch-geometric >= 2.5
- numpy >= 1.24
- tqdm >= 4.66
- tensorboard >= 2.14
