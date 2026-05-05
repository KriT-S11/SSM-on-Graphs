# Linearizing Temporal Graph Dynamics with State Space Models (LTG-SSM)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

This is the official PyTorch implementation of the paper **"Linearizing Temporal Graph Dynamics with State Space Models"**.

## 🗂️ Repository Structure

```text
ltg_ssm/
├── layers/
│   ├── gnn.py           # Spatial Diffusion using SAGEConv/GCNConv
│   ├── mixing.py        # Temporal Feature Mixing via Interpolation Gating
│   └── ssm.py           # Adaptive State Space Model layer (Diagonal Approximation)
├── models/
│   └── ltg_ssm.py       # Full end-to-end LTG-SSM architecture
├── test_model.py        # Dummy pipeline for gradient and forward pass verification
├── environment.yml      # Conda environment configuration
├── requirements.txt     # Pip dependencies
└── README.md
```

## ⚙️ Installation

You can set up the required environment using either `conda` or `pip`.

### Option 1: Conda (Recommended)
This will create an environment named `ltg_ssm` with PyTorch and PyTorch Geometric installed.
```bash
conda env create -f environment.yml
conda activate ltg_ssm
```

### Option 2: Pip
If you are using a standard virtual environment:
```bash
pip install -r requirements.txt
```

## 🚀 Usage

You can verify that the core `LTG-SSM` architecture initializes correctly and successfully backpropagates gradients by running the included synthetic test script:

```bash
python test_model.py
```

This will generate a synthetic temporal graph sequence across multiple snapshots, pass it through the GNN -> Mixing -> SSM pipeline, and compute a dummy cross-entropy loss.

## 📈 Supported Datasets (To-Do)
The architecture has been designed to support temporal node classification on large-scale datasets. Data loaders and training loops for the following datasets will be added:
- **DBLP-3**
- **Brain**
- **Tmall** (Massive-scale E-commerce dataset)

## 📝 Citation

If you find this code or our paper useful in your research, please consider citing:

```bibtex
@article{ltg_ssm_2024,
  title={Linearizing Temporal Graph Dynamics with State Space Models},
  author={Sinha, Alankrit and others},
  journal={arXiv preprint},
  year={2024}
}
```

## 📄 License
This project is released under the MIT License.
