## Repository Structure

REFER TO BRANCH CODEBASE
```text
ltg_ssm/
├── layers/
│   ├── gnn.py           
│   ├── mixing.py        
│   └── ssm.py           
├── models/
│   └── ltg_ssm.py       
├── test_model.py       
├── environment.yml      
├── requirements.txt     
└── README.md
```

## Installation

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

## Supported Datasets (To-Do)
The architecture has been designed to support temporal node classification on large-scale datasets. Data loaders and training loops for the following datasets will be added:
- **DBLP-3**
- **Brain**
- **Tmall** (Massive-scale E-commerce dataset)
