## 📖 Overview 
MethylFormer is a WGBS-based methylation foundation model, and our primary objective is to elucidate the relationship between DNA methylation and CTCF expression. Furthermore, we aim to leverage this relationship to gain insights into how DNA methylation influences gene expression.
## 🚀 Quick Setup

### Prerequisites

- Python 3.9.10

### Installation Instruction

```bash
# Clone the repository
git clone https://github.com/deepdoctor/MethylFormerDev.git
# Create conda env
conda create -n mehthylFormer python=3.9.10
# "Install CpGpt to obtain its dependent packages.
pip install CpGPT
```

## 🧪 Tutorial
Modify the `root_dir` in `scripts/train_ctcf_ddp.py` to your working directory.
### Single GPU (or CPU) plain python:

```
python scripts/train_ctcf_ddp.py --distributed none --config configs/train_ctcf_single_gpu.json
```

### Single machine, 4 GPUs (DDP):

```bash
torchrun --nproc_per_node=4 scripts/train_ctcf_ddp.py --distributed ddp --config configs/train_ctcf_ddp_4gpu.json
```

