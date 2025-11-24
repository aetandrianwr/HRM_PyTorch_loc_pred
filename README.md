# Hierarchical Reasoning Model (PyTorch)

This repository implements HRM with ACT (Adaptive Computation Time) for Sudoku puzzles in PyTorch, converted from the original MLX Swift implementation.

## Project Structure

```
HierarchicalReasoningModel_PyTorch/
├── src/
│   └── hrm/
│       ├── modeling/          # Core model components
│       │   ├── attention.py
│       │   ├── swiglu.py
│       │   ├── embedding.py
│       │   ├── rmsnorm.py
│       │   ├── linear.py
│       │   ├── rotary.py
│       │   ├── init_utils.py
│       │   └── hrm.py
│       ├── utils/             # Utilities
│       │   ├── sudoku.py
│       │   └── training.py
│       ├── train.py           # Training script
│       ├── infer.py           # Inference script
│       └── __init__.py
├── tests/
│   ├── unit/                  # Unit tests
│   ├── integration/           # Integration tests
│   └── e2e/                   # End-to-end tests
├── checkpoints/               # Model checkpoints
├── requirements.txt           # Python dependencies
└── setup.py                   # Package setup
```

## Setup

### Create Virtual Environment

```bash
cd HierarchicalReasoningModel_PyTorch
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Training

To train the model:

```bash
python -m src.hrm.train
```

### Inference

To run inference on a random puzzle with a specific checkpoint and difficulty:

```bash
python -m src.hrm.infer <checkpoint_path> <difficulty>
```

Difficulty levels: `very-easy`, `easy`, `medium`, `hard`, `extreme`

Example:
```bash
python -m src.hrm.infer checkpoints/checkpoint-1000.pt medium
```

## Testing

Run all tests:
```bash
pytest tests/
```

Run specific test suites:
```bash
pytest tests/unit/           # Unit tests only
pytest tests/integration/    # Integration tests only
pytest tests/e2e/            # End-to-end tests only
```

## GPU Support

The model runs on GPU by default if CUDA is available. You can verify GPU usage:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

## Implementation Details

This PyTorch implementation faithfully replicates the original MLX Swift codebase:

- **Exact architecture**: All model components (Attention, SwiGLU, RMSNorm, etc.) are implemented with identical behavior
- **Adaptive Computation Time (ACT)**: Q-learning based halting mechanism
- **Hierarchical reasoning**: Two-level (high-level and low-level) reasoning cycles
- **Curriculum learning**: Progressive difficulty increase during training
- **Truncated normal initialization**: Custom weight initialization matching the original
- **Rotary Position Embeddings (RoPE)**: Positional encoding mechanism
- **bfloat16 support**: Mixed precision training for efficiency

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@misc{wang2025hrm,
  title         = {Hierarchical Reasoning Model},
  author        = {Wang, Guan and Li, Jin and Sun, Yuhao and Chen, Xing and Liu, Changling and Wu, Yue and Lu, Meng and Song, Sen and Abbasi Yadkori, Yasin},
  year          = {2025},
  eprint        = {2506.21734},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  doi           = {10.48550/arXiv.2506.21734},
  url           = {https://arxiv.org/abs/2506.21734}
}
```

## License

This is a faithful PyTorch conversion of the original MLX Swift implementation.
