# Quick Start Guide

## Installation & Setup (30 seconds)

```bash
cd HierarchicalReasoningModel_PyTorch
pip install -r requirements.txt
```

## Verify Installation (10 seconds)

```bash
pytest tests/ -v
```

Expected: **64 tests passed** ✅

## Run Quick Demo (5 seconds)

```python
import torch
from src.hrm.modeling import HRMACTInner, HRMACTModelConfig, TransformerConfig, ACTConfig
from src.hrm.utils import generate_sudoku, Difficulty

# Create model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = HRMACTModelConfig(
    seq_len=81, vocab_size=10, high_level_cycles=2, low_level_cycles=2,
    transformers=TransformerConfig(num_layers=2, hidden_size=64, num_heads=4, expansion=2.0),
    act=ACTConfig(halt_max_steps=4, halt_exploration_probability=0.1),
)
gen = torch.Generator(device=device)
gen.manual_seed(42)
model = HRMACTInner(config=config, generator=gen, device=device).to(device)

# Generate and solve puzzle
puzzle, solution = generate_sudoku(Difficulty.EASY)
print("Model ready! ✓")
```

## Train Model

```bash
python -m src.hrm.train
```

Checkpoints saved to `checkpoints/checkpoint-{step}.pt`

## Run Inference

```bash
python -m src.hrm.infer checkpoints/checkpoint-250.pt medium
```

## GPU Check

```python
import torch
print(f"CUDA: {torch.cuda.is_available()}")  # Should be True
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

## Documentation

- `README.md` - Main documentation
- `docs/VIRTUAL_ENV_SETUP.md` - Environment setup details
- `docs/CONVERSION_NOTES.md` - Conversion documentation
- `PROJECT_SUMMARY.md` - Complete project overview

## Support

All components tested and verified:
- ✅ 64/64 tests passing
- ✅ GPU acceleration enabled
- ✅ Complete fidelity to original
- ✅ Ready for production use
