# Virtual Environment Setup Documentation

## Overview

This document provides detailed instructions for setting up the virtual environment for the HRM PyTorch project.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- virtualenv or venv module (usually included with Python)
- CUDA-compatible GPU (optional, for GPU acceleration)

## Automatic Setup (Recommended)

The easiest way to set up the environment is to use the provided setup script:

```bash
cd HierarchicalReasoningModel_PyTorch
chmod +x setup_venv.sh
./setup_venv.sh
```

This script will:
1. Create a virtual environment in `venv/`
2. Activate the environment
3. Upgrade pip
4. Install all required dependencies
5. Install the package in editable mode

## Manual Setup

If you prefer to set up manually or are on Windows:

### Step 1: Create Virtual Environment

**On Linux/macOS:**
```bash
python3 -m venv venv
```

**On Windows:**
```cmd
python -m venv venv
```

### Step 2: Activate Virtual Environment

**On Linux/macOS:**
```bash
source venv/bin/activate
```

**On Windows (CMD):**
```cmd
venv\Scripts\activate.bat
```

**On Windows (PowerShell):**
```powershell
venv\Scripts\Activate.ps1
```

### Step 3: Upgrade pip

```bash
pip install --upgrade pip
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Install Package

```bash
pip install -e .
```

## Verifying Installation

After installation, verify that everything is working:

```python
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

Expected output:
```
PyTorch: 2.x.x
CUDA available: True  # or False if no GPU
```

## Running Tests

To verify the installation is complete and correct:

```bash
# Run all tests
pytest tests/

# Run specific test suites
pytest tests/unit/           # Unit tests only
pytest tests/integration/    # Integration tests only
pytest tests/e2e/            # End-to-end tests only

# Run with verbose output
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html
```

## GPU Setup

### CUDA Installation

If you have an NVIDIA GPU, install CUDA-enabled PyTorch:

```bash
# For CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Verify GPU

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Device name: {torch.cuda.get_device_name(0)}")
```

## Troubleshooting

### Issue: "Command not found: python3"

**Solution:** Use `python` instead of `python3`, or install Python 3.

### Issue: "No module named 'torch'"

**Solution:** Ensure you've activated the virtual environment and installed dependencies.

### Issue: "CUDA out of memory"

**Solution:** Reduce batch size in training configuration or use CPU.

### Issue: Tests failing

**Solution:** 
1. Ensure all dependencies are installed
2. Check Python version (>= 3.8)
3. Run tests individually to isolate issues

## Deactivating Environment

When you're done working:

```bash
deactivate
```

## Removing Environment

To completely remove the virtual environment:

```bash
rm -rf venv/
```

Then you can recreate it using the setup steps above.

## Dependencies

The project requires:

- **torch** (>=2.0.0): Deep learning framework
- **numpy** (>=1.24.0): Numerical computing
- **pytest** (>=7.4.0): Testing framework
- **safetensors** (>=0.4.0): Safe tensor serialization
- **tqdm** (>=4.65.0): Progress bars

See `requirements.txt` for complete list with version constraints.

## Development Dependencies

For development work, install additional packages:

```bash
pip install pytest-cov  # Test coverage
pip install black       # Code formatting
pip install flake8      # Linting
pip install mypy        # Type checking
```

## Environment Variables

Optional environment variables:

- `CUDA_VISIBLE_DEVICES`: Control which GPUs are visible (e.g., `CUDA_VISIBLE_DEVICES=0,1`)
- `OMP_NUM_THREADS`: Number of OpenMP threads for CPU operations

Example:
```bash
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
```

## Next Steps

After setting up the environment:

1. Review the main README.md for usage instructions
2. Run the test suite to verify installation
3. Try training: `python -m src.hrm.train`
4. Try inference: `python -m src.hrm.infer <checkpoint> <difficulty>`
