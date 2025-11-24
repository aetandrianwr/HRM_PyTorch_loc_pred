#!/bin/bash

# Virtual Environment Setup Script for HRM PyTorch
# This script creates a virtual environment and installs all dependencies

set -e

echo "=========================================="
echo "HRM PyTorch - Virtual Environment Setup"
echo "=========================================="
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Detected Python version: $python_version"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Install package in editable mode
echo ""
echo "Installing package in editable mode..."
pip install -e .

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To activate the virtual environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To deactivate, run:"
echo "  deactivate"
echo ""
echo "To run training:"
echo "  python -m src.hrm.train"
echo ""
echo "To run tests:"
echo "  pytest tests/"
echo ""
