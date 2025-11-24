from setuptools import setup, find_packages

setup(
    name="hierarchical-reasoning-model",
    version="1.0.0",
    description="Hierarchical Reasoning Model with ACT for Sudoku - PyTorch Implementation",
    author="Converted from MLX Swift implementation",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "safetensors>=0.4.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
        ]
    },
)
