# UV Setup Guide

This project uses `uv` for fast, reliable Python package management.

## Installation

### Install UV

**Windows:**
```bash
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Verify Installation
```bash
uv --version
```

## Project Setup

### 1. Create Virtual Environment with UV
```bash
cd "e:/Foundora/New folder/TextExtractor"
uv venv
```

### 2. Activate Virtual Environment

**Windows (bash):**
```bash
source .venv/Scripts/activate
```

**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
source .venv/bin/activate
```

### 3. Install Dependencies

**For CPU-only PyTorch:**
```bash
uv pip install -e .
```

**For GPU (CUDA 12.4) - Recommended for 4GB GPU:**
```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
uv pip install -e .
```

**For GPU (CUDA 11.8):**
```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
uv pip install -e .
```

## Quick Start

### Extract Text from PDF
```bash
uv run python pdf_text_extractor.py path/to/your.pdf
```

### Check GPU Setup
```bash
uv run python check_gpu.py
```

### Train Model
```bash
uv run python train.py
```

### Generate Text
```bash
uv run python generate.py --prompt "The boy" --max_length 500
```

## UV Commands Cheat Sheet

### Package Management
```bash
# Add a package
uv pip install package-name

# Add dev dependency
uv pip install --dev package-name

# Update all packages
uv pip install --upgrade -e .

# List installed packages
uv pip list

# Show package info
uv pip show package-name
```

### Running Scripts
```bash
# Run any Python script with uv
uv run python script.py

# Run with specific Python version
uv run --python 3.11 python script.py
```

### Virtual Environment
```bash
# Create venv
uv venv

# Create with specific Python version
uv venv --python 3.11

# Remove venv
rm -rf .venv
```

## Why UV?

✅ **10-100x faster** than pip
✅ **Reliable** dependency resolution
✅ **Cross-platform** compatibility
✅ **pip-compatible** interface
✅ **Minimal configuration** needed

## Benefits for This Project

1. **Fast installation**: PyTorch installs in minutes instead of 10-20 minutes
2. **Consistent environments**: Lock files ensure reproducible builds
3. **Better caching**: Shared package cache across projects
4. **Faster CI/CD**: Dramatically speeds up GitHub Actions

## Troubleshooting

### UV not found after installation
Close and reopen your terminal, or add to PATH:
```bash
# Windows
set PATH=%USERPROFILE%\.cargo\bin;%PATH%

# macOS/Linux
export PATH="$HOME/.cargo/bin:$PATH"
```

### CUDA version mismatch
Check your CUDA version:
```bash
nvidia-smi
```
Then install matching PyTorch version (see Step 3 above).

### Virtual environment activation issues
Make sure you're in the project directory:
```bash
cd "e:/Foundora/New folder/TextExtractor"
```

## Migration from pip

If you have existing pip environment:
```bash
# Deactivate current environment
deactivate

# Remove old venv
rm -rf venv/

# Create new UV environment
uv venv
source .venv/Scripts/activate  # Windows bash
uv pip install -e .
```

## Additional Resources

- UV Documentation: https://github.com/astral-sh/uv
- PyTorch Installation: https://pytorch.org/get-started/locally/
