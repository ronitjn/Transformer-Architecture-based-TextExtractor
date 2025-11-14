# GPU Setup Instructions for 4GB NVIDIA GPU

## Step 1: Check Your CUDA Version

First, check which CUDA version your GPU drivers support:

```bash
nvidia-smi
```

Look for the "CUDA Version" in the top right corner (e.g., 11.8, 12.1, etc.)

## Step 2: Install UV (if not already installed)

```bash
# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Step 3: Setup Virtual Environment

```bash
# Create venv
uv venv

# Activate
source .venv/Scripts/activate  # Windows (bash)
# OR
source .venv/bin/activate  # macOS/Linux
```

## Step 4: Install CUDA-enabled PyTorch

### For CUDA 11.8:
```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### For CUDA 12.1:
```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### For CUDA 12.4:
```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### Install other dependencies:
```bash
uv pip install -e .
```

Visit https://pytorch.org/get-started/locally/ for other versions.

## Step 5: Verify GPU is Working

```bash
uv run python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('GPU Name:', torch.cuda.get_device_name(0))"

# OR use the check script
uv run python check_gpu.py
```

Should output:
```
CUDA Available: True
GPU Name: [Your GPU Model]
```

## Configuration Optimized for 4GB GPU

The training configuration has been updated:

- **seq_length**: 128 (reduced from 256)
- **batch_size**: 64 (increased for GPU efficiency)
- **num_layers**: 4 (reduced from 6)
- **Estimated GPU usage**: ~3.5GB

## Training Speed Comparison

- **CPU**: ~8 seconds per iteration
- **GPU (4GB)**: ~0.1-0.3 seconds per iteration ⚡
- **Speedup**: ~25-80x faster!

## Memory Tips

If you run out of GPU memory during training:

1. Reduce `batch_size`: Try 48, 32, or 16
2. Reduce `seq_length`: Try 96 or 64
3. Reduce `num_layers`: Try 3
4. Reduce `d_model`: Try 128

## After Installing CUDA PyTorch

Simply run:
```bash
uv run python train.py
```

The script will automatically detect and use your GPU!

## Monitoring GPU Usage

While training, open another terminal:
```bash
# Watch GPU usage in real-time
nvidia-smi -l 1
```

This will show:
- GPU utilization %
- Memory usage
- Temperature
