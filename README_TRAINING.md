# Transformer Language Model Training

Train a Transformer-based language model (based on "Attention Is All You Need") on text data.

## Project Structure

```
TextExtractor/
├── pdf_text_extractor.py      # PDF text extraction
├── transformer_model.py        # Transformer model implementation
├── data_processor.py           # Data processing and tokenization
├── train.py                    # Training script
├── generate.py                 # Text generation script
├── requirements_training.txt   # Training dependencies
└── harrypotter_extracted.txt   # Extracted text data
```

## Installation

This project uses **UV** for fast package management.

### 1. Install UV
```bash
# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Setup Project
```bash
# Create virtual environment
uv venv

# Activate
source .venv/Scripts/activate  # Windows (bash)
# OR
source .venv/bin/activate  # macOS/Linux

# Install for CPU
uv pip install -e .

# OR Install for GPU (CUDA 12.4)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
uv pip install -e .
```

See [UV Setup Guide](README_UV_SETUP.md) for details.

## Model Architecture

The implementation follows the "Attention Is All You Need" paper:

- **Multi-Head Self-Attention**: 8 attention heads
- **Position-wise Feed-Forward Networks**: 2-layer FFN with ReLU
- **Positional Encoding**: Sinusoidal position embeddings
- **Layer Normalization**: After each sub-layer
- **Residual Connections**: Around each sub-layer
- **Causal Masking**: For autoregressive generation

### Default Configuration

- **d_model**: 256 (embedding dimension)
- **num_heads**: 8 (attention heads)
- **num_layers**: 6 (transformer blocks)
- **d_ff**: 1024 (feedforward dimension)
- **dropout**: 0.1
- **seq_length**: 256 (context window)

## Training

### Basic Training

```bash
uv run python train.py
```

### Configuration

Edit the `config` dictionary in `train.py`:

```python
config = {
    # Data
    'text_path': 'harrypotter_extracted.txt',
    'level': 'char',  # 'char' or 'word' level tokenization
    'vocab_size': None,  # None for unlimited vocab
    'seq_length': 256,  # Context length
    'batch_size': 32,
    
    # Model
    'd_model': 256,
    'num_heads': 8,
    'num_layers': 6,
    'd_ff': 1024,
    'dropout': 0.1,
    
    # Training
    'lr': 0.0001,
    'num_epochs': 50,
    'grad_clip': 1.0,
}
```

### Training Features

- **Automatic checkpointing**: Saves best and latest models
- **TensorBoard logging**: Track loss, perplexity, and samples
- **Learning rate scheduling**: ReduceLROnPlateau
- **Gradient clipping**: Prevents exploding gradients
- **Validation**: Regular validation with perplexity calculation

### Monitor Training

```bash
tensorboard --logdir logs
```

## Text Generation

### Generate text from trained model:

```bash
# Basic generation
uv run python generate.py --prompt "The boy"

# Custom parameters
uv run python generate.py \
    --checkpoint checkpoints/best.pt \
    --prompt "Once upon a time" \
    --max_length 500 \
    --temperature 0.8 \
    --top_k 50
```

### Generation Parameters

- **temperature**: Controls randomness (0.1-2.0)
  - Lower = more conservative/repetitive
  - Higher = more creative/random
- **top_k**: Sample from top K most likely tokens
  - Lower = more focused
  - Higher = more diverse

## Tokenization

Two tokenization levels are supported:

### Character-Level (default)
- Vocabulary: ~100-200 unique characters
- Pros: No unknown tokens, smaller vocab
- Cons: Longer sequences, slower training

### Word-Level
```python
config['level'] = 'word'
config['vocab_size'] = 10000  # Limit vocabulary
```
- Vocabulary: Most common words
- Pros: Shorter sequences, faster training
- Cons: Unknown words, larger vocab

## Model Sizes

Approximate parameter counts:

| Configuration | Parameters | GPU Memory |
|--------------|------------|------------|
| Small (d=128, layers=4) | ~2M | ~2GB |
| Medium (d=256, layers=6) | ~8M | ~4GB |
| Large (d=512, layers=8) | ~40M | ~8GB |

## Training Tips

1. **Start small**: Begin with fewer layers/smaller d_model
2. **Monitor perplexity**: Lower is better (good: <10, great: <5)
3. **Adjust learning rate**: If loss plateaus, reduce LR
4. **Increase context**: Longer sequences = better coherence
5. **More data**: More training data = better results

## Expected Results

After training on ~6M characters:

- **Epoch 10-20**: Model learns basic patterns, grammar
- **Epoch 20-40**: Coherent sentences, some style mimicry
- **Epoch 40+**: Longer coherent passages, good style

## Troubleshooting

### Out of Memory
- Reduce `batch_size`
- Reduce `seq_length`
- Reduce `d_model` or `num_layers`
- Use gradient checkpointing

### Poor Generation Quality
- Train longer (more epochs)
- Increase model size
- Adjust temperature/top_k
- Use more training data
- Try word-level tokenization

### Loss Not Decreasing
- Check learning rate (try 1e-4 to 1e-5)
- Verify data preprocessing
- Check for NaN gradients
- Reduce batch size

## References

- Vaswani et al. (2017) "Attention Is All You Need"
- https://arxiv.org/abs/1706.03762

## Notes on Copyright

This implementation provides the technical framework for training language models. Users are responsible for ensuring they have appropriate rights to use any training data, especially for commercial purposes.
