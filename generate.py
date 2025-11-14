"""
Text generation script using trained model
"""

import torch
import argparse
from transformer_model import TransformerLanguageModel
from data_processor import Tokenizer


def load_model(checkpoint_path, tokenizer_path, device='cpu'):
    """Load trained model and tokenizer"""
    
    # Load tokenizer
    tokenizer = Tokenizer.load(tokenizer_path)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    # Create model
    model = TransformerLanguageModel(
        vocab_size=len(tokenizer.token2idx),
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        d_ff=config['d_ff'],
        max_seq_len=config['seq_length'],
        dropout=0.0  # No dropout during inference
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded from {checkpoint_path}")
    print(f"Trained for {checkpoint['epoch']} epochs")
    print(f"Best validation loss: {checkpoint['val_loss']:.4f}")
    
    return model, tokenizer


def generate_text(model, tokenizer, prompt, max_length=500, temperature=0.8, top_k=50, device='cpu'):
    """Generate text from prompt"""
    
    # Encode prompt
    tokens = tokenizer.encode(prompt)
    input_tensor = torch.tensor([tokens]).to(device)
    
    print(f"\nPrompt: {prompt}")
    print(f"Generating {max_length} tokens...\n")
    
    # Generate
    with torch.no_grad():
        generated = model.generate(
            input_tensor,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k
        )
    
    # Decode
    generated_text = tokenizer.decode(generated[0].cpu().tolist())
    
    return generated_text


def main():
    parser = argparse.ArgumentParser(description='Generate text using trained model')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--tokenizer', type=str, default='tokenizer.json',
                       help='Path to tokenizer')
    parser.add_argument('--prompt', type=str, default='The boy',
                       help='Text prompt to start generation')
    parser.add_argument('--max_length', type=int, default=500,
                       help='Maximum length to generate')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Sampling temperature (higher = more random)')
    parser.add_argument('--top_k', type=int, default=50,
                       help='Top-k sampling parameter')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cpu/cuda/auto)')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load model
    model, tokenizer = load_model(args.checkpoint, args.tokenizer, device)
    
    # Generate
    generated = generate_text(
        model, tokenizer, args.prompt,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device
    )
    
    print("="*80)
    print("GENERATED TEXT:")
    print("="*80)
    print(generated)
    print("="*80)


if __name__ == "__main__":
    main()
