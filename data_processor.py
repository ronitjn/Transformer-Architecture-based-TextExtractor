"""
Data processing utilities for text datasets
Tokenization and dataset preparation
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import re


class Tokenizer:
    """Simple character or word-level tokenizer"""
    
    def __init__(self, level='char', vocab_size=None):
        """
        Args:
            level: 'char' for character-level or 'word' for word-level
            vocab_size: Maximum vocabulary size (None = unlimited)
        """
        self.level = level
        self.vocab_size = vocab_size
        self.token2idx = {}
        self.idx2token = {}
        
        # Special tokens
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'
        self.sos_token = '<SOS>'
        self.eos_token = '<EOS>'
        
        self.special_tokens = [self.pad_token, self.unk_token, self.sos_token, self.eos_token]
    
    def build_vocab(self, text):
        """Build vocabulary from text"""
        if self.level == 'char':
            tokens = list(text)
        else:  # word level
            tokens = self._tokenize_words(text)
        
        # Count token frequencies
        token_counts = Counter(tokens)
        
        # Add special tokens first
        for i, token in enumerate(self.special_tokens):
            self.token2idx[token] = i
            self.idx2token[i] = token
        
        # Add most common tokens
        if self.vocab_size:
            most_common = token_counts.most_common(self.vocab_size - len(self.special_tokens))
        else:
            most_common = token_counts.most_common()
        
        for token, _ in most_common:
            if token not in self.token2idx:
                idx = len(self.token2idx)
                self.token2idx[token] = idx
                self.idx2token[idx] = token
        
        print(f"Vocabulary size: {len(self.token2idx)}")
        print(f"Most common tokens: {most_common[:10]}")
    
    def _tokenize_words(self, text):
        """Simple word tokenization"""
        # Convert to lowercase and split on whitespace/punctuation
        text = text.lower()
        tokens = re.findall(r'\b\w+\b|[.,!?;]', text)
        return tokens
    
    def encode(self, text):
        """Convert text to token indices"""
        if self.level == 'char':
            tokens = list(text)
        else:
            tokens = self._tokenize_words(text)
        
        indices = [
            self.token2idx.get(token, self.token2idx[self.unk_token])
            for token in tokens
        ]
        return indices
    
    def decode(self, indices):
        """Convert token indices back to text"""
        tokens = [self.idx2token.get(idx, self.unk_token) for idx in indices]
        
        if self.level == 'char':
            return ''.join(tokens)
        else:
            # Simple word-level decoding
            text = ' '.join(tokens)
            # Fix spacing around punctuation
            text = re.sub(r'\s+([.,!?;])', r'\1', text)
            return text
    
    def save(self, path):
        """Save tokenizer vocabulary"""
        data = {
            'level': self.level,
            'vocab_size': self.vocab_size,
            'token2idx': self.token2idx,
            'idx2token': {int(k): v for k, v in self.idx2token.items()}
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        print(f"Tokenizer saved to {path}")
    
    @classmethod
    def load(cls, path):
        """Load tokenizer vocabulary"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        tokenizer = cls(level=data['level'], vocab_size=data['vocab_size'])
        tokenizer.token2idx = data['token2idx']
        tokenizer.idx2token = {int(k): v for k, v in data['idx2token'].items()}
        
        print(f"Tokenizer loaded from {path}")
        print(f"Vocabulary size: {len(tokenizer.token2idx)}")
        return tokenizer


class TextDataset(Dataset):
    """PyTorch dataset for text data"""
    
    def __init__(self, text, tokenizer, seq_length=128, stride=64):
        """
        Args:
            text: Raw text string
            tokenizer: Tokenizer instance
            seq_length: Length of each sequence
            stride: Stride for creating overlapping sequences
        """
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        
        # Encode entire text
        self.encoded = tokenizer.encode(text)
        
        # Create sequences with stride
        self.sequences = []
        for i in range(0, len(self.encoded) - seq_length, stride):
            seq = self.encoded[i:i + seq_length + 1]  # +1 for target
            if len(seq) == seq_length + 1:
                self.sequences.append(seq)
        
        print(f"Created {len(self.sequences)} sequences of length {seq_length}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        # Input is all tokens except last
        x = torch.tensor(sequence[:-1], dtype=torch.long)
        # Target is all tokens except first (shifted by 1)
        y = torch.tensor(sequence[1:], dtype=torch.long)
        return x, y


def load_text_file(filepath):
    """Load text from file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    return text


def prepare_data(text_path, tokenizer_path=None, level='char', vocab_size=None, 
                 seq_length=128, stride=64, batch_size=32, train_split=0.9):
    """
    Prepare data for training
    
    Args:
        text_path: Path to text file
        tokenizer_path: Path to save/load tokenizer
        level: 'char' or 'word'
        vocab_size: Maximum vocabulary size
        seq_length: Sequence length
        stride: Stride for sequences
        batch_size: Batch size for DataLoader
        train_split: Fraction of data for training
    
    Returns:
        train_loader, val_loader, tokenizer
    """
    # Load text
    print(f"Loading text from {text_path}")
    text = load_text_file(text_path)
    print(f"Loaded {len(text):,} characters")
    
    # Create or load tokenizer
    if tokenizer_path and os.path.exists(tokenizer_path):
        tokenizer = Tokenizer.load(tokenizer_path)
    else:
        tokenizer = Tokenizer(level=level, vocab_size=vocab_size)
        tokenizer.build_vocab(text)
        if tokenizer_path:
            tokenizer.save(tokenizer_path)
    
    # Split data
    split_idx = int(len(text) * train_split)
    train_text = text[:split_idx]
    val_text = text[split_idx:]
    
    print(f"Train text: {len(train_text):,} characters")
    print(f"Val text: {len(val_text):,} characters")
    
    # Create datasets
    train_dataset = TextDataset(train_text, tokenizer, seq_length, stride)
    val_dataset = TextDataset(val_text, tokenizer, seq_length, stride)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, val_loader, tokenizer


if __name__ == "__main__":
    import os
    
    # Example usage
    text = "This is a sample text. " * 100
    
    # Character-level tokenizer
    print("\n=== Character-level tokenization ===")
    char_tokenizer = Tokenizer(level='char')
    char_tokenizer.build_vocab(text)
    
    sample = "Hello world!"
    encoded = char_tokenizer.encode(sample)
    decoded = char_tokenizer.decode(encoded)
    print(f"Original: {sample}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    
    # Word-level tokenizer
    print("\n=== Word-level tokenization ===")
    word_tokenizer = Tokenizer(level='word', vocab_size=1000)
    word_tokenizer.build_vocab(text)
    
    encoded = word_tokenizer.encode(sample)
    decoded = word_tokenizer.decode(encoded)
    print(f"Original: {sample}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
