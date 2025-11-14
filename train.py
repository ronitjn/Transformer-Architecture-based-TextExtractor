"""
Training script for Transformer language model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import time
from tqdm import tqdm

from transformer_model import TransformerLanguageModel, count_parameters
from data_processor import prepare_data, Tokenizer


class Trainer:
    """Trainer class for language model"""
    
    def __init__(self, model, train_loader, val_loader, tokenizer, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        self.config = config
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        print(f"Using device: {self.device}")
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token2idx[tokenizer.pad_token])
        self.optimizer = optim.Adam(model.parameters(), lr=config['lr'], betas=(0.9, 0.98), eps=1e-9)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
        )
        
        # Logging
        self.writer = SummaryWriter(config['log_dir'])
        
        # Checkpointing
        self.checkpoint_dir = config['checkpoint_dir']
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.best_val_loss = float('inf')
        self.global_step = 0
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(inputs)
            
            # Calculate loss
            loss = self.criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
            
            self.optimizer.step()
            
            # Logging
            total_loss += loss.item()
            self.global_step += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            # Log to tensorboard
            if batch_idx % 100 == 0:
                self.writer.add_scalar('Train/Loss', loss.item(), self.global_step)
                self.writer.add_scalar('Train/LearningRate', 
                                     self.optimizer.param_groups[0]['lr'], 
                                     self.global_step)
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for inputs, targets in tqdm(self.val_loader, desc="Validation"):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                logits = self.model(inputs)
                loss = self.criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return avg_loss, perplexity
    
    def generate_sample(self, prompt="The ", max_length=200):
        """Generate sample text"""
        self.model.eval()
        
        # Encode prompt
        tokens = self.tokenizer.encode(prompt)
        input_tensor = torch.tensor([tokens]).to(self.device)
        
        # Generate
        with torch.no_grad():
            generated = self.model.generate(
                input_tensor, 
                max_length=max_length,
                temperature=0.8,
                top_k=50
            )
        
        # Decode
        generated_text = self.tokenizer.decode(generated[0].cpu().tolist())
        return generated_text
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }
        
        # Save latest
        path = os.path.join(self.checkpoint_dir, 'latest.pt')
        torch.save(checkpoint, path)
        
        # Save best
        if is_best:
            path = os.path.join(self.checkpoint_dir, 'best.pt')
            torch.save(checkpoint, path)
            print(f"Saved best model with val_loss: {val_loss:.4f}")
    
    def train(self, num_epochs):
        """Main training loop"""
        print(f"\nStarting training for {num_epochs} epochs")
        print(f"Model parameters: {count_parameters(self.model):,}")
        
        for epoch in range(1, num_epochs + 1):
            start_time = time.time()
            
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss, perplexity = self.validate()
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Log
            epoch_time = time.time() - start_time
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Perplexity: {perplexity:.2f}")
            print(f"Time: {epoch_time:.2f}s")
            
            self.writer.add_scalar('Val/Loss', val_loss, epoch)
            self.writer.add_scalar('Val/Perplexity', perplexity, epoch)
            
            # Generate sample
            if epoch % 5 == 0:
                sample = self.generate_sample()
                print(f"\nGenerated sample:\n{sample}\n")
                self.writer.add_text('Generated', sample, epoch)
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            self.save_checkpoint(epoch, val_loss, is_best)
        
        self.writer.close()
        print("\nTraining complete!")


def main():
    """Main training function"""
    
    # Configuration - Optimized for 4GB GPU
    config = {
        # Data
        'text_path': 'harrypotter_extracted.txt',
        'tokenizer_path': 'tokenizer.json',
        'level': 'char',  # 'char' or 'word'
        'vocab_size': None,  # None for unlimited
        'seq_length': 128,  # Reduced from 256 for 4GB GPU
        'stride': 64,
        'batch_size': 64,  # Increased batch size (more efficient on GPU)
        'train_split': 0.95,
        
        # Model - Optimized for 4GB GPU
        'd_model': 256,
        'num_heads': 8,
        'num_layers': 4,  # Reduced from 6 to fit in 4GB
        'd_ff': 1024,
        'dropout': 0.1,
        
        # Training
        'lr': 0.0003,  # Slightly higher LR for faster convergence
        'num_epochs': 50,
        'grad_clip': 1.0,
        
        # Logging
        'log_dir': 'logs',
        'checkpoint_dir': 'checkpoints'
    }
    
    # Prepare data
    print("Preparing data...")
    train_loader, val_loader, tokenizer = prepare_data(
        text_path=config['text_path'],
        tokenizer_path=config['tokenizer_path'],
        level=config['level'],
        vocab_size=config['vocab_size'],
        seq_length=config['seq_length'],
        stride=config['stride'],
        batch_size=config['batch_size'],
        train_split=config['train_split']
    )
    
    # Create model
    print("\nCreating model...")
    model = TransformerLanguageModel(
        vocab_size=len(tokenizer.token2idx),
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        d_ff=config['d_ff'],
        max_seq_len=config['seq_length'],
        dropout=config['dropout']
    )
    
    # Create trainer
    trainer = Trainer(model, train_loader, val_loader, tokenizer, config)
    
    # Train
    trainer.train(config['num_epochs'])


if __name__ == "__main__":
    main()
