#!/usr/bin/env python3
"""
Training script for the Melody-to-Chord Transformer model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pickle
from pathlib import Path
import time
from typing import Dict, List, Tuple
import json

# Import our model
import sys
sys.path.append('model')
from transformer_model import MelodyToChordTransformer, MelodyChordDataset, create_model, count_parameters

class MelodyChordTrainer:
    """Trainer class for the melody-to-chord model."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Optimizer and scheduler
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=50,  # Adjust based on number of epochs
            eta_min=1e-6
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rate': []
        }
        
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            # Print progress
            if batch_idx % 50 == 0:
                print(f'  Batch {batch_idx:3d}/{len(self.train_loader)} | '
                     f'Loss: {loss.item():.4f} | '
                     f'Acc: {100. * correct / total:.1f}%')
                
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
                
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(self, num_epochs: int, save_dir: str = 'model_checkpoints') -> Dict:
        """Train the model for multiple epochs."""
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        best_val_acc = 0.0
        best_model_path = save_path / 'best_model.pth'
        
        print(f"🚀 Starting training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Model parameters: {count_parameters(self.model):,}")
        print("-" * 60)
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Train
            print(f"Epoch {epoch+1}/{num_epochs}")
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Update scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rate'].append(current_lr)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss
                }, best_model_path)
                print(f"  💾 Saved new best model (val_acc: {val_acc:.2f}%)")
            
            # Print epoch summary
            epoch_time = time.time() - start_time
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            print(f"  LR: {current_lr:.2e} | Time: {epoch_time:.1f}s")
            print("-" * 60)
            
        # Save final model and history
        torch.save({
            'epoch': num_epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }, save_path / 'final_model.pth')
        
        # Save training history
        with open(save_path / 'training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"✅ Training complete!")
        print(f"Best validation accuracy: {best_val_acc:.2f}%")
        print(f"Models saved to: {save_path}")
        
        return self.history


def main():
    """Main training function."""
    print("🎼 Melody-to-Chord Transformer Training")
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load metadata
    print("Loading metadata...")
    with open('processed_data/metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    
    num_chords = len(metadata['chord_classes'])
    print(f"Number of chord classes: {num_chords}")
    print(f"Dataset info: {metadata['dataset_info']}")
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = MelodyChordDataset('processed_data/features_full.npz', train=True, train_split=0.8)
    val_dataset = MelodyChordDataset('processed_data/features_full.npz', train=False, train_split=0.8)
    
    # Create data loaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Create model
    print("Creating model...")
    model = create_model(num_chords)
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Create trainer
    trainer = MelodyChordTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=1e-4,
        weight_decay=1e-5
    )
    
    # Train
    num_epochs = 20  # Start with fewer epochs for testing
    history = trainer.train(num_epochs)
    
    # Print final results
    final_train_acc = history['train_acc'][-1]
    final_val_acc = history['val_acc'][-1]
    best_val_acc = max(history['val_acc'])
    
    print(f"🎯 Final Results:")
    print(f"  Final Train Accuracy: {final_train_acc:.2f}%")
    print(f"  Final Val Accuracy:   {final_val_acc:.2f}%")
    print(f"  Best Val Accuracy:    {best_val_acc:.2f}%")


if __name__ == "__main__":
    main()
