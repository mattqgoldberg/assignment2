#!/usr/bin/env python3
"""
Main training script for the MIDI Melody-to-Chord Transformer model.
Consolidated version with improved architecture and class balancing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import Counter
import json
import time
from pathlib import Path

class SimpleTransformer(nn.Module):
    """Simplified but effective transformer for MIDI melody-to-chord prediction."""
    
    def __init__(self, input_dim=4, d_model=128, num_classes=50, seq_len=32, num_heads=8, num_layers=3, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(seq_len, d_model) * 0.1)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        batch_size, seq_len = x.shape[:2]
        
        # Project input and add positional encoding
        x = self.input_proj(x)  # (batch, seq_len, d_model)
        x = x + self.pos_encoding[:seq_len].unsqueeze(0)
        x = self.dropout(x)
        
        # Transformer encoding
        x = self.transformer(x)  # (batch, seq_len, d_model)
        
        # Global average pooling
        x = x.mean(dim=1)  # (batch, d_model)
        
        # Classification
        return self.classifier(x)

def load_and_prepare_data(data_path, num_classes=30, train_ratio=0.8):
    """Load data and prepare balanced training/validation sets."""
    print(f"📊 Loading data from {data_path}...")
    
    data = np.load(data_path)
    X, y = data['X'], data['y']
    print(f"Original data: {X.shape}, {y.shape}")
    
    # Select top N most common classes to avoid extreme imbalance
    counts = Counter(y)
    top_classes = [cls for cls, _ in counts.most_common(num_classes)]
    print(f"Using top {num_classes} classes out of {len(counts)} total")
    
    # Filter data to selected classes
    mask = np.isin(y, top_classes)
    X_filtered = X[mask]
    y_filtered = y[mask]
    
    # Remap labels to continuous range 0 to num_classes-1
    class_map = {old: new for new, old in enumerate(sorted(top_classes))}
    y_remapped = np.array([class_map[label] for label in y_filtered])
    
    print(f"Filtered data: {X_filtered.shape}, classes: 0-{num_classes-1}")
    
    # Show class distribution
    final_counts = Counter(y_remapped)
    print(f"Class distribution:")
    for cls in sorted(final_counts.keys())[:10]:
        print(f"  Class {cls}: {final_counts[cls]} samples")
    if len(final_counts) > 10:
        print(f"  ... and {len(final_counts)-10} more classes")
    
    # Train/validation split
    n_samples = len(X_filtered)
    n_train = int(n_samples * train_ratio)
    
    np.random.seed(42)
    indices = np.random.permutation(n_samples)
    
    train_data = {
        'X': torch.FloatTensor(X_filtered[indices[:n_train]]),
        'y': torch.LongTensor(y_remapped[indices[:n_train]])
    }
    
    val_data = {
        'X': torch.FloatTensor(X_filtered[indices[n_train:]]),
        'y': torch.LongTensor(y_remapped[indices[n_train:]])
    }
    
    print(f"Train: {len(train_data['X'])}, Validation: {len(val_data['X'])}")
    
    return train_data, val_data, class_map

def train_model(
    model, 
    train_data, 
    val_data, 
    device, 
    num_epochs=20, 
    batch_size=32, 
    learning_rate=1e-3,
    patience=5
):
    """Train the model with early stopping."""
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    best_val_acc = 0
    patience_counter = 0
    history = {'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': []}
    
    print(f"\n🚀 Starting training for up to {num_epochs} epochs...")
    print(f"Device: {device}, Batch size: {batch_size}, Learning rate: {learning_rate}")
    print("-" * 80)
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Training phase
        model.train()
        train_correct = 0
        train_total = 0
        train_loss_sum = 0
        num_batches = 0
        
        for i in range(0, len(train_data['X']), batch_size):
            batch_X = train_data['X'][i:i+batch_size].to(device)
            batch_y = train_data['y'][i:i+batch_size].to(device)
            
            if len(batch_X) < 2:  # Skip tiny batches
                continue
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss_sum += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
            num_batches += 1
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss_sum = 0
        val_batches = 0
        
        with torch.no_grad():
            for i in range(0, len(val_data['X']), batch_size):
                batch_X = val_data['X'][i:i+batch_size].to(device)
                batch_y = val_data['y'][i:i+batch_size].to(device)
                
                if len(batch_X) < 1:
                    continue
                
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                val_loss_sum += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
                val_batches += 1
        
        # Calculate metrics
        train_acc = 100 * train_correct / train_total if train_total > 0 else 0
        val_acc = 100 * val_correct / val_total if val_total > 0 else 0
        train_loss = train_loss_sum / num_batches if num_batches > 0 else 0
        val_loss = val_loss_sum / val_batches if val_batches > 0 else 0
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save history
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch+1:2d}/{num_epochs}: "
              f"Train Acc: {train_acc:5.1f}% | Val Acc: {val_acc:5.1f}% | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Time: {epoch_time:.1f}s")
        
        # Early stopping and model saving
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            # Save best model
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'epoch': epoch,
                'history': history
            }, 'model_checkpoints/best_model.pth')
            
            print(f"  💾 New best model saved (val_acc: {val_acc:.1f}%)")
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"  🛑 Early stopping triggered (patience: {patience})")
            break
        
        # Stop if we achieve good performance
        if val_acc > 70:
            print(f"  🎯 Excellent performance achieved! Stopping early.")
            break
    
    print(f"\n✅ Training complete!")
    print(f"Best validation accuracy: {best_val_acc:.1f}%")
    
    return history, best_val_acc

def main():
    """Main training function."""
    print("🎵 MIDI Melody-to-Chord Transformer Training")
    print("=" * 60)
    
    # Setup
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    Path('model_checkpoints').mkdir(exist_ok=True)
    
    # Load and prepare data
    train_data, val_data, class_map = load_and_prepare_data(
        'processed_data/features_full.npz',
        num_classes=30,  # Use top 30 classes for good balance
        train_ratio=0.85
    )
    
    # Create model
    model = SimpleTransformer(
        input_dim=4,
        d_model=128,
        num_classes=30,
        seq_len=32,
        num_heads=8,
        num_layers=3,
        dropout=0.1
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Train model
    history, best_val_acc = train_model(
        model=model,
        train_data=train_data,
        val_data=val_data,
        device=device,
        num_epochs=50,
        batch_size=32,
        learning_rate=1e-3,
        patience=7
    )
    
    # Save final results
    results = {
        'best_val_acc': float(best_val_acc),
        'class_map': {str(k): int(v) for k, v in class_map.items()},
        'model_config': {
            'input_dim': 4,
            'd_model': 128,
            'num_classes': 30,
            'seq_len': 32,
            'num_heads': 8,
            'num_layers': 3,
            'dropout': 0.1
        },
        'training_history': history
    }
    
    with open('model_checkpoints/training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 Results saved to model_checkpoints/training_results.json")
    print(f"🎯 Final validation accuracy: {best_val_acc:.1f}%")

if __name__ == "__main__":
    main()
