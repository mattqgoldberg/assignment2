#!/usr/bin/env python3
"""
Simple training script for melody-to-chord prediction.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pickle
from pathlib import Path
import time
import math

# Simple positional encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]

# Simple transformer model
class SimpleMelodyChordModel(nn.Module):
    def __init__(self, num_features=4, num_chords=95, d_model=128, nhead=4, num_layers=3):
        super().__init__()
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(num_features, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_chords)
        )
        
    def forward(self, x):
        # x: (batch, seq_len, features)
        x = self.input_projection(x) * math.sqrt(self.d_model)
        x = x.transpose(0, 1)  # (seq_len, batch, d_model)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # (batch, seq_len, d_model)
        
        x = self.transformer(x)
        x = x[:, -1, :]  # Take last timestep
        x = self.classifier(x)
        return x

# Dataset class
class MelodyDataset(Dataset):
    def __init__(self, features_path, train=True, split=0.8):
        data = np.load(features_path)
        X, y = data['X'], data['y']
        
        # Split
        n = len(X)
        split_idx = int(n * split)
        if train:
            self.X, self.y = X[:split_idx], y[:split_idx]
        else:
            self.X, self.y = X[split_idx:], y[split_idx:]
            
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx]), torch.LongTensor([self.y[idx]]).squeeze()

def train_model():
    print("🎼 Starting Simple Melody-to-Chord Training")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Data
    print("Loading data...")
    train_dataset = MelodyDataset('processed_data/features_full.npz', train=True)
    val_dataset = MelodyDataset('processed_data/features_full.npz', train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Model
    model = SimpleMelodyChordModel(num_chords=95).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training
    num_epochs = 10
    best_val_acc = 0
    
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pred = output.argmax(dim=1)
            train_correct += pred.eq(target).sum().item()
            train_total += target.size(0)
            
            if batch_idx % 50 == 0:
                print(f'  Epoch {epoch+1} Batch {batch_idx:3d} | Loss: {loss.item():.4f}')
        
        # Validate
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                pred = output.argmax(dim=1)
                val_correct += pred.eq(target).sum().item()
                val_total += target.size(0)
        
        # Statistics
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        print(f"Epoch {epoch+1:2d}: Train Acc: {train_acc:5.1f}% | Val Acc: {val_acc:5.1f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            Path('model_checkpoints').mkdir(exist_ok=True)
            torch.save(model.state_dict(), 'model_checkpoints/best_simple_model.pth')
            print(f"  💾 New best model saved! (Val Acc: {val_acc:.1f}%)")
    
    print(f"\\n✅ Training complete! Best validation accuracy: {best_val_acc:.1f}%")
    
    # Test some predictions
    print("\\n🔮 Testing predictions...")
    model.eval()
    with torch.no_grad():
        # Get a few samples
        for i in range(3):
            x, y_true = val_dataset[i]
            x = x.unsqueeze(0).to(device)
            
            output = model(x)
            y_pred = output.argmax(dim=1).item()
            confidence = torch.softmax(output, dim=1).max().item()
            
            print(f"Sample {i+1}: True={y_true}, Pred={y_pred}, Confidence={confidence:.3f}")

if __name__ == "__main__":
    train_model()
