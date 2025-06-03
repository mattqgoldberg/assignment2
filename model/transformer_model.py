#!/usr/bin/env python3
"""
Transformer-based model for MIDI melody-to-chord progression prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from pathlib import Path
from typing import Tuple, Optional
import math

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer sequences."""
    
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


class MelodyFeatureEmbedding(nn.Module):
    """Embedding layer for melody features."""
    
    def __init__(self, num_features: int, d_model: int):
        super().__init__()
        self.num_features = num_features
        self.d_model = d_model
        
        # Linear projection for each feature
        self.feature_projections = nn.ModuleList([
            nn.Linear(1, d_model // num_features) 
            for _ in range(num_features)
        ])
        
        # Final projection to ensure correct dimension
        self.final_projection = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, num_features)
        batch_size, seq_len, _ = x.shape
        
        # Project each feature separately
        feature_embeddings = []
        for i in range(self.num_features):
            feature = x[:, :, i:i+1]  # (batch_size, seq_len, 1)
            embedded = self.feature_projections[i](feature)  # (batch_size, seq_len, d_model//num_features)
            feature_embeddings.append(embedded)
            
        # Concatenate all feature embeddings
        combined = torch.cat(feature_embeddings, dim=-1)  # (batch_size, seq_len, d_model)
        
        # Final projection and normalization
        output = self.final_projection(combined)
        output = self.layer_norm(output)
        
        return output


class MelodyToChordTransformer(nn.Module):
    """Transformer model for melody-to-chord progression prediction."""
    
    def __init__(
        self,
        num_features: int = 4,
        num_chords: int = 100,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_seq_length: int = 32
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_chords = num_chords
        self.max_seq_length = max_seq_length
        
        # Feature embedding
        self.feature_embedding = MelodyFeatureEmbedding(num_features, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_encoder_layers
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim_feedforward // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, num_chords)
        )
        
        # Pooling strategy
        self.pooling = "last"  # Options: "last", "mean", "attention"
        if self.pooling == "attention":
            self.attention_pool = nn.Linear(d_model, 1)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x shape: (batch_size, seq_len, num_features)
        seq_len = x.size(1)
        
        # Feature embedding
        embedded = self.feature_embedding(x)  # (batch_size, seq_len, d_model)
        
        # Add positional encoding
        embedded = embedded.transpose(0, 1)  # (seq_len, batch_size, d_model)
        embedded = self.pos_encoding(embedded)
        embedded = embedded.transpose(0, 1)  # (batch_size, seq_len, d_model)
        
        # Transformer encoding
        encoded = self.transformer_encoder(embedded, src_key_padding_mask=mask)
        
        # Pooling
        if self.pooling == "last":
            # Use the last time step
            pooled = encoded[:, -1, :]  # (batch_size, d_model)
        elif self.pooling == "mean":
            # Average pooling
            if mask is not None:
                # Handle padding
                mask_expanded = mask.unsqueeze(-1).expand_as(encoded)
                encoded_masked = encoded.masked_fill(mask_expanded, 0)
                pooled = encoded_masked.sum(dim=1) / (~mask).sum(dim=1, keepdim=True).float()
            else:
                pooled = encoded.mean(dim=1)
        elif self.pooling == "attention":
            # Attention pooling
            attention_weights = torch.softmax(self.attention_pool(encoded), dim=1)
            pooled = (encoded * attention_weights).sum(dim=1)
        
        # Classification
        logits = self.classifier(pooled)  # (batch_size, num_chords)
        
        return logits
    
    def predict_chord(self, melody_sequence: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """Predict chord for a single melody sequence."""
        self.eval()
        with torch.no_grad():
            if melody_sequence.dim() == 2:
                melody_sequence = melody_sequence.unsqueeze(0)  # Add batch dimension
                
            logits = self.forward(melody_sequence)
            probabilities = F.softmax(logits, dim=-1)
            predicted_chord = torch.argmax(logits, dim=-1)
            
            return predicted_chord.item(), probabilities.squeeze()


class MelodyChordDataset(torch.utils.data.Dataset):
    """Dataset class for melody-chord pairs."""
    
    def __init__(self, features_path: str, train: bool = True, train_split: float = 0.8):
        # Load features
        data = np.load(features_path)
        self.X = data['X'].astype(np.float32)  # (num_samples, seq_len, num_features)
        self.y = data['y'].astype(np.int64)    # (num_samples,)
        
        # Train/test split
        num_samples = len(self.X)
        split_idx = int(num_samples * train_split)
        
        if train:
            self.X = self.X[:split_idx]
            self.y = self.y[:split_idx]
        else:
            self.X = self.X[split_idx:]
            self.y = self.y[split_idx:]
            
        print(f"{'Train' if train else 'Test'} dataset: {len(self.X)} samples")
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx])


def create_model(num_chords: int) -> MelodyToChordTransformer:
    """Create and return a model instance."""
    model = MelodyToChordTransformer(
        num_features=4,
        num_chords=num_chords,
        d_model=256,
        nhead=8,
        num_encoder_layers=4,  # Reduced for faster training
        dim_feedforward=512,   # Reduced for faster training
        dropout=0.1,
        max_seq_length=32
    )
    return model


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    """Test model creation and forward pass."""
    print("🤖 Testing Melody-to-Chord Transformer Model")
    
    # Load metadata to get number of chords
    with open('processed_data/metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    
    num_chords = len(metadata['chord_classes'])
    print(f"Number of chord classes: {num_chords}")
    
    # Create model
    model = create_model(num_chords)
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Test forward pass
    batch_size = 4
    seq_len = 32
    num_features = 4
    
    # Create dummy input
    x = torch.randn(batch_size, seq_len, num_features)
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(x)
        print(f"Output shape: {output.shape}")
        print(f"Output sample: {output[0][:10]}")
        
        # Test prediction
        predicted_chord, probabilities = model.predict_chord(x[0])
        print(f"Predicted chord class: {predicted_chord}")
        print(f"Top 5 probabilities: {torch.topk(probabilities, 5)}")
    
    print("✅ Model test successful!")


if __name__ == "__main__":
    main()
