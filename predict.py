#!/usr/bin/env python3
"""
Inference script for the MIDI Melody-to-Chord Transformer model.
Load a trained model and predict chord progressions from melody sequences.
"""

import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
import math

class SimpleTransformer(nn.Module):
    """Simple Transformer model matching the training architecture."""
    
    def __init__(self, input_dim=4, d_model=128, num_heads=8, num_layers=3, 
                 num_classes=30, max_seq_len=32, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Positional encoding (learnable)
        self.pos_encoding = nn.Parameter(torch.randn(max_seq_len, d_model))
        self.dropout = nn.Dropout(dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
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

class ChordPredictor:
    """Class for loading and using trained chord prediction models."""
    
    def __init__(self, model_path='model_checkpoints/best_model.pth'):
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.model = None
        self.class_map = None
        self.reverse_class_map = None
        
        if Path(model_path).exists():
            self.load_model(model_path)
        else:
            print(f"⚠️  Model file {model_path} not found. Train a model first.")
    
    def load_model(self, model_path):
        """Load a trained model from checkpoint."""
        print(f"📦 Loading model from {model_path}...")
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Try to load training results for model config
            results_path = Path(model_path).parent / 'training_results.json'
            if results_path.exists():
                with open(results_path, 'r') as f:
                    results = json.load(f)
                    config = results.get('model_config', {})
                    self.class_map = results.get('class_map', {})
            else:
                # Default config
                config = {
                    'input_dim': 4,
                    'd_model': 128,
                    'num_classes': 30,
                    'num_heads': 8,
                    'num_layers': 3,
                    'dropout': 0.1
                }
                self.class_map = {}
            
            # Create model with correct architecture
            self.model = SimpleTransformer(
                input_dim=config.get('input_dim', 4),
                d_model=config.get('d_model', 128),
                num_heads=config.get('num_heads', 8),  
                num_layers=config.get('num_layers', 3),
                num_classes=config.get('num_classes', 30),
                dropout=config.get('dropout', 0.1)
            )
            
            # Load weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            # Create reverse class mapping for predictions
            if self.class_map:
                self.reverse_class_map = {v: k for k, v in self.class_map.items()}
            
            val_acc = checkpoint.get('val_acc', 'unknown')
            print(f"✅ Model loaded successfully! Validation accuracy: {val_acc:.1f}%")
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            self.model = None
    
    def predict(self, melody_sequence):
        """
        Predict chord progression for a melody sequence.
        
        Args:
            melody_sequence: numpy array of shape (seq_len, input_dim) or (1, seq_len, input_dim)
            
        Returns:
            Predicted chord class and confidence
        """
        if self.model is None:
            return None, 0.0
        
        # Ensure correct shape
        if melody_sequence.ndim == 2:
            melody_sequence = melody_sequence[np.newaxis, :]  # Add batch dimension
        
        # Convert to tensor
        x = torch.FloatTensor(melody_sequence).to(self.device)
        
        with torch.no_grad():
            logits = self.model(x)
            probabilities = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(logits, dim=-1).item()
            confidence = probabilities[0, predicted_class].item()
        
        # Map back to original class if possible
        if self.reverse_class_map and predicted_class in self.reverse_class_map:
            original_class = self.reverse_class_map[predicted_class]
        else:
            original_class = predicted_class
        
        return original_class, confidence
    
    def predict_sequence(self, melody_sequences):
        """
        Predict chord progressions for multiple melody sequences.
        
        Args:
            melody_sequences: numpy array of shape (batch_size, seq_len, input_dim)
            
        Returns:
            List of (predicted_class, confidence) tuples
        """
        if self.model is None:
            return []
        
        x = torch.FloatTensor(melody_sequences).to(self.device)
        
        with torch.no_grad():
            logits = self.model(x)
            probabilities = torch.softmax(logits, dim=-1)
            predicted_classes = torch.argmax(logits, dim=-1).cpu().numpy()
            confidences = torch.max(probabilities, dim=-1)[0].cpu().numpy()
        
        results = []
        for pred_class, conf in zip(predicted_classes, confidences):
            if self.reverse_class_map and pred_class in self.reverse_class_map:
                original_class = self.reverse_class_map[pred_class]
            else:
                original_class = pred_class
            results.append((original_class, conf))
        
        return results

def demo():
    """Demonstration of the chord predictor."""
    print("🎵 MIDI Chord Prediction Demo")
    print("=" * 40)
    
    # Initialize predictor
    predictor = ChordPredictor()
    
    if predictor.model is None:
        print("❌ No trained model available. Please train a model first.")
        return
    
    print(f"📊 Model loaded with {predictor.model.classifier[-1].out_features} chord classes")
    
    # Load some test data
    try:
        data = np.load('processed_data/features_full.npz')
        X_test = data['X'][:10]  # Use first 10 sequences for demo
        
        print(f"\n🎹 Predicting chords for {len(X_test)} melody sequences...")
        
        results = predictor.predict_sequence(X_test)
        
        print(f"\n📋 Predictions:")
        for i, (chord_class, confidence) in enumerate(results):
            print(f"  Sequence {i+1}: Chord {chord_class} (confidence: {confidence:.3f})")
        
        # Show average confidence
        avg_confidence = np.mean([conf for _, conf in results])
        print(f"\n📈 Average confidence: {avg_confidence:.3f}")
        
    except FileNotFoundError:
        print("❌ No test data found. Please run the data pipeline first.")
    except Exception as e:
        print(f"❌ Error during prediction: {e}")

if __name__ == "__main__":
    demo()
