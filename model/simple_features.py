#!/usr/bin/env python3
"""
Simplified feature engineering for testing.
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from collections import Counter
from sklearn.preprocessing import LabelEncoder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Feature engineering pipeline for melody-to-chord progression prediction."""
    
    def __init__(self, vocab_size: int = 128, max_sequence_length: int = 32):
        """Initialize feature engineer."""
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.chord_encoder = LabelEncoder()
        
    def load_processed_data(self, data_path: str) -> List[Dict]:
        """Load preprocessed melody-chord pairs."""
        logger.info(f"Loading processed data from {data_path}")
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        logger.info(f"Loaded {len(data)} melody-chord pairs")
        return data
        
    def create_simple_features(self, data: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Create simple features for initial testing."""
        logger.info("Creating simple features...")
        
        # Extract basic features
        melody_sequences = []
        chord_labels = []
        chord_strings = []
        
        for pair in data[:1000]:  # Limit for testing
            try:
                # Get melody notes
                melody_notes = pair.get('melody_notes', [])
                if not melody_notes:
                    continue
                    
                # Simple melody features: just pitch sequence
                pitches = [note.get('pitch', 60) for note in melody_notes[:self.max_sequence_length]]
                
                # Pad or truncate to fixed length
                melody_seq = np.zeros(self.max_sequence_length)
                seq_len = min(len(pitches), self.max_sequence_length)
                melody_seq[:seq_len] = pitches[:seq_len]
                
                # Normalize to 0-1 range
                melody_seq = melody_seq / 127.0
                
                # Get chord info
                chord_info = pair.get('chord_info', {})
                root_pitch = chord_info.get('root_pitch', 60) % 12  # Reduce to pitch class
                quality = chord_info.get('quality', 'unknown')
                chord_string = f"{root_pitch}_{quality}"
                
                melody_sequences.append(melody_seq)
                chord_strings.append(chord_string)
                
            except Exception as e:
                logger.warning(f"Error processing pair: {e}")
                continue
                
        if not chord_strings:
            logger.error("No valid chord strings found!")
            return np.array([]), np.array([])
            
        # Fit chord encoder
        logger.info(f"Fitting encoder with {len(set(chord_strings))} unique chords")
        self.chord_encoder.fit(chord_strings)
        
        # Encode chord labels
        chord_labels = self.chord_encoder.transform(chord_strings)
        
        logger.info(f"Created {len(melody_sequences)} sequences")
        logger.info(f"Unique chords: {len(self.chord_encoder.classes_)}")
        
        return np.array(melody_sequences), np.array(chord_labels)
        
    def save_features(self, X, y, output_dir: str):
        """Save features and encoders."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save features
        np.savez(output_path / 'features.npz', X=X, y=y)
        
        # Save encoders
        encoders = {
            'chord_encoder': self.chord_encoder,
            'vocab_size': self.vocab_size,
            'max_sequence_length': self.max_sequence_length
        }
        
        with open(output_path / 'encoders.pkl', 'wb') as f:
            pickle.dump(encoders, f)
            
        logger.info(f"Saved features and encoders to {output_path}")


def main():
    """Main function for testing."""
    print("Starting feature engineering...")
    
    # Initialize feature engineer
    feature_engineer = FeatureEngineer(max_sequence_length=32)
    
    # Load processed data
    data = feature_engineer.load_processed_data('processed_data/processed_data.pkl')
    
    # Create features
    X, y = feature_engineer.create_simple_features(data)
    
    if len(X) > 0:
        print(f"Created feature matrices:")
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape}")
        print(f"  Unique chord labels: {len(np.unique(y))}")
        
        # Save features
        feature_engineer.save_features(X, y, 'processed_data/')
        print("Features saved successfully!")
    else:
        print("No features created!")


if __name__ == "__main__":
    main()
