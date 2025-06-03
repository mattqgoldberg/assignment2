#!/usr/bin/env python3
"""
Feature engineering for MIDI melody-to-chord progression prediction.
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

class MelodyChordFeatureEngineer:
    """Feature engineering pipeline for melody-to-chord progression prediction."""
    
    def __init__(self, max_sequence_length: int = 32):
        """Initialize feature engineer."""
        self.max_sequence_length = max_sequence_length
        self.chord_encoder = LabelEncoder()
        self.root_encoder = LabelEncoder()
        self.quality_encoder = LabelEncoder()
        
    def load_processed_data(self, data_path: str) -> List[Dict]:
        """Load preprocessed melody-chord pairs."""
        logger.info(f"Loading processed data from {data_path}")
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        logger.info(f"Loaded {len(data)} files with melody-chord pairs")
        return data
        
    def extract_sequences(self, data: List[Dict], limit: int = None) -> Tuple[List, List]:
        """Extract melody sequences and chord labels from processed data."""
        logger.info("Extracting sequences from aligned pairs...")
        
        melody_sequences = []
        chord_labels = []
        
        files_processed = 0
        for file_data in data:
            if limit and files_processed >= limit:
                break
                
            aligned_pairs = file_data.get('aligned_pairs', [])
            if not aligned_pairs:
                continue
                
            # Create sequences of consecutive melody-chord pairs
            for i in range(0, len(aligned_pairs), self.max_sequence_length // 2):
                sequence = aligned_pairs[i:i + self.max_sequence_length]
                if len(sequence) < 4:  # Skip very short sequences
                    continue
                    
                # Extract melody features
                melody_features = self._extract_melody_sequence(sequence)
                
                # Extract target chord (last chord in sequence)
                target_chord = self._extract_chord_label(sequence[-1])
                
                melody_sequences.append(melody_features)
                chord_labels.append(target_chord)
                
            files_processed += 1
            
        logger.info(f"Extracted {len(melody_sequences)} sequences from {files_processed} files")
        return melody_sequences, chord_labels
        
    def _extract_melody_sequence(self, aligned_pairs: List[Dict]) -> np.ndarray:
        """Extract melody features from a sequence of aligned pairs."""
        # Feature channels: pitch, duration, interval, rhythm_position
        num_features = 4
        sequence = np.zeros((self.max_sequence_length, num_features))
        
        seq_len = min(len(aligned_pairs), self.max_sequence_length)
        
        pitches = []
        durations = []
        offsets = []
        
        for i in range(seq_len):
            pair = aligned_pairs[i]
            pitch = pair.get('melody_pitch', 60)
            duration = pair.get('melody_duration', 0.25)
            offset = pair.get('melody_offset', 0.0)
            
            pitches.append(pitch)
            durations.append(duration)
            offsets.append(offset)
            
        # Normalize features
        for i in range(seq_len):
            # Pitch (normalized to 0-1)
            sequence[i, 0] = pitches[i] / 127.0
            
            # Duration (clamped and normalized)
            sequence[i, 1] = min(durations[i], 4.0) / 4.0
            
            # Interval (difference from previous pitch)
            if i > 0:
                interval = pitches[i] - pitches[i-1]
                sequence[i, 2] = np.clip(interval, -12, 12) / 12.0
            else:
                sequence[i, 2] = 0.0
                
            # Rhythmic position (beat position within measure)
            sequence[i, 3] = (offsets[i] % 4.0) / 4.0
            
        return sequence
        
    def _extract_chord_label(self, aligned_pair: Dict) -> str:
        """Extract chord label from aligned pair."""
        root = aligned_pair.get('chord_root', 'C')
        quality = aligned_pair.get('chord_quality', 'major')
        return f"{root}_{quality}"
        
    def encode_features(self, melody_sequences: List, chord_labels: List) -> Tuple[np.ndarray, np.ndarray]:
        """Encode features for machine learning."""
        logger.info("Encoding features...")
        
        # Convert melody sequences to numpy array
        X = np.array(melody_sequences)
        
        # Fit and transform chord labels
        self.chord_encoder.fit(chord_labels)
        y = self.chord_encoder.transform(chord_labels)
        
        # Also fit individual encoders for analysis
        roots = [label.split('_')[0] for label in chord_labels]
        qualities = ['_'.join(label.split('_')[1:]) for label in chord_labels]
        
        self.root_encoder.fit(roots)
        self.quality_encoder.fit(qualities)
        
        logger.info(f"Feature shapes: X={X.shape}, y={y.shape}")
        logger.info(f"Unique chords: {len(self.chord_encoder.classes_)}")
        logger.info(f"Unique roots: {len(self.root_encoder.classes_)}")
        logger.info(f"Unique qualities: {len(self.quality_encoder.classes_)}")
        
        return X, y
        
    def save_features(self, X: np.ndarray, y: np.ndarray, output_dir: str):
        """Save features and encoders."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save features
        np.savez(output_path / 'features.npz', X=X, y=y)
        logger.info(f"Saved features to {output_path / 'features.npz'}")
        
        # Save encoders
        encoders = {
            'chord_encoder': self.chord_encoder,
            'root_encoder': self.root_encoder,
            'quality_encoder': self.quality_encoder,
            'max_sequence_length': self.max_sequence_length,
            'num_features': 4
        }
        
        with open(output_path / 'encoders.pkl', 'wb') as f:
            pickle.dump(encoders, f)
        logger.info(f"Saved encoders to {output_path / 'encoders.pkl'}")
        
    def analyze_dataset(self, chord_labels: List[str]):
        """Analyze the dataset distribution."""
        logger.info("Dataset Analysis:")
        
        # Chord distribution
        chord_counts = Counter(chord_labels)
        logger.info(f"Total chord instances: {len(chord_labels)}")
        logger.info(f"Unique chords: {len(chord_counts)}")
        
        logger.info("Top 10 most common chords:")
        for chord, count in chord_counts.most_common(10):
            pct = (count / len(chord_labels)) * 100
            logger.info(f"  {chord:20} {count:>6} ({pct:.1f}%)")
            
        # Root distribution
        roots = [label.split('_')[0] for label in chord_labels]
        root_counts = Counter(roots)
        logger.info(f"\nTop 10 most common roots:")
        for root, count in root_counts.most_common(10):
            pct = (count / len(roots)) * 100
            logger.info(f"  {root:10} {count:>6} ({pct:.1f}%)")
            
        # Quality distribution
        qualities = ['_'.join(label.split('_')[1:]) for label in chord_labels]
        quality_counts = Counter(qualities)
        logger.info(f"\nTop 10 most common qualities:")
        for quality, count in quality_counts.most_common(10):
            pct = (count / len(qualities)) * 100
            logger.info(f"  {quality:20} {count:>6} ({pct:.1f}%)")


def main():
    """Main function for feature engineering."""
    logger.info("🎼 Starting Melody-to-Chord Feature Engineering")
    
    # Initialize feature engineer
    fe = MelodyChordFeatureEngineer(max_sequence_length=32)
    
    # Load processed data
    data = fe.load_processed_data('processed_data/processed_data.pkl')
    
    # Extract sequences (limit to reasonable size for initial testing)
    melody_sequences, chord_labels = fe.extract_sequences(data, limit=50)
    
    if not melody_sequences:
        logger.error("No sequences extracted!")
        return
        
    # Analyze dataset
    fe.analyze_dataset(chord_labels)
    
    # Encode features
    X, y = fe.encode_features(melody_sequences, chord_labels)
    
    # Save features
    fe.save_features(X, y, 'processed_data/')
    
    logger.info("✅ Feature engineering complete!")
    logger.info(f"📊 Generated {len(X)} training examples")
    logger.info(f"🎯 {len(np.unique(y))} unique chord classes")


if __name__ == "__main__":
    main()
