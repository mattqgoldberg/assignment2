#!/usr/bin/env python3
"""
Feature engineering module for MIDI melody-to-chord progression prediction.
Converts raw melody-chord pairs into features suitable for Transformer training.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from collections import defaultdict, Counter
from dataclasses import dataclass
from music21 import pitch, interval, key, meter
import torch
from sklearn.preprocessing import LabelEncoder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MelodyFeatures:
    """Container for melody features"""
    pitch_sequence: List[int]      # MIDI note numbers
    rhythm_sequence: List[float]   # Note durations in quarters 
    interval_sequence: List[int]   # Intervals between consecutive notes
    contour_sequence: List[int]    # Melodic contour (-1, 0, 1)
    scale_degrees: List[int]       # Scale degrees relative to key
    position_in_measure: List[float]  # Beat positions (0.0-4.0 typically)

@dataclass
class ChordFeatures:
    """Container for chord features"""
    root_pitch: int               # Root note as MIDI number
    chord_quality: str           # Chord quality (major, minor, etc.)
    bass_pitch: int              # Bass note as MIDI number
    chord_id: int                # Encoded chord identifier
    roman_numeral: str           # Roman numeral analysis
    function: str                # Harmonic function (tonic, dominant, etc.)

class FeatureEngineer:
    """
    Feature engineering pipeline for melody-to-chord progression prediction.
    """
    
    def __init__(self, vocab_size: int = 128, max_sequence_length: int = 32):
        """
        Initialize feature engineer.
        
        Args:
            vocab_size: Size of pitch vocabulary (typically 128 for MIDI)
            max_sequence_length: Maximum sequence length for model input
        """
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        
        # Initialize encoders
        self.chord_encoder = LabelEncoder()
        self.quality_encoder = LabelEncoder()
        self.function_encoder = LabelEncoder()
        
        # Pitch class to scale degree mappings for major/minor keys
        self.major_scale_degrees = {0: 1, 2: 2, 4: 3, 5: 4, 7: 5, 9: 6, 11: 7}
        self.minor_scale_degrees = {0: 1, 2: 2, 3: 3, 5: 4, 7: 5, 8: 6, 10: 7}
        
        # Roman numeral mappings
        self.major_roman_numerals = ['I', 'ii', 'iii', 'IV', 'V', 'vi', 'vii°']
        self.minor_roman_numerals = ['i', 'ii°', 'III', 'iv', 'v', 'VI', 'VII']
        
        # Harmonic function mappings
        self.harmonic_functions = {
            'I': 'tonic', 'i': 'tonic',
            'ii': 'subdominant', 'ii°': 'subdominant', 'IV': 'subdominant', 'iv': 'subdominant',
            'V': 'dominant', 'v': 'dominant', 'vii°': 'dominant', 'VII': 'dominant',
            'iii': 'mediant', 'III': 'mediant', 'vi': 'submediant', 'VI': 'submediant'
        }
        
    def load_processed_data(self, data_path: str) -> List[Dict]:
        """Load preprocessed melody-chord pairs."""
        logger.info(f"Loading processed data from {data_path}")
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        logger.info(f"Loaded {len(data)} melody-chord pairs")
        return data
        
    def extract_melody_features(self, melody_notes: List[Dict], key_sig: str) -> MelodyFeatures:
        """
        Extract comprehensive features from melody sequence.
        
        Args:
            melody_notes: List of note dictionaries with pitch, duration, offset
            key_sig: Key signature string (e.g., 'C major', 'a minor')
            
        Returns:
            MelodyFeatures object with extracted features
        """
        if not melody_notes:
            return MelodyFeatures([], [], [], [], [], [])
            
        pitches = []
        durations = []
        offsets = []
        
        for note in melody_notes:
            pitches.append(note['pitch'])
            durations.append(note['duration'])
            offsets.append(note['offset'])
            
        # Calculate intervals between consecutive notes
        intervals = []
        for i in range(1, len(pitches)):
            intervals.append(pitches[i] - pitches[i-1])
        intervals = [0] + intervals  # Pad first note with 0
        
        # Calculate melodic contour
        contour = []
        for interval_val in intervals:
            if interval_val > 0:
                contour.append(1)   # ascending
            elif interval_val < 0:
                contour.append(-1)  # descending
            else:
                contour.append(0)   # same
                
        # Calculate scale degrees relative to key
        scale_degrees = self._calculate_scale_degrees(pitches, key_sig)
        
        # Calculate position in measure (assuming 4/4 time for now)
        positions = [offset % 4.0 for offset in offsets]
        
        return MelodyFeatures(
            pitch_sequence=pitches,
            rhythm_sequence=durations,
            interval_sequence=intervals,
            contour_sequence=contour,
            scale_degrees=scale_degrees,
            position_in_measure=positions
        )
        
    def extract_chord_features(self, chord_info: Dict, key_sig: str) -> ChordFeatures:
        """
        Extract features from chord information.
        
        Args:
            chord_info: Dictionary with chord information
            key_sig: Key signature string
            
        Returns:
            ChordFeatures object with extracted features
        """
        root_pitch = chord_info.get('root_pitch', 60)  # Default to C4
        chord_quality = chord_info.get('quality', 'unknown')
        bass_pitch = chord_info.get('bass_pitch', root_pitch)
        
        # Generate Roman numeral analysis
        roman_numeral = self._get_roman_numeral(root_pitch, chord_quality, key_sig)
        
        # Determine harmonic function
        function = self.harmonic_functions.get(roman_numeral, 'other')
        
        return ChordFeatures(
            root_pitch=root_pitch,
            chord_quality=chord_quality,
            bass_pitch=bass_pitch,
            chord_id=0,  # Will be set during encoding
            roman_numeral=roman_numeral,
            function=function
        )
        
    def _calculate_scale_degrees(self, pitches: List[int], key_sig: str) -> List[int]:
        """Calculate scale degrees for pitches relative to key signature."""
        try:
            # Parse key signature
            key_parts = key_sig.split()
            if len(key_parts) >= 2:
                key_name = key_parts[0].upper()
                mode = key_parts[1].lower()
            else:
                return [1] * len(pitches)  # Default to tonic
                
            # Get tonic pitch class
            tonic_pitch = pitch.Pitch(key_name).pitchClass
            
            # Choose scale degree mapping
            if mode == 'major':
                scale_map = self.major_scale_degrees
            else:  # minor
                scale_map = self.minor_scale_degrees
                
            scale_degrees = []
            for p in pitches:
                pc = p % 12
                # Transpose to key
                relative_pc = (pc - tonic_pitch) % 12
                scale_degree = scale_map.get(relative_pc, 1)  # Default to tonic
                scale_degrees.append(scale_degree)
                
            return scale_degrees
            
        except Exception as e:
            logger.warning(f"Could not calculate scale degrees for key {key_sig}: {e}")
            return [1] * len(pitches)  # Default to all tonic
            
    def _get_roman_numeral(self, root_pitch: int, quality: str, key_sig: str) -> str:
        """Generate Roman numeral analysis for chord."""
        try:
            key_parts = key_sig.split()
            if len(key_parts) >= 2:
                key_name = key_parts[0].upper()
                mode = key_parts[1].lower()
            else:
                return 'I'
                
            # Get tonic and root pitch classes
            tonic_pitch = pitch.Pitch(key_name).pitchClass
            root_pc = root_pitch % 12
            
            # Calculate scale degree (1-based)
            scale_degree = ((root_pc - tonic_pitch) % 12)
            degree_mapping = {0: 0, 2: 1, 4: 2, 5: 3, 7: 4, 9: 5, 11: 6}
            degree_idx = degree_mapping.get(scale_degree, 0)
            
            # Choose Roman numeral based on mode
            if mode == 'major':
                roman = self.major_roman_numerals[degree_idx]
            else:
                roman = self.minor_roman_numerals[degree_idx]
                
            return roman
            
        except Exception as e:
            logger.warning(f"Could not generate Roman numeral: {e}")
            return 'I'
            
    def create_sequences(self, data: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create input sequences for model training.
        
        Args:
            data: List of melody-chord pair dictionaries
            
        Returns:
            Tuple of (melody_sequences, chord_labels) as numpy arrays
        """
        logger.info("Creating training sequences...")
        
        # Collect all chord types for encoding
        all_chords = []
        all_qualities = []
        all_functions = []
        
        processed_pairs = []
        
        for pair in data:
            try:
                melody_features = self.extract_melody_features(
                    pair['melody_notes'], 
                    pair.get('key_signature', 'C major')
                )
                
                chord_features = self.extract_chord_features(
                    pair['chord_info'],
                    pair.get('key_signature', 'C major')
                )
                
                if len(melody_features.pitch_sequence) > 0:
                    processed_pairs.append((melody_features, chord_features))
                    all_chords.append(f"{chord_features.root_pitch % 12}_{chord_features.chord_quality}")
                    all_qualities.append(chord_features.chord_quality)
                    all_functions.append(chord_features.function)
                    
            except Exception as e:
                logger.warning(f"Error processing pair: {e}")
                continue
                
        # Fit encoders
        logger.info("Fitting label encoders...")
        self.chord_encoder.fit(all_chords)
        self.quality_encoder.fit(all_qualities)
        self.function_encoder.fit(all_functions)
        
        # Create sequences
        melody_sequences = []
        chord_labels = []
        
        for melody_features, chord_features in processed_pairs:
            # Create melody sequence (multiple features concatenated)
            melody_seq = self._create_melody_sequence(melody_features)
            
            # Encode chord label
            chord_label = f"{chord_features.root_pitch % 12}_{chord_features.chord_quality}"
            chord_id = self.chord_encoder.transform([chord_label])[0]
            
            melody_sequences.append(melody_seq)
            chord_labels.append(chord_id)
            
        logger.info(f"Created {len(melody_sequences)} training sequences")
        logger.info(f"Vocabulary sizes - Chords: {len(self.chord_encoder.classes_)}, "
                   f"Qualities: {len(self.quality_encoder.classes_)}, "
                   f"Functions: {len(self.function_encoder.classes_)}")
        
        return np.array(melody_sequences), np.array(chord_labels)
        
    def _create_melody_sequence(self, melody_features: MelodyFeatures) -> np.ndarray:
        """
        Create a fixed-length melody sequence with multiple feature channels.
        
        Args:
            melody_features: MelodyFeatures object
            
        Returns:
            Numpy array of shape (max_sequence_length, num_features)
        """
        seq_len = min(len(melody_features.pitch_sequence), self.max_sequence_length)
        
        # Feature channels: pitch, duration, interval, contour, scale_degree, position
        num_features = 6
        sequence = np.zeros((self.max_sequence_length, num_features))
        
        if seq_len > 0:
            # Normalize features
            pitches = np.array(melody_features.pitch_sequence[:seq_len]) / 127.0  # Normalize MIDI
            durations = np.clip(melody_features.rhythm_sequence[:seq_len], 0, 4) / 4.0  # Normalize durations
            intervals = np.clip(melody_features.interval_sequence[:seq_len], -12, 12) / 12.0  # Normalize intervals
            contours = np.array(melody_features.contour_sequence[:seq_len])  # Already -1, 0, 1
            scale_degrees = (np.array(melody_features.scale_degrees[:seq_len]) - 1) / 6.0  # Normalize 1-7 to 0-1
            positions = np.array(melody_features.position_in_measure[:seq_len]) / 4.0  # Normalize positions
            
            # Fill sequence
            sequence[:seq_len, 0] = pitches
            sequence[:seq_len, 1] = durations
            sequence[:seq_len, 2] = intervals
            sequence[:seq_len, 3] = contours
            sequence[:seq_len, 4] = scale_degrees
            sequence[:seq_len, 5] = positions
            
        return sequence
        
    def save_encoders(self, output_dir: str):
        """Save fitted encoders for inference."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        encoders = {
            'chord_encoder': self.chord_encoder,
            'quality_encoder': self.quality_encoder,
            'function_encoder': self.function_encoder,
            'vocab_size': self.vocab_size,
            'max_sequence_length': self.max_sequence_length
        }
        
        with open(output_path / 'encoders.pkl', 'wb') as f:
            pickle.dump(encoders, f)
            
        logger.info(f"Saved encoders to {output_path / 'encoders.pkl'}")
        
    def load_encoders(self, encoders_path: str):
        """Load pre-fitted encoders."""
        with open(encoders_path, 'rb') as f:
            encoders = pickle.load(f)
            
        self.chord_encoder = encoders['chord_encoder']
        self.quality_encoder = encoders['quality_encoder']
        self.function_encoder = encoders['function_encoder']
        self.vocab_size = encoders['vocab_size']
        self.max_sequence_length = encoders['max_sequence_length']
        
        logger.info(f"Loaded encoders from {encoders_path}")


def main():
    """Main function for testing feature engineering."""
    # Initialize feature engineer
    feature_engineer = FeatureEngineer(max_sequence_length=32)
    
    # Load processed data
    data = feature_engineer.load_processed_data('processed_data/processed_data.pkl')
    
    # Create training sequences
    X, y = feature_engineer.create_sequences(data)
    
    print(f"Created feature matrices:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Unique chord labels: {len(np.unique(y))}")
    
    # Save encoders
    feature_engineer.save_encoders('processed_data/')
    
    # Save feature matrices
    np.savez('processed_data/features.npz', X=X, y=y)
    print("Saved features to processed_data/features.npz")


if __name__ == "__main__":
    main()
