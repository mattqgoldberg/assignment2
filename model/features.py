#!/usr/bin/env python3
"""
Working feature engineering for MIDI melody-to-chord progression prediction.
"""

import numpy as np
import pickle
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from collections import Counter

def create_melody_chord_features():
    """Create features for melody-to-chord prediction."""
    print("🎼 Starting Melody-to-Chord Feature Engineering")
    
    # Load processed data
    print("Loading processed data...")
    with open('processed_data/processed_data.pkl', 'rb') as f:
        data = pickle.load(f)
    print(f"Loaded {len(data)} files with melody-chord pairs")
    
    # Parameters
    max_sequence_length = 32
    num_features = 4  # pitch, duration, interval, rhythm_position
    
    # Extract sequences
    print("Extracting melody sequences and chord labels...")
    melody_sequences = []
    chord_labels = []
    
    files_processed = 0
    total_sequences = 0
    
    for file_data in data:
        aligned_pairs = file_data.get('aligned_pairs', [])
        if not aligned_pairs:
            continue
            
        # Create overlapping sequences from each file
        step_size = max_sequence_length // 4  # 25% overlap
        for i in range(0, len(aligned_pairs) - max_sequence_length + 1, step_size):
            sequence = aligned_pairs[i:i + max_sequence_length]
            
            # Extract melody features
            melody_features = np.zeros((max_sequence_length, num_features))
            
            pitches = []
            for j, pair in enumerate(sequence):
                pitch = pair.get('melody_pitch', 60)
                duration = pair.get('melody_duration', 0.25)
                offset = pair.get('melody_offset', 0.0)
                
                pitches.append(pitch)
                
                # Feature extraction
                melody_features[j, 0] = pitch / 127.0  # Normalized pitch
                melody_features[j, 1] = min(duration, 4.0) / 4.0  # Normalized duration
                melody_features[j, 3] = (offset % 4.0) / 4.0  # Rhythmic position
                
                # Interval (difference from previous pitch)
                if j > 0:
                    interval = pitch - pitches[j-1]
                    melody_features[j, 2] = np.clip(interval, -12, 12) / 12.0
                
            # Target chord (last chord in sequence)
            target_pair = sequence[-1]
            root = target_pair.get('chord_root', 'C')
            quality = target_pair.get('chord_quality', 'major')
            chord_label = f"{root}_{quality}"
            
            melody_sequences.append(melody_features)
            chord_labels.append(chord_label)
            total_sequences += 1
            
        files_processed += 1
        if files_processed % 10 == 0:
            print(f"  Processed {files_processed} files, extracted {total_sequences} sequences")
    
    print(f"Extracted {len(melody_sequences)} sequences from {files_processed} files")
    
    # Analyze dataset
    print("\\nDataset Analysis:")
    chord_counts = Counter(chord_labels)
    print(f"Total sequences: {len(chord_labels)}")
    print(f"Unique chords: {len(chord_counts)}")
    
    print("\\nTop 15 most common chords:")
    for chord, count in chord_counts.most_common(15):
        pct = (count / len(chord_labels)) * 100
        print(f"  {chord:25} {count:>6} ({pct:.1f}%)")
    
    # Encode features
    print("\\nEncoding features...")
    X = np.array(melody_sequences)
    
    chord_encoder = LabelEncoder()
    y = chord_encoder.fit_transform(chord_labels)
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Label vector shape: {y.shape}")
    print(f"Number of unique chord classes: {len(chord_encoder.classes_)}")
    
    # Save features
    print("\\nSaving features...")
    output_dir = Path('processed_data')
    
    # Save feature matrices
    np.savez(output_dir / 'features.npz', X=X, y=y)
    print(f"Saved features to {output_dir / 'features.npz'}")
    
    # Save encoders and metadata
    encoders = {
        'chord_encoder': chord_encoder,
        'max_sequence_length': max_sequence_length,
        'num_features': num_features,
        'chord_classes': chord_encoder.classes_.tolist(),
        'dataset_stats': {
            'total_sequences': len(chord_labels),
            'num_files': files_processed,
            'unique_chords': len(chord_counts),
            'top_chords': chord_counts.most_common(10)
        }
    }
    
    with open(output_dir / 'encoders.pkl', 'wb') as f:
        pickle.dump(encoders, f)
    print(f"Saved encoders to {output_dir / 'encoders.pkl'}")
    
    print("\\n✅ Feature engineering complete!")
    return X, y, chord_encoder

if __name__ == "__main__":
    X, y, encoder = create_melody_chord_features()
