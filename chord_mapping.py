#!/usr/bin/env python3
"""
Chord mapping utilities for converting model predictions to chord names and audio.
"""

import json
import pickle
from pathlib import Path

class ChordMapper:
    """Maps model output indices to chord names and provides audio-friendly formats."""
    
    def __init__(self, model_checkpoints_dir="model_checkpoints", processed_data_dir="processed_data"):
        self.model_checkpoints_dir = model_checkpoints_dir
        self.processed_data_dir = processed_data_dir
        self._load_mappings()
    
    def _load_mappings(self):
        """Load chord mappings from training results and metadata."""
        # Load class mapping from training results
        training_results_path = Path(self.model_checkpoints_dir) / "training_results.json"
        with open(training_results_path, 'r') as f:
            data = json.load(f)
            class_map = data['class_map']
        
        # Load chord names from metadata
        metadata_path = Path(self.processed_data_dir) / "metadata.pkl"
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
            chord_classes = metadata['chord_classes']
        
        # Create reverse mapping (model_index -> original_index -> chord_name)
        self.reverse_map = {v: int(k) for k, v in class_map.items()}
        
        # Create the final mapping: model_index -> chord_name
        self.index_to_chord = {}
        for model_idx in range(30):
            original_idx = self.reverse_map[model_idx]
            chord_name = chord_classes[original_idx]
            self.index_to_chord[model_idx] = chord_name
    
    def get_chord_name(self, model_index):
        """Get chord name from model output index."""
        return self.index_to_chord.get(model_index, f"Unknown_{model_index}")
    
    def get_chord_name_simple(self, model_index):
        """Get simplified chord name suitable for audio playback."""
        full_name = self.get_chord_name(model_index)
        
        # Parse the chord name: format is typically "ROOT_TYPE" or "ROOT_TYPE_EXTENSION"
        parts = full_name.split('_')
        if len(parts) >= 2:
            root = parts[0]
            chord_type = parts[1]
            
            # Convert flats (-)
            if '-' in root:
                root = root.replace('-', 'b')
            
            # Simplify chord types for audio playback
            if chord_type == 'major':
                return root  # Just the root for major chords
            elif chord_type == 'minor':
                return f"{root}m"
            elif chord_type == 'power':
                return f"{root}5"  # Power chord
            elif chord_type in ['3rd', 'perfect']:
                # For intervals, just return the root
                return root
            else:
                # For unknown or other types, return root
                return root
        
        return full_name  # Fallback
    
    def get_all_mappings(self):
        """Get all mappings as a dictionary."""
        return self.index_to_chord.copy()
    
    def print_mappings(self):
        """Print all chord mappings."""
        print("Model Index -> Chord Name")
        print("=" * 40)
        for idx in range(30):
            full_name = self.get_chord_name(idx)
            simple_name = self.get_chord_name_simple(idx)
            print(f"{idx:2d} -> {full_name:<20} (audio: {simple_name})")

# Convenience functions
def get_chord_name(model_index):
    """Convenience function to get chord name."""
    mapper = ChordMapper()
    return mapper.get_chord_name(model_index)

def get_chord_name_simple(model_index):
    """Convenience function to get simplified chord name for audio."""
    mapper = ChordMapper()
    return mapper.get_chord_name_simple(model_index)

if __name__ == "__main__":
    # Test the mapper
    mapper = ChordMapper()
    mapper.print_mappings()
