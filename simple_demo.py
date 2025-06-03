#!/usr/bin/env python3
"""
Simple demonstration of the MIDI melody-to-chord prediction system.
This script loads the trained model and makes predictions on sample melodies.
"""

import torch
import numpy as np
import json
from pathlib import Path

# Import the model from predict.py
import sys
sys.path.append('.')
from predict import ChordPredictor

def create_sample_melodies():
    """Create some sample melodies for demonstration."""
    samples = {}
    
    # C Major Scale melody (ascending)
    samples["C Major Scale"] = np.array([
        [60, 1, 0.5, 0.0],  # C4
        [62, 1, 0.5, 0.5],  # D4
        [64, 1, 0.5, 1.0],  # E4
        [65, 1, 0.5, 1.5],  # F4
        [67, 1, 0.5, 2.0],  # G4
        [69, 1, 0.5, 2.5],  # A4
        [71, 1, 0.5, 3.0],  # B4
        [72, 1, 1.0, 3.5],  # C5
    ])
    
    # Simple arpeggio (C major chord)
    samples["C Major Arpeggio"] = np.array([
        [60, 1, 0.5, 0.0],  # C4
        [64, 1, 0.5, 0.5],  # E4
        [67, 1, 0.5, 1.0],  # G4
        [72, 1, 1.0, 1.5],  # C5
        [67, 1, 0.5, 2.5],  # G4
        [64, 1, 0.5, 3.0],  # E4
        [60, 1, 1.0, 3.5],  # C4
    ])
    
    # Simple blues-style melody
    samples["Blues Melody"] = np.array([
        [60, 1, 0.5, 0.0],  # C4
        [63, 1, 0.5, 0.5],  # Eb4
        [65, 1, 0.5, 1.0],  # F4
        [66, 1, 0.5, 1.5],  # F#4
        [67, 1, 1.0, 2.0],  # G4
        [65, 1, 0.5, 3.0],  # F4
        [63, 1, 0.5, 3.5],  # Eb4
        [60, 1, 1.0, 4.0],  # C4
    ])
    
    # A minor melody
    samples["A Minor Melody"] = np.array([
        [57, 1, 0.5, 0.0],  # A3
        [60, 1, 0.5, 0.5],  # C4
        [62, 1, 0.5, 1.0],  # D4
        [64, 1, 0.5, 1.5],  # E4
        [65, 1, 0.5, 2.0],  # F4
        [67, 1, 0.5, 2.5],  # G4
        [69, 1, 1.0, 3.0],  # A4
    ])
    
    # Jazz-style melody
    samples["Jazz Melody"] = np.array([
        [60, 1, 0.25, 0.0],   # C4
        [64, 1, 0.25, 0.25],  # E4
        [67, 1, 0.5, 0.5],    # G4
        [71, 1, 0.25, 1.0],   # B4
        [72, 1, 0.75, 1.25],  # C5
        [69, 1, 0.25, 2.0],   # A4
        [65, 1, 0.5, 2.25],   # F4
        [62, 1, 0.75, 2.75],  # D4
    ])
    
    return samples

def midi_note_to_name(midi_note):
    """Convert MIDI note number to note name."""
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (midi_note // 12) - 1
    note = note_names[midi_note % 12]
    return f"{note}{octave}"

def display_melody(melody, name):
    """Display a melody in a readable format."""
    print(f"\n{name}:")
    print("-" * 40)
    for i, note in enumerate(melody):
        pitch, velocity, duration, onset = note
        note_name = midi_note_to_name(int(pitch))
        print(f"  Note {i+1}: {note_name} (MIDI {int(pitch)}) - Duration: {duration:.2f}s")

def main():
    """Main demonstration function."""
    print("🎵 MIDI Melody-to-Chord Prediction System Demo 🎵")
    print("=" * 60)
    
    # Check if model files exist
    model_path = Path("model_checkpoints/best_model.pth")
    results_path = Path("model_checkpoints/training_results.json")
    
    if not model_path.exists() or not results_path.exists():
        print("❌ Error: Model files not found!")
        print("Make sure you have:")
        print("  - model_checkpoints/best_model.pth")
        print("  - model_checkpoints/training_results.json")
        return
    
    print("✅ Loading trained model...")
    
    # Initialize the chord predictor
    predictor = ChordPredictor()
    
    if predictor.model is None:
        print("❌ Failed to load model!")
        return
    
    # Load the model configuration and class mapping
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    print(f"📊 Model achieved {results['best_val_acc']:.1f}% validation accuracy")
    
    # Create sample melodies
    samples = create_sample_melodies()
    
    print(f"\n🎼 Created {len(samples)} sample melodies for demonstration:")
    for name in samples.keys():
        print(f"  • {name}")
    
    print("\n" + "="*60)
    print("Making Chord Predictions...")
    print("="*60)
    
    # Make predictions for each sample
    for name, melody in samples.items():
        display_melody(melody, name)
        
        try:
            # Get prediction using the ChordPredictor
            predicted_chord, confidence = predictor.predict(melody)
            print(f"  🎹 Predicted Chord: {predicted_chord}")
            print(f"  📈 Confidence: {confidence:.1f}%")
            
        except Exception as e:
            print(f"  ❌ Error predicting chord: {e}")
        
        print()
    
    print("="*60)
    print("Demo completed! 🎉")
    print("\nTo make predictions on your own melodies:")
    print("  python predict.py --melody your_melody.mid")

if __name__ == "__main__":
    main()
