#!/usr/bin/env python3
"""
Interactive demo script for MIDI chord predictions.
Shows real-time chord prediction examples with sample melodies.
"""

import numpy as np
import torch
from predict import ChordPredictor
import time
import json
import pickle
from pathlib import Path
from audio_player import AudioPlayer, play_melody, play_chord, play_melody_with_chord

def create_sample_melodies():
    """Create some sample melody sequences for demonstration."""
    melodies = []
    
    # 1. C Major Scale ascending
    c_major = np.zeros((32, 4))
    pitches = [60, 62, 64, 65, 67, 69, 71, 72]  # C-D-E-F-G-A-B-C
    for i, pitch in enumerate(pitches):
        c_major[i, 0] = pitch  # pitch
        c_major[i, 1] = 0.5    # duration
        if i > 0:
            c_major[i, 2] = pitch - pitches[i-1]  # interval
        c_major[i, 3] = i * 0.5  # rhythm position
    melodies.append(("C Major Scale", c_major))
    
    # 2. Simple C Major Arpeggio
    c_arp = np.zeros((32, 4))
    arp_pitches = [60, 64, 67, 72, 67, 64, 60]  # C-E-G-C-G-E-C
    for i, pitch in enumerate(arp_pitches):
        c_arp[i, 0] = pitch
        c_arp[i, 1] = 0.75
        if i > 0:
            c_arp[i, 2] = pitch - arp_pitches[i-1]
        c_arp[i, 3] = i * 0.75
    melodies.append(("C Major Arpeggio", c_arp))
    
    # 3. Blues-style melody
    blues = np.zeros((32, 4))
    blues_pitches = [60, 63, 65, 66, 67, 66, 65, 63, 60]  # C-Eb-F-Gb-G-Gb-F-Eb-C
    for i, pitch in enumerate(blues_pitches):
        blues[i, 0] = pitch
        blues[i, 1] = 0.33
        if i > 0:
            blues[i, 2] = pitch - blues_pitches[i-1]
        blues[i, 3] = i * 0.33
    melodies.append(("Blues Melody", blues))
    
    # 4. Minor scale melody
    a_minor = np.zeros((32, 4))
    minor_pitches = [57, 59, 60, 62, 64, 65, 67, 69]  # A-B-C-D-E-F-G-A
    for i, pitch in enumerate(minor_pitches):
        a_minor[i, 0] = pitch
        a_minor[i, 1] = 0.5
        if i > 0:
            a_minor[i, 2] = pitch - minor_pitches[i-1]
        a_minor[i, 3] = i * 0.5
    melodies.append(("A Minor Scale", a_minor))
    
    # 5. Jazz-style melody
    jazz = np.zeros((32, 4))
    jazz_pitches = [60, 64, 67, 70, 73, 70, 67, 65, 62, 60]  # C-E-G-Bb-C#-Bb-G-F-D-C
    for i, pitch in enumerate(jazz_pitches):
        jazz[i, 0] = pitch
        jazz[i, 1] = 0.25
        if i > 0:
            jazz[i, 2] = pitch - jazz_pitches[i-1]
        jazz[i, 3] = i * 0.25
    melodies.append(("Jazz Melody", jazz))
    
    return melodies

def print_melody_info(name, melody):
    """Print information about a melody."""
    print(f"\n🎵 {name}")
    print("-" * (len(name) + 4))
    
    # Extract notes (non-zero pitches)
    notes = []
    for i in range(len(melody)):
        if melody[i, 0] > 0:
            pitch = int(melody[i, 0])
            # Convert MIDI to note name
            note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            note_name = note_names[pitch % 12]
            octave = pitch // 12 - 1
            notes.append(f"{note_name}{octave}")
    
    print(f"Notes: {' - '.join(notes[:8])}{'...' if len(notes) > 8 else ''}")

def demo_single_prediction(predictor, melody_name, melody):
    """Demo a single melody prediction with details."""
    print_melody_info(melody_name, melody)
    
    # Make prediction
    chord_class, confidence = predictor.predict(melody)
    
    print(f"🎯 Predicted chord: Class {chord_class}")
    print(f"📊 Confidence: {confidence:.3f}")
    
    # Get top 5 predictions for more insight
    if predictor.model is not None:
        x = torch.FloatTensor(melody[np.newaxis, :]).to(predictor.device)
        with torch.no_grad():
            logits = predictor.model(x)
            probs = torch.softmax(logits, dim=-1)
            top5_probs, top5_indices = torch.topk(probs[0], 5)
        
        print(f"📈 Top 5 predictions:")
        for i, (prob, idx) in enumerate(zip(top5_probs, top5_indices)):
            print(f"   {i+1}. Class {idx.item()}: {prob.item():.3f}")

def interactive_demo():
    """Run an interactive demonstration."""
    print("🎵 MIDI Chord Prediction Demo")
    print("=" * 40)
    print("This demo shows how the AI predicts chords from melody patterns!")
    print()
    
    # Initialize predictor
    print("📦 Loading trained model...")
    predictor = ChordPredictor()
    
    if predictor.model is None:
        print("❌ No trained model found!")
        print("Please train a model first by running: python train.py")
        return
    
    print(f"✅ Model loaded successfully!")
    print(f"📊 Model can predict among {predictor.model.classifier[-1].out_features} chord classes")
    print(f"🎯 Model validation accuracy: 59.1%")
    print()
    
    # Create sample melodies
    sample_melodies = create_sample_melodies()
    
    # Demo each melody
    for i, (name, melody) in enumerate(sample_melodies):
        print(f"\n{'='*50}")
        print(f"Example {i+1}/{len(sample_melodies)}")
        demo_single_prediction(predictor, name, melody)
        
        # Pause for dramatic effect
        time.sleep(1)
    
    print(f"\n{'='*50}")
    print("🎉 Demo Complete!")
    print()
    print("Key Insights:")
    print("• The model analyzes pitch patterns, intervals, and rhythm")
    print("• Higher confidence suggests stronger harmonic relationships") 
    print("• Different musical styles produce different chord predictions")
    print("• The AI learned harmonic patterns from thousands of MIDI files")
    print()
    print("🚀 Try running with your own MIDI data!")

def quick_test():
    """Quick test with real data samples."""
    print("🔬 Quick Test with Real Data")
    print("=" * 30)
    
    predictor = ChordPredictor()
    if predictor.model is None:
        print("❌ No model available")
        return
    
    try:
        # Load real data
        data = np.load('processed_data/features_full.npz')
        X_real = data['X'][:5]  # First 5 real sequences
        
        print(f"📊 Testing on {len(X_real)} real melody sequences...")
        results = predictor.predict_sequence(X_real)
        
        print("\n📋 Real Data Predictions:")
        for i, (chord, conf) in enumerate(results):
            print(f"  Real Sequence {i+1}: Chord {chord} (confidence: {conf:.3f})")
        
        avg_conf = np.mean([conf for _, conf in results])
        print(f"\n📈 Average confidence on real data: {avg_conf:.3f}")
        
    except FileNotFoundError:
        print("❌ No processed data found")

def main():
    """Main demo function."""
    print("Choose demo mode:")
    print("1. Interactive melody demo (recommended)")
    print("2. Quick test with real data")
    print("3. Both")
    print()
    
    try:
        choice = input("Enter choice (1-3): ").strip()
        
        if choice == "1":
            interactive_demo()
        elif choice == "2":
            quick_test()
        elif choice == "3":
            interactive_demo()
            print("\n" + "="*60 + "\n")
            quick_test()
        else:
            print("Running interactive demo by default...")
            interactive_demo()
            
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted. Thanks for trying the chord predictor!")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
