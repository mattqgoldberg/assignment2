# Gradient Boosting model for melody harmonization
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from music21 import converter, note, chord, analysis
from collections import Counter
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

# Import the HarmonizationModel class to reuse its functionality
from hmm_model import HarmonizationModel, get_note_name, get_chord_quality, get_pitch_class

def main():
    """Train and evaluate a Gradient Boosting model for harmonization"""
    melody_dir = os.path.join("data", "MIDI", "melody")
    chord_dir = os.path.join("data", "MIDI", "chords")
    
    # Initialize Gradient Boosting harmonization model
    model_type = "gb"  # Gradient Boosting
    harmonizer = HarmonizationModel(model_type=model_type)
    
    print(f"Extracting sequences for {model_type} model...")
    X_data, Y, file_ids, measure_positions = harmonizer.extract_sequences(melody_dir, chord_dir)
    print(f"Loaded {len(X_data)} aligned melody/chord pairs.")
    
    X_enc, Y_enc = harmonizer.encode_sequences(X_data, Y, file_ids, measure_positions)
    print(f"Melody vocab size: {len(harmonizer.note_vocab)} | Chord vocab size: {len(harmonizer.chord_vocab)}")
    
    # Split into train/test
    X_train, X_test, Y_train, Y_test = train_test_split(X_enc, Y_enc, test_size=0.2, random_state=42)
    print(f"Training on {len(X_train)} examples, testing on {len(X_test)} examples")
    
    # Train model with hyperparameter tuning
    print("Training Gradient Boosting model...")
    harmonizer.train(X_train, Y_train, tune_hyperparams=True)
    
    # Save model
    model_filename = f"results/harmonizer_{model_type}_model.pkl"
    harmonizer.save(model_filename)
    
    # Evaluate with visualizations
    print("Evaluating Gradient Boosting model...")
    harmonizer.evaluate(X_test, Y_test, output_fig=True)
    
    print(f"Gradient Boosting model trained and saved to {model_filename}")

if __name__ == "__main__":
    main()
