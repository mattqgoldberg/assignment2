# Optimized HMM-based harmonization model for melody-to-chord prediction
# This implementation uses proper Hidden Markov Models for sequence modeling

import os
import numpy as np
import pickle
from collections import Counter, defaultdict
from music21 import converter, note, chord, analysis, key, interval, meter
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from hmmlearn import hmm
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')  # Suppress hmmlearn convergence warnings

def get_note_name(n):
    """Extract note name from a music21 note or chord object"""
    if isinstance(n, note.Note):
        return n.name  # keep melody as note name only (C, D, E, etc.)
    elif isinstance(n, chord.Chord):
        # Use full chord symbol if available, else root name
        if hasattr(n, 'figure') and n.figure:
            return n.figure
        elif hasattr(n, 'pitchedCommonName') and n.pitchedCommonName:
            return n.pitchedCommonName
        elif hasattr(n, 'fullName') and n.fullName:
            return n.fullName
        else:
            return n.root().name
    return None

def get_pitch_class(note_name):
    """Extract pitch class (0-11) from note name"""
    if note_name is None or len(note_name) == 0:
        return 0
    
    pitch_classes = {'C': 0, 'C#': 1, 'D-': 1, 'D': 2, 'D#': 3, 'E-': 3, 
                     'E': 4, 'F': 5, 'F#': 6, 'G-': 6, 'G': 7, 
                     'G#': 8, 'A-': 8, 'A': 9, 'A#': 10, 'B-': 10, 'B': 11}
    
    # Extract just the pitch name (without octave)
    if len(note_name) > 1 and note_name[1] in ['#', '-']:
        pitch = note_name[:2]
    else:
        pitch = note_name[0]
        
    return pitch_classes.get(pitch, 0)

def get_chord_quality(chord_name):
    """Extract chord quality from chord name"""
    if chord_name is None:
        return "unknown"
        
    if "minor" in chord_name or "min" in chord_name or "-minor" in chord_name or "m" in chord_name:
        return "minor"
    elif "major" in chord_name or "-major" in chord_name or "maj" in chord_name:
        return "major"
    elif "dominant" in chord_name or "7" in chord_name:
        return "dominant"
    elif "diminished" in chord_name or "dim" in chord_name:
        return "diminished"
    elif "augmented" in chord_name or "aug" in chord_name:
        return "augmented"
    else:
        return "unknown"

class OptimizedHMMModel:
    """Optimized HMM-based harmonization model with comprehensive musical features"""
    
    def __init__(self, n_components=8, n_iter=100):
        self.model = None
        self.note_vocab = None
        self.chord_vocab = None
        self.n_components = n_components  # Number of hidden states in HMM
        self.n_iter = n_iter              # Number of EM iterations
        self.is_trained = False
        self.chord2idx = None
        self.idx2chord = None
        self.note2idx = None
        self.idx2note = None
        
    def extract_sequences(self, melody_dir, chord_dir):
        """Extract measure-level melody and chord sequences from MIDI files for HMM"""
        # This is a simplified version that works at the measure level
        # It's more similar to the approach in hmm_model.py but maintains proper sequence structure
        
        melody_sequences = []  # Will contain sequences of melody notes per measure
        chord_sequences = []   # Will contain chord labels per measure
        file_boundaries = []   # Will store the boundaries between different songs
        
        files = sorted(os.listdir(melody_dir))
        current_position = 0
        
        for file_id, f in enumerate(files):
            if not f.endswith('.mid') or not os.path.exists(os.path.join(chord_dir, f)):
                continue
                
            try:
                # Parse the files
                melody_score = converter.parse(os.path.join(melody_dir, f))
                chord_score = converter.parse(os.path.join(chord_dir, f))
                
                # Use measures (bars) as time units - this is more reliable
                melody_measures = melody_score.parts[0].getElementsByClass('Measure')
                chord_measures = chord_score.parts[0].getElementsByClass('Measure')
                
                if len(melody_measures) == 0 or len(chord_measures) == 0:
                    continue
                    
                min_len = min(len(melody_measures), len(chord_measures))
                file_sequence = []  # Melody/chord sequence for this file
                
                # Extract and analyze each measure
                for i in range(min_len):
                    # Get melody notes
                    melody_notes = [n for n in melody_measures[i].notes if isinstance(n, note.Note)]
                    if not melody_notes:
                        continue
                        
                    # Get chord information 
                    chords_in_measure = [c for c in chord_measures[i].notes if isinstance(c, chord.Chord)]
                    if not chords_in_measure:
                        continue
                        
                    # Extract basic note and chord information
                    melody_note_names = [get_note_name(n) for n in melody_notes]
                    chord_labels = [get_note_name(c) for c in chords_in_measure]
                    chord_label = max(set(chord_labels), key=chord_labels.count)
                    
                    # For each note in the measure, add a (note, chord) pair
                    for note_name in melody_note_names:
                        melody_sequences.append(note_name)
                        chord_sequences.append(chord_label)
                        file_sequence.append((note_name, chord_label))
                
                if file_sequence:
                    # Store the boundary position after each file
                    file_boundaries.append(current_position + len(file_sequence))
                    current_position += len(file_sequence)
            
            except Exception as e:
                print(f"Error processing {f}: {str(e)}")
                continue
        
        return melody_sequences, chord_sequences, file_boundaries
    
    def encode_sequences(self, melody_sequences, chord_sequences):
        """Encode melody and chord sequences for HMM"""
        # Build vocabularies for melody and chord
        self.note_vocab = sorted(set(melody_sequences))
        self.chord_vocab = sorted(set(chord_sequences))
        
        # Create mapping dictionaries
        self.note2idx = {n: i for i, n in enumerate(self.note_vocab)}
        self.idx2note = {i: n for i, n in enumerate(self.note_vocab)}
        self.chord2idx = {c: i for i, c in enumerate(self.chord_vocab)}
        self.idx2chord = {i: c for i, c in enumerate(self.chord_vocab)}
        
        # Convert to numerical sequences
        melody_encoded = np.array([self.note2idx[n] for n in melody_sequences])
        chord_encoded = np.array([self.chord2idx[c] for c in chord_sequences])
        
        return melody_encoded, chord_encoded
    
    def train(self, melody_encoded, chord_encoded, file_boundaries=None):
        """Train the HMM model on encoded sequences"""
        print(f"Training HMM with {self.n_components} hidden states...")
        
        if len(melody_encoded) == 0:
            print("Error: No training data available")
            return False
        
        # Let's use GaussianHMM which is more stable in hmmlearn
        # For categorical data, we'll use one-hot encoding
        self.model = hmm.GaussianHMM(
            n_components=self.n_components, 
            n_iter=self.n_iter,
            covariance_type="diag",  # Diagonal covariance for efficiency
            random_state=42
        )
        
        # One-hot encode the melody notes
        n_notes = len(self.note_vocab)
        X_one_hot = np.zeros((len(melody_encoded), n_notes))
        for i, idx in enumerate(melody_encoded):
            X_one_hot[i, idx] = 1.0
            
        # Define lengths of independent sequences if file boundaries are provided
        if file_boundaries:
            # Calculate lengths of sequences between boundaries
            lengths = []
            prev = 0
            for boundary in file_boundaries:
                lengths.append(boundary - prev)
                prev = boundary
            lengths.append(len(melody_encoded) - prev)
            
            # Filter out zero-length sequences
            lengths = [length for length in lengths if length > 0]
        else:
            lengths = None
        
        # Train the model
        try:
            self.model.fit(X_one_hot, lengths=lengths)
            self.is_trained = True
            print("HMM training completed successfully.")
            
            # Store the training data for later analysis
            self.training_data = (melody_encoded, chord_encoded)
            
            # Evaluate the learned hidden states
            self._analyze_hidden_states(melody_encoded, chord_encoded)
            return True
            
        except Exception as e:
            print(f"HMM training error: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def _analyze_hidden_states(self, melody_encoded, chord_encoded):
        """Analyze what the hidden states represent in terms of chords and notes"""
        # One-hot encode the melody notes for prediction
        n_notes = len(self.note_vocab)
        X_one_hot = np.zeros((len(melody_encoded), n_notes))
        for i, idx in enumerate(melody_encoded):
            X_one_hot[i, idx] = 1.0
            
        # Predict hidden states
        try:
            hidden_states = self.model.predict(X_one_hot)
            
            # For each hidden state, see which chords occur most frequently
            state_chord_counters = [Counter() for _ in range(self.n_components)]
            for state, chord_idx in zip(hidden_states, chord_encoded):
                state_chord_counters[state][self.idx2chord[chord_idx]] += 1
            
            print("\nHidden State Analysis:")
            for state, counter in enumerate(state_chord_counters):
                if counter:
                    top_chords = counter.most_common(3)
                    print(f"State {state}: {[c for c, _ in top_chords]}")
                    
            # Also analyze which melody notes trigger each state
            state_note_counters = [Counter() for _ in range(self.n_components)]
            for state, note_idx in zip(hidden_states, melody_encoded):
                state_note_counters[state][self.idx2note[note_idx]] += 1
                
            print("\nCommon Melody Notes per State:")
            for state, counter in enumerate(state_note_counters):
                if counter:
                    top_notes = counter.most_common(3)
                    print(f"State {state}: {[n for n, _ in top_notes]}")
                    
            # Analyze transitions between states
            transitions = defaultdict(Counter)
            for i in range(1, len(hidden_states)):
                from_state = hidden_states[i-1]
                to_state = hidden_states[i]
                transitions[from_state][to_state] += 1
                
            print("\nCommon State Transitions:")
            for from_state, counters in sorted(transitions.items()):
                top_transitions = counters.most_common(2)
                if top_transitions:
                    print(f"State {from_state} → {[f'State {to}' for to, _ in top_transitions]}")
                    
        except Exception as e:
            print(f"Error analyzing hidden states: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def predict_chords(self, melody_sequence):
        """Predict chords for a new melody sequence"""
        if not self.is_trained:
            print("Model not trained yet.")
            return []
        
        # Encode melody sequence
        try:
            melody_encoded = []
            for note in melody_sequence:
                if note in self.note2idx:
                    melody_encoded.append(self.note2idx[note])
                else:
                    # Handle unknown notes by finding closest match based on pitch class
                    pitch_class = get_pitch_class(note)
                    closest_note = min(self.note2idx.keys(), 
                                     key=lambda x: abs(get_pitch_class(x) - pitch_class))
                    melody_encoded.append(self.note2idx[closest_note])
            melody_encoded = np.array(melody_encoded)
        except Exception as e:
            print(f"Error encoding melody: {str(e)}")
            return []
        
        # One-hot encode for the GaussianHMM
        n_notes = len(self.note_vocab)
        X_one_hot = np.zeros((len(melody_encoded), n_notes))
        for i, idx in enumerate(melody_encoded):
            X_one_hot[i, idx] = 1.0
        
        # Use Viterbi algorithm to find most likely hidden state sequence
        try:
            hidden_states = self.model.predict(X_one_hot)
            
            # Count which chords appear most often for each state in training
            if not hasattr(self, 'state_chord_probs'):
                # Lazy calculation - only done the first time prediction is needed
                print("Calculating state-chord probabilities...")
                self._calculate_state_chord_probs()
                
            # Use the pre-calculated probabilities to predict chords
            chord_predictions = []
            for state in hidden_states:
                chord_idx = self.state_chord_probs[state]
                chord_predictions.append(self.idx2chord[chord_idx])
                
            return chord_predictions
            
        except Exception as e:
            print(f"Error predicting states: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
    
    def _calculate_state_chord_probs(self):
        """Calculate which chord is most likely for each hidden state"""
        # We need to re-run the Viterbi algorithm on the training data
        # This is a post-training step to improve the model's chord predictions
        
        if not hasattr(self, 'training_data'):
            print("Warning: No training data available for chord probability calculation.")
            # Default to naive mapping
            self.state_chord_probs = {i: i % len(self.chord_vocab) for i in range(self.n_components)}
            return
        
        try:    
            melody_encoded, chord_encoded = self.training_data
            
            # One-hot encode the melody notes for prediction
            n_notes = len(self.note_vocab)
            X_one_hot = np.zeros((len(melody_encoded), n_notes))
            for i, idx in enumerate(melody_encoded):
                X_one_hot[i, idx] = 1.0
                
            # Run Viterbi
            hidden_states = self.model.predict(X_one_hot)
            
            # For each state, count chord occurrences
            state_chord_counts = defaultdict(Counter)
            for state, chord_idx in zip(hidden_states, chord_encoded):
                state_chord_counts[state][chord_idx] += 1
                
            # For each state, pick the most common chord
            self.state_chord_probs = {}
            for state in range(self.n_components):
                counts = state_chord_counts[state]
                if counts:
                    self.state_chord_probs[state] = counts.most_common(1)[0][0]
                else:
                    # If we never saw this state in training, use the most common chord overall
                    all_chord_counts = Counter()
                    for _, chord_idx in zip(hidden_states, chord_encoded):
                        all_chord_counts[chord_idx] += 1
                    self.state_chord_probs[state] = all_chord_counts.most_common(1)[0][0] if all_chord_counts else 0
                    
            # Print state to chord mapping for transparency
            print("\nHidden State to Chord Mapping:")
            for state in range(self.n_components):
                chord_idx = self.state_chord_probs[state]
                chord_name = self.idx2chord[chord_idx]
                print(f"State {state} → {chord_name}")
                
        except Exception as e:
            print(f"Error calculating state-chord probabilities: {str(e)}")
            import traceback
            traceback.print_exc()
            # Use fallback naive mapping
            self.state_chord_probs = {i: i % len(self.chord_vocab) for i in range(self.n_components)}
    
    def evaluate(self, melody_sequences, true_chords):
        """Evaluate model performance on test data"""
        if not self.is_trained:
            print("Model not trained yet.")
            return 0
            
        # Predict chords
        pred_chords = self.predict_chords(melody_sequences)
        
        # Calculate accuracy
        correct = sum(1 for true, pred in zip(true_chords, pred_chords) if true == pred)
        total = len(true_chords)
        accuracy = correct / total if total > 0 else 0
        
        print(f"HMM model accuracy: {accuracy:.3f} ({correct}/{total})")
        
        # Analyze common confusions
        confusions = [(true, pred) for true, pred in zip(true_chords, pred_chords) if true != pred]
        confusion_counter = Counter(confusions).most_common(5)
        
        print("\nCommon chord confusions:")
        for (true, pred), count in confusion_counter:
            print(f"True: {true}, Predicted: {pred}, Count: {count}")
            
        return accuracy
    
    def save(self, filepath):
        """Save the trained model to a file"""
        if not self.is_trained:
            print("Warning: Saving untrained model")
            
        with open(filepath, "wb") as f:
            pickle.dump({
                'model': self.model,
                'note_vocab': self.note_vocab,
                'chord_vocab': self.chord_vocab,
                'note2idx': self.note2idx,
                'idx2note': self.idx2note,
                'chord2idx': self.chord2idx,
                'idx2chord': self.idx2chord,
                'n_components': self.n_components,
                'n_iter': self.n_iter,
                'is_trained': self.is_trained
            }, f)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """Load a trained model from a file"""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            
        instance = cls(
            n_components=data['n_components'],
            n_iter=data['n_iter']
        )
        
        instance.model = data['model']
        instance.note_vocab = data['note_vocab']
        instance.chord_vocab = data['chord_vocab']
        instance.note2idx = data['note2idx']
        instance.idx2note = data['idx2note']
        instance.chord2idx = data['chord2idx']
        instance.idx2chord = data['idx2chord']
        instance.is_trained = data['is_trained']
        
        return instance

def main():
    """Train and evaluate the optimized HMM model"""
    melody_dir = os.path.join("data", "MIDI", "melody")
    chord_dir = os.path.join("data", "MIDI", "chords")
    
    # Create HMM model with tuneable parameters
    n_components = 12  # Number of hidden states - corresponds to keys in music (12 pitch classes)
    n_iter = 100       # Number of EM iterations
    
    harmonizer = OptimizedHMMModel(n_components=n_components, n_iter=n_iter)
    
    print("Extracting sequences...")
    melody_sequences, chord_sequences, file_boundaries = harmonizer.extract_sequences(melody_dir, chord_dir)
    print(f"Extracted {len(melody_sequences)} melody-chord pairs from files")
    
    if not melody_sequences:
        print("Error: No sequences extracted. Check data paths.")
        return
    
    # Encode sequences
    melody_encoded, chord_encoded = harmonizer.encode_sequences(melody_sequences, chord_sequences)
    print(f"Melody vocabulary size: {len(harmonizer.note_vocab)}")
    print(f"Chord vocabulary size: {len(harmonizer.chord_vocab)}")
    
    # Store training data for later analysis
    harmonizer.training_data = (melody_encoded, chord_encoded)
    
    # Split data into train/test sets - use simple random split for reliability
    melody_train, melody_test, chord_train, chord_test = train_test_split(
        melody_encoded, chord_encoded, test_size=0.2, random_state=42
    )
    
    # Use a smaller subset for training if dataset is very large
    # This helps avoid memory issues and speeds up development
    max_train_size = 50000
    if len(melody_train) > max_train_size:
        print(f"Training data is large ({len(melody_train)} examples), using {max_train_size} random samples")
        indices = np.random.choice(len(melody_train), max_train_size, replace=False)
        melody_train = melody_train[indices]
        chord_train = chord_train[indices]
        
    print(f"Training on {len(melody_train)} examples, testing on {len(melody_test)} examples")
    
    # Train the model - no boundaries for now to simplify
    success = harmonizer.train(melody_train, chord_train)
    
    if success:
        # Save the trained model
        model_filename = "results/hmm_optimized_model.pkl"
        harmonizer.save(model_filename)
        
        # Evaluate on test data
        print("\nEvaluating on test data...")
        test_melody_sequences = [harmonizer.idx2note[idx] for idx in melody_test]
        test_chord_sequences = [harmonizer.idx2chord[idx] for idx in chord_test]
        harmonizer.evaluate(test_melody_sequences, test_chord_sequences)
    else:
        print("Training failed. Check error messages.")

if __name__ == "__main__":
    main()
