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
        """Encode melody and chord sequences for HMM with enhanced musical features"""
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
        
        # Create additional musical features
        self.pitch_class_map = {}
        for note in self.note_vocab:
            self.pitch_class_map[note] = get_pitch_class(note)
            
        self.chord_root_map = {}
        self.chord_type_map = {}
        for chord in self.chord_vocab:
            # Extract chord root (usually the first part before any quality indicators)
            if "-" in chord:
                root = chord.split("-")[0]
            else:
                root = chord.split(" ")[0] if " " in chord else chord
            
            # Get pitch class of root
            self.chord_root_map[chord] = get_pitch_class(root)
            
            # Get chord quality/type
            self.chord_type_map[chord] = get_chord_quality(chord)
        
        return melody_encoded, chord_encoded
    
    def train(self, melody_encoded, chord_encoded, file_boundaries=None):
        """Train the HMM model on encoded sequences with enhanced musical features"""
        print(f"Training HMM with {self.n_components} hidden states...")
        
        if len(melody_encoded) == 0:
            print("Error: No training data available")
            return False
        
        # Enhanced feature encoding that combines:
        # 1. One-hot encoding of notes
        # 2. Pitch class information (circular distance relationships)
        # 3. Recent note history to capture melodic contour
        n_notes = len(self.note_vocab)
        n_features = n_notes + 3  # One-hot + 3 musical features
        
        # Create enhanced feature arrays
        X_features = np.zeros((len(melody_encoded), n_features))
        
        # One-hot encode the melody notes
        for i, idx in enumerate(melody_encoded):
            # Basic one-hot encoding
            X_features[i, idx] = 1.0
            
            # Add pitch class as a feature (normalized to 0-1)
            note_name = self.idx2note[idx]
            pitch_class = self.pitch_class_map.get(note_name, 0)
            X_features[i, n_notes] = pitch_class / 12.0
            
            # Add melodic movement features (if not at the beginning)
            if i > 0:
                # Pitch distance from previous note (-1 to +1 scale)
                prev_note = self.idx2note[melody_encoded[i-1]]
                prev_pitch = self.pitch_class_map.get(prev_note, 0)
                pitch_diff = (pitch_class - prev_pitch) % 12
                if pitch_diff > 6:
                    pitch_diff = pitch_diff - 12  # Map to -6 to +6 range
                X_features[i, n_notes+1] = pitch_diff / 6.0
                
                # Direction of melodic movement (-1: down, 0: same, 1: up)
                if pitch_diff == 0:
                    movement = 0
                else:
                    movement = 1 if pitch_diff > 0 else -1
                X_features[i, n_notes+2] = movement
        
        # Let's use GaussianHMM which is more stable in hmmlearn
        # For categorical data, we'll use enhanced feature representations
        self.model = hmm.GaussianHMM(
            n_components=self.n_components, 
            n_iter=self.n_iter,
            covariance_type="diag",  # Use diagonal covariance for more robustness
            random_state=42,
            init_params="stc",  # Initialize starts, transitions, and covars
            params="stmc"       # Train all parameters
        )
        
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
            print(f"Training on {len(lengths)} separate sequences")
        else:
            lengths = None
            
        # Initialize with simple but effective musical knowledge
        # Use one state per pitch class for direct mapping
        # This provides a strong musical basis for the model
        self.model.startprob_ = np.ones(self.n_components) / self.n_components
        self.model.transmat_ = np.ones((self.n_components, self.n_components)) / self.n_components
        
        # Initialize means and covars based on pitch classes
        for state in range(self.n_components):
            # Each state corresponds to a pitch class (0-11)
            pitch_class = state % 12
            
            # Initialize means with zeros
            self.model.means_[state] = np.zeros(n_features)
            
            # Set means for one-hot part based on notes in this pitch class
            for note_idx, note in self.idx2note.items():
                note_pc = self.pitch_class_map.get(note, 0)
                if note_pc == pitch_class:
                    self.model.means_[state, note_idx] = 1.0
                    
            # Also set the musical feature for pitch class
            self.model.means_[state, n_notes] = pitch_class / 12.0
            
            # Set different covariance values for stability
            self.model.covars_[state] = np.ones(n_features) * 0.1
        
        # Train the model
        try:
            # Use the enhanced features, not the undefined X_one_hot
            self.model.fit(X_features, lengths=lengths)
            self.is_trained = True
            print("HMM training completed successfully.")
            
            # Store the training data and features for later analysis
            self.training_data = (melody_encoded, chord_encoded)
            self.X_features = X_features
            
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
        # Use the same feature encoding as in training for consistency
        if hasattr(self, 'X_features'):
            X_features = self.X_features
        else:
            # Create enhanced features
            n_notes = len(self.note_vocab)
            feature_dim = n_notes + 3
            X_features = np.zeros((len(melody_encoded), feature_dim))
            
            for i, idx in enumerate(melody_encoded):
                # One-hot encoding
                X_features[i, idx] = 1.0
                
                # Add pitch class feature
                note_name = self.idx2note[idx]
                pitch_class = self.pitch_class_map.get(note_name, 0)
                X_features[i, n_notes] = pitch_class / 12.0
                
                # Add melodic movement features
                if i > 0:
                    prev_note = self.idx2note[melody_encoded[i-1]]
                    prev_pitch = self.pitch_class_map.get(prev_note, 0)
                    pitch_diff = (pitch_class - prev_pitch) % 12
                    if pitch_diff > 6:
                        pitch_diff = pitch_diff - 12
                    X_features[i, n_notes+1] = pitch_diff / 6.0
                    
                    movement = 0 if pitch_diff == 0 else (1 if pitch_diff > 0 else -1)
                    X_features[i, n_notes+2] = movement
            
        # Predict hidden states with enhanced features
        try:
            hidden_states = self.model.predict(X_features)
            
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
            
            # Analyze musical patterns in states
            state_musical_features = [[] for _ in range(self.n_components)]
            for state, melody_idx, chord_idx in zip(hidden_states, melody_encoded, chord_encoded):
                note = self.idx2note[melody_idx]
                chord = self.idx2chord[chord_idx]
                note_pc = self.pitch_class_map.get(note, 0)
                chord_root = self.chord_root_map.get(chord, 0)
                chord_type = self.chord_type_map.get(chord, "unknown")
                
                # Calculate melodic-harmonic relationship
                interval_to_root = (chord_root - note_pc) % 12
                
                state_musical_features[state].append((interval_to_root, chord_type))
            
            print("\nMelody-Chord Relationships per State:")
            for state in range(self.n_components):
                if state_musical_features[state]:
                    # Analyze intervals between melody and chord root
                    intervals = [i for i, _ in state_musical_features[state]]
                    common_intervals = Counter(intervals).most_common(2)
                    
                    # Analyze chord types
                    chord_types = [t for _, t in state_musical_features[state]]
                    common_types = Counter(chord_types).most_common(1)
                    
                    interval_names = {
                        0: "unison", 1: "minor 2nd", 2: "major 2nd", 
                        3: "minor 3rd", 4: "major 3rd", 5: "perfect 4th",
                        6: "tritone", 7: "perfect 5th", 8: "minor 6th", 
                        9: "major 6th", 10: "minor 7th", 11: "major 7th"
                    }
                    
                    interval_desc = [f"{interval_names.get(i, i)}" for i, _ in common_intervals]
                    print(f"State {state}: most common interval={interval_desc}, " 
                          f"chord type={common_types[0][0] if common_types else 'unknown'}")
                    
        except Exception as e:
            print(f"Error analyzing hidden states: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def predict_chords(self, melody_sequence):
        """Predict chords for a new melody sequence using advanced harmonic analysis"""
        if not self.is_trained:
            print("Model not trained yet.")
            return []
        
        # Encode melody sequence
        try:
            melody_encoded = []
            melody_pitch_classes = []
            
            for note in melody_sequence:
                if note in self.note2idx:
                    melody_encoded.append(self.note2idx[note])
                    melody_pitch_classes.append(self.pitch_class_map.get(note, 0))
                else:
                    # Handle unknown notes by finding closest match based on pitch class
                    pitch_class = get_pitch_class(note)
                    melody_pitch_classes.append(pitch_class)
                    closest_note = min(self.note2idx.keys(), 
                                      key=lambda x: abs(get_pitch_class(x) - pitch_class))
                    melody_encoded.append(self.note2idx[closest_note])
            
            melody_encoded = np.array(melody_encoded)
        except Exception as e:
            print(f"Error encoding melody: {str(e)}")
            return []
        
        # Create enhanced features for prediction
        n_notes = len(self.note_vocab)
        n_features = n_notes + 3  # One-hot + pitch class + movement features
        X_features = np.zeros((len(melody_encoded), n_features))
        
        # Create the feature vectors
        for i, idx in enumerate(melody_encoded):
            # Basic one-hot encoding
            X_features[i, idx] = 1.0
            
            # Add pitch class as a feature (normalized to 0-1)
            note_name = self.idx2note[idx]
            pitch_class = self.pitch_class_map.get(note_name, 0)
            X_features[i, n_notes] = pitch_class / 12.0
            
            # Add melodic movement features (if not at the beginning)
            if i > 0:
                # Pitch distance from previous note
                prev_note = self.idx2note[melody_encoded[i-1]]
                prev_pitch = self.pitch_class_map.get(prev_note, 0)
                pitch_diff = (pitch_class - prev_pitch) % 12
                if pitch_diff > 6:
                    pitch_diff = pitch_diff - 12  # Map to -6 to +6 range
                X_features[i, n_notes+1] = pitch_diff / 6.0
                
                # Direction of melodic movement
                if pitch_diff == 0:
                    movement = 0
                else:
                    movement = 1 if pitch_diff > 0 else -1
                X_features[i, n_notes+2] = movement
        
        # Use Viterbi algorithm to find most likely hidden state sequence
        try:
            hidden_states = self.model.predict(X_features)
            
            # Ensure we have state-chord probabilities
            if not hasattr(self, 'state_chord_probs'):
                print("Calculating state-chord probabilities...")
                self._calculate_state_chord_probs()
            
            # Find key center to help with harmonic analysis
            # A simple approximation is to look at pitch class distribution
            key_weights = np.zeros(12)
            for pc in melody_pitch_classes:
                key_weights[pc] += 1
                
            # Major scale pattern: emphasize diatonic tones
            major_pattern = np.array([6.0, 1.0, 2.0, 1.0, 3.0, 2.0, 1.0, 4.0, 1.0, 2.0, 1.0, 2.0])
            minor_pattern = np.array([6.0, 1.0, 2.0, 5.0, 1.0, 2.0, 1.0, 4.0, 3.0, 1.0, 2.0, 1.0])
            
            # Find best matching key by correlating with patterns
            major_correlation = np.zeros(12)
            minor_correlation = np.zeros(12)
            
            for i in range(12):
                major_correlation[i] = np.corrcoef(np.roll(major_pattern, i), key_weights)[0, 1]
                minor_correlation[i] = np.corrcoef(np.roll(minor_pattern, i), key_weights)[0, 1]
                
            # Find highest correlation
            best_major_idx = np.argmax(major_correlation)
            best_minor_idx = np.argmax(minor_correlation)
            
            is_major = major_correlation[best_major_idx] > minor_correlation[best_minor_idx]
            key_center = best_major_idx if is_major else best_minor_idx
            key_quality = "major" if is_major else "minor"
            
            # Create chord progressions with musical context awareness
            chord_predictions = []
            last_chord_idx = None
            
            # Create a sliding window for chord analysis (3 notes)
            window_size = min(3, len(hidden_states))
            
            for i in range(len(hidden_states)):
                # Calculate window boundaries
                start = max(0, i - window_size // 2)
                end = min(len(hidden_states), i + window_size // 2 + 1)
                
                # Get window of states and weights by distance from center
                window_states = hidden_states[start:end]
                window_weights = [1.0 - 0.2 * abs(j - i) for j in range(start, end)]
                
                # Track state counts for window with weights
                window_state_counts = defaultdict(float)
                for state, weight in zip(window_states, window_weights):
                    window_state_counts[state] += weight
                
                # Get the most common state in the window
                primary_state = max(window_state_counts.items(), key=lambda x: x[1])[0]
                
                # Get candidate chords for the primary state
                candidates = self.state_chord_candidates[primary_state]
                
                # Prepare to score candidates
                scored_candidates = []
                
                for chord_idx, base_prob in candidates:
                    chord_name = self.idx2chord[chord_idx]
                    chord_root = self.chord_root_map.get(chord_name, 0)
                    chord_quality = self.chord_type_map.get(chord_name, "unknown")
                    
                    # Start with base probability from the model
                    score = base_prob
                    
                    # Factor 1: Diatonic compatibility with estimated key
                    # Major keys: I, ii, iii, IV, V, vi, viio
                    # Minor keys: i, iio, III, iv, v, VI, VII
                    major_diatonic = [(key_center) % 12, 
                                      (key_center + 2) % 12, 
                                      (key_center + 4) % 12,
                                      (key_center + 5) % 12,
                                      (key_center + 7) % 12,
                                      (key_center + 9) % 12,
                                      (key_center + 11) % 12]
                                      
                    minor_diatonic = [(key_center) % 12, 
                                      (key_center + 2) % 12,
                                      (key_center + 3) % 12,
                                      (key_center + 5) % 12,
                                      (key_center + 7) % 12,
                                      (key_center + 8) % 12, 
                                      (key_center + 10) % 12]
                                      
                    diatonic_roots = major_diatonic if is_major else minor_diatonic
                    
                    if chord_root in diatonic_roots:
                        diatonic_bonus = 1.5
                        # Give extra weight to most important scale degrees
                        if chord_root == key_center:  # Tonic (I/i)
                            diatonic_bonus *= 2.0
                        elif chord_root == (key_center + 7) % 12:  # Dominant (V)
                            diatonic_bonus *= 1.75
                        elif chord_root == (key_center + 5) % 12:  # Subdominant (IV/iv)
                            diatonic_bonus *= 1.5
                            
                        score *= diatonic_bonus
                    
                    # Factor 2: Chord progression voice leading
                    if last_chord_idx is not None:
                        last_chord = self.idx2chord[last_chord_idx] 
                        last_root = self.chord_root_map.get(last_chord, 0)
                        
                        # Circle of fifths movement (root movement by P5/P4)
                        if (chord_root - last_root) % 12 == 7:  # Up P5
                            score *= 2.0  # Strong progression
                        elif (chord_root - last_root) % 12 == 5:  # Up P4 (down P5)
                            score *= 1.75  # Strong progression
                        elif abs((chord_root - last_root) % 12) <= 2:  # Step-wise motion
                            score *= 1.25  # Smooth voice leading
                        
                        # Avoid repeating same chord too much
                        if chord_idx == last_chord_idx:
                            # Only repeat if probability is very high
                            if base_prob > 0.5:
                                score *= 0.9  # Small penalty for repetition
                            else:
                                score *= 0.5  # Larger penalty for repetition
                                
                    # Factor 3: Fit with melody note
                    current_pc = melody_pitch_classes[i]
                    
                    # Check if melody note is in the chord (root, third, fifth, seventh)
                    root_matches = current_pc == chord_root
                    third_matches = current_pc == (chord_root + (4 if chord_quality == "major" else 3)) % 12
                    fifth_matches = current_pc == (chord_root + 7) % 12
                    
                    if root_matches:
                        score *= 1.5  # Strong match with chord root
                    elif third_matches:
                        score *= 1.3  # Good match with chord third
                    elif fifth_matches:
                        score *= 1.2  # Decent match with chord fifth
                    
                    scored_candidates.append((chord_idx, score))
                
                # Choose the best candidate
                best_chord_idx = max(scored_candidates, key=lambda x: x[1])[0]
                chord_predictions.append(self.idx2chord[best_chord_idx])
                last_chord_idx = best_chord_idx
            
            return chord_predictions
            
        except Exception as e:
            print(f"Error predicting chords: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
            
        except Exception as e:
            print(f"Error predicting states: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
    
    def _calculate_state_chord_probs(self):
        """Calculate which chord is most likely for each hidden state with improved mapping and weighting"""
        # This is a post-training step to improve the model's chord predictions by
        # analyzing the relationship between hidden states and chords
        
        if not hasattr(self, 'training_data'):
            print("Warning: No training data available for chord probability calculation.")
            # Default to naive mapping
            self.state_chord_probs = {i: i % len(self.chord_vocab) for i in range(self.n_components)}
            self.state_chord_candidates = {i: [(i % len(self.chord_vocab), 1.0)] for i in range(self.n_components)}
            return
        
        try:    
            melody_encoded, chord_encoded = self.training_data
            
            # Recalculate features
            n_notes = len(self.note_vocab)
            n_features = n_notes + 3
            X_features = np.zeros((len(melody_encoded), n_features))
            
            for i, idx in enumerate(melody_encoded):
                # One-hot encoding
                X_features[i, idx] = 1.0
                
                # Add pitch class feature
                note_name = self.idx2note[idx]
                pitch_class = self.pitch_class_map.get(note_name, 0)
                X_features[i, n_notes] = pitch_class / 12.0
                
                # Add melodic movement features
                if i > 0:
                    prev_note = self.idx2note[melody_encoded[i-1]]
                    prev_pitch = self.pitch_class_map.get(prev_note, 0)
                    pitch_diff = (pitch_class - prev_pitch) % 12
                    if pitch_diff > 6:
                        pitch_diff = pitch_diff - 12
                    X_features[i, n_notes+1] = pitch_diff / 6.0
                    
                    movement = 0 if pitch_diff == 0 else (1 if pitch_diff > 0 else -1)
                    X_features[i, n_notes+2] = movement
            
            # Run Viterbi with our enhanced features to predict hidden states for training data
            hidden_states = self.model.predict(X_features)
            
            # Extract musical features for all chords for better mapping
            chord_feature_vectors = {}
            for chord_idx in range(len(self.chord_vocab)):
                chord = self.idx2chord[chord_idx]
                
                # Get chord root
                chord_root = self.chord_root_map.get(chord, 0)
                
                # Get chord quality
                chord_quality = self.chord_type_map.get(chord, "unknown")
                
                # Convert quality to numeric values
                quality_values = {
                    "major": 1.0,
                    "minor": 0.7, 
                    "dominant": 0.8,
                    "diminished": 0.5,
                    "augmented": 0.6,
                    "unknown": 0.5
                }
                quality_value = quality_values.get(chord_quality, 0.5)
                
                # Create feature vector for each chord
                chord_feature_vectors[chord_idx] = (chord_root, quality_value)
            
            # Extract musical features for each state based on its melodic content
            state_feature_vectors = {}
            for state in range(self.n_components):
                # Find all positions where this state was predicted
                state_positions = [i for i, s in enumerate(hidden_states) if s == state]
                
                if not state_positions:
                    # Default to some reasonable values if we never saw this state
                    state_feature_vectors[state] = (0, 0.5)  # Default root C, unknown quality
                    continue
                    
                # Get the most common note in this state
                state_notes = [melody_encoded[i] for i in state_positions]
                common_note_idx = Counter(state_notes).most_common(1)[0][0]
                common_note = self.idx2note[common_note_idx]
                note_pc = self.pitch_class_map.get(common_note, 0)
                
                # Analyze melodic movements in this state
                movements = []
                for pos in state_positions:
                    if pos > 0:
                        prev_note = self.idx2note[melody_encoded[pos-1]]
                        prev_pc = self.pitch_class_map.get(prev_note, 0)
                        current_note = self.idx2note[melody_encoded[pos]]
                        current_pc = self.pitch_class_map.get(current_note, 0)
                        
                        # Calculate direction
                        diff = (current_pc - prev_pc) % 12
                        if diff > 6:
                            diff -= 12  # Convert to -6 to 6 range
                        movements.append(diff)
                
                # Calculate average movement
                avg_movement = sum(movements) / len(movements) if movements else 0
                
                # Calculate quality value based on movement patterns
                # Upward movements often suggest major, downward often minor
                quality_value = 0.8 if avg_movement > 0 else 0.7
                
                # Store feature vector
                state_feature_vectors[state] = (note_pc, quality_value)
            
            # For each state, count chord occurrences 
            state_chord_counts = defaultdict(Counter)
            for state, chord_idx in zip(hidden_states, chord_encoded):
                state_chord_counts[state][chord_idx] += 1
            
            # Calculate chord probabilities with musical knowledge enhancement
            self.state_chord_probs = {}
            self.state_chord_candidates = {}
            
            for state in range(self.n_components):
                counts = state_chord_counts[state]
                state_root, state_quality = state_feature_vectors.get(state, (0, 0.5))
                
                # Enhanced probabilities using both counts and musical compatibility
                enhanced_probs = {}
                
                for chord_idx in range(len(self.chord_vocab)):
                    # Base probability from counts (with add-one smoothing)
                    base_count = counts.get(chord_idx, 0) + 0.1
                    
                    # Musical compatibility score
                    chord_root, chord_quality = chord_feature_vectors.get(chord_idx, (0, 0.5))
                    
                    # Calculate compatibility based on music theory
                    # 1. Root compatibility (higher for chord roots that make sense with the notes)
                    root_distance = min((chord_root - state_root) % 12, (state_root - chord_root) % 12)
                    
                    # Perfect fifth, fourth and third relationships are strongest
                    root_compat = 1.0
                    if root_distance == 0:  # Unison/octave
                        root_compat = 5.0
                    elif root_distance == 7:  # Perfect fifth
                        root_compat = 4.0
                    elif root_distance == 5:  # Perfect fourth
                        root_compat = 3.5
                    elif root_distance in [3, 4]:  # Third
                        root_compat = 3.0
                    else:
                        root_compat = 2.0 - (root_distance / 12.0)
                    
                    # 2. Quality compatibility
                    quality_compat = 1.0 - abs(chord_quality - state_quality)
                    
                    # Combine the factors
                    compat_score = root_compat * quality_compat
                    
                    # Final probability combines observed counts and music theory
                    enhanced_probs[chord_idx] = base_count * compat_score
                
                # Normalize probabilities
                total_prob = sum(enhanced_probs.values())
                if total_prob > 0:
                    enhanced_probs = {k: v/total_prob for k, v in enhanced_probs.items()}
                
                # Get the most likely chord and candidates
                if enhanced_probs:
                    most_likely = max(enhanced_probs.items(), key=lambda x: x[1])
                    self.state_chord_probs[state] = most_likely[0]
                    
                    # Store top candidates with probabilities
                    candidates = sorted(enhanced_probs.items(), key=lambda x: x[1], reverse=True)[:5]
                    self.state_chord_candidates[state] = candidates
                else:
                    # Fallback to most common chord overall
                    all_chord_counts = Counter(chord_encoded)
                    self.state_chord_probs[state] = all_chord_counts.most_common(1)[0][0] if all_chord_counts else 0
                    self.state_chord_candidates[state] = [(self.state_chord_probs[state], 1.0)]
            
            # Print state to chord mapping with probabilities
            print("\nHidden State to Chord Mapping:")
            for state in range(self.n_components):
                chord_idx = self.state_chord_probs[state]
                chord_name = self.idx2chord[chord_idx]
                prob = self.state_chord_candidates[state][0][1]
                print(f"State {state} → {chord_name} (probability: {prob:.3f})")
                
            # Group states by chord quality for analysis
            state_by_quality = defaultdict(list)
            for state in range(self.n_components):
                chord_idx = self.state_chord_probs[state]
                chord = self.idx2chord[chord_idx]
                quality = self.chord_type_map.get(chord, "unknown")
                state_by_quality[quality].append(state)
                
            print("\nState Distribution by Chord Quality:")
            for quality, states in state_by_quality.items():
                print(f"{quality}: {len(states)} states")
                
        except Exception as e:
            print(f"Error calculating state-chord probabilities: {str(e)}")
            import traceback
            traceback.print_exc()
            # Use fallback naive mapping
            self.state_chord_probs = {i: i % len(self.chord_vocab) for i in range(self.n_components)}
    
    def evaluate(self, melody_sequences, true_chords):
        """Evaluate model performance on test data with enhanced metrics"""
        if not self.is_trained:
            print("Model not trained yet.")
            return 0
            
        # Predict chords
        pred_chords = self.predict_chords(melody_sequences)
        
        if len(pred_chords) == 0:
            print("No predictions generated. Check for errors.")
            return 0
            
        # Calculate overall accuracy
        correct = sum(1 for true, pred in zip(true_chords, pred_chords) if true == pred)
        total = len(true_chords)
        accuracy = correct / total if total > 0 else 0
        
        print(f"HMM model accuracy: {accuracy:.3f} ({correct}/{total})")
        
        # Analyze common confusions
        confusions = [(true, pred) for true, pred in zip(true_chords, pred_chords) if true != pred]
        confusion_counter = Counter(confusions).most_common(5)
        
        print("\nCommon chord confusions:")
        for (true, pred), count in confusion_counter[:5]:  # Limit to avoid empty list issues
            print(f"True: {true}, Predicted: {pred}, Count: {count}")
        
        try:
            # Calculate advanced metrics
            # 1. Chord root accuracy (if roots match, consider partially correct)
            root_correct = 0
            for true, pred in zip(true_chords, pred_chords):
                true_root = self.chord_root_map.get(true, None)
                pred_root = self.chord_root_map.get(pred, None)
                if true_root is not None and pred_root is not None and true_root == pred_root:
                    root_correct += 1
            
            root_accuracy = root_correct / total if total > 0 else 0
            print(f"Chord root accuracy: {root_accuracy:.3f} ({root_correct}/{total})")
            
            # 2. Chord quality accuracy (if chord qualities match)
            quality_correct = 0
            for true, pred in zip(true_chords, pred_chords):
                true_quality = self.chord_type_map.get(true, None)
                pred_quality = self.chord_type_map.get(pred, None)
                if true_quality is not None and pred_quality is not None and true_quality == pred_quality:
                    quality_correct += 1
            
            quality_accuracy = quality_correct / total if total > 0 else 0
            print(f"Chord quality accuracy: {quality_accuracy:.3f} ({quality_correct}/{total})")
            
            # Check if we have enough predictions for bigram analysis
            if len(pred_chords) > 1 and len(pred_chords) == len(true_chords):
                # 3. Chord progression accuracy (count correct bigrams)
                progression_correct = 0
                for i in range(1, len(true_chords)):
                    if true_chords[i-1] == pred_chords[i-1] and true_chords[i] == pred_chords[i]:
                        progression_correct += 1
                
                progression_total = len(true_chords) - 1
                progression_accuracy = progression_correct / progression_total if progression_total > 0 else 0
                print(f"Chord progression accuracy: {progression_accuracy:.3f} ({progression_correct}/{progression_total})")
                
                # 4. Functional harmony accuracy (evaluate based on chord function)
                # This is an approximation based on root movement
                function_correct = 0
                for i in range(1, len(true_chords)):
                    true_curr_root = self.chord_root_map.get(true_chords[i], None)
                    true_prev_root = self.chord_root_map.get(true_chords[i-1], None)
                    pred_curr_root = self.chord_root_map.get(pred_chords[i], None)
                    pred_prev_root = self.chord_root_map.get(pred_chords[i-1], None)
                    
                    if (true_curr_root is not None and true_prev_root is not None and 
                        pred_curr_root is not None and pred_prev_root is not None):
                        
                        # Calculate root movements (intervals)
                        true_movement = (true_curr_root - true_prev_root) % 12
                        pred_movement = (pred_curr_root - pred_prev_root) % 12
                        
                        # If the root movement matches, consider functionally equivalent
                        if true_movement == pred_movement:
                            function_correct += 1
                
                function_accuracy = function_correct / progression_total if progression_total > 0 else 0
                print(f"Functional harmony accuracy: {function_accuracy:.3f} ({function_correct}/{progression_total})")
            else:
                print("Not enough predictions for progression analysis")
            
            # Analyze by chord type
            chord_type_accuracy = defaultdict(lambda: [0, 0])  # [correct, total]
            for true, pred in zip(true_chords, pred_chords):
                chord_type = self.chord_type_map.get(true, "unknown")
                chord_type_accuracy[chord_type][1] += 1
                if true == pred:
                    chord_type_accuracy[chord_type][0] += 1
            
            print("\nAccuracy by chord quality:")
            for chord_type, (correct, total) in sorted(chord_type_accuracy.items()):
                if total > 0:
                    acc = correct / total
                    print(f"{chord_type}: {acc:.3f} ({correct}/{total})")
        
        except Exception as e:
            print(f"Error during evaluation metrics: {str(e)}")
            import traceback
            traceback.print_exc()
            
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
    
    # Create HMM model with enhanced tuneable parameters
    # Use more conservative values to ensure stability
    n_components = 12  # Fewer components for stability - one per pitch class
    n_iter = 100       # Fewer iterations for faster convergence
    
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
