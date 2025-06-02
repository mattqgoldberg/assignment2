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
    """Extract chord quality from chord name with enhanced recognition"""
    if chord_name is None:
        return "unknown"
    
    # Enhanced detection pattern matching
    if any(x in chord_name.lower() for x in ["minor", "min", "-minor", "m ", "m-", "m/"]) and not "maj" in chord_name.lower():
        return "minor"
    elif any(x in chord_name.lower() for x in ["dim", "diminish", "°", "o"]):
        return "diminished"
    elif any(x in chord_name.lower() for x in ["aug", "augment", "+"]):
        return "augmented"
    elif any(x in chord_name.lower() for x in ["dom", "7", "9", "11", "13"]) and not "maj7" in chord_name.lower():
        return "dominant"
    elif any(x in chord_name.lower() for x in ["major", "maj", "-major", "M", "Δ"]):
        return "major"
    elif any(x in chord_name.lower() for x in ["sus", "suspension"]):
        return "suspended"
    elif any(x in chord_name.lower() for x in ["maj7", "Δ7", "M7"]):
        return "major7"
    elif any(x in chord_name.lower() for x in ["min7", "m7"]):
        return "minor7"
    elif any(x in chord_name.lower() for x in ["half-diminish", "ø", "m7b5"]):
        return "half-diminished"
    else:
        # Default to major if just a note name (e.g. "C", "G", etc.)
        if len(chord_name) <= 2 or (len(chord_name) <= 3 and chord_name[1] in ['#', '-']):
            return "major"
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
        """Extract measure-level melody and chord sequences from MIDI files for HMM
        with enhanced musical context extraction"""
        
        melody_sequences = []  # Will contain sequences of melody notes per measure
        chord_sequences = []   # Will contain chord labels per measure
        file_boundaries = []   # Will store the boundaries between different songs
        key_signatures = []    # Will store key signature information for musical context
        meter_signatures = []  # Will store meter/time signature information
        
        # Check if the directories exist and are not the same
        if not os.path.exists(melody_dir):
            raise FileNotFoundError(f"Melody directory {melody_dir} does not exist")
        
        if not os.path.exists(chord_dir):
            raise FileNotFoundError(f"Chord directory {chord_dir} does not exist")
        
        # Get the list of files - handle both separate and same directory case
        melody_files = []
        
        # Check if we have a dedicated melody directory
        if os.path.exists(os.path.join(melody_dir, "melody")):
            melody_subdir = os.path.join(melody_dir, "melody")
            melody_files = sorted([f for f in os.listdir(melody_subdir) if f.endswith('.mid')])
            # Adjust paths for the files
            melody_dir = melody_subdir
            chord_dir = os.path.join(chord_dir, "chords")
        else:
            melody_files = sorted([f for f in os.listdir(melody_dir) if f.endswith('.mid')])
        
        current_position = 0
        
        for file_id, f in enumerate(melody_files):
            chord_file = f
            
            # Try to find corresponding chord file with different prefixes/suffixes
            if not os.path.exists(os.path.join(chord_dir, chord_file)):
                # Try with _chord or _harmony suffix
                base = os.path.splitext(f)[0]
                for suffix in ['_chord', '_chords', '_harmony']:
                    potential_file = base + suffix + '.mid'
                    if os.path.exists(os.path.join(chord_dir, potential_file)):
                        chord_file = potential_file
                        break
            
            if not os.path.exists(os.path.join(chord_dir, chord_file)):
                print(f"Warning: No chord file found for {f}, skipping...")
                continue
                
            try:
                # Parse the files
                melody_score = converter.parse(os.path.join(melody_dir, f))
                chord_score = converter.parse(os.path.join(chord_dir, f))
                
                # Extract global musical context
                # Try to find key signature
                file_key_sig = None
                ks_elements = melody_score.flat.getElementsByClass(key.KeySignature)
                if ks_elements:
                    file_key_sig = ks_elements[0]
                    
                # Try to analyze key if no key signature found
                if not file_key_sig:
                    try:
                        file_key = analysis.discrete.analyzeStream(melody_score, 'key')
                        if file_key:
                            file_key_sig = file_key
                    except:
                        pass
                
                # Try to get time signature
                file_meter = None
                ts_elements = melody_score.flat.getElementsByClass(meter.TimeSignature)
                if ts_elements:
                    file_meter = ts_elements[0]
                
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
                    
                    # Check for local key changes in this measure
                    measure_key_sig = file_key_sig
                    local_ks = melody_measures[i].getElementsByClass(key.KeySignature)
                    if local_ks:
                        measure_key_sig = local_ks[0]
                    
                    # Check for local time signature changes
                    measure_meter = file_meter
                    local_ts = melody_measures[i].getElementsByClass(meter.TimeSignature)
                    if local_ts:
                        measure_meter = local_ts[0]
                    
                    # Extract basic note and chord information
                    melody_note_names = [get_note_name(n) for n in melody_notes]
                    chord_labels = [get_note_name(c) for c in chords_in_measure]
                    chord_label = max(set(chord_labels), key=chord_labels.count)
                    
                    # For each note in the measure, add a (note, chord) pair with musical context
                    for note_name in melody_note_names:
                        melody_sequences.append(note_name)
                        chord_sequences.append(chord_label)
                        key_signatures.append(measure_key_sig)
                        meter_signatures.append(measure_meter)
                        file_sequence.append((note_name, chord_label))
                
                if file_sequence:
                    # Store the boundary position after each file
                    file_boundaries.append(current_position + len(file_sequence))
                    current_position += len(file_sequence)
            
            except Exception as e:
                print(f"Error processing {f}: {str(e)}")
                continue
        
        # Store musical context as object attributes for later use
        self.key_signatures = key_signatures
        self.meter_signatures = meter_signatures
        
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
        
        # Enhance chord representation with more detailed analysis
        self.chord_root_map = {}
        self.chord_type_map = {}
        self.chord_components = {}  # Store chord components for richer representation
        
        # For common chord types, define their component intervals
        chord_interval_templates = {
            "major": [0, 4, 7],                   # Root, major 3rd, perfect 5th
            "minor": [0, 3, 7],                   # Root, minor 3rd, perfect 5th 
            "diminished": [0, 3, 6],              # Root, minor 3rd, diminished 5th
            "augmented": [0, 4, 8],               # Root, major 3rd, augmented 5th
            "dominant": [0, 4, 7, 10],            # Root, major 3rd, perfect 5th, minor 7th
            "major7": [0, 4, 7, 11],              # Root, major 3rd, perfect 5th, major 7th
            "minor7": [0, 3, 7, 10],              # Root, minor 3rd, perfect 5th, minor 7th
            "half-diminished": [0, 3, 6, 10],     # Root, minor 3rd, diminished 5th, minor 7th
            "suspended": [0, 5, 7]                # Root, perfect 4th, perfect 5th
        }
        
        for chord in self.chord_vocab:
            # Extract chord root (usually the first part before any quality indicators)
            if "-" in chord:
                root = chord.split("-")[0]
            else:
                root = chord.split(" ")[0] if " " in chord else chord
            
            # Get pitch class of root
            root_pc = get_pitch_class(root)
            self.chord_root_map[chord] = root_pc
            
            # Get chord quality/type
            chord_quality = get_chord_quality(chord)
            self.chord_type_map[chord] = chord_quality
            
            # Build chord component representation
            if chord_quality in chord_interval_templates:
                # Add all component pitch classes based on the template
                components = [(root_pc + interval) % 12 for interval in chord_interval_templates[chord_quality]]
                self.chord_components[chord] = components
            else:
                # Default to just the root if we can't determine components
                self.chord_components[chord] = [root_pc]
        
        # Create circle of fifths distance map for enhanced chord relationships
        self.circle_of_fifths_map = {}
        for pc1 in range(12):  # For each pitch class
            self.circle_of_fifths_map[pc1] = {}
            for pc2 in range(12):  # Calculate distance on circle of fifths
                # Number of steps around circle of fifths (perfect 5ths)
                # Each step is 7 semitones
                steps = 0
                current = pc1
                while current != pc2:
                    current = (current + 7) % 12  # Move by perfect fifth
                    steps += 1
                    if steps > 12:  # Safety check
                        steps = 0
                        break
                self.circle_of_fifths_map[pc1][pc2] = min(steps, 12 - steps)  # Take shortest distance
        
        # Calculate additional musical features, like melodic intervals
        if len(melody_encoded) > 1:
            self.melodic_intervals = []
            for i in range(1, len(melody_sequences)):
                prev_pc = self.pitch_class_map[melody_sequences[i-1]]
                curr_pc = self.pitch_class_map[melody_sequences[i]]
                interval = (curr_pc - prev_pc) % 12
                if interval > 6:  # Convert to smallest interval (-6 to 6)
                    interval = interval - 12
                self.melodic_intervals.append(interval)
            self.melodic_intervals.append(0)  # Add a zero for the last note
        else:
            self.melodic_intervals = [0] * len(melody_encoded)
        
        return melody_encoded, chord_encoded
    
    def train(self, melody_encoded, chord_encoded, file_boundaries=None):
        """Train the HMM model on encoded sequences with enhanced musical features and stable initialization"""
        print(f"Training HMM with {self.n_components} hidden states...")
        
        if len(melody_encoded) == 0:
            print("Error: No training data available")
            return False
        
        # Make sure data is not too large to avoid numerical issues
        max_train_size = 5000  # Reasonable size for stability
        if len(melody_encoded) > max_train_size:
            print(f"Limiting training data to {max_train_size} samples for stability")
            indices = np.random.choice(len(melody_encoded), max_train_size, replace=False)
            melody_encoded = melody_encoded[indices]
            chord_encoded = chord_encoded[indices]
        
        # Enhanced feature encoding that combines:
        # 1. One-hot encoding of notes
        # 2. Pitch class information (circular distance relationships)
        # 3. Recent note history to capture melodic contour
        # 4. Chord component membership (for each pitch class)
        n_notes = len(self.note_vocab)
        n_features = n_notes + 6  # One-hot + 6 musical features
        
        # Create enhanced feature arrays
        X_features = np.zeros((len(melody_encoded), n_features))
        
        # One-hot encode the melody notes with enhanced context
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
                
                # Add additional features:
                # 1. Is this note in current chord?
                chord_name = self.idx2chord[chord_encoded[i]]
                chord_components = self.chord_components.get(chord_name, [self.chord_root_map.get(chord_name, 0)])
                X_features[i, n_notes+3] = 1.0 if pitch_class in chord_components else 0.0
                
                # 2. Circle of fifths distance between note and chord root
                chord_root = self.chord_root_map.get(chord_name, 0)
                cf_distance = self.circle_of_fifths_map.get(pitch_class, {}).get(chord_root, 6)
                X_features[i, n_notes+4] = cf_distance / 6.0  # Normalize to 0-1
                
                # 3. Is the note a chord tone in the previous chord?
                if i > 0:
                    prev_chord_name = self.idx2chord[chord_encoded[i-1]]
                    prev_chord_components = self.chord_components.get(
                        prev_chord_name, [self.chord_root_map.get(prev_chord_name, 0)])
                    X_features[i, n_notes+5] = 1.0 if pitch_class in prev_chord_components else 0.0
        
        # Use GaussianHMM with simpler, more stable initialization
        # We'll use fewer hidden states (10-12) to avoid numerical instability
        # This also matches our musical understanding (roughly one state per pitch class is sufficient)
        self.model = hmm.GaussianHMM(
            n_components=self.n_components, 
            n_iter=self.n_iter,
            covariance_type="diag",  # Diagonal covariance for stability
            random_state=42,
            init_params="stmc",      # Initialize all parameters with default values
            params="stmc",           # Train all parameters
            verbose=True,            # Show training progress
            tol=0.01,                # More lenient convergence tolerance
        )
        
        # Define lengths of independent sequences if file boundaries are provided
        if file_boundaries:
            # Calculate lengths of sequences between boundaries
            lengths = []
            prev = 0
            for boundary in file_boundaries:
                lengths.append(boundary - prev)
                prev = boundary
            
            # Only add the final piece if there are notes left
            if len(melody_encoded) > prev:
                lengths.append(len(melody_encoded) - prev)
            
            # Filter out zero-length sequences
            lengths = [length for length in lengths if length > 0]
            print(f"Training on {len(lengths)} separate sequences")
        else:
            lengths = None
        
        # Fit the model once with default initialization to set up the arrays
        # This will establish the proper dimensions for all parameters
        self.model.fit(X_features[:100])  # Just use a small sample to initialize
        
        # Now we can set our custom initialization parameters
        # 1. Start probabilities - uniform across states
        self.model.startprob_ = np.ones(self.n_components) / self.n_components
        
        # 2. Transition matrix - more sophisticated structure
        # We'll use musical knowledge about chord progressions
        self.model.transmat_ = np.zeros((self.n_components, self.n_components))
        
        # We'll create a transition matrix where:
        # - Each state has higher transition probability to neighboring states
        # - States that represent common chord tones have higher transition probability to each other
        # - Some transitions are more common in music (fifth relationships, etc.)
        
        # First, identify states corresponding to musical degrees (based on component count)
        for i in range(self.n_components):
            # Higher self-transition for stability
            self.model.transmat_[i, i] = 0.3
            
            # Distribute remaining probability among musically sensible transitions
            for j in range(self.n_components):
                if i != j:  # Skip self-transition (already set)
                    # Base probability - will be refined
                    self.model.transmat_[i, j] = 0.05
                    
                    # Enhance transitions between musically related states
                    # For example, give higher probabilities to circle of fifths relationships
                    i_class = i % 12  # Map to pitch class (0-11)
                    j_class = j % 12
                    
                    # Circle of fifths relationship (perfect fifth)
                    if (i_class + 7) % 12 == j_class:  # Perfect fifth up
                        self.model.transmat_[i, j] += 0.1
                    elif (i_class + 5) % 12 == j_class:  # Perfect fourth up (fifth down)
                        self.model.transmat_[i, j] += 0.08
                    # Stepwise motion (common in melodies)
                    elif (i_class + 1) % 12 == j_class or (i_class + 11) % 12 == j_class:
                        self.model.transmat_[i, j] += 0.06
                    # Major/minor third relationships (triad components)
                    elif (i_class + 3) % 12 == j_class or (i_class + 4) % 12 == j_class:
                        self.model.transmat_[i, j] += 0.04
                        
        # Create circle of fifths distance map if not created earlier
        if not hasattr(self, 'circle_of_fifths_map'):
            self.circle_of_fifths_map = {}
            for pc1 in range(12):  # For each pitch class
                self.circle_of_fifths_map[pc1] = {}
                for pc2 in range(12):  # Calculate distance on circle of fifths
                    # Number of steps around circle of fifths (perfect 5ths)
                    # Each step is 7 semitones
                    steps = 0
                    current = pc1
                    while current != pc2:
                        current = (current + 7) % 12  # Move by perfect fifth
                        steps += 1
                        if steps > 12:  # Safety check
                            steps = 0
                            break
                    self.circle_of_fifths_map[pc1][pc2] = min(steps, 12 - steps)  # Take shortest distance
        
        # Normalize the transition matrix (rows must sum to 1)
        for i in range(self.n_components):
            row_sum = self.model.transmat_[i].sum()
            if row_sum > 0:
                self.model.transmat_[i] /= row_sum
        
        # 3. Initialize means vectors for emission distributions
        # Clean out any existing initialization
        self.model.means_ = np.zeros((self.n_components, n_features))
        
        # Initialize based on musical knowledge
        for state in range(self.n_components):
            # Each state primarily corresponds to a pitch class
            pitch_class = state % 12
            
            # Set means for one-hot part based on notes in this pitch class
            for note_idx, note in self.idx2note.items():
                note_pc = self.pitch_class_map.get(note, 0)
                if note_pc == pitch_class:
                    self.model.means_[state, note_idx] = 0.8
                
            # Set musical feature means
            self.model.means_[state, n_notes] = pitch_class / 12.0  # Pitch class
            
            # Create musically meaningful initializations for other features
            if state < self.n_components // 2:
                # For first half of states - ascending melodic patterns
                self.model.means_[state, n_notes+1] = 0.3  # Positive interval
                self.model.means_[state, n_notes+2] = 0.7  # Upward direction
            else:
                # For second half - descending melodic patterns
                self.model.means_[state, n_notes+1] = -0.3  # Negative interval
                self.model.means_[state, n_notes+2] = -0.7  # Downward direction
                
            # Initialize chord-relatedness features
            self.model.means_[state, n_notes+3] = 0.6  # Note tends to be in chord
            self.model.means_[state, n_notes+4] = 0.3  # Close on circle of fifths
            self.model.means_[state, n_notes+5] = 0.5  # Might be in previous chord
        
        # 4. Initialize covariance matrices for stability
        # Start with uniform moderate covariance
        for state in range(self.n_components):
            # Set a base value for all features
            self.model.covars_[state] = 0.5
            
            # Use larger variances for one-hot part (more flexibility)
            for j in range(n_notes):
                self.model.covars_[state, j] = 2.0
                
            # Use smaller variances for musical features (more constrained)
            for j in range(n_notes, n_features):
                self.model.covars_[state, j] = 0.25
        
        # Train the model with early stopping and restarts to avoid poor local minima
        try:
            # Train with the enhanced features - simplified for stability
            try:
                # Just do a single, reliable training pass
                self.model.fit(X_features)
                log_likelihood = self.model.score(X_features)
                print(f"Log likelihood: {log_likelihood:.2f}")
                
                self.is_trained = True
                print(f"HMM training completed successfully with {self.n_components} hidden states.")
                
                # Store the training data and features for later analysis
                self.training_data = (melody_encoded, chord_encoded)
                self.X_features = X_features
                
                # Ensure we have the state-chord mapping ready
                print("Calculating state-chord probabilities...")
                self._calculate_state_chord_probs()
                
                # Evaluate the learned hidden states
                self._analyze_hidden_states(melody_encoded, chord_encoded)
                return True
                
            except Exception as e:
                print(f"HMM training error: {str(e)}")
                import traceback
                traceback.print_exc()
                return False
            
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
            # Match the feature dimension used during training
            n_features = n_notes + 6  # Ensuring consistency with training features
            X_features = np.zeros((len(melody_encoded), n_features))
            
            for i, idx in enumerate(melody_encoded):
                # One-hot encoding
                X_features[i, idx] = 1.0
                
                # Add pitch class feature
                note_name = self.idx2note[idx]
                pitch_class = self.pitch_class_map.get(note_name, 0)
                X_features[i, n_notes] = pitch_class / 12.0
                
                # Initialize all additional features with neutral values
                X_features[i, n_notes+1:] = 0.5
                
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
        """Predict chords for a new melody sequence using advanced harmonic analysis and music theory"""
        if not self.is_trained:
            print("Model not trained yet.")
            return []
            
        # Handle empty input
        if not melody_sequence:
            print("Warning: Empty melody sequence provided")
            return []
            
        # Ensure we have everything needed for prediction
        if not all(hasattr(self, attr) for attr in ['note2idx', 'pitch_class_map', 'chord_root_map', 'chord_type_map']):
            print("Warning: Model missing required attributes. Initializing with defaults.")
            
            # Initialize missing attributes with defaults if needed
            if not hasattr(self, 'pitch_class_map'):
                self.pitch_class_map = {note: get_pitch_class(note) for note in self.note_vocab} if hasattr(self, 'note_vocab') else {}
                
            if not hasattr(self, 'chord_root_map'):
                self.chord_root_map = {}
                if hasattr(self, 'chord_vocab'):
                    for chord in self.chord_vocab:
                        root = chord.split(" ")[0] if " " in chord else chord
                        self.chord_root_map[chord] = get_pitch_class(root)
                        
            if not hasattr(self, 'chord_type_map'):
                self.chord_type_map = {}
                if hasattr(self, 'chord_vocab'):
                    for chord in self.chord_vocab:
                        self.chord_type_map[chord] = get_chord_quality(chord)
        
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
                    
                    # Find closest note or use a default if vocabulary is empty
                    if self.note2idx:
                        closest_note = min(self.note2idx.keys(), 
                                          key=lambda x: abs(get_pitch_class(x) - pitch_class))
                        melody_encoded.append(self.note2idx[closest_note])
                    else:
                        print("Warning: Empty note vocabulary. Using default encoding.")
                        melody_encoded.append(0)  # Default to first index
            
            melody_encoded = np.array(melody_encoded)
            
            # Double-check we have data to work with
            if len(melody_encoded) == 0:
                print("Warning: No valid notes found in melody sequence")
                return []
                
        except Exception as e:
            print(f"Error encoding melody: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
        
        # Create enhanced features for prediction
        n_notes = len(self.note_vocab)
        # Must match the number of features used in training
        n_features = n_notes + 6  # One-hot + 6 musical features
        X_features = np.zeros((len(melody_encoded), n_features))
        
        # Track melodic context
        scale_degrees = []
        intervals = []
        
        # Create the feature vectors with the same structure used in training
        for i, idx in enumerate(melody_encoded):
            # Basic one-hot encoding
            X_features[i, idx] = 1.0
            
            # Add pitch class as a feature (normalized to 0-1)
            note_name = self.idx2note[idx]
            pitch_class = self.pitch_class_map.get(note_name, 0)
            X_features[i, n_notes] = pitch_class / 12.0
            
            # Track scale degree information for later analysis
            scale_degrees.append(pitch_class)
            
            # Create placeholder values for the remaining features
            # These will be properly set after key/chord analysis
            # For now, set to neutral values
            X_features[i, n_notes+1:] = 0.5  
            
            # Add melodic movement features (if not at the beginning)
            if i > 0:
                # Pitch distance from previous note
                prev_note = self.idx2note[melody_encoded[i-1]]
                prev_pitch = self.pitch_class_map.get(prev_note, 0)
                pitch_diff = (pitch_class - prev_pitch) % 12
                if pitch_diff > 6:
                    pitch_diff = pitch_diff - 12  # Map to -6 to +6 range
                
                X_features[i, n_notes+1] = pitch_diff / 6.0
                intervals.append(pitch_diff)
                
                # Direction of melodic movement
                if pitch_diff == 0:
                    movement = 0
                else:
                    movement = 1 if pitch_diff > 0 else -1
                X_features[i, n_notes+2] = movement
        
        # Advanced key analysis - improved algorithm
        # First detect key using Krumhansl-Schmuckler key-finding algorithm with the Aarden-Essen weightings
        # These are empirically derived weights for each scale degree's importance in key identification
        major_profile = np.array([17.7661, 0.145624, 14.9265, 0.160186, 19.8049, 11.3587, 0.291248, 
                                  22.062, 0.145624, 8.15494, 0.232998, 4.95122])
        minor_profile = np.array([18.2648, 0.737619, 14.0499, 16.8599, 0.702494, 14.4362, 0.702494, 
                                  18.6161, 4.56621, 1.93186, 7.37619, 1.75623])
        
        # Count occurrences of each pitch class
        pitch_counts = np.zeros(12)
        for pc in melody_pitch_classes:
            pitch_counts[pc] += 1
            
        # Normalize counts
        if sum(pitch_counts) > 0:
            pitch_counts = pitch_counts / sum(pitch_counts)
        
        # Calculate correlation with each possible key
        key_correlations = []
        for i in range(12):  # For each possible tonic pitch class
            # Shift the pitch count vector to the key's frame of reference
            shifted_counts = np.roll(pitch_counts, -i)
            
            # Compute correlation with major and minor profiles
            major_corr = np.corrcoef(shifted_counts, major_profile)[0, 1]
            minor_corr = np.corrcoef(shifted_counts, minor_profile)[0, 1]
            
            key_correlations.append((i, True, major_corr))   # Major key
            key_correlations.append((i, False, minor_corr))  # Minor key
        
        # Sort by correlation coefficient
        key_correlations.sort(key=lambda x: x[2], reverse=True)
        
        # Get the most likely key
        key_center, is_major, key_correlation = key_correlations[0]
        key_quality = "major" if is_major else "minor"
        print(f"Detected key: {(['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][key_center])} {key_quality}")
        
        # Use Viterbi algorithm to find most likely hidden state sequence
        try:
            # First predict the hidden state sequence
            hidden_states = self.model.predict(X_features)
            
            # Ensure we have state-chord probabilities
            if not hasattr(self, 'state_chord_probs'):
                print("Calculating state-chord probabilities...")
                self._calculate_state_chord_probs()
            
            # Now use sophisticated methods to generate chord progression
            
            # Define scale degrees for the detected key
            if is_major:
                # Major key scale degrees: 1, 2, 3, 4, 5, 6, 7
                scale_degrees = [(key_center) % 12, 
                                 (key_center + 2) % 12, 
                                 (key_center + 4) % 12,
                                 (key_center + 5) % 12,
                                 (key_center + 7) % 12,
                                 (key_center + 9) % 12,
                                 (key_center + 11) % 12]
                
                # Common chord qualities for each scale degree in major keys
                degree_qualities = {
                    0: ["major", "major7"],            # I - tonic
                    1: ["minor", "minor7"],            # ii - supertonic 
                    2: ["minor", "minor7"],            # iii - mediant
                    3: ["major", "major7"],            # IV - subdominant
                    4: ["major", "dominant"],          # V - dominant
                    5: ["minor", "minor7"],            # vi - submediant
                    6: ["diminished", "half-diminished"]  # vii° - leading tone
                }
            else:
                # Minor key scale degrees: 1, 2, b3, 4, 5, b6, b7/7
                scale_degrees = [(key_center) % 12, 
                                 (key_center + 2) % 12,
                                 (key_center + 3) % 12,
                                 (key_center + 5) % 12,
                                 (key_center + 7) % 12,
                                 (key_center + 8) % 12, 
                                 (key_center + 10) % 12]
                
                # Common chord qualities for each scale degree in minor keys
                degree_qualities = {
                    0: ["minor", "minor7"],            # i - tonic
                    1: ["diminished", "half-diminished"],  # ii° - supertonic
                    2: ["major", "major7"],            # III - mediant
                    3: ["minor", "minor7"],            # iv - subdominant
                    4: ["minor", "dominant"],          # v/V - dominant
                    5: ["major", "major7"],            # VI - submediant
                    6: ["major", "dominant"]           # VII - subtonic
                }
            
            # Define common chord progressions as transition weights
            # Based on functional harmony principles
            # From -> To: Tonic(0), Subdominant(3), Dominant(4), Secondary Dominants, etc.
            progression_weights = {
                # From tonic
                0: {3: 2.0, 4: 1.8, 5: 1.5, 1: 1.3, 2: 1.2, 6: 1.0},
                
                # From supertonic
                1: {4: 2.0, 0: 1.5, 3: 1.3, 6: 1.2, 5: 1.0},
                
                # From mediant
                2: {5: 1.8, 3: 1.5, 0: 1.3, 4: 1.2, 1: 1.0},
                
                # From subdominant
                3: {4: 2.5, 0: 1.8, 1: 1.5, 2: 1.3, 5: 1.2, 6: 1.0},
                
                # From dominant
                4: {0: 3.0, 5: 1.8, 3: 1.5, 2: 1.3, 1: 1.0},
                
                # From submediant
                5: {1: 1.8, 4: 1.5, 0: 1.3, 2: 1.2, 3: 1.0},
                
                # From leading tone/subtonic
                6: {0: 2.5, 4: 2.0, 3: 1.5, 5: 1.3, 2: 1.2}
            }
            
            # Create a more sophisticated chord progression using both HMM states and music theory
            chord_predictions = []
            last_chord_idx = None
            last_scale_degree_idx = None  # Track position in the scale
            
            # First pass - use Hidden Markov Model states to produce initial predictions
            hmm_chord_candidates = [] # List of lists of candidates for each position
            
            # Ensure we have state_chord_candidates attribute
            if not hasattr(self, 'state_chord_candidates') or not self.state_chord_candidates:
                # Initialize state_chord_candidates if it doesn't exist
                self.state_chord_candidates = {}
                for s in range(self.n_components):
                    if hasattr(self, 'state_chord_probs') and s in self.state_chord_probs:
                        chord_idx = self.state_chord_probs[s]
                        self.state_chord_candidates[s] = [(chord_idx, 1.0)]
                    else:
                        # Fallback: map each state to multiple chord candidates with probabilities
                        candidates = []
                        for chord_idx in range(min(5, len(self.chord_vocab))):
                            # Diminishing probabilities
                            prob = 1.0 / (chord_idx + 1)
                            candidates.append((chord_idx, prob))
                        self.state_chord_candidates[s] = candidates
            
            # For each position, get top chord candidates from the HMM
            for i, state in enumerate(hidden_states):
                # Get candidate chords for this state
                if state in self.state_chord_candidates:
                    candidates = self.state_chord_candidates[state]
                    hmm_chord_candidates.append(candidates)
                else:
                    # Use fallback if this state has no candidates
                    hmm_chord_candidates.append([(0, 1.0)])  # Default to first chord with weight 1.0
            
            # Second pass - refine chord progression using harmonic rules and context
            # Use a sliding window approach (typical progression is 4-8 measures)
            progression_window = min(8, len(melody_encoded))
            
            # Process in overlapping windows to ensure coherence
            for i in range(len(melody_encoded)):
                # Calculate window boundaries
                start = max(0, i - progression_window // 2)
                end = min(len(melody_encoded), i + progression_window // 2) 
                
                # Get local pitch context
                local_pitches = melody_pitch_classes[start:end]
                
                # Score chord candidates for this position
                candidates = []
                
                # Get HMM candidates for this position
                if i < len(hmm_chord_candidates):
                    hmm_candidates = hmm_chord_candidates[i]
                else:
                    # Fallback if we're missing candidates
                    hmm_candidates = [(0, 1.0)]
                
                # Analyze melodic context more deeply
                current_pc = melody_pitch_classes[i]
                
                # Find the scale degree of this melody note (0-6)
                current_degree = -1
                for j, degree_pc in enumerate(scale_degrees):
                    if degree_pc == current_pc:
                        current_degree = j
                        break
                
                # If not a scale degree, find closest one
                if current_degree == -1:
                    # Find closest scale degree
                    current_degree = min(range(7), key=lambda j: min((current_pc - scale_degrees[j]) % 12,
                                                                    (scale_degrees[j] - current_pc) % 12))
                
                # For each candidate from the HMM
                for chord_idx, hmm_prob in hmm_candidates:
                    chord_name = self.idx2chord[chord_idx]
                    chord_root = self.chord_root_map.get(chord_name, 0)
                    chord_quality = self.chord_type_map.get(chord_name, "unknown")
                    
                    # 1. Start with base probability from HMM
                    score = hmm_prob * 3.0  # Weight the HMM prediction strongly
                    
                    # 2. Determine functional role of chord
                    # Find which scale degree this chord root corresponds to
                    chord_degree = -1
                    for j, degree_pc in enumerate(scale_degrees):
                        if degree_pc == chord_root:
                            chord_degree = j
                            break
                    
                    # 3. Calculate diatonic compatibility score
                    if chord_degree != -1:  # In the key
                        # Check if chord quality matches expected quality for this scale degree
                        expected_qualities = degree_qualities.get(chord_degree, [])
                        if chord_quality in expected_qualities:
                            # Perfect match for both root and quality
                            score *= 2.0
                        else:
                            # Just the root matches
                            score *= 1.5
                            
                        # Apply functional harmony progression weights
                        if last_scale_degree_idx is not None:
                            # Get weight for this progression
                            progression_weight = progression_weights.get(last_scale_degree_idx, {}).get(chord_degree, 1.0)
                            score *= progression_weight
                    
                    # 4. Melodic-harmonic relationship
                    # Check if melody note is in the chord
                    chord_name = self.idx2chord[chord_idx]
                    if chord_name in self.chord_components:
                        chord_components = self.chord_components[chord_name]
                        if current_pc in chord_components:
                            # Melody note is a chord tone
                            score *= 1.7
                            
                            # Is it the root, third, fifth?
                            if current_pc == chord_root:  # Root
                                score *= 1.2
                            elif (current_pc - chord_root) % 12 in [3, 4]:  # Third
                                score *= 1.15
                            elif (current_pc - chord_root) % 12 == 7:  # Fifth
                                score *= 1.1
                    
                    # 5. Consider voice leading and smooth progression
                    if last_chord_idx is not None:
                        last_chord = self.idx2chord[last_chord_idx]
                        last_root = self.chord_root_map.get(last_chord, 0)
                        
                        # Avoid direct repetition unless high probability
                        if chord_idx == last_chord_idx:
                            if hmm_prob > 0.7:
                                # Allow repetition with smaller penalty if very likely
                                score *= 0.85
                            else:
                                # Strong penalty for unlikely repetition
                                score *= 0.4
                        
                        # Prefer smooth root motion
                        root_distance = min((chord_root - last_root) % 12, (last_root - chord_root) % 12)
                        
                        # Circle of fifths movement (strongest)
                        if root_distance == 7:  # Perfect 5th
                            score *= 1.8
                        elif root_distance == 5:  # Perfect 4th
                            score *= 1.6
                        # Step-wise root motion (smooth)
                        elif root_distance <= 2:
                            score *= 1.4
                        # Thirds (triadic relationships)
                        elif root_distance in [3, 4, 8, 9]:
                            score *= 1.3
                    
                    candidates.append((chord_idx, score))
                
                # Choose the best candidate
                if candidates:
                    best_chord_idx, _ = max(candidates, key=lambda x: x[1])
                    best_chord = self.idx2chord[best_chord_idx]
                    
                    # Update the progression state
                    best_root = self.chord_root_map.get(best_chord, 0)
                    
                    # Find which scale degree the best chord corresponds to
                    best_degree = -1
                    for j, degree_pc in enumerate(scale_degrees):
                        if degree_pc == best_root:
                            best_degree = j
                            break
                    
                    # Store results
                    chord_predictions.append(best_chord)
                    last_chord_idx = best_chord_idx
                    last_scale_degree_idx = best_degree
                else:
                    # Fallback - use most common chord in the detected key
                    fallback_degree = 0  # Default to tonic
                    fallback_root = scale_degrees[fallback_degree]
                    
                    # Find a chord with this root
                    fallback_chord = None
                    for chord in self.chord_vocab:
                        if self.chord_root_map.get(chord, -1) == fallback_root:
                            fallback_chord = chord
                            break
                    
                    if fallback_chord is None and self.chord_vocab:
                        fallback_chord = self.chord_vocab[0]
                    
                    if fallback_chord:
                        chord_predictions.append(fallback_chord)
                        last_chord_idx = self.chord2idx.get(fallback_chord, 0)
                        last_scale_degree_idx = fallback_degree
            
            return chord_predictions
            
        except Exception as e:
            print(f"Error predicting chords: {str(e)}")
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
            
            # Use the same feature dimensions that were used during training
            n_notes = len(self.note_vocab) 
            n_features = n_notes + 6  # Must match what was used in training
            X_features = np.zeros((len(melody_encoded), n_features))
            
            for i, idx in enumerate(melody_encoded):
                # One-hot encoding
                X_features[i, idx] = 1.0
                
                # Add pitch class feature
                note_name = self.idx2note[idx]
                pitch_class = self.pitch_class_map.get(note_name, 0)
                X_features[i, n_notes] = pitch_class / 12.0
                
                # Add placeholder values for other features to match dimensions
                X_features[i, n_notes+1:] = 0.5  # Neutral values
                
                # Add melodic movement features (if not at the beginning)
                if i > 0:
                    prev_note = self.idx2note[melody_encoded[i-1]]
                    prev_pitch = self.pitch_class_map.get(prev_note, 0)
                    pitch_diff = (pitch_class - prev_pitch) % 12
                    if pitch_diff > 6:
                        pitch_diff = pitch_diff - 12
                    X_features[i, n_notes+1] = pitch_diff / 6.0
                    
                    movement = 0 if pitch_diff == 0 else (1 if pitch_diff > 0 else -1)
                    X_features[i, n_notes+2] = movement
            
            # Initialize structures
            self.state_chord_probs = {}
            self.state_chord_candidates = {}
            
            # Predict hidden states for the training data using the current model
            hidden_states = self.model.predict(X_features)
            
            # Instead of using Viterbi which could mismatch dimensions, we'll count directly
            # This is more stable and gives similar results
            
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
    """Train and evaluate the optimized HMM model with enhanced musical features and stability"""
    # Use absolute paths to ensure files are found correctly
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    melody_dir = os.path.join(base_dir, "data", "MIDI")
    chord_dir = os.path.join(base_dir, "data", "MIDI")
    
    # Check if directories exist
    if not os.path.exists(melody_dir):
        print(f"Error: Melody directory {melody_dir} does not exist.")
        print(f"Current working directory: {os.getcwd()}")
        print("Creating a test dataset for demonstration...")
        
        # Create a synthetic dataset for testing
        os.makedirs(os.path.join(base_dir, "data", "MIDI", "melody"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "data", "MIDI", "chords"), exist_ok=True)
        generate_test_dataset(base_dir)
        
        # Update paths to use the test dataset
        melody_dir = os.path.join(base_dir, "data", "MIDI", "melody")
        chord_dir = os.path.join(base_dir, "data", "MIDI", "chords")
    
    # Create HMM model with optimal parameters for music harmony
    # Use 12 states (one per pitch class) to create a musically meaningful model
    # This prevents overfitting and numerical instability while maintaining musical relevance
    n_components = 12  # One per pitch class - musically meaningful states
    n_iter = 100       # Reasonable number of EM iterations to ensure convergence
    
    print("=== Enhanced Hidden Markov Model for Music Harmonization ===")
    print("This model uses music theory principles with machine learning to")
    print("accurately predict chord progressions from melody sequences.")
    
    harmonizer = OptimizedHMMModel(n_components=n_components, n_iter=n_iter)
    
    print("\nExtracting sequences with musical context...")
    melody_sequences, chord_sequences, file_boundaries = harmonizer.extract_sequences(melody_dir, chord_dir)
    print(f"Extracted {len(melody_sequences)} melody-chord pairs from files")
    
    if not melody_sequences:
        print("Error: No sequences extracted. Check data paths.")
        return
    
    # Encode sequences with enhanced musical features
    print("Encoding sequences with enhanced musical features...")
    melody_encoded, chord_encoded = harmonizer.encode_sequences(melody_sequences, chord_sequences)
    print(f"Melody vocabulary size: {len(harmonizer.note_vocab)}")
    print(f"Chord vocabulary size: {len(harmonizer.chord_vocab)}")
    
    # Analyze chord types for better understanding
    chord_qualities = {}
    for chord in harmonizer.chord_vocab:
        quality = harmonizer.chord_type_map.get(chord, "unknown")
        chord_qualities[quality] = chord_qualities.get(quality, 0) + 1
    
    print("\nChord types in dataset:")
    for quality, count in sorted(chord_qualities.items(), key=lambda x: x[1], reverse=True):
        print(f"  {quality}: {count} chords")
    
    # Store training data for later analysis
    harmonizer.training_data = (melody_encoded, chord_encoded)
    
    # Split data into train/test sets with stratification by chord types
    # This ensures we have representative samples of all chord types
    try:
        # Create labels for stratification based on chord roots
        chord_roots = [harmonizer.chord_root_map.get(harmonizer.idx2chord[idx], 0) for idx in chord_encoded]
        melody_train, melody_test, chord_train, chord_test = train_test_split(
            melody_encoded, chord_encoded, test_size=0.2, random_state=42,
            stratify=chord_roots  # Stratify by chord roots for balanced splits
        )
    except Exception:
        # Fallback to regular split if stratification fails
        print("Using standard train/test split")
        melody_train, melody_test, chord_train, chord_test = train_test_split(
            melody_encoded, chord_encoded, test_size=0.2, random_state=42
        )
    
    # Use a much smaller subset for training to ensure model stability
    max_train_size = 10000  # Much smaller for stability with rich feature set
    if len(melody_train) > max_train_size:
        print(f"Training data is large ({len(melody_train)} examples), using {max_train_size} representative samples")
        
        # Simple random sampling for stability
        indices = np.random.choice(len(melody_train), max_train_size, replace=False)
        melody_train = melody_train[indices]
        chord_train = chord_train[indices]
        
    print(f"Training on {len(melody_train)} examples, testing on {len(melody_test)} examples")
    
    # Train the model without boundaries for better stability
    print("\nTraining the enhanced HMM with musical initialization...")
    # Don't pass file_boundaries to avoid sequence length issues
    success = harmonizer.train(melody_train, chord_train)
    
    if success:
        # Create results directory if it doesn't exist
        results_dir = os.path.join(base_dir, "results")
        os.makedirs(results_dir, exist_ok=True)
        
        # Save the trained model
        model_filename = os.path.join(results_dir, "hmm_optimized_model.pkl")
        harmonizer.save(model_filename)
        
        # Evaluate on test data with comprehensive metrics
        print("\nEvaluating on test data...")
        test_melody_sequences = [harmonizer.idx2note[idx] for idx in melody_test]
        test_chord_sequences = [harmonizer.idx2chord[idx] for idx in chord_test]
        accuracy = harmonizer.evaluate(test_melody_sequences, test_chord_sequences)
        
        # Compare with baseline (30.7%)
        baseline = 0.307
        improvement = ((accuracy - baseline) / baseline) * 100
        print(f"\nBaseline accuracy: {baseline:.3f}")
        print(f"New model accuracy: {accuracy:.3f}")
        print(f"Improvement: {improvement:.1f}%")
        
        # Try to generate some example progressions
        print("\nGenerating example chord progression...")
        try:
            # Take a short melody segment from test data
            example_length = 16
            example_melody = test_melody_sequences[:example_length]
            example_true_chords = test_chord_sequences[:example_length]
            
            # Predict chords
            example_pred_chords = harmonizer.predict_chords(example_melody)
            
            print("Example prediction:")
            print("Melody | True Chord | Predicted Chord | Match?")
            print("-" * 60)
            for i in range(min(example_length, len(example_pred_chords))):
                match = "✓" if example_true_chords[i] == example_pred_chords[i] else "✗"
                print(f"{example_melody[i]:6} | {example_true_chords[i]:10} | {example_pred_chords[i]:15} | {match}")
        except Exception as e:
            print(f"Error generating example: {str(e)}")
    else:
        print("Training failed. Check error messages.")

def generate_test_dataset(base_dir):
    """Generate a simple test dataset for HMM model demonstration"""
    from music21 import stream, note, chord, metadata
    import random
    
    print("Creating synthetic dataset for testing...")
    
    # Define common chord progressions in different keys
    progressions = [
        # I-IV-V-I in C
        [('C', 'major'), ('F', 'major'), ('G', 'major'), ('C', 'major')],
        # ii-V-I in C
        [('D', 'minor'), ('G', 'dominant'), ('C', 'major'), ('C', 'major')],
        # I-vi-IV-V in G
        [('G', 'major'), ('E', 'minor'), ('C', 'major'), ('D', 'major')],
        # I-V-vi-IV in A
        [('A', 'major'), ('E', 'major'), ('F#', 'minor'), ('D', 'major')]
    ]
    
    # Note ranges for melodies in different keys
    melody_notes = {
        'C': ['C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5'],
        'G': ['G3', 'A3', 'B3', 'C4', 'D4', 'E4', 'F#4', 'G4'],
        'A': ['A3', 'B3', 'C#4', 'D4', 'E4', 'F#4', 'G#4', 'A4']
    }
    
    # Create 10 sample pieces
    for i in range(1, 11):
        # Create a piece with 4 measures, each with a different chord
        piece_key = random.choice(['C', 'G', 'A'])
        progression = random.choice(progressions)
        
        # Create melody stream
        melody_stream = stream.Stream()
        melody_stream.append(metadata.Metadata())
        melody_stream.metadata.title = f"Test Piece {i}"
        
        # Create chord stream with same metadata
        chord_stream = stream.Stream()
        chord_stream.append(metadata.Metadata())
        chord_stream.metadata.title = f"Test Piece {i}"
        
        # Fill each measure
        for chord_data in progression:
            chord_root, chord_quality = chord_data
            
            # Add a chord to the chord stream
            if chord_quality == 'major':
                c = chord.Chord([chord_root + '3', chord_root + '4', chord_root + '5'], quarterLength=4.0)
                c.pitchedCommonName = chord_root
            elif chord_quality == 'minor':
                c = chord.Chord([chord_root + '3', chord_root + 'm4', chord_root + '5'], quarterLength=4.0)
                c.pitchedCommonName = chord_root + 'm'
            elif chord_quality == 'dominant':
                c = chord.Chord([chord_root + '3', chord_root + '4', chord_root + '5', chord_root + '7'], quarterLength=4.0)
                c.pitchedCommonName = chord_root + '7'
            else:
                c = chord.Chord([chord_root + '3', chord_root + '4', chord_root + '5'], quarterLength=4.0)
                c.pitchedCommonName = chord_root
                
            chord_stream.append(c)
            
            # Add 4 quarter notes to the melody for this measure
            for j in range(4):
                n = note.Note(random.choice(melody_notes[piece_key]))
                n.quarterLength = 1.0
                melody_stream.append(n)
        
        # Save the MIDI files
        melody_path = os.path.join(base_dir, "data", "MIDI", "melody", f"test_piece_{i}.mid")
        chord_path = os.path.join(base_dir, "data", "MIDI", "chords", f"test_piece_{i}.mid")
        
        melody_stream.write('midi', fp=melody_path)
        chord_stream.write('midi', fp=chord_path)
        
    print(f"Generated 10 test pieces in {os.path.join(base_dir, 'data', 'MIDI')}")

if __name__ == "__main__":
    main()
