# HMM-based harmonization model for melody-to-chord prediction
# Uses hmmlearn to train a discrete HMM on aligned melody/chord sequences
import os
from music21 import converter, note, chord, analysis, key, interval
from collections import Counter, defaultdict
import numpy as np
import pickle
from music21 import meter
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def get_note_name(n):
    if isinstance(n, note.Note):
        return n.name  # keep melody as note name only
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
        
    if "minor" in chord_name or "-minor" in chord_name:
        return "minor"
    elif "major" in chord_name or "-major" in chord_name:
        return "major"
    elif "dominant" in chord_name or "7" in chord_name:
        return "dominant"
    elif "diminished" in chord_name:
        return "diminished"
    elif "augmented" in chord_name:
        return "augmented"
    else:
        return "unknown"

class HarmonizationModel:
    def __init__(self, model_type="rf"):
        self.model = None
        self.note2idx = None
        self.chord2idx = None
        self.idx2chord = None
        self.note_vocab = None
        self.chord_vocab = None
        self.is_enhanced = True
        self.model_type = model_type
        self.scaler = None
    
    def extract_sequences(self, melody_dir, chord_dir):
        """Extract melody and chord sequences from MIDI files with enhanced music features"""
        X_data = []  # Will store all features
        Y = []       # Will store chord labels
        file_ids = []
        measure_positions = []
        
        files = sorted(os.listdir(melody_dir))
        for file_id, f in enumerate(files):
            if not f.endswith('.mid') or not os.path.exists(os.path.join(chord_dir, f)):
                continue
                
            try:
                # Parse the files
                melody_score = converter.parse(os.path.join(melody_dir, f))
                chord_score = converter.parse(os.path.join(chord_dir, f))
                
                # Try to detect key signature
                try:
                    k = analysis.discrete.analyzeStream(melody_score, 'key')
                    key_sig = k.tonic.name + (' minor' if k.mode == 'minor' else ' major')
                    key_pitch_class = get_pitch_class(k.tonic.name)
                    is_minor = 1 if k.mode == 'minor' else 0
                except:
                    key_sig = 'C major'  # Default if detection fails
                    key_pitch_class = 0
                    is_minor = 0
                
                # Use measures (bars) as time units
                melody_measures = melody_score.parts[0].getElementsByClass('Measure')
                chord_measures = chord_score.parts[0].getElementsByClass('Measure')
                
                if len(melody_measures) == 0 or len(chord_measures) == 0:
                    continue
                    
                min_len = min(len(melody_measures), len(chord_measures))
                
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
                    
                    # Basic feature: set of notes in the measure
                    basic_feature = tuple(melody_note_names)
                    
                    # ADVANCED FEATURES
                    
                    # 1. Rhythmic features
                    note_durations = [n.quarterLength for n in melody_notes]
                    avg_duration = sum(note_durations) / len(note_durations) if note_durations else 0
                    max_duration = max(note_durations) if note_durations else 0
                    min_duration = min(note_durations) if note_durations else 0
                    
                    # 2. Pitch-based features
                    pitch_classes = [get_pitch_class(name) for name in melody_note_names]
                    unique_pitch_classes = len(set(pitch_classes))
                    
                    # Most common pitch class in measure
                    most_common_pc = Counter(pitch_classes).most_common(1)[0][0] if pitch_classes else 0
                    
                    # 3. Interval features
                    intervals = []
                    for j in range(len(melody_notes)-1):
                        try:
                            i1 = interval.Interval(melody_notes[j], melody_notes[j+1])
                            intervals.append(i1.semitones)
                        except:
                            intervals.append(0)
                            
                    avg_interval = sum(intervals) / len(intervals) if intervals else 0
                    max_interval = max(intervals) if intervals else 0
                    
                    # 4. Relationship to key
                    distance_from_key = (most_common_pc - key_pitch_class) % 12
                    
                    # 5. Chord quality of the current chord
                    chord_quality = get_chord_quality(chord_label)
                    
                    # Store all this information
                    X_data.append({
                        'basic_feature': basic_feature, 
                        'avg_duration': avg_duration,
                        'max_duration': max_duration,
                        'min_duration': min_duration,
                        'unique_pitch_classes': unique_pitch_classes,
                        'most_common_pc': most_common_pc,
                        'avg_interval': avg_interval,
                        'max_interval': max_interval,
                        'distance_from_key': distance_from_key,
                        'key_pitch_class': key_pitch_class,
                        'is_minor': is_minor,
                        'position': i / min_len,  # Normalized position
                        'is_first': 1 if i == 0 else 0
                    })
                    
                    Y.append(chord_label)
                    file_ids.append(file_id)
                    measure_positions.append(i)
                    
            except Exception as e:
                print(f"Error processing {f}: {str(e)}")
                continue
        
        return X_data, Y, file_ids, measure_positions
    
    def extract_features(self, melody_notes, position=0, is_first=True, prev_chord_idx=-1, 
                         key_pitch_class=0, is_minor=0):
        """Extract features for a single melody measure with enhanced music features"""
        # Convert to data format expected by the model
        if isinstance(melody_notes, tuple) or isinstance(melody_notes, list):
            pitch_classes = [get_pitch_class(name) for name in melody_notes]
            
            # Basic features: histogram of notes
            basic_features = np.bincount([self.note2idx.get(n, 0) for n in melody_notes if n in self.note2idx], 
                                        minlength=len(self.note_vocab))
            
            # Advanced features (similar to those in extract_sequences)
            unique_pitch_classes = len(set(pitch_classes))
            most_common_pc = Counter(pitch_classes).most_common(1)[0][0] if pitch_classes else 0
            distance_from_key = (most_common_pc - key_pitch_class) % 12
            
            # Build feature vector
            features = list(basic_features)
            
            # Previous chord
            features.append(prev_chord_idx)
            
            # Position features
            features.append(int(is_first))
            features.append(position)
            
            # Melody complexity
            features.append(unique_pitch_classes)
            
            # Key relationship
            features.append(distance_from_key)
            features.append(key_pitch_class)
            features.append(is_minor)
            
            # Handle any feature standardization if used
            if self.scaler is not None:
                features = self.scaler.transform([features])[0]
                
            return np.array(features)
        else:
            # Handle the case where we're passed the data dictionary from extract_sequences
            data = melody_notes
            basic_feature = data['basic_feature']
            return self.extract_features(
                basic_feature,
                data['position'],
                data['is_first'] == 1,
                prev_chord_idx,
                data['key_pitch_class'],
                data['is_minor']
            )
    
    def encode_sequences(self, X_data, Y, file_ids, measure_positions):
        """Encode sequences and build vocabularies with enhanced features"""
        # Extract just the basic features for vocabulary building
        X_basic = [data['basic_feature'] for data in X_data]
        
        # Melody and chord vocabularies
        self.note_vocab = sorted(set(n for seq in X_basic for n in seq))
        self.chord_vocab = sorted(set(Y))
        self.note2idx = {n: i for i, n in enumerate(self.note_vocab)}
        self.chord2idx = {c: i for i, c in enumerate(self.chord_vocab)}
        self.idx2chord = {i: c for c, i in self.chord2idx.items()}
        
        # Enhanced features
        X_enhanced = []
        
        for i, data in enumerate(X_data):
            prev_chord_idx = -1
            if i > 0 and file_ids[i] == file_ids[i-1] and measure_positions[i] == measure_positions[i-1] + 1:
                prev_chord_idx = self.chord2idx[Y[i-1]]
                
            # Extract features using our feature extraction method
            features = self.extract_features(
                data,
                data['position'], 
                data['is_first'] == 1,
                prev_chord_idx,
                data['key_pitch_class'],
                data['is_minor']
            )
            
            X_enhanced.append(features)
        
        # Convert to numpy arrays
        X_enhanced = np.array(X_enhanced)
        Y_enc = np.array([self.chord2idx[c] for c in Y])
        
        # Standardize features (optional - better for neural networks)
        if self.model_type == "nn":
            self.scaler = StandardScaler()
            X_enhanced = self.scaler.fit_transform(X_enhanced)
        
        return X_enhanced, Y_enc
    
    def train(self, X_enc, Y_enc, tune_hyperparams=False):
        """Train the model with support for multiple model types and hyperparameter tuning"""
        print(f"Training {self.model_type} classifier...")
        
        if tune_hyperparams:
            print("Performing hyperparameter tuning...")
            if self.model_type == "rf":
                base_model = RandomForestClassifier(random_state=42)
                param_grid = {
                    'n_estimators': [100, 200],
                    'max_depth': [None, 20, 30],
                    'min_samples_split': [2, 5]
                }
            elif self.model_type == "gb":
                base_model = GradientBoostingClassifier(random_state=42)
                param_grid = {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 5],
                    'learning_rate': [0.05, 0.1]
                }
            elif self.model_type == "nn":
                base_model = MLPClassifier(random_state=42, max_iter=500, early_stopping=True)
                param_grid = {
                    'hidden_layer_sizes': [(100,), (100, 50)],
                    'alpha': [0.0001, 0.001],
                    'learning_rate': ['constant', 'adaptive']
                }
            else:
                # Default to RandomForest
                base_model = RandomForestClassifier(random_state=42)
                param_grid = {
                    'n_estimators': [100, 200],
                    'max_depth': [None, 20]
                }
                
            # Use a smaller subset for tuning if dataset is large
            if len(X_enc) > 10000:
                indices = np.random.choice(len(X_enc), 10000, replace=False)
                X_tune = X_enc[indices]
                Y_tune = Y_enc[indices]
            else:
                X_tune = X_enc
                Y_tune = Y_enc
                
            # Run grid search
            grid_search = GridSearchCV(base_model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
            grid_search.fit(X_tune, Y_tune)
            
            # Print best parameters
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best CV score: {grid_search.best_score_:.3f}")
            
            # Train final model with best parameters
            if self.model_type == "rf":
                self.model = RandomForestClassifier(random_state=42, **grid_search.best_params_)
            elif self.model_type == "gb":
                self.model = GradientBoostingClassifier(random_state=42, **grid_search.best_params_)
            elif self.model_type == "nn":
                self.model = MLPClassifier(random_state=42, max_iter=500, **grid_search.best_params_)
            else:
                self.model = RandomForestClassifier(random_state=42, **grid_search.best_params_)
                
        else:
            # Without hyperparameter tuning
            if self.model_type == "rf":
                self.model = RandomForestClassifier(n_estimators=200, max_depth=30, random_state=42)
            elif self.model_type == "gb":
                self.model = GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42)
            elif self.model_type == "nn":
                self.model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
            else:
                self.model = RandomForestClassifier(n_estimators=200, max_depth=30, random_state=42)
        
        # Train final model
        self.model.fit(X_enc, Y_enc)
        return self.model
    
    def predict_chords(self, melody_bars, previous_chords=None, key_sig="C major"):
        """Predict chords for a sequence of melody bars with enhanced music features"""
        # Extract key information
        key_tonic = key_sig.split()[0]
        key_pitch_class = get_pitch_class(key_tonic)
        is_minor = 1 if "minor" in key_sig else 0
        
        # Extract features for each bar
        features = []
        
        for i, bar in enumerate(melody_bars):
            # Determine previous chord index
            prev_chord_idx = -1
            if previous_chords and i > 0:
                try:
                    prev_chord_idx = list(self.idx2chord.keys())[list(self.idx2chord.values()).index(previous_chords[i-1])]
                except ValueError:
                    pass
            
            # Extract features
            position = i / max(1, len(melody_bars))
            is_first = (i == 0)
            
            # Create a simplified data dictionary for feature extraction
            bar_data = {
                'basic_feature': bar,
                'position': position,
                'is_first': 1 if is_first else 0,
                'key_pitch_class': key_pitch_class,
                'is_minor': is_minor
            }
            
            bar_features = self.extract_features(bar_data, position, is_first, prev_chord_idx, key_pitch_class, is_minor)
            features.append(bar_features)
        
        # Make predictions
        features = np.array(features)
        
        # Apply scaling if needed
        if self.scaler is not None:
            features = self.scaler.transform(features)
            
        chord_indices = self.model.predict(features)
        chord_labels = [self.idx2chord[idx] for idx in chord_indices]
        
        return chord_labels
    
    def evaluate(self, X_enc, Y_enc, output_fig=False):
        """Evaluate model performance with enhanced visualizations"""
        if self.model is None:
            print("Model not trained yet.")
            return
            
        Y_pred = self.model.predict(X_enc)
        correct = np.sum(Y_pred == Y_enc)
        total = len(Y_enc)
        acc = correct / total if total > 0 else 0
        print(f"Model accuracy: {acc:.3f} ({correct}/{total})")
        
        # Feature importance for tree-based models
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            top_k = min(10, len(importances))
            indices = np.argsort(importances)[-top_k:]
            print("\nTop features by importance:")
            for i in indices[::-1]:
                print(f"Feature {i}: {importances[i]:.4f}")
                
            if output_fig:
                # Plot feature importance
                plt.figure(figsize=(10, 6))
                plt.barh(range(top_k), importances[indices])
                plt.yticks(range(top_k), [f"Feature {i}" for i in indices])
                plt.xlabel("Feature Importance")
                plt.title("Top Features by Importance")
                plt.tight_layout()
                plt.savefig("results/feature_importance.png")
                print("Feature importance plot saved to results/feature_importance.png")
                
        # Analyze common chords
        chord_counts = Counter(Y_enc)
        print("\nMost common chords in dataset:")
        for chord_idx, count in chord_counts.most_common(5):
            chord_name = self.idx2chord[chord_idx]
            accuracy = np.mean(Y_pred[Y_enc == chord_idx] == chord_idx)
            print(f"Chord {chord_name}: {count} instances, accuracy: {accuracy:.3f}")
        
        # Analyze errors
        errors = [(self.idx2chord[Y_enc[i]], self.idx2chord[Y_pred[i]]) 
                 for i in range(len(Y_enc)) if Y_pred[i] != Y_enc[i]]
        error_counts = Counter(errors).most_common(5)
        
        print("\nCommon misclassifications:")
        for (true_chord, pred_chord), count in error_counts:
            print(f"True: {true_chord}, Predicted: {pred_chord}, Count: {count}")
            
        if output_fig:
            # Plot confusion matrix for top chords
            top_chord_indices = [idx for idx, _ in chord_counts.most_common(10)]
            top_chord_names = [self.idx2chord[idx] for idx in top_chord_indices]
            
            # Filter data to only include top chords
            mask = np.isin(Y_enc, top_chord_indices)
            Y_enc_filtered = Y_enc[mask]
            Y_pred_filtered = Y_pred[mask]
            
            # Remap indices to 0-(n-1) for confusion matrix
            idx_map = {old_idx: new_idx for new_idx, old_idx in enumerate(top_chord_indices)}
            Y_enc_remapped = np.array([idx_map[idx] for idx in Y_enc_filtered])
            Y_pred_remapped = np.array([idx_map.get(idx, -1) for idx in Y_pred_filtered])
            
            # Remove any predictions that aren't in our top chords
            valid_mask = Y_pred_remapped != -1
            Y_enc_remapped = Y_enc_remapped[valid_mask]
            Y_pred_remapped = Y_pred_remapped[valid_mask]
            
            # Generate confusion matrix
            cm = confusion_matrix(Y_enc_remapped, Y_pred_remapped)
            
            # Plot
            plt.figure(figsize=(12, 10))
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title("Confusion Matrix for Top Chords")
            plt.colorbar()
            tick_marks = np.arange(len(top_chord_names))
            plt.xticks(tick_marks, top_chord_names, rotation=45, ha="right")
            plt.yticks(tick_marks, top_chord_names)
            plt.tight_layout()
            plt.ylabel('True Chord')
            plt.xlabel('Predicted Chord')
            plt.savefig("results/confusion_matrix.png")
            print("Confusion matrix saved to results/confusion_matrix.png")
    
    def save(self, filepath):
        """Save model to file"""
        with open(filepath, "wb") as f:
            pickle.dump({
                'model': self.model,
                'note2idx': self.note2idx,
                'chord2idx': self.chord2idx,
                'note_vocab': self.note_vocab,
                'chord_vocab': self.chord_vocab,
                'is_enhanced': self.is_enhanced,
                'model_type': self.model_type,
                'scaler': self.scaler
            }, f)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """Load model from file"""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        
        # Handle old model format
        if isinstance(data, tuple):
            model, note2idx, chord2idx, note_vocab, chord_vocab = data
            is_enhanced = True
            model_type = "rf"
            scaler = None
        else:
            model = data['model']
            note2idx = data['note2idx']
            chord2idx = data['chord2idx'] 
            note_vocab = data['note_vocab']
            chord_vocab = data['chord_vocab']
            is_enhanced = data.get('is_enhanced', True)
            model_type = data.get('model_type', 'rf')
            scaler = data.get('scaler', None)
            
        instance = cls(model_type=model_type)
        instance.model = model
        instance.note2idx = note2idx
        instance.chord2idx = chord2idx
        instance.note_vocab = note_vocab
        instance.chord_vocab = chord_vocab
        instance.idx2chord = {i: c for c, i in chord2idx.items()}
        instance.is_enhanced = is_enhanced
        instance.scaler = scaler
        
        return instance

def main():
    melody_dir = os.path.join("data", "MIDI", "melody")
    chord_dir = os.path.join("data", "MIDI", "chords")
    
    # Choose model type: "rf" (Random Forest), "gb" (Gradient Boosting), "nn" (Neural Network)
    model_type = "gb"  # Gradient Boosting generally performs very well
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
    print("Training model...")
    harmonizer.train(X_train, Y_train, tune_hyperparams=True)
    
    # Save model
    model_filename = f"results/harmonizer_{model_type}_model.pkl"
    harmonizer.save(model_filename)
    
    # Evaluate with visualizations
    print("Evaluating model...")
    harmonizer.evaluate(X_test, Y_test, output_fig=True)

if __name__ == "__main__":
    main()
