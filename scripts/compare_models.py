# Script to compare different harmonization models on the same test data
import os
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from collections import Counter

# Import models
from hmm_model import HarmonizationModel, get_note_name
from hmm_optimized import OptimizedHMMModel

def extract_test_data(melody_dir, chord_dir, n_files=20):
    """Extract test data from the last n files"""
    melody_files = sorted([f for f in os.listdir(melody_dir) if f.endswith('.mid')])
    chord_files = sorted([f for f in os.listdir(chord_dir) if f.endswith('.mid')])
    
    # Select the last n files for testing
    test_files = melody_files[-n_files:]
    
    # Extract test data using the barwise extraction from HarmonizationModel
    all_melody_bars = []
    all_real_chords = []
    all_melody_measures = []
    
    for f in test_files:
        if not os.path.exists(os.path.join(chord_dir, f)):
            continue
            
        # Use the extract_barwise function from play_harmonization.py
        from play_harmonization import extract_barwise
        bars, real_chords, melody_measures = extract_barwise(
            os.path.join(melody_dir, f), 
            os.path.join(chord_dir, f)
        )
        
        if bars and real_chords:
            all_melody_bars.extend(bars)
            all_real_chords.extend(real_chords)
            all_melody_measures.extend(melody_measures)
    
    print(f"Extracted {len(all_melody_bars)} test bars from {len(test_files)} files")
    return all_melody_bars, all_real_chords, all_melody_measures

def load_model(model_path, model_class=HarmonizationModel):
    """Load a harmonization model from file"""
    try:
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            return None
            
        if model_class == OptimizedHMMModel:
            return OptimizedHMMModel.load(model_path)
        else:
            model = None
            # First try the normal load method
            try:
                model = HarmonizationModel.load(model_path)
            except:
                # If that fails, try loading as a dictionary (for baseline model)
                with open(model_path, "rb") as f:
                    model_dict = pickle.load(f)
                    
                # Wrap the baseline model dict in a compatible object
                class BaselineWrapper:
                    def __init__(self, model_dict):
                        self.model_dict = model_dict
                        
                    def predict_chords(self, melody_bars, **kwargs):
                        predictions = []
                        for bar in melody_bars:
                            # For each bar, find the most common chord for each note
                            chord_predictions = []
                            for note in bar:
                                chord_predictions.append(self.model_dict.get(note, "C-major triad"))  # Default to C if not found
                            
                            # Use most common chord for the bar
                            if chord_predictions:
                                most_common = Counter(chord_predictions).most_common(1)[0][0]
                                predictions.append(most_common)
                            else:
                                predictions.append("C-major triad")  # Default 
                                
                        return predictions
                
                if isinstance(model_dict, dict):
                    model = BaselineWrapper(model_dict)
                    
            return model
    except Exception as e:
        print(f"Error loading model {model_path}: {str(e)}")
        return None

def prepare_sequence_data(all_melody_bars, all_real_chords):
    """Convert barwise data to sequential data for HMM model"""
    # Flatten the bars into a sequence of notes
    melody_sequence = []
    chord_sequence = []
    
    for bar, chord in zip(all_melody_bars, all_real_chords):
        # Each bar contains multiple notes
        for note in bar:
            melody_sequence.append(note)
            chord_sequence.append(chord)  # Same chord for all notes in the bar
    
    return melody_sequence, chord_sequence

def evaluate_models(models, melody_bars, real_chords, melody_sequence=None, chord_sequence=None):
    """Evaluate multiple harmonization models on the same test data"""
    results = {}
    
    for name, model in models.items():
        if model is None:
            print(f"Skipping {name} - model not loaded")
            continue
        
        print(f"\nEvaluating {name}...")
        
        try:
            # For the optimized HMM model, use sequential prediction
            if isinstance(model, OptimizedHMMModel) and melody_sequence and len(melody_sequence) > 0:
                pred_chords = model.predict_chords(melody_sequence)
                
                # We need to convert back to barwise structure for comparison
                bar_preds = []
                i = 0
                for bar in melody_bars:
                    # For each bar, take the most common predicted chord
                    bar_length = len(bar)
                    bar_chords = pred_chords[i:i+bar_length] if i+bar_length <= len(pred_chords) else []
                    if bar_chords:
                        most_common = Counter(bar_chords).most_common(1)[0][0]
                        bar_preds.append(most_common)
                    else:
                        # Fallback if we've run out of predictions
                        bar_preds.append(real_chords[0])
                    i += bar_length
                pred_chords = bar_preds
            elif hasattr(model, 'predict_chords'):
                # For classifier-based models with predict_chords method
                pred_chords = model.predict_chords(melody_bars)
            elif hasattr(model, 'get'):
                # For the baseline model (dictionary)
                pred_chords = []
                for bar in melody_bars:
                    # For each bar, find the most common chord for each note
                    chord_predictions = []
                    for note in bar:
                        chord_predictions.append(model.get(note, "C-major triad"))  # Default to C if not found
                    
                    # Use most common chord for the bar
                    if chord_predictions:
                        most_common = Counter(chord_predictions).most_common(1)[0][0]
                        pred_chords.append(most_common)
                    else:
                        pred_chords.append("C-major triad")  # Default 
            else:
                print(f"Unsupported model type: {type(model)}")
                continue
        except Exception as e:
            print(f"Error predicting with {name}: {str(e)}")
            continue
        
        # Calculate accuracy
        correct = sum(1 for true, pred in zip(real_chords, pred_chords) if true == pred)
        total = len(real_chords)
        accuracy = correct / total if total > 0 else 0
        
        print(f"Accuracy: {accuracy:.3f} ({correct}/{total})")
        
        # Calculate chord-wise stats
        chord_stats = {}
        for chord in set(real_chords):
            indices = [i for i, c in enumerate(real_chords) if c == chord]
            if indices:
                chord_correct = sum(1 for i in indices if pred_chords[i] == real_chords[i])
                chord_accuracy = chord_correct / len(indices)
                chord_stats[chord] = {
                    'count': len(indices),
                    'accuracy': chord_accuracy,
                    'correct': chord_correct
                }
        
        # Find common errors
        errors = [(real_chords[i], pred_chords[i]) for i in range(len(real_chords)) 
                  if real_chords[i] != pred_chords[i]]
        error_counts = Counter(errors).most_common(5)
        
        print("Common errors:")
        for (true, pred), count in error_counts:
            print(f"True: {true}, Pred: {pred}, Count: {count}")
        
        # Store results
        results[name] = {
            'accuracy': accuracy,
            'pred_chords': pred_chords,
            'chord_stats': chord_stats,
            'error_counts': error_counts
        }
    
    return results

def plot_comparison(results, output_path):
    """Plot accuracy comparison of different models"""
    names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in names]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, accuracies)
    plt.ylabel('Accuracy')
    plt.title('Harmonization Model Comparison')
    plt.ylim(0, 1.0)
    
    # Add value labels
    for bar, acc in zip(bars, accuracies):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f'{acc:.3f}',
            ha='center',
            fontsize=10
        )
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Comparison chart saved to {output_path}")

def export_comparison(test_data, results, output_dir="results/comparisons"):
    """Export examples of harmonization from each model"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    melody_bars, real_chords, melody_measures = test_data
    
    if len(melody_bars) == 0:
        print("No test data to export comparisons from")
        return
    
    # Select a few examples to showcase
    n_examples = min(5, len(melody_bars) // 8)
    n_examples = max(1, n_examples)  # Ensure at least one example if we have data
    
    for i in range(n_examples):
        # Take consecutive bars for a meaningful musical example
        start_idx = i * 8
        end_idx = min(start_idx + 8, len(melody_bars))
        example_bars = melody_bars[start_idx:end_idx]
        example_real_chords = real_chords[start_idx:end_idx]
        example_measures = melody_measures[start_idx:end_idx]
        
        # Generate harmonization from each model
        example_results = {}
        example_results['Real'] = example_real_chords
        
        for name, model_result in results.items():
            try:
                example_results[name] = model_result['pred_chords'][start_idx:end_idx]
            except Exception as e:
                print(f"Error extracting predictions for {name}: {str(e)}")
                example_results[name] = example_real_chords  # Fall back to real chords
        
        # Export music21 scores for each harmonization
        from play_harmonization import build_score
        
        for name, chords in example_results.items():
            try:
                # Build and export score
                score = build_score(example_measures, chords, f"Example {i+1}: {name} Harmonization")
                
                # Write to MIDI
                safe_name = name.replace(' ', '_').replace('/', '_')
                filename = f"example{i+1}_{safe_name}"
                midi_path = os.path.join(output_dir, f"{filename}.mid")
                
                # Ensure directory exists
                os.makedirs(os.path.dirname(midi_path), exist_ok=True)
                
                # Export MIDI file
                score.write('midi', fp=midi_path)
                print(f"Exported MIDI to {midi_path}")
                
                # Try to convert to MP3 if timidity and ffmpeg are available
                try:
                    wav_path = midi_path.replace('.mid', '.wav')
                    mp3_path = midi_path.replace('.mid', '.mp3')
                    
                    timidity_result = os.system(f"timidity {midi_path} -Ow -o {wav_path}")
                    if timidity_result == 0:  # Command succeeded
                        ffmpeg_result = os.system(f"ffmpeg -y -i {wav_path} {mp3_path}")
                        if ffmpeg_result == 0:
                            print(f"Exported MP3 to {mp3_path}")
                            # Clean up WAV file
                            os.remove(wav_path)
                except Exception as audio_err:
                    print(f"Audio conversion error (not critical): {str(audio_err)}")
            except Exception as e:
                print(f"Error exporting {name} harmonization: {str(e)}")

def main():
    # Set paths
    melody_dir = os.path.join("data", "MIDI", "melody")
    chord_dir = os.path.join("data", "MIDI", "chords")
    
    # Check if the paths exist
    if not os.path.exists(melody_dir):
        print(f"Error: Melody directory {melody_dir} does not exist")
        return
    if not os.path.exists(chord_dir):
        print(f"Error: Chord directory {chord_dir} does not exist")
        return
        
    print(f"Using melody directory: {melody_dir}")
    print(f"Using chord directory: {chord_dir}")
    
    # Extract test data
    test_data = extract_test_data(melody_dir, chord_dir, n_files=20)
    melody_bars, real_chords, melody_measures = test_data
    
    # Also prepare sequential data for HMM model
    melody_sequence, chord_sequence = prepare_sequence_data(melody_bars, real_chords)
    
    # Load models - make sure to use the models we actually have
    models = {}
    
    # Check if the baseline model exists
    if os.path.exists("results/baseline_model.pkl"):
        models["Baseline"] = load_model("results/baseline_model.pkl")
    
    # Check if standard HMM model exists
    if os.path.exists("results/hmm_model.pkl"):
        models["Standard HMM"] = load_model("results/hmm_model.pkl")
    
    # Check for Random Forest model
    rf_model_path = "results/harmonizer_rf_model.pkl"
    if os.path.exists(rf_model_path):
        from hmm_model import HarmonizationModel
        models["Random Forest"] = HarmonizationModel.load(rf_model_path)
        
    # Check for Gradient Boosting model
    gb_model_path = "results/harmonizer_gb_model.pkl"
    if os.path.exists(gb_model_path):
        from hmm_model import HarmonizationModel
        models["Gradient Boosting"] = HarmonizationModel.load(gb_model_path)
        
    # Check for Neural Network model
    nn_model_path = "results/harmonizer_nn_model.pkl"
    if os.path.exists(nn_model_path):
        from hmm_model import HarmonizationModel
        models["Neural Network"] = HarmonizationModel.load(nn_model_path)
    
    # Always include our optimized HMM
    models["Optimized HMM"] = load_model("results/hmm_optimized_model.pkl", OptimizedHMMModel)
    
    # If optimized HMM is not yet trained, train it
    if models["Optimized HMM"] is None:
        print("Training Optimized HMM model...")
        from hmm_optimized import main as train_hmm
        train_hmm()
        models["Optimized HMM"] = load_model("results/hmm_optimized_model.pkl", OptimizedHMMModel)
    
    # Evaluate models - only with those that loaded correctly
    active_models = {k: v for k, v in models.items() if v is not None}
    if active_models:
        results = evaluate_models(active_models, melody_bars, real_chords, melody_sequence, chord_sequence)
        
        # Plot comparison
        plot_comparison(results, "results/model_comparison.png")
        
        # Export example harmonizations
        export_comparison(test_data, results)
    else:
        print("No models could be loaded for comparison.")

if __name__ == "__main__":
    main()
