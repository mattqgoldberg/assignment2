import os
import pickle
from music21 import converter, note, chord, stream, instrument, harmony, metadata
import numpy as np
from hmm_model import get_note_name, HarmonizationModel

def extract_barwise(melody_path, chord_path):
    """Extract melody notes and real chords bar by bar"""
    melody_score = converter.parse(melody_path)
    chord_score = converter.parse(chord_path)
    melody_measures = melody_score.parts[0].getElementsByClass('Measure')
    chord_measures = chord_score.parts[0].getElementsByClass('Measure')
    min_len = min(len(melody_measures), len(chord_measures))
    bars = []
    real_chords = []
    for i in range(min_len):
        melody_notes = [get_note_name(n) for n in melody_measures[i].notes if isinstance(n, note.Note)]
        chords_in_measure = [c for c in chord_measures[i].notes if isinstance(c, chord.Chord)]
        if not melody_notes or not chords_in_measure:
            continue
        chord_labels = [get_note_name(c) for c in chords_in_measure]
        chord_label = max(set(chord_labels), key=chord_labels.count)
        bars.append(tuple(melody_notes))
        real_chords.append(chord_label)
    return bars, real_chords, melody_measures[:len(bars)]

def build_score(melody_measures, chords, title):
    """Build a music21 score from melody measures and chord labels"""
    s = stream.Score()
    melody_part = stream.Part()
    melody_part.append(instrument.Piano())
    chord_part = stream.Part()
    chord_part.append(instrument.Piano())
    for m, c_label in zip(melody_measures, chords):
        m2 = m.flat.notes.stream()
        melody_part.append(m2)
        # Try to create a ChordSymbol, else fallback to root note
        try:
            c = harmony.ChordSymbol(c_label)
        except Exception:
            root = c_label.split('-')[0] if '-' in c_label else c_label
            c = chord.Chord([root])
        c.quarterLength = m.barDuration.quarterLength if hasattr(m, 'barDuration') else 4.0
        c.offset = 0
        chord_measure = stream.Measure()
        chord_measure.append(c)
        chord_part.append(chord_measure)
    s.insert(0, melody_part)
    s.insert(0, chord_part)
    s.metadata = metadata.Metadata()
    s.metadata.title = title
    return s
    
def harmonize_and_play(melody_path, chord_path, harmonizer, n_bars=8):
    """Generate and play harmonization for a melody"""
    # Extract melody and real chords
    bars, real_chords, melody_measures = extract_barwise(melody_path, chord_path)
    
    # Predict chords using the harmonizer model
    pred_chords = harmonizer.predict_chords(bars, real_chords)

    # Build scores for real and predicted harmonizations
    real_score = build_score(melody_measures[:n_bars], real_chords[:n_bars], "Real Harmonization")
    pred_score = build_score(melody_measures[:n_bars], pred_chords[:n_bars], "Predicted Harmonization")
    
    # Export to MIDI files
    os.makedirs("results/mp3", exist_ok=True)
    real_midi_path = "results/mp3/real_harmonization.mid"
    pred_midi_path = "results/mp3/predicted_harmonization.mid"
    real_score.write('midi', fp=real_midi_path)
    pred_score.write('midi', fp=pred_midi_path)
    print(f"Exported real harmonization to {real_midi_path}")
    print(f"Exported predicted harmonization to {pred_midi_path}")
    
    # Optionally, convert to mp3 using timidity and ffmpeg if available
    for midi_path, mp3_path in [
        (real_midi_path, "results/mp3/real_harmonization.mp3"),
        (pred_midi_path, "results/mp3/predicted_harmonization.mp3")]:
        wav_path = midi_path.replace('.mid', '.wav')
        # Convert MIDI to WAV
        os.system(f"timidity {midi_path} -Ow -o {wav_path}")
        # Convert WAV to MP3
        os.system(f"ffmpeg -y -i {wav_path} {mp3_path}")
        print(f"Exported {mp3_path}")

if __name__ == "__main__":
    melody_dir = os.path.join("data", "MIDI", "melody")
    chord_dir = os.path.join("data", "MIDI", "chords")
    
    # Try to use newer model if it exists
    if os.path.exists("results/harmonizer_gb_model.pkl"):
        model_path = "results/harmonizer_gb_model.pkl"
        print("Using Gradient Boosting model")
        from hmm_model import HarmonizationModel
        harmonizer_class = HarmonizationModel
    elif os.path.exists("results/hmm_optimized_model.pkl"):
        model_path = "results/hmm_optimized_model.pkl"
        print("Using optimized HMM model")
        from hmm_optimized import OptimizedHMMModel
        harmonizer_class = OptimizedHMMModel
    elif os.path.exists("results/harmonizer_model.pkl"):
        model_path = "results/harmonizer_model.pkl"
        print("Using new harmonizer model")
        from hmm_model import HarmonizationModel
        harmonizer_class = HarmonizationModel
    elif os.path.exists("results/enhanced_model.pkl"):
        model_path = "results/enhanced_model.pkl"
        print("Using enhanced model")
        from hmm_model import HarmonizationModel
        harmonizer_class = HarmonizationModel
    else:
        model_path = "results/hmm_model.pkl"
        print("Using original model")
        from hmm_model import HarmonizationModel
        harmonizer_class = HarmonizationModel
    
    # Load the model
    harmonizer = harmonizer_class.load(model_path)
    
    # Process melody files
    files = sorted(os.listdir(melody_dir))
    for f in files:
        if not f.endswith('.mid') or not os.path.exists(os.path.join(chord_dir, f)):
            continue
        print(f"Processing example: {f}")
        melody_path = os.path.join(melody_dir, f)
        chord_path = os.path.join(chord_dir, f)
        harmonize_and_play(melody_path, chord_path, harmonizer, n_bars=8)
        break  # Remove this break to process more examples
