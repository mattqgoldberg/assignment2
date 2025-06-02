# Baseline harmonization model: Predict the most common chord for each melody note
# This script builds a simple lookup-based model and evaluates it on the dataset
import os
from collections import Counter, defaultdict
from music21 import converter, note, chord
import pickle

def get_note_name(n):
    if isinstance(n, note.Note):
        return n.nameWithOctave
    elif isinstance(n, chord.Chord):
        # Use root note for chord
        return n.root().nameWithOctave
    return None

def extract_pairs(melody_path, chord_path):
    melody_score = converter.parse(melody_path)
    chord_score = converter.parse(chord_path)
    melody_notes = [n for n in melody_score.flat.notes]
    chord_notes = [c for c in chord_score.flat.notes]
    # Pair by index (assume same length, simple baseline)
    pairs = list(zip(melody_notes, chord_notes))
    return [(get_note_name(m), get_note_name(c)) for m, c in pairs]

def build_lookup(melody_dir, chord_dir):
    lookup = defaultdict(Counter)
    files = sorted(os.listdir(melody_dir))
    for f in files:
        if not f.endswith('.mid') or not os.path.exists(os.path.join(chord_dir, f)):
            continue
        pairs = extract_pairs(os.path.join(melody_dir, f), os.path.join(chord_dir, f))
        for m, c in pairs:
            lookup[m][c] += 1
    # For each melody note, pick the most common chord
    model = {m: c.most_common(1)[0][0] for m, c in lookup.items() if c}
    return model

def evaluate(model, melody_dir, chord_dir):
    total, correct = 0, 0
    files = sorted(os.listdir(melody_dir))
    for f in files:
        if not f.endswith('.mid') or not os.path.exists(os.path.join(chord_dir, f)):
            continue
        pairs = extract_pairs(os.path.join(melody_dir, f), os.path.join(chord_dir, f))
        for m, c in pairs:
            pred = model.get(m, None)
            if pred == c:
                correct += 1
            total += 1
    acc = correct / total if total > 0 else 0
    print(f"Baseline accuracy: {acc:.3f} ({correct}/{total})")

def main():
    melody_dir = os.path.join("data", "MIDI", "melody")
    chord_dir = os.path.join("data", "MIDI", "chords")
    print("Building baseline model...")
    model = build_lookup(melody_dir, chord_dir)
    with open("results/baseline_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print(f"Model size: {len(model)} melody notes mapped.")
    print("Evaluating baseline model...")
    evaluate(model, melody_dir, chord_dir)

if __name__ == "__main__":
    main()
