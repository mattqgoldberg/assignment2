# HMM-based harmonization model for melody-to-chord prediction
# Uses hmmlearn to train a discrete HMM on aligned melody/chord sequences
import os
from music21 import converter, note, chord
from collections import Counter, defaultdict
import numpy as np
from hmmlearn import hmm
import pickle
from music21 import meter
from sklearn.naive_bayes import MultinomialNB

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

def extract_sequences(melody_dir, chord_dir):
    X, Y = [], []
    files = sorted(os.listdir(melody_dir))
    for f in files:
        if not f.endswith('.mid') or not os.path.exists(os.path.join(chord_dir, f)):
            continue
        melody_score = converter.parse(os.path.join(melody_dir, f))
        chord_score = converter.parse(os.path.join(chord_dir, f))
        # Use measures (bars) as time units
        melody_measures = melody_score.parts[0].getElementsByClass('Measure')
        chord_measures = chord_score.parts[0].getElementsByClass('Measure')
        if len(melody_measures) == 0 or len(chord_measures) == 0:
            continue
        min_len = min(len(melody_measures), len(chord_measures))
        for i in range(min_len):
            melody_notes = [get_note_name(n) for n in melody_measures[i].notes if isinstance(n, note.Note)]
            # Use the first chord in the measure as the label (or the most common if multiple)
            chords_in_measure = [c for c in chord_measures[i].notes if isinstance(c, chord.Chord)]
            if not melody_notes or not chords_in_measure:
                continue
            # Use the most common chord in the measure as the label
            chord_labels = [get_note_name(c) for c in chords_in_measure]
            chord_label = max(set(chord_labels), key=chord_labels.count)
            # Represent the melody by the set of notes in the measure (as a tuple)
            X.append(tuple(melody_notes))
            Y.append(chord_label)
    return X, Y

def encode_sequences(X, Y):
    # Melody as tuple of notes per bar, chord as label
    note_vocab = sorted(set(n for seq in X for n in seq))
    chord_vocab = sorted(set(Y))
    note2idx = {n: i for i, n in enumerate(note_vocab)}
    chord2idx = {c: i for i, c in enumerate(chord_vocab)}
    # Represent each bar as a histogram of notes
    X_enc = [np.bincount([note2idx[n] for n in seq], minlength=len(note_vocab)) for seq in X]
    Y_enc = [chord2idx[c] for c in Y]
    X_enc = np.array(X_enc)
    Y_enc = np.array(Y_enc)
    return X_enc, Y_enc, note2idx, chord2idx, note_vocab, chord_vocab

def train_hmm(X_enc, Y_enc, n_chords):
    # For beat-level, use a classifier instead of HMM (since HMM expects sequences)
    # We'll use a simple multinomial Naive Bayes as a baseline
    model = MultinomialNB()
    model.fit(X_enc, Y_enc)
    return model

def evaluate_hmm(model, X_enc, Y_enc):
    Y_pred = model.predict(X_enc)
    correct = np.sum(Y_pred == Y_enc)
    total = len(Y_enc)
    acc = correct / total if total > 0 else 0
    print(f"Beat-level classifier accuracy: {acc:.3f} ({correct}/{total})")

def main():
    melody_dir = os.path.join("data", "MIDI", "melody")
    chord_dir = os.path.join("data", "MIDI", "chords")
    print("Extracting sequences...")
    X, Y = extract_sequences(melody_dir, chord_dir)
    print(f"Loaded {len(X)} aligned melody/chord pairs.")
    X_enc, Y_enc, note2idx, chord2idx, note_vocab, chord_vocab = encode_sequences(X, Y)
    print(f"Melody vocab size: {len(note_vocab)} | Chord vocab size: {len(chord_vocab)}")
    print("Training HMM...")
    model = train_hmm(X_enc, Y_enc, n_chords=len(chord_vocab))
    with open("results/hmm_model.pkl", "wb") as f:
        pickle.dump((model, note2idx, chord2idx, note_vocab, chord_vocab), f)
    print("Evaluating HMM...")
    evaluate_hmm(model, X_enc, Y_enc)

if __name__ == "__main__":
    main()
