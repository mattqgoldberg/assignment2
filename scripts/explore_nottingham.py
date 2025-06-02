# This script explores the Nottingham Dataset and prints basic statistics about the melodies and chords.
import os
from music21 import converter

def explore_midi_folder(melody_dir, chord_dir):
    melody_files = sorted([f for f in os.listdir(melody_dir) if f.endswith('.mid')])
    chord_files = sorted([f for f in os.listdir(chord_dir) if f.endswith('.mid')])
    print(f"Number of melody files: {len(melody_files)}")
    print(f"Number of chord files: {len(chord_files)}")
    print(f"Example melody file: {melody_files[0]}")
    print(f"Example chord file: {chord_files[0]}")
    # Load and print info for one example
    melody_score = converter.parse(os.path.join(melody_dir, melody_files[0]))
    chord_score = converter.parse(os.path.join(chord_dir, chord_files[0]))
    print("Melody parts:", melody_score.parts)
    print("Chord parts:", chord_score.parts)
    print("Melody notes:", [n for n in melody_score.flat.notes])
    print("Chord notes:", [n for n in chord_score.flat.notes])

if __name__ == "__main__":
    melody_dir = os.path.join("..", "data", "MIDI", "melody")
    chord_dir = os.path.join("..", "data", "MIDI", "chords")
    # Fix: Use correct relative path from project root
    if not os.path.exists(melody_dir):
        melody_dir = os.path.join("data", "MIDI", "melody")
    if not os.path.exists(chord_dir):
        chord_dir = os.path.join("data", "MIDI", "chords")
    explore_midi_folder(melody_dir, chord_dir)
