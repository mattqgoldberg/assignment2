# Harmonization Project

This project performs symbolic, conditioned music generation: given a melody, it generates a harmonizing chord sequence.

## Structure
- `data/` — for datasets (e.g., Nottingham)
- `scripts/` — for code (data processing, training, generation)
- `results/` — for generated outputs (models, MIDI, MP3, etc.)

## Dataset
We use the Nottingham Dataset, which contains melody and chord pairs in symbolic format (MIDI and ABC). You can download it from:
- [Nottingham Dataset (official)](http://www-etud.iro.umontreal.ca/~boulanni/icml2012)
- [Alternative mirror (Google Drive)](https://drive.google.com/file/d/0B7pQmmH1cWnQbWl1Q1ZkZl9kV0E/view)

After downloading, extract the contents into the `data/` folder so you have `data/MIDI/` and `data/ABC_cleaned/`.

## Setup
1. Download the Nottingham Dataset and place it in `data/` as described above.
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
   (Requires: music21, torch, numpy, hmmlearn, scikit-learn, scipy)

## Scripts
- `scripts/explore_nottingham.py`: Data exploration — prints basic stats and inspects MIDI files.
- `scripts/baseline_model.py`: Baseline model — maps each melody note to its most common chord (lookup table). Prints accuracy and saves the model.
- `scripts/hmm_model.py`: Beat/bar-level harmonization using advanced ML classifiers (Random Forest, Gradient Boosting, Neural Networks). Trains and evaluates a model to predict chords for each measure given the melody notes. Saves the model and prints accuracy.
- `scripts/hmm_optimized.py`: True sequence-based HMM implementation using the hmmlearn library. Models the chord progression as a proper Markov process with hidden states. Incorporates music theory and properly models the temporal aspects of harmonization.
- `scripts/compare_models.py`: Compares different harmonization models (Baseline, HMM, RF, GB, and Optimized HMM) on the same test data, producing comparative accuracy metrics and example outputs.
- `scripts/play_harmonization.py`: Loads a trained model and outputs both the real and predicted harmonizations for a melody as MIDI and MP3 files (in `results/mp3/`). Requires `timidity` and `ffmpeg` for MP3 export (optional).

## Usage
1. **Explore the data:**
   ```sh
   python scripts/explore_nottingham.py
   ```
2. **Train the baseline model:**
   ```sh
   python scripts/baseline_model.py
   ```
3. **Train and evaluate the standard harmonization models:**
   ```sh
   python scripts/hmm_model.py      # Standard HMM model
   python scripts/rf_model.py       # Random Forest model
   python scripts/gb_model.py       # Gradient Boosting model
   ```
4. **Train and evaluate the optimized HMM model:**
   ```sh
   python scripts/hmm_optimized.py
   ```
5. **Compare all harmonization models:**
   ```sh
   python scripts/compare_models.py
   ```
   This will generate comparison metrics and example outputs in `results/comparisons/`.
6. **Generate and export harmonizations:**
   ```sh
   python scripts/play_harmonization.py
   ```
   This will create MIDI and (optionally) MP3 files in `results/mp3/`.

## Models
- **Baseline**: Simple lookup table that maps each melody note to its most common chord across the dataset.
- **Standard HMM**: Uses the MultinomialNB classifier to predict chords based on melody notes at the measure level.
- **Random Forest**: Machine learning model that uses ensemble decision trees to predict chords based on melody features.
- **Gradient Boosting**: Advanced machine learning model that uses boosted trees for chord prediction with improved accuracy.
- **Optimized HMM**: True HMM implementation using the hmmlearn library. Models the harmonic progression as a proper Markov process with hidden states that capture musical structure.

## Key Features of Optimized HMM
- **Proper sequence modeling**: Uses the Viterbi algorithm to find the optimal sequence of hidden states.
- **Musical structure awareness**: Preserves the temporal nature of music with sequence-based prediction.
- **Hidden states interpretation**: Maps hidden states to musical concepts (like harmonic functions).
- **Improved alignment**: Better alignment between melody notes and chord changes.
- **Cross-song boundaries**: Respects song boundaries when training to avoid creating false transitions.

## Notes
- For MP3 export, install Timidity and FFmpeg:
  ```sh
  brew install timidity ffmpeg
  ```
- You can adjust the number of bars or which files are used in `play_harmonization.py`.
- All scripts are designed for Python 3.8+ and tested on macOS.
