# MIDI Melody to Chord Progression Generator

## Project Overview
This project implements a machine learning model that takes MIDI melody input and generates appropriate chord progressions. The model learns harmonic relationships between melodies and their accompanying chords to create musically coherent accompaniments.

## Dataset

### Primary Dataset: The Lakh MIDI Dataset (LMD)
- **Source**: The Lakh MIDI Dataset - a collection of 176,581 unique MIDI files
- **Download**: Available at https://colinraffel.com/projects/lmd/
- **Size**: ~45,000 matched MIDI files after cleaning
- **License**: Various (many public domain and creative commons)

### Alternative/Supplementary Datasets:
1. **Maestro Dataset**: High-quality piano performances (Google's dataset)
2. **JSB Chorales**: Bach chorale dataset for classical harmonic patterns
3. **Nottingham Music Database**: Folk tunes with chord annotations

### Dataset Rationale:
- **Diversity**: LMD provides a wide variety of musical genres and styles
- **Scale**: Large enough dataset for deep learning approaches
- **Quality**: MIDI format preserves precise timing and pitch information
- **Preprocessing**: Can extract melody (highest pitch) and chord (lower voices) automatically

## Model Architecture and Methodology

### Model Type: Sequence-to-Sequence Transformer with Musical Domain Knowledge

#### Primary Architecture: Music Transformer
- **Base Model**: Transformer encoder-decoder architecture
- **Input**: Melody sequence (pitch, duration, timing)
- **Output**: Chord progression (root, quality, inversion, timing)

#### Key Design Decisions:
1. **Tokenization**: 
   - Melody: (pitch, duration, position_in_bar) tuples
   - Chords: (root_note, chord_quality, inversion, duration) tuples

2. **Positional Encoding**: 
   - Standard transformer positional encoding
   - Additional musical positional encoding (beat position, measure position)

3. **Attention Mechanism**:
   - Self-attention for melody context
   - Cross-attention between melody and chord sequences

#### Alternative Approaches Considered:
- **LSTM-based**: Good for sequential data but less parallelizable
- **CNN-based**: Good for local patterns but misses long-term dependencies
- **Rule-based**: Limited creativity and adaptability

### Model Components:

1. **Melody Encoder**:
   - Processes input melody sequences
   - Embeddings for pitch, duration, and timing
   - Multi-head self-attention layers

2. **Chord Decoder**:
   - Generates chord progressions autoregressively
   - Cross-attention to melody representations
   - Chord quality prediction head

3. **Musical Constraint Layer**:
   - Ensures harmonic consistency
   - Key signature awareness
   - Voice leading constraints

## Features for the Model

### Input Features (Melody):
1. **Pitch Information**:
   - MIDI note numbers (0-127)
   - Relative pitch within key
   - Interval from previous note

2. **Rhythmic Features**:
   - Note duration (quarter notes, eighth notes, etc.)
   - Position within measure
   - Syncopation indicators

3. **Temporal Features**:
   - Time since last note
   - Beat strength (downbeat, upbeat)
   - Measure position

4. **Musical Context**:
   - Estimated key signature
   - Mode (major/minor)
   - Tempo markings

### Output Features (Chords):
1. **Harmonic Content**:
   - Root note (C, D, E, F, G, A, B)
   - Chord quality (major, minor, diminished, augmented, 7th, etc.)
   - Extensions (9th, 11th, 13th)
   - Inversions (root position, first inversion, second inversion)

2. **Timing Information**:
   - Chord duration
   - Onset time relative to melody
   - Harmonic rhythm patterns

### Derived Features:
1. **Melodic Analysis**:
   - Melodic contour (up, down, static)
   - Intervallic patterns
   - Phrase boundaries

2. **Harmonic Analysis**:
   - Functional harmony labels (I, V, vi, etc.)
   - Circle of fifths relationships
   - Cadence patterns

## End Goal and Success Metrics

### Primary Objective:
Create a model that generates musically coherent and stylistically appropriate chord progressions for given melodies, suitable for real-time accompaniment or composition assistance.

### Success Metrics:

#### Quantitative Metrics:
1. **Harmonic Accuracy**: Percentage of chords that fit within the established key
2. **Voice Leading Quality**: Smoothness of chord transitions (measured by semitone movement)
3. **Perplexity**: Model confidence in chord predictions
4. **BLEU Score**: Similarity to human-composed reference chord progressions

#### Qualitative Metrics:
1. **Musical Coherence**: Human evaluation of harmonic logic
2. **Style Consistency**: Adherence to genre-specific harmonic patterns
3. **Creative Appropriateness**: Balance between predictability and surprise

### Target Applications:
1. **Real-time Accompaniment**: Live performance assistance for musicians
2. **Composition Tool**: Aid for songwriters and composers
3. **Music Education**: Teaching harmonic relationships and chord progressions
4. **Game/Film Scoring**: Automated background music generation

### Performance Targets:
- **Latency**: < 50ms for real-time applications
- **Accuracy**: >85% harmonic appropriateness on test set
- **User Satisfaction**: >4/5 rating from musician evaluators

## Implementation Phases:

### Phase 1: Data Preparation and Analysis
- Download and preprocess MIDI datasets
- Extract melody and chord pairs
- Analyze harmonic patterns and statistics

### Phase 2: Model Development
- Implement baseline models (LSTM, simple Transformer)
- Develop full Music Transformer architecture
- Train and validate models

### Phase 3: Evaluation and Refinement
- Quantitative evaluation on test sets
- Human evaluation with musicians
- Model optimization and hyperparameter tuning

### Phase 4: Deployment and Integration
- Real-time inference optimization
- User interface development
- Plugin/API development for DAWs

## Technical Requirements:
- **Framework**: PyTorch or TensorFlow
- **Hardware**: GPU with >=8GB VRAM for training
- **Dependencies**: music21, pretty_midi, mido for MIDI processing
- **Storage**: ~50GB for dataset and model checkpoints

---

*This project aims to bridge the gap between music theory and machine learning, creating a practical tool for musicians while advancing the field of computational music generation.*
