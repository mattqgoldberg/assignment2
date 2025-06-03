# Data Pipeline for MIDI Melody to Chord Progression

This directory contains the data pipeline for downloading and preprocessing the Lakh MIDI Dataset (LMD) for training a melody-to-chord progression model.

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r ../requirements.txt
   ```

2. **Run the complete pipeline:**
   ```bash
   python run_pipeline.py
   ```

3. **Or run individual steps:**
   ```bash
   # Download only
   python run_pipeline.py --download-only
   
   # Preprocess only (if dataset already downloaded)
   python run_pipeline.py --preprocess-only
   ```

## Pipeline Components

### 1. Dataset Download (`download_dataset.py`)
- Downloads the Lakh MIDI Dataset from official sources
- Supports three dataset variants:
  - `lmd_matched`: Subset with metadata matches (~3.8GB) - **Recommended for starting**
  - `lmd_full`: Complete dataset (~35GB)
  - `lmd_aligned`: Time-aligned version (~35GB)
- Validates downloaded files
- Extracts archives and manages storage

### 2. MIDI Preprocessing (`preprocess_midi.py`)
- Parses MIDI files using music21 library
- Extracts melody lines (highest voice/pitch)
- Identifies chord progressions and harmonic content
- Aligns melody notes with corresponding chords
- Analyzes musical features:
  - Key signatures
  - Time signatures
  - Chord qualities (major, minor, 7th, etc.)
  - Rhythmic patterns

### 3. Pipeline Orchestration (`run_pipeline.py`)
- Coordinates download and preprocessing steps
- Provides command-line interface with options
- Generates processing statistics and reports
- Handles error recovery and logging

## Output Data Structure

The preprocessing pipeline produces:

### Processed Data (`processed_data/processed_data.pkl`)
Python pickle file containing list of dictionaries, each representing one MIDI file:

```python
{
    'file_path': str,                    # Original MIDI file path
    'aligned_pairs': [                   # List of melody-chord alignments
        {
            'melody_pitch': int,         # MIDI note number (0-127)
            'melody_offset': float,      # Time offset in quarter notes
            'melody_duration': float,    # Note duration in quarter notes
            'chord_root': str,           # Chord root note (C, D, E, etc.)
            'chord_quality': str,        # Chord type (major, minor, etc.)
            'chord_offset': float,       # Chord onset time
            'chord_duration': float      # Chord duration
        },
        # ... more pairs
    ],
    'key_signature': str,                # Detected key signature
    'time_signature': str,               # Time signature (e.g., "4/4")
    'total_melody_notes': int,           # Total melody notes in file
    'total_chords': int                  # Total chords in file
}
```

### Processing Statistics (`processed_data/processing_stats.pkl`)
Contains analysis of the processed dataset:
- File processing success/failure rates
- Chord distribution across the dataset
- Key signature distribution
- Overall dataset statistics

## Configuration Options

### Command Line Arguments

```bash
python run_pipeline.py [OPTIONS]

Options:
  --dataset {lmd_matched,lmd_full,lmd_aligned}
                        Dataset to download (default: lmd_matched)
  --max-files MAX_FILES
                        Maximum number of files to process (for testing)
  --data-dir DATA_DIR   Directory to store raw data (default: ./data)
  --output-dir OUTPUT_DIR
                        Directory for processed data (default: ./processed_data)
  --download-only       Only download, don't preprocess
  --preprocess-only     Only preprocess existing data
```

### Examples

```bash
# Download and process the matched dataset (recommended start)
python run_pipeline.py --dataset lmd_matched

# Process only first 100 files for quick testing
python run_pipeline.py --max-files 100

# Use custom directories
python run_pipeline.py --data-dir /path/to/data --output-dir /path/to/output

# Download the full dataset (large!)
python run_pipeline.py --dataset lmd_full
```

## Data Quality and Filtering

The preprocessing pipeline includes several quality filters:

1. **MIDI Validation**: Files must be parseable by both `mido` and `music21`
2. **Content Requirements**: Files must contain both melody and chord information
3. **Temporal Alignment**: Melody notes must align with chord progressions
4. **Musical Validity**: Basic harmonic analysis must be possible

Files that fail these checks are logged but excluded from the final dataset.

## Performance Notes

- **Processing Speed**: ~10-50 files per minute depending on file complexity
- **Memory Usage**: ~100-500MB during processing
- **Storage Requirements**: 
  - Raw dataset: 3.8GB (matched) to 35GB (full)
  - Processed data: ~10-20% of raw dataset size

## Troubleshooting

### Common Issues

1. **Download Failures**: Check internet connection and disk space
2. **Processing Errors**: Often due to corrupted or non-standard MIDI files
3. **Memory Issues**: Reduce `--max-files` for testing, or increase system RAM

### Logs and Debugging

The pipeline provides detailed logging. Check terminal output for:
- Download progress and validation
- File processing success/failure rates
- Error messages for failed files

### Recovery

The pipeline is designed to be resumable:
- Downloaded archives are preserved for re-extraction
- Processed data is saved incrementally
- Failed files are logged for review

## Next Steps

After running the pipeline:

1. **Explore the data**: Use Jupyter notebooks to analyze processed data
2. **Feature Engineering**: Extract additional features from the aligned pairs
3. **Model Training**: Use the processed data to train melody-to-chord models
4. **Evaluation**: Validate model performance on held-out test sets
