#!/usr/bin/env python3
"""
Main data pipeline script for the MIDI Melody to Chord Progression project.
This script orchestrates the download and preprocessing of the Lakh MIDI Dataset.
"""

import sys
import argparse
from pathlib import Path

# Add the data_pipeline directory to Python path
sys.path.append(str(Path(__file__).parent))

from download_dataset import LakhMIDIDownloader
from preprocess_midi import MIDIPreprocessor

def main():
    parser = argparse.ArgumentParser(description="MIDI Melody to Chord Data Pipeline")
    parser.add_argument("--dataset", choices=["lmd_matched", "lmd_full", "lmd_aligned"], 
                       default="lmd_matched", help="Dataset to download")
    parser.add_argument("--max-files", type=int, help="Maximum number of files to process")
    parser.add_argument("--data-dir", default="./data", help="Directory to store data")
    parser.add_argument("--output-dir", default="./processed_data", help="Directory for processed data")
    parser.add_argument("--download-only", action="store_true", help="Only download, don't preprocess")
    parser.add_argument("--preprocess-only", action="store_true", help="Only preprocess existing data")
    
    args = parser.parse_args()
    
    print("🎵 MIDI Melody to Chord Progression Data Pipeline")
    print("=" * 50)
    
    # Initialize components
    downloader = LakhMIDIDownloader(data_dir=args.data_dir)
    preprocessor = MIDIPreprocessor(output_dir=args.output_dir)
    
    # Step 1: Download dataset (unless preprocess-only)
    if not args.preprocess_only:
        print(f"\n📥 Step 1: Downloading {args.dataset} dataset...")
        try:
            dataset_dir = downloader.download_dataset(args.dataset)
            print(f"✅ Dataset downloaded to: {dataset_dir}")
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return 1
    else:
        dataset_dir = Path(args.data_dir) / args.dataset
        if not dataset_dir.exists():
            print(f"❌ Dataset directory {dataset_dir} not found!")
            return 1
    
    # Exit if download-only
    if args.download_only:
        print("✅ Download complete. Exiting as requested.")
        return 0
    
    # Step 2: Preprocess MIDI files
    print(f"\n🔄 Step 2: Preprocessing MIDI files...")
    try:
        midi_files = downloader.get_midi_files(dataset_dir)
        
        if not midi_files:
            print("❌ No MIDI files found in dataset!")
            return 1
        
        print(f"Found {len(midi_files)} MIDI files")
        
        # Process files
        processed_data = preprocessor.process_dataset(midi_files, args.max_files)
        
        print(f"✅ Preprocessing complete!")
        print(f"📊 Successfully processed {len(processed_data)} files")
        print(f"💾 Processed data saved to: {preprocessor.output_dir}")
        
        # Show quick statistics
        if processed_data:
            total_pairs = sum(len(d['aligned_pairs']) for d in processed_data)
            print(f"🎼 Total melody-chord pairs: {total_pairs}")
            print(f"📈 Average pairs per file: {total_pairs / len(processed_data):.1f}")
        
    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")
        return 1
    
    print(f"\n🎉 Pipeline complete! Data ready for model training.")
    return 0

if __name__ == "__main__":
    exit(main())
