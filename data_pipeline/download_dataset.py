import os
import requests
import zipfile
import tarfile
from pathlib import Path
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LakhMIDIDownloader:
    """
    Downloads and extracts the Lakh MIDI Dataset (LMD).
    
    The LMD is available in several versions:
    - LMD-full: Complete dataset (~35GB compressed)
    - LMD-matched: Subset with metadata matches (~3.8GB compressed)
    - LMD-aligned: Time-aligned version (~35GB compressed)
    """
    
    def __init__(self, data_dir="./data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Dataset URLs (these are the official links from Colin Raffel's website)
        self.urls = {
            "lmd_matched": "http://hog.ee.columbia.edu/craffel/lmd/lmd_matched.tar.gz",
            "lmd_full": "http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz",
            "lmd_aligned": "http://hog.ee.columbia.edu/craffel/lmd/lmd_aligned.tar.gz"
        }
        
        # Start with the matched dataset as it's smaller and has metadata
        self.default_dataset = "lmd_matched"
    
    def download_file(self, url, filename):
        """Download a file with progress bar."""
        logger.info(f"Downloading {filename} from {url}")
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filename, 'wb') as file, tqdm(
            desc=filename.name,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for chunk in response.iter_content(chunk_size=8192):
                size = file.write(chunk)
                progress_bar.update(size)
        
        logger.info(f"Downloaded {filename}")
    
    def extract_archive(self, archive_path, extract_to):
        """Extract tar.gz archive."""
        logger.info(f"Extracting {archive_path} to {extract_to}")
        
        with tarfile.open(archive_path, 'r:gz') as tar:
            # Get total number of members for progress bar
            members = tar.getmembers()
            
            with tqdm(total=len(members), desc="Extracting") as progress_bar:
                for member in members:
                    tar.extract(member, extract_to)
                    progress_bar.update(1)
        
        logger.info(f"Extraction complete")
    
    def download_dataset(self, dataset_name=None):
        """
        Download and extract the specified dataset.
        
        Args:
            dataset_name (str): One of 'lmd_matched', 'lmd_full', 'lmd_aligned'
                               Defaults to 'lmd_matched'
        """
        if dataset_name is None:
            dataset_name = self.default_dataset
        
        if dataset_name not in self.urls:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(self.urls.keys())}")
        
        url = self.urls[dataset_name]
        filename = self.data_dir / f"{dataset_name}.tar.gz"
        extract_dir = self.data_dir / dataset_name
        
        # Check if already downloaded and extracted
        if extract_dir.exists() and any(extract_dir.iterdir()):
            logger.info(f"Dataset {dataset_name} already exists at {extract_dir}")
            return extract_dir
        
        # Download if not exists
        if not filename.exists():
            try:
                self.download_file(url, filename)
            except Exception as e:
                logger.error(f"Failed to download {dataset_name}: {e}")
                raise
        
        # Extract
        try:
            extract_dir.mkdir(exist_ok=True)
            self.extract_archive(filename, extract_dir)
        except Exception as e:
            logger.error(f"Failed to extract {dataset_name}: {e}")
            raise
        
        # Clean up archive file to save space (optional)
        cleanup = input(f"Delete archive file {filename} to save space? (y/n): ").lower().strip()
        if cleanup == 'y':
            filename.unlink()
            logger.info(f"Deleted archive file {filename}")
        
        return extract_dir
    
    def get_midi_files(self, dataset_dir):
        """
        Get list of all MIDI files in the dataset directory.
        
        Args:
            dataset_dir (Path): Directory containing the extracted dataset
            
        Returns:
            list: List of Path objects pointing to MIDI files
        """
        midi_extensions = ['.mid', '.midi', '.MID', '.MIDI']
        midi_files = []
        
        for ext in midi_extensions:
            midi_files.extend(dataset_dir.rglob(f'*{ext}'))
        
        logger.info(f"Found {len(midi_files)} MIDI files")
        return midi_files
    
    def validate_dataset(self, dataset_dir):
        """
        Validate the downloaded dataset by checking file counts and structure.
        
        Args:
            dataset_dir (Path): Directory containing the extracted dataset
        """
        midi_files = self.get_midi_files(dataset_dir)
        
        if len(midi_files) == 0:
            logger.error("No MIDI files found in dataset!")
            return False
        
        # Check file sizes
        total_size = sum(f.stat().st_size for f in midi_files)
        logger.info(f"Dataset contains {len(midi_files)} MIDI files")
        logger.info(f"Total size: {total_size / (1024**2):.2f} MB")
        
        # Sample a few files to check they're valid
        sample_files = midi_files[:min(5, len(midi_files))]
        for midi_file in sample_files:
            try:
                # Try to read with mido to validate
                import mido
                mid = mido.MidiFile(midi_file)
                logger.info(f"Validated {midi_file.name}: {len(mid.tracks)} tracks, {len(mid)} messages")
            except Exception as e:
                logger.warning(f"Could not validate {midi_file}: {e}")
        
        return True


def main():
    """Main function to download and validate the dataset."""
    downloader = LakhMIDIDownloader()
    
    print("Lakh MIDI Dataset Downloader")
    print("============================")
    print("Available datasets:")
    print("1. lmd_matched - Subset with metadata matches (~3.8GB)")
    print("2. lmd_full - Complete dataset (~35GB)")
    print("3. lmd_aligned - Time-aligned version (~35GB)")
    print()
    
    choice = input("Which dataset would you like to download? (1-3, default=1): ").strip()
    
    dataset_map = {
        "1": "lmd_matched",
        "2": "lmd_full", 
        "3": "lmd_aligned",
        "": "lmd_matched"  # default
    }
    
    dataset_name = dataset_map.get(choice)
    if not dataset_name:
        print("Invalid choice. Using default (lmd_matched)")
        dataset_name = "lmd_matched"
    
    try:
        # Download and extract
        dataset_dir = downloader.download_dataset(dataset_name)
        
        # Validate
        if downloader.validate_dataset(dataset_dir):
            print(f"\n✅ Successfully downloaded and validated {dataset_name}")
            print(f"📁 Dataset location: {dataset_dir}")
            
            # Get MIDI files for further processing
            midi_files = downloader.get_midi_files(dataset_dir)
            print(f"🎵 Found {len(midi_files)} MIDI files ready for processing")
            
        else:
            print(f"❌ Dataset validation failed")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
