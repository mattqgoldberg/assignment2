import mido
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from tqdm import tqdm
import pickle
from collections import defaultdict
import music21
from music21 import converter, stream, note, chord, key, meter

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MIDIPreprocessor:
    """
    Preprocesses MIDI files to extract melody and chord features for training.
    """
    
    def __init__(self, output_dir="./processed_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Musical parameters
        self.time_resolution = 480  # Ticks per quarter note
        self.min_duration = 0.125   # Minimum note duration (32nd note)
        self.max_sequence_length = 512  # Maximum sequence length for model
        
        # Chord vocabulary - common chord types
        self.chord_types = [
            'major', 'minor', 'diminished', 'augmented',
            'major7', 'minor7', 'dominant7', 'diminished7',
            'major6', 'minor6', 'sus2', 'sus4'
        ]
        
        # Note names for root detection
        self.note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    def midi_to_music21(self, midi_path):
        """
        Convert MIDI file to music21 stream for easier analysis.
        
        Args:
            midi_path (Path): Path to MIDI file
            
        Returns:
            music21.stream.Stream: Parsed music21 stream
        """
        try:
            return converter.parse(str(midi_path))
        except Exception as e:
            logger.warning(f"Could not parse {midi_path} with music21: {e}")
            return None
    
    def extract_melody_from_stream(self, stream_obj):
        """
        Extract melody line from music21 stream (typically highest voice).
        
        Args:
            stream_obj: music21 stream object
            
        Returns:
            list: List of melody notes with (pitch, duration, offset) tuples
        """
        melody_notes = []
        
        # Get all notes and chords from all parts
        all_notes = []
        for element in stream_obj.flat.notesAndRests:
            if isinstance(element, note.Note):
                all_notes.append((element.pitch.midi, element.offset, element.duration.quarterLength))
            elif isinstance(element, chord.Chord):
                # For chords, take the highest note as melody
                highest_pitch = max([p.midi for p in element.pitches])
                all_notes.append((highest_pitch, element.offset, element.duration.quarterLength))
        
        # Sort by offset time
        all_notes.sort(key=lambda x: x[1])
        
        # Filter to get melody line (remove notes that are too close in time)
        if all_notes:
            melody_notes = [all_notes[0]]  # Start with first note
            
            for pitch, offset, duration in all_notes[1:]:
                # If this note is significantly later than the last melody note, add it
                last_offset = melody_notes[-1][1]
                if offset >= last_offset + self.min_duration:
                    melody_notes.append((pitch, offset, duration))
        
        return melody_notes
    
    def extract_chords_from_stream(self, stream_obj):
        """
        Extract chord progressions from music21 stream by analyzing simultaneous notes.
        
        Args:
            stream_obj: music21 stream object
            
        Returns:
            list: List of chord progressions with (root, quality, offset, duration) tuples
        """
        chord_progression = []
        
        try:
            # First try to find explicit chord objects
            chords = stream_obj.flatten().getElementsByClass(chord.Chord)
            
            if chords:
                # Process explicit chords
                for c in chords:
                    try:
                        root_note = c.root().name
                        chord_quality = self.classify_chord_quality(c)
                        
                        chord_progression.append({
                            'root': root_note,
                            'quality': chord_quality,
                            'offset': c.offset,
                            'duration': c.duration.quarterLength,
                            'notes': [p.midi for p in c.pitches]
                        })
                    except Exception as e:
                        logger.debug(f"Could not analyze chord: {e}")
                        continue
            else:
                # No explicit chords found, construct from simultaneous notes
                chord_progression = self.construct_chords_from_notes(stream_obj)
                    
        except Exception as e:
            logger.warning(f"Could not extract chords: {e}")
        
        return chord_progression
    
    def construct_chords_from_notes(self, stream_obj):
        """
        Construct chords from simultaneous notes in different parts/tracks.
        
        Args:
            stream_obj: music21 stream object
            
        Returns:
            list: List of constructed chord progressions
        """
        chord_progression = []
        
        try:
            # Get all notes from all parts
            all_notes = []
            
            if hasattr(stream_obj, 'parts') and stream_obj.parts:
                # Multi-part stream
                for part in stream_obj.parts:
                    notes = part.flatten().getElementsByClass(note.Note)
                    for n in notes:
                        all_notes.append({
                            'pitch': n.pitch.midi,
                            'offset': n.offset,
                            'duration': n.duration.quarterLength,
                            'part': part
                        })
            else:
                # Single part stream
                notes = stream_obj.flatten().getElementsByClass(note.Note)
                for n in notes:
                    all_notes.append({
                        'pitch': n.pitch.midi,
                        'offset': n.offset,
                        'duration': n.duration.quarterLength,
                        'part': None
                    })
            
            if not all_notes:
                return chord_progression
            
            # Sort by offset
            all_notes.sort(key=lambda x: x['offset'])
            
            # Group notes by time windows to form chords
            time_window = 0.5  # 0.5 quarter note window for simultaneity
            current_chord_notes = []
            current_offset = all_notes[0]['offset']
            
            for note_info in all_notes:
                # If note is within time window, add to current chord
                if note_info['offset'] <= current_offset + time_window:
                    current_chord_notes.append(note_info)
                else:
                    # Process current chord and start new one
                    if len(current_chord_notes) >= 2:  # Need at least 2 notes for a chord
                        chord_info = self.analyze_chord_from_notes(current_chord_notes, current_offset)
                        if chord_info:
                            chord_progression.append(chord_info)
                    
                    # Start new chord
                    current_chord_notes = [note_info]
                    current_offset = note_info['offset']
            
            # Process final chord
            if len(current_chord_notes) >= 2:
                chord_info = self.analyze_chord_from_notes(current_chord_notes, current_offset)
                if chord_info:
                    chord_progression.append(chord_info)
                    
        except Exception as e:
            logger.warning(f"Could not construct chords from notes: {e}")
        
        return chord_progression
    
    def analyze_chord_from_notes(self, note_list, offset):
        """
        Analyze a group of simultaneous notes to determine chord root and quality.
        
        Args:
            note_list: List of note dictionaries
            offset: Timing offset of the chord
            
        Returns:
            dict: Chord information or None
        """
        try:
            # Extract unique pitches and sort
            pitches = sorted(list(set([n['pitch'] for n in note_list])))
            
            if len(pitches) < 2:
                return None
            
            # Create a music21 chord object for analysis
            chord_obj = chord.Chord([note.Note(midi=p) for p in pitches])
            
            # Analyze chord
            root_note = chord_obj.root().name
            chord_quality = self.classify_chord_quality(chord_obj)
            
            # Calculate average duration
            avg_duration = sum([n['duration'] for n in note_list]) / len(note_list)
            
            return {
                'root': root_note,
                'quality': chord_quality,
                'offset': offset,
                'duration': avg_duration,
                'notes': pitches
            }
            
        except Exception as e:
            logger.debug(f"Could not analyze chord from notes: {e}")
            return None
    
    def classify_chord_quality(self, chord_obj):
        """
        Classify the quality of a music21 chord object.
        
        Args:
            chord_obj: music21 chord object
            
        Returns:
            str: Chord quality classification
        """
        try:
            # Get chord intervals
            intervals = [p.midi for p in chord_obj.pitches]
            intervals.sort()
            
            if len(intervals) < 2:
                return 'unknown'
            
            # Normalize to root position
            root = intervals[0]
            normalized = [(interval - root) % 12 for interval in intervals]
            normalized = list(set(normalized))  # Remove duplicates
            normalized.sort()
            
            # Common chord patterns (semitones from root)
            chord_patterns = {
                # Triads
                (0, 4, 7): 'major',
                (0, 3, 7): 'minor',
                (0, 3, 6): 'diminished',
                (0, 4, 8): 'augmented',
                
                # 7th chords
                (0, 4, 7, 11): 'major7',
                (0, 3, 7, 10): 'minor7',
                (0, 4, 7, 10): 'dominant7',
                (0, 3, 6, 9): 'diminished7',
                (0, 3, 6, 10): 'half_diminished7',
                (0, 4, 8, 10): 'augmented7',
                
                # 6th chords
                (0, 4, 7, 9): 'major6',
                (0, 3, 7, 9): 'minor6',
                
                # Suspended chords
                (0, 2, 7): 'sus2',
                (0, 5, 7): 'sus4',
                
                # Extended chords (simplified)
                (0, 4, 7, 10, 14): 'dominant9',
                (0, 3, 7, 10, 14): 'minor9',
                (0, 4, 7, 11, 14): 'major9',
                
                # Power chord (5th)
                (0, 7): 'power',
                
                # Simple intervals
                (0, 3): 'minor_3rd',
                (0, 4): 'major_3rd',
                (0, 5): 'perfect_4th',
            }
            
            # Try exact pattern matches first
            pattern = tuple(normalized)
            if pattern in chord_patterns:
                return chord_patterns[pattern]
            
            # Try subsets (for chords with extra notes)
            for chord_pattern, quality in chord_patterns.items():
                if len(chord_pattern) <= len(normalized):
                    if all(note in normalized for note in chord_pattern):
                        # Check if it's the most likely match (contains core intervals)
                        if chord_pattern == (0, 4, 7) or chord_pattern == (0, 3, 7):  # Major/minor priority
                            return quality
            
            # Fallback analysis for basic triads
            if len(normalized) >= 3:
                has_major_third = 4 in normalized
                has_minor_third = 3 in normalized
                has_perfect_fifth = 7 in normalized
                has_diminished_fifth = 6 in normalized
                has_augmented_fifth = 8 in normalized
                
                if has_perfect_fifth:
                    if has_major_third:
                        return 'major'
                    elif has_minor_third:
                        return 'minor'
                elif has_diminished_fifth and has_minor_third:
                    return 'diminished'
                elif has_augmented_fifth and has_major_third:
                    return 'augmented'
            
            # For two-note intervals
            if len(normalized) == 2:
                interval = normalized[1]
                if interval == 7:
                    return 'power'
                elif interval == 4:
                    return 'major_3rd'
                elif interval == 3:
                    return 'minor_3rd'
                elif interval == 5:
                    return 'perfect_4th'
            
            return 'unknown'
            
        except Exception as e:
            logger.debug(f"Could not classify chord: {e}")
            return 'unknown'
    
    def align_melody_chords(self, melody_notes, chord_progression):
        """
        Align melody notes with their corresponding chords based on timing.
        
        Args:
            melody_notes: List of melody note tuples
            chord_progression: List of chord dictionaries
            
        Returns:
            list: List of aligned (melody_note, chord) pairs
        """
        aligned_pairs = []
        
        if not melody_notes or not chord_progression:
            return aligned_pairs
        
        # Sort both by time
        melody_notes.sort(key=lambda x: x[1])  # Sort by offset
        chord_progression.sort(key=lambda x: x['offset'])
        
        for melody_note in melody_notes:
            pitch, offset, duration = melody_note
            
            # Find the chord that's active at this melody note's time
            active_chord = None
            for chord_info in chord_progression:
                chord_start = chord_info['offset']
                chord_end = chord_start + chord_info['duration']
                
                # Check if melody note falls within chord duration
                if chord_start <= offset < chord_end:
                    active_chord = chord_info
                    break
            
            # If no exact match, find closest preceding chord
            if active_chord is None:
                for chord_info in reversed(chord_progression):
                    if chord_info['offset'] <= offset:
                        active_chord = chord_info
                        break
            
            if active_chord:
                aligned_pairs.append({
                    'melody_pitch': pitch,
                    'melody_offset': offset,
                    'melody_duration': duration,
                    'chord_root': active_chord['root'],
                    'chord_quality': active_chord['quality'],
                    'chord_offset': active_chord['offset'],
                    'chord_duration': active_chord['duration']
                })
        
        return aligned_pairs
    
    def process_midi_file(self, midi_path):
        """
        Process a single MIDI file to extract melody-chord pairs.
        
        Args:
            midi_path (Path): Path to MIDI file
            
        Returns:
            dict: Processed data with melody-chord alignments
        """
        try:
            # Convert to music21 stream
            stream_obj = self.midi_to_music21(midi_path)
            if stream_obj is None:
                return None
            
            # Extract melody and chords
            melody_notes = self.extract_melody_from_stream(stream_obj)
            chord_progression = self.extract_chords_from_stream(stream_obj)
            
            # Align melody with chords
            aligned_pairs = self.align_melody_chords(melody_notes, chord_progression)
            
            if not aligned_pairs:
                return None
            
            # Extract additional metadata
            try:
                key_sig = stream_obj.analyze('key')
                time_sig = stream_obj.flat.getElementsByClass(meter.TimeSignature)
                time_signature = str(time_sig[0]) if time_sig else "4/4"
            except:
                key_sig = None
                time_signature = "4/4"
            
            return {
                'file_path': str(midi_path),
                'aligned_pairs': aligned_pairs,
                'key_signature': str(key_sig) if key_sig else 'unknown',
                'time_signature': time_signature,
                'total_melody_notes': len(melody_notes),
                'total_chords': len(chord_progression)
            }
            
        except Exception as e:
            logger.warning(f"Error processing {midi_path}: {e}")
            return None
    
    def process_dataset(self, midi_files, max_files=None):
        """
        Process multiple MIDI files and save processed data.
        
        Args:
            midi_files (list): List of MIDI file paths
            max_files (int): Maximum number of files to process (for testing)
            
        Returns:
            list: List of processed data dictionaries
        """
        if max_files:
            midi_files = midi_files[:max_files]
        
        processed_data = []
        failed_files = []
        
        logger.info(f"Processing {len(midi_files)} MIDI files...")
        
        for midi_file in tqdm(midi_files, desc="Processing MIDI files"):
            result = self.process_midi_file(midi_file)
            
            if result:
                processed_data.append(result)
            else:
                failed_files.append(str(midi_file))
        
        logger.info(f"Successfully processed {len(processed_data)} files")
        logger.info(f"Failed to process {len(failed_files)} files")
        
        # Save processed data
        output_file = self.output_dir / "processed_data.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump(processed_data, f)
        
        # Save statistics
        self.save_statistics(processed_data, failed_files)
        
        return processed_data
    
    def save_statistics(self, processed_data, failed_files):
        """
        Save processing statistics and data analysis.
        
        Args:
            processed_data (list): List of processed data dictionaries
            failed_files (list): List of failed file paths
        """
        stats = {
            'total_files_processed': len(processed_data),
            'total_files_failed': len(failed_files),
            'total_melody_chord_pairs': sum(len(d['aligned_pairs']) for d in processed_data),
            'failed_files': failed_files[:100],  # Save first 100 failed files
        }
        
        # Analyze chord distribution
        chord_counts = defaultdict(int)
        key_counts = defaultdict(int)
        
        for data in processed_data:
            key_counts[data['key_signature']] += 1
            for pair in data['aligned_pairs']:
                chord_key = f"{pair['chord_root']}_{pair['chord_quality']}"
                chord_counts[chord_key] += 1
        
        # Convert to Counter objects for most_common functionality
        from collections import Counter
        stats['chord_distribution'] = dict(Counter(chord_counts).most_common(50))
        stats['key_distribution'] = dict(Counter(key_counts).most_common(20))
        
        # Save statistics
        stats_file = self.output_dir / "processing_stats.pkl"
        with open(stats_file, 'wb') as f:
            pickle.dump(stats, f)
        
        logger.info(f"Saved processing statistics to {stats_file}")


def main():
    """Main function to preprocess MIDI data."""
    from download_dataset import LakhMIDIDownloader
    
    # Initialize components
    downloader = LakhMIDIDownloader()
    preprocessor = MIDIPreprocessor()
    
    # Check if dataset exists, if not download it
    dataset_dir = Path("./data/lmd_matched")
    if not dataset_dir.exists():
        print("Dataset not found. Downloading...")
        dataset_dir = downloader.download_dataset("lmd_matched")
    
    # Get MIDI files
    midi_files = downloader.get_midi_files(dataset_dir)
    
    # Ask user about processing limits
    print(f"\nFound {len(midi_files)} MIDI files")
    max_files_input = input("Enter max files to process (or press Enter for all): ").strip()
    max_files = int(max_files_input) if max_files_input.isdigit() else None
    
    # Process dataset
    processed_data = preprocessor.process_dataset(midi_files, max_files)
    
    print(f"\n✅ Processing complete!")
    print(f"📊 Processed {len(processed_data)} files")
    print(f"💾 Data saved to {preprocessor.output_dir}")


if __name__ == "__main__":
    main()
