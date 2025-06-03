#!/usr/bin/env python3
"""
Audio playback module for MIDI melodies and chord progressions.
Provides functionality to hear the model's predictions in action.
"""

import numpy as np
import pygame
import pygame.midi
import time
import threading
from pathlib import Path
import tempfile
import subprocess
import sys

# Try different MIDI/audio approaches
try:
    import mido
    MIDO_AVAILABLE = True
except ImportError:
    MIDO_AVAILABLE = False

try:
    from music21 import stream, note, chord, duration, meter, tempo, midi
    MUSIC21_AVAILABLE = True
except ImportError:
    MUSIC21_AVAILABLE = False

try:
    import pretty_midi
    PRETTY_MIDI_AVAILABLE = True
except ImportError:
    PRETTY_MIDI_AVAILABLE = False

class AudioPlayer:
    """Class for playing MIDI melodies and chord progressions."""
    
    def __init__(self):
        self.playing = False
        self.current_thread = None
        
        # Try to initialize pygame for audio playback
        try:
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            pygame.midi.init()
            self.pygame_available = True
            print("🎵 Audio system initialized with pygame")
        except Exception as e:
            self.pygame_available = False
            print(f"⚠️  Pygame not available: {e}")
        
    def __del__(self):
        """Clean up pygame resources."""
        try:
            if self.pygame_available:
                pygame.midi.quit()
                pygame.mixer.quit()
        except:
            pass
    
    def midi_note_to_frequency(self, midi_note):
        """Convert MIDI note number to frequency in Hz."""
        return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))
    
    def create_tone(self, frequency, duration_ms, sample_rate=22050):
        """Create a simple sine wave tone."""
        frames = int(duration_ms * sample_rate / 1000)
        arr = np.zeros(frames)
        
        # Generate sine wave
        for i in range(frames):
            arr[i] = np.sin(2 * np.pi * frequency * i / sample_rate) * 0.3
        
        # Apply fade in/out to prevent clicks
        fade_frames = min(100, frames // 10)
        for i in range(fade_frames):
            fade = i / fade_frames
            arr[i] *= fade
            arr[-(i+1)] *= fade
        
        return (arr * 32767).astype(np.int16)
    
    def create_chord_tone(self, midi_notes, duration_ms, sample_rate=22050):
        """Create a chord tone by mixing multiple frequencies."""
        if not midi_notes:
            return np.zeros(0, dtype=np.int16)
        
        frames = int(duration_ms * sample_rate / 1000)
        arr = np.zeros(frames)
        
        # Mix frequencies for each note in the chord
        for midi_note in midi_notes:
            freq = self.midi_note_to_frequency(midi_note)
            for i in range(frames):
                arr[i] += np.sin(2 * np.pi * freq * i / sample_rate) * (0.3 / len(midi_notes))
        
        # Apply fade in/out
        fade_frames = min(100, frames // 10)
        for i in range(fade_frames):
            fade = i / fade_frames
            arr[i] *= fade
            arr[-(i+1)] *= fade
        
        return (arr * 32767).astype(np.int16)
    
    def play_tone_pygame(self, tone_data):
        """Play a tone using pygame."""
        if not self.pygame_available:
            return
        
        try:
            # Create stereo sound (duplicate mono to both channels)
            stereo_data = np.zeros((len(tone_data), 2), dtype=np.int16)
            stereo_data[:, 0] = tone_data
            stereo_data[:, 1] = tone_data
            
            sound = pygame.sndarray.make_sound(stereo_data)
            sound.play()
            
            # Wait for the sound to finish
            time.sleep(len(tone_data) / 22050.0)
            
        except Exception as e:
            print(f"Error playing tone: {e}")
    
    def play_melody_simple(self, melody_sequence, note_duration=0.5):
        """
        Play a melody sequence using simple tones.
        
        Args:
            melody_sequence: numpy array with shape (seq_len, 4) or list of note info
            note_duration: duration for each note in seconds
        """
        if not self.pygame_available:
            print("🔇 Audio playback not available")
            return
        
        self.playing = True
        print("🎵 Playing melody...")
        
        try:
            for i, note_info in enumerate(melody_sequence):
                if not self.playing:
                    break
                    
                # Extract MIDI note number
                if isinstance(note_info, (list, np.ndarray)) and len(note_info) > 0:
                    midi_note = int(note_info[0])  # First element is pitch
                else:
                    continue
                
                # Skip silent notes (pitch 0 or invalid)
                if midi_note <= 0 or midi_note > 127:
                    time.sleep(note_duration)
                    continue
                
                # Generate and play tone
                frequency = self.midi_note_to_frequency(midi_note)
                tone = self.create_tone(frequency, note_duration * 1000)
                self.play_tone_pygame(tone)
                
                # Small gap between notes
                time.sleep(0.05)
        
        except Exception as e:
            print(f"Error playing melody: {e}")
        
        finally:
            self.playing = False
            print("🎵 Melody finished")
    
    def chord_name_to_midi_notes(self, chord_name, root_octave=4):
        """
        Convert a chord name to MIDI note numbers.
        This is a basic implementation - can be enhanced with more chord types.
        """
        if not chord_name or chord_name == "unknown":
            return []
        
        # Basic note name to MIDI number mapping (C4 = 60)
        note_map = {
            'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4, 'F': 5,
            'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11
        }
        
        # Parse chord name to get root note
        chord_name = str(chord_name).strip()
        root_note = None
        
        # Try to extract root note (handle sharps and flats)
        if len(chord_name) >= 2 and chord_name[1] in ['#', 'b']:
            root_note = chord_name[:2]
        elif len(chord_name) >= 1:
            root_note = chord_name[0]
        
        if root_note not in note_map:
            # If we can't parse the chord, return a default C major chord
            print(f"⚠️  Unknown chord: {chord_name}, using C major")
            root_note = 'C'
        
        root_midi = note_map[root_note] + (root_octave * 12)
        
        # Determine chord type based on name
        chord_lower = chord_name.lower()
        
        if 'min' in chord_lower or 'm' in chord_lower:
            # Minor chord: root, minor third, fifth
            return [root_midi, root_midi + 3, root_midi + 7]
        elif 'dim' in chord_lower:
            # Diminished chord: root, minor third, diminished fifth
            return [root_midi, root_midi + 3, root_midi + 6]
        elif 'aug' in chord_lower:
            # Augmented chord: root, major third, augmented fifth
            return [root_midi, root_midi + 4, root_midi + 8]
        elif '7' in chord_name:
            # Seventh chord: root, major third, fifth, minor seventh
            return [root_midi, root_midi + 4, root_midi + 7, root_midi + 10]
        else:
            # Default to major chord: root, major third, fifth
            return [root_midi, root_midi + 4, root_midi + 7]
    
    def play_chord_simple(self, chord_name, duration=1.0):
        """
        Play a chord using simple tones.
        
        Args:
            chord_name: string name of the chord
            duration: duration to play the chord in seconds
        """
        if not self.pygame_available:
            print("🔇 Audio playback not available")
            return
        
        midi_notes = self.chord_name_to_midi_notes(chord_name)
        if not midi_notes:
            return
        
        print(f"🎹 Playing chord: {chord_name} {midi_notes}")
        
        try:
            tone = self.create_chord_tone(midi_notes, duration * 1000)
            self.play_tone_pygame(tone)
        except Exception as e:
            print(f"Error playing chord: {e}")
    
    def play_chord_progression(self, chord_names, chord_duration=1.0):
        """
        Play a sequence of chords.
        
        Args:
            chord_names: list of chord names
            chord_duration: duration for each chord in seconds
        """
        if not self.pygame_available:
            print("🔇 Audio playback not available")
            return
        
        self.playing = True
        print(f"🎼 Playing chord progression: {' - '.join(map(str, chord_names))}")
        
        try:
            for i, chord_name in enumerate(chord_names):
                if not self.playing:
                    break
                
                print(f"  [{i+1}/{len(chord_names)}] {chord_name}")
                self.play_chord_simple(chord_name, chord_duration)
                
                # Small gap between chords
                time.sleep(0.1)
        
        except Exception as e:
            print(f"Error playing chord progression: {e}")
        
        finally:
            self.playing = False
            print("🎼 Chord progression finished")
    
    def play_melody_and_chords(self, melody_sequence, chord_names, note_duration=0.5):
        """
        Play melody and chords together (chords change with melody rhythm).
        
        Args:
            melody_sequence: numpy array with melody notes
            chord_names: list of chord names (one per melody note or fewer)
            note_duration: duration for each note in seconds
        """
        if not self.pygame_available:
            print("🔇 Audio playback not available")
            return
        
        self.playing = True
        print(f"🎵 Playing melody with {len(chord_names)} chords...")
        
        try:
            for i, note_info in enumerate(melody_sequence):
                if not self.playing:
                    break
                
                # Get chord for this position (cycle if fewer chords than notes)
                if chord_names and len(chord_names) > 0:
                    chord_idx = min(i, len(chord_names) - 1)
                    chord_name = chord_names[chord_idx]
                else:
                    chord_name = None
                
                # Extract melody note
                if isinstance(note_info, (list, np.ndarray)) and len(note_info) > 0:
                    midi_note = int(note_info[0])
                else:
                    continue
                
                # Create melody and chord tones
                melody_tone = None
                chord_tone = None
                
                # Generate melody tone
                if midi_note > 0 and midi_note <= 127:
                    freq = self.midi_note_to_frequency(midi_note)
                    melody_tone = self.create_tone(freq, note_duration * 1000)
                
                # Generate chord tone
                if chord_name:
                    chord_notes = self.chord_name_to_midi_notes(chord_name, root_octave=3)  # Lower octave
                    if chord_notes:
                        chord_tone = self.create_chord_tone(chord_notes, note_duration * 1000)
                
                # Mix melody and chord
                if melody_tone is not None and chord_tone is not None:
                    # Ensure same length
                    min_len = min(len(melody_tone), len(chord_tone))
                    mixed_tone = melody_tone[:min_len] + (chord_tone[:min_len] * 0.3).astype(np.int16)
                    mixed_tone = np.clip(mixed_tone, -32767, 32767).astype(np.int16)
                    self.play_tone_pygame(mixed_tone)
                elif melody_tone is not None:
                    self.play_tone_pygame(melody_tone)
                elif chord_tone is not None:
                    self.play_tone_pygame(chord_tone)
                else:
                    time.sleep(note_duration)
                
                # Progress indicator
                if i % 4 == 0:
                    progress = (i + 1) / len(melody_sequence) * 100
                    print(f"  Progress: {progress:.0f}%")
                
                # Small gap between notes
                time.sleep(0.05)
        
        except Exception as e:
            print(f"Error playing melody with chords: {e}")
        
        finally:
            self.playing = False
            print("🎵 Playback finished")
    
    def stop(self):
        """Stop current playback."""
        self.playing = False
        if self.current_thread and self.current_thread.is_alive():
            self.current_thread.join(timeout=1.0)
    
    def play_async(self, play_function, *args, **kwargs):
        """Play audio in a separate thread to avoid blocking."""
        if self.current_thread and self.current_thread.is_alive():
            self.stop()
        
        self.current_thread = threading.Thread(
            target=play_function, 
            args=args, 
            kwargs=kwargs
        )
        self.current_thread.start()
        return self.current_thread

# Convenience functions for easy use
def play_melody(melody_sequence, note_duration=0.5, async_play=True):
    """
    Convenience function to play a melody.
    
    Args:
        melody_sequence: numpy array or list of melody notes
        note_duration: duration for each note in seconds
        async_play: if True, play in background thread
    """
    player = AudioPlayer()
    
    if async_play:
        thread = player.play_async(player.play_melody_simple, melody_sequence, note_duration)
        return player, thread
    else:
        player.play_melody_simple(melody_sequence, note_duration)
        return player, None

def play_chords(chord_names, chord_duration=1.0, async_play=True):
    """
    Convenience function to play a chord progression.
    
    Args:
        chord_names: list of chord names
        chord_duration: duration for each chord in seconds
        async_play: if True, play in background thread
    """
    player = AudioPlayer()
    
    if async_play:
        thread = player.play_async(player.play_chord_progression, chord_names, chord_duration)
        return player, thread
    else:
        player.play_chord_progression(chord_names, chord_duration)
        return player, None

def play_melody_with_chords(melody_sequence, chord_names, note_duration=0.5, async_play=True):
    """
    Convenience function to play melody with chord accompaniment.
    
    Args:
        melody_sequence: numpy array or list of melody notes
        chord_names: list of chord names
        note_duration: duration for each note in seconds
        async_play: if True, play in background thread
    """
    player = AudioPlayer()
    
    if async_play:
        thread = player.play_async(player.play_melody_and_chords, melody_sequence, chord_names, note_duration)
        return player, thread
    else:
        player.play_melody_and_chords(melody_sequence, chord_names, note_duration)
        return player, None

if __name__ == "__main__":
    # Simple test
    print("🎵 Testing Audio Player...")
    
    # Test melody (C major scale)
    test_melody = np.array([
        [60, 1, 0.5, 0.0],  # C4
        [62, 1, 0.5, 0.5],  # D4
        [64, 1, 0.5, 1.0],  # E4
        [65, 1, 0.5, 1.5],  # F4
        [67, 1, 0.5, 2.0],  # G4
        [69, 1, 0.5, 2.5],  # A4
        [71, 1, 0.5, 3.0],  # B4
        [72, 1, 1.0, 3.5],  # C5
    ])
    
    # Test chords
    test_chords = ["C", "F", "G", "C"]
    
    player = AudioPlayer()
    
    print("\n1. Testing melody playback...")
    player.play_melody_simple(test_melody, note_duration=0.4)
    
    print("\n2. Testing chord playback...")
    player.play_chord_progression(test_chords, chord_duration=0.8)
    
    print("\n3. Testing melody with chords...")
    player.play_melody_and_chords(test_melody, test_chords, note_duration=0.4)
    
    print("\n✅ Audio player test complete!")
