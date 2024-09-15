import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.signal import hilbert
import time
import os
import pygame

# Load music file
y, sr = librosa.load('song1.mp3')

# Extract tempo, beats, and amplitude envelope
tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
amplitude_envelope = np.abs(hilbert(y))  # Amplitude envelope for energy-based transitions

# Get beat times
beat_times = librosa.frames_to_time(beats, sr=sr)

# Define more detailed ASCII art poses with better dance sequences
poses = [
    """
     O
    /|\\
    / \\
    """,  # Pose 1 - Standing with arms up
    """
     O
    /|
    / \\
    """,  # Pose 2 - Standing with one arm down
    """
    \\O/
     |
    / \\
    """,  # Pose 3 - Arms wide open
    """
     O
    /|\\
     |
    """,  # Pose 4 - Dancing with legs still
    """
    \\O/
     |
    /|\\
    """,  # Pose 5 - Arms in motion
    """
    \\O
     |
    / \\
    """,  # Pose 6 - Leaning back
    """
     O/
    /|
     \\
    """,  # Pose 7 - Leaning to one side
    """
     O
    /|\\
    /\\
    """,  # Pose 8 - Jumping
    """
     O
    /|
    / \\
    """,  # Pose 9 - Arms down
    """
    \\O/
     |
    /|
    """,  # Pose 10 - Ending pose
]

def get_pose(energy):
    """Select a pose based on the energy (amplitude) of the beat."""
    if energy < 0.1:
        return poses[0]  # Minimal motion
    elif 0.1 <= energy < 0.2:
        return poses[1]
    elif 0.2 <= energy < 0.3:
        return poses[2]
    elif 0.3 <= energy < 0.4:
        return poses[3]
    elif 0.4 <= energy < 0.5:
        return poses[4]
    elif 0.5 <= energy < 0.6:
        return poses[5]
    elif 0.6 <= energy < 0.7:
        return poses[6]
    elif 0.7 <= energy < 0.8:
        return poses[7]
    elif 0.8 <= energy < 0.9:
        return poses[8]
    else:
        return poses[9]  # Most energetic

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

# Initialize pygame mixer for audio playback
pygame.mixer.init()
pygame.mixer.music.load('song1.mp3')
pygame.mixer.music.play()

# Real-time processing loop with amplitude-driven transitions
for i, beat_time in enumerate(beat_times[:-1]):
    start = librosa.time_to_frames(beat_time, sr=sr)
    end = librosa.time_to_frames(beat_times[i + 1], sr=sr)
    
    # Get average amplitude between two beats to drive the pose
    energy = np.mean(amplitude_envelope[start:end])
    
    # Get the corresponding pose based on the energy level
    pose = get_pose(energy)
    
    clear_screen()
    print(pose)
    time.sleep(0.5)  # Sync with the beat intervals

pygame.mixer.music.stop()
