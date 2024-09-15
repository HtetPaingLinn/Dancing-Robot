import wave
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import time
import threading
from scipy.signal import butter, lfilter
from pydub import AudioSegment
import os
import subprocess  # Import subprocess module

def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except FileNotFoundError:
        return False

def update_plot(frames, sample_rate):
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots(figsize=(10, 4))

    chunk_size = int(sample_rate * 0.1)  # 100 ms chunks
    time_data = []
    amplitude_data = []

    ax.set_xlim(0, len(frames) / sample_rate)
    ax.set_ylim(-np.max(frames), np.max(frames))
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Waveform of Audio File")

    for start in range(0, len(frames), chunk_size):
        end = min(start + chunk_size, len(frames))
        time_data.extend(np.linspace(start / sample_rate, end / sample_rate, num=end - start))
        amplitude_data.extend(frames[start:end])

        ax.clear()  # Clear previous plot
        ax.plot(time_data, amplitude_data, color="#008080", lw=0.5)
        ax.set_xlim(0, len(frames) / sample_rate)
        ax.set_ylim(-np.max(frames), np.max(frames))
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Amplitude")
        ax.set_title("Waveform of Audio File")
        plt.pause(0.1)  # Pause to update plot

    plt.ioff()  # Turn off interactive mode
    plt.show()

def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, data)
    return y

def convert_mp3_to_wav(mp3_file_path):
    if not check_ffmpeg():
        raise RuntimeError("FFmpeg is not installed or not found in PATH")
    
    try:
        audio = AudioSegment.from_mp3(mp3_file_path)
        wav_file_path = mp3_file_path.replace('.mp3', '.wav')
        audio.export(wav_file_path, format='wav')
        return wav_file_path
    except Exception as e:
        print(f"Error converting MP3 to WAV: {e}")
        return None

def load_audio_file(file_path):
    if file_path.endswith('.mp3'):
        wav_file_path = convert_mp3_to_wav(file_path)
        if wav_file_path is None:
            raise ValueError("Failed to convert MP3 to WAV")
        file_path = wav_file_path
    
    try:
        wav = wave.open(file_path, 'r')
    except wave.Error as e:
        print(f"Error opening WAV file: {e}")
        raise

    raw = wav.readframes(-1)
    raw = np.frombuffer(raw, dtype=np.int16)
    sample_rate = wav.getframerate()

    if wav.getnchannels() == 2:
        raw = raw.reshape(-1, 2)
        raw = raw.mean(axis=1).astype(np.int16)

    wav.close()
    return raw, sample_rate

def play_audio(frames, sample_rate):
    sd.play(frames, sample_rate)
    sd.wait()

audio_file_path = './Believer.mp3'  # Example MP3 file

try:
    raw, sample_rate = load_audio_file(audio_file_path)
    
    audio_thread = threading.Thread(target=play_audio, args=(raw, sample_rate))
    audio_thread.start()
    
    update_plot(raw, sample_rate)
    
    audio_thread.join()

    if audio_file_path.endswith('.mp3'):
        wav_file_path = audio_file_path.replace('.mp3', '.wav')
        if os.path.exists(wav_file_path):
            os.remove(wav_file_path)

except Exception as e:
    print(f"An error occurred: {e}")
