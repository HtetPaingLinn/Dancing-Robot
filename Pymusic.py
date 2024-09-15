from pydub import AudioSegment
import numpy as np
import matplotlib.pyplot as plt

# Function to convert MP3 to an array of samples
def load_audio(mp3_file):
    # Load audio file
    audio = AudioSegment.from_mp3(mp3_file)
    
    # Convert to raw audio data
    samples = np.array(audio.get_array_of_samples())
    
    # If stereo, take only one channel
    if audio.channels == 2:
        samples = samples[::2]
    
    return samples, audio.frame_rate

# Function to plot the waveform
def plot_waveform(samples, frame_rate):
    # Generate time axis in seconds
    time_axis = np.linspace(0, len(samples) / frame_rate, num=len(samples))
    
    # Plot the waveform
    plt.figure(figsize=(10, 4))
    plt.plot(time_axis, samples, color='blue')
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Waveform")
    plt.grid()
    plt.show()

def main(mp3_file):
    samples, frame_rate = load_audio(mp3_file)
    plot_waveform(samples, frame_rate)

if __name__ == "__main__":
    mp3_file = "./Believer.mp3"  # Replace with your MP3 file path
    main(mp3_file)
