import os
import warnings
import numpy as np
import tensorflow as tf
import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import librosa
from tensorflow.image import resize
import matplotlib.pyplot as plt
import io
from PIL import Image
from pydub import AudioSegment
import requests

# Suppress TensorFlow and Python warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

# Load the trained model
model = tf.keras.models.load_model("./Trained_model.h5")

# List of genre classes
classes = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

# Spotify API setup
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id="fac2ebdc6da1419ca21579c37951f5ea", client_secret="6738671cfa6245f98156c4a56ceb38ca"))

# Function to load and preprocess audio data
def load_and_preprocess_data(file_path, target_shape=(150, 150)):
    data = []
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    
    chunk_duration = 4  # seconds
    overlap_duration = 2  # seconds
    chunk_samples = chunk_duration * sample_rate
    overlap_samples = overlap_duration * sample_rate
    num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1

    for i in range(num_chunks):
        start = i * (chunk_samples - overlap_samples)
        end = start + chunk_samples
        chunk = audio_data[start:end]
        mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
        mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
        data.append(mel_spectrogram)

    return np.array(data)

# Function to predict the genre
def model_prediction(X_test):
    y_pred = model.predict(X_test)
    predicted_categories = np.argmax(y_pred, axis=1)
    unique_elements, counts = np.unique(predicted_categories, return_counts=True)
    max_elements = unique_elements[counts == np.max(counts)]
    return max_elements[0]

# Function to recommend similar music based on audio features
def recommend_similar_music_by_features(audio_features):
    results = sp.recommendations(seed_genres=['pop'], limit=5, 
                                 target_energy=audio_features['energy'], 
                                 target_tempo=audio_features['tempo'])
    tracks = results['tracks']
    recommendations = [{'name': track['name'], 'artist': track['artists'][0]['name'], 
                        'url': track['external_urls']['spotify']} for track in tracks]
    return recommendations

# Function to analyze the uploaded music file for tempo and energy level
def analyze_music(audio_data, sample_rate):
    tempo, _ = librosa.beat.beat_track(y=audio_data, sr=sample_rate)
    energy = np.mean(librosa.feature.rms(y=audio_data))
    return {'tempo': tempo, 'energy': energy}

# Function to convert MP3 to WAV if needed
def convert_mp3_to_wav(mp3_file_path):
    audio = AudioSegment.from_mp3(mp3_file_path)
    wav_file_path = mp3_file_path.replace('.mp3', '.wav')
    audio.export(wav_file_path, format='wav')
    return wav_file_path

# Function to load and plot waveform
def plot_waveform(audio_data, sample_rate):
    plt.figure(figsize=(10, 4))
    plt.plot(np.linspace(0, len(audio_data) / sample_rate, num=len(audio_data)), audio_data, color="#008080")
    plt.title('Waveform of Audio File')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.xlim(0, len(audio_data) / sample_rate)
    plt.ylim(-np.max(np.abs(audio_data)), np.max(np.abs(audio_data)))
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf


# Streamlit UI
st.title('Music Genre Classification, Visualization, and Song Recommendation')

# File uploader
uploaded_file = st.file_uploader("Upload a music file", type=["mp3", "wav", "ogg"])

if uploaded_file is not None:
    # Save uploaded file to a temporary location
    temp_file_path = "./temp_music_file"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Convert MP3 to WAV if necessary
    if temp_file_path.endswith('.mp3'):
        temp_file_path = convert_mp3_to_wav(temp_file_path)
    
    # Load and preprocess the audio file
    audio_data, sample_rate = librosa.load(temp_file_path, sr=None)
    X_test = load_and_preprocess_data(temp_file_path)
    
    # Predict genre
    c_index = model_prediction(X_test)
    st.success(f"Predicted Music Genre: {classes[c_index]}")

    # Display waveform
    waveform_img = plot_waveform(audio_data, sample_rate)
    st.image(waveform_img, caption='Waveform of Audio File', use_column_width=True)

    # Advanced music analysis: tempo and energy level
    audio_features = analyze_music(audio_data, sample_rate)
    st.write(f"Tempo: {audio_features['tempo']} BPM")
    st.write(f"Energy Level: {audio_features['energy']}")

    # Recommend similar music based on audio features
    st.subheader("Recommended Music Based on Audio Features")
    recommendations = recommend_similar_music_by_features(audio_features)
    
    if recommendations:
        for rec in recommendations:
            st.write(f"{rec['name']} by {rec['artist']} - [Listen on Spotify]({rec['url']})")


    
    # Optional: Play the uploaded file
    st.audio(uploaded_file)

else:
    st.warning("Please upload a music file to predict its genre and get recommendations.")
