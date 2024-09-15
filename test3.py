import os
import warnings
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.image import resize

# Suppress TensorFlow and Python warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

# Loading the trained model
model = tf.keras.models.load_model("./Trained_model.h5")

# List of genre classes
classes = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

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

# File path to the test music file
file_path = "./song.mp3"

# Process the test file and make predictions
X_test = load_and_preprocess_data(file_path)
c_index = model_prediction(X_test)

# Output the predicted genre
print(classes[c_index])
