import pandas as pd
import patoolib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import parselmouth
from parselmouth.praat import call

def preprocess_audio(audio_path):
    # Load the audio into a Parselmouth Sound object
    sound = parselmouth.Sound(audio_path)

    # Extract the features
    pitch = call(sound, "To Pitch", 0.0, 75, 600)
    pointProcess = call(sound, "To PointProcess (periodic, cc)", 75, 600)
    
    jitter_local = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    jitter_local_absolute = call(pointProcess, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
    jitter_rap = call(pointProcess, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
    jitter_ppq5 = call(pointProcess, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
    jitter_ddp = call(pointProcess, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)
    shimmer_apq11 = call([sound, pointProcess], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    standard_deviation = np.std(sound.values)
    max_pitch = call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic")
    num_unvoiced_frames = call(pointProcess, "Get number of periods", 0, 0, 0.0001, 0.02, 1.3)
    total_frames = call(pointProcess, "Get number of points")
    fraction_locally_unvoiced_frames = num_unvoiced_frames / total_frames if total_frames else 0

    # Pack the features into a numpy array
    features = np.array([jitter_local, jitter_local_absolute, jitter_rap, jitter_ppq5, jitter_ddp, shimmer_apq11,
                         standard_deviation, max_pitch, fraction_locally_unvoiced_frames])

    # Expand dimensions to match the model's input shape
    features = np.expand_dims(features, axis=0)

    return features

def create_audio_model():
    audioModel = Sequential([
        Dense(16, activation='relu', input_shape=(9,)),
        Dense(32, activation='relu'),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(2, activation='softmax')
        ])

    audioModel.load_weights('PDaudio1Model_weights.h5')
    return audioModel


