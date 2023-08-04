import os
import numpy as np
import scipy.io.wavfile
#from IPython.display import Audio
import tensorflow as tf
from sklearn import metrics
import librosa, librosa.display
import pandas as pd
import parselmouth
import pickle

def preprocess_ad_audio(path):
    #(sample_rate, signal) = scipy.io.wavfile.read(path)

    sound = parselmouth.Sound(path)
    pp = parselmouth.praat.call(sound, "To PointProcess (periodic, cc)...", 1, 200)
    #jitter = parselmouth.praat.call(pp, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)
    #shimmer = parselmouth.praat.call([sound, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.2, 1.3, 1.6)
    num_unvoiced_frames = parselmouth.praat.call(pp, "Get number of periods", 0, 0, 0.0001, 0.02, 1.3)
    total_frames = parselmouth.praat.call(pp, "Get number of points")
    fraction_locally_unvoiced_frames = num_unvoiced_frames / total_frames if total_frames else 0

    X_features = np.array([fraction_locally_unvoiced_frames)
    
    return X_features


def ad_audio_model():
    with open("ADmodel2.pkl", "rb") as file:
        model = pickle.load(file) 
    return model
