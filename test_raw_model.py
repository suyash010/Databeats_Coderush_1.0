# --- Cell 1: Imports ---
import pickle
import numpy as np
import tensorflow as tf
import mne
import os

# --- Cell 2: Helper Functions ---
def minmax_scale_channel(channel_data):
    channel_reshaped = np.reshape(channel_data, (2, -1))
    min_vals = np.min(channel_reshaped, axis=1, keepdims=True)
    max_vals = np.max(channel_reshaped, axis=1, keepdims=True)
    scaled_channel = (channel_reshaped - min_vals) / (max_vals - min_vals)
    scaled_channel = np.reshape(scaled_channel, channel_data.shape)
    return scaled_channel

def select_channel(raw, selected_channel=['Fp1', 'Fp2']):
    chan_indices = [raw.ch_names.index(channel) for channel in selected_channel]
    return raw.get_data(picks=chan_indices)

def division(new_raw, samp_freq=250, recording_time=10):
    trail_length = samp_freq * recording_time
    trail_length_shape = np.shape(new_raw)[1]
    total_chunks = trail_length_shape // trail_length
    new_raw = new_raw[:, 0:total_chunks * trail_length]
    chunks_raw = np.split(new_raw, total_chunks, axis=1)
    return chunks_raw

def preprocess_edf(edf_path):
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=0)
    raw.pick_types(eeg=True)
    raw.filter(l_freq=0.5, h_freq=50, method='iir', iir_params={'order': 2, 'ftype': 'butter'})
    new_raw = select_channel(raw)
    chunks = division(new_raw)
    X = np.array([minmax_scale_channel(chunk) for chunk in chunks])
    X = np.reshape(X, (*X.shape, 1))
    return X

# --- Cell 3: User Input ---
model_path = input("Enter path to your trained model (.keras): ").strip()
edf_path = input("Enter path to your EEG .edf file for testing: ").strip()

# --- Cell 4: Load Model and Preprocess Data ---
model = tf.keras.models.load_model(model_path)
X_test = preprocess_edf(edf_path)

# --- Cell 5: Predict and Display Results ---
preds = model.predict(X_test)
pred_classes = np.round(preds.flatten()).astype(int)
print("Predictions for each segment (0=Healthy, 1=Schizophrenia):")
print(pred_classes)
healthy_count = np.sum(pred_classes == 0)
schizo_count = np.sum(pred_classes == 1)
print(f"\nHealthy segments: {healthy_count}")
print(f"Schizophrenia segments: {schizo_count}")
if healthy_count > schizo_count:
    print("\nFinal classification: HEALTHY")
elif schizo_count > healthy_count:
    print("\nFinal classification: SCHIZOPHRENIA")
else:
    print("\nFinal classification: INCONCLUSIVE (equal segments)")