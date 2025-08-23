import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import pyedflib
import mne
import os
from tensorflow.keras.models import load_model

# ---- MODEL FUNCTION ----
def classify_edf_file(edf_file_path: str):
    """
    Preprocesses data from an EDF file and classifies it using a pre-trained model.
    This function loads an EDF file, applies minmax scaling and reshaping to prepare the data
    for the model, loads a pre-trained Keras model, makes predictions on the preprocessed
    data, and returns a single binary classification result for the entire file.

    Args:
        edf_file_path: The path to the EDF file.
    Returns:
        A single integer (0 or 1) representing the final classification result for the file,
        or None if an error occurs during file loading, preprocessing, or model loading/prediction.
    """
    try:
        raw = mne.io.read_raw_edf(edf_file_path, preload=True)
        edf_data = raw.get_data()

        def minmax_scale_channel_safe(channel_data):
            if channel_data.ndim < 2:
                channel_data = np.expand_dims(channel_data, axis=0)
            min_vals = np.min(channel_data, axis=1, keepdims=True)
            max_vals = np.max(channel_data, axis=1, keepdims=True)
            scaled_channel = (channel_data - min_vals) / (max_vals - min_vals + 1e-8)
            return scaled_channel

        scaled_edf_data = minmax_scale_channel_safe(edf_data)
        num_channels, num_time_steps = scaled_edf_data.shape
        chunk_time_size = 2500
        channels_per_sample = 2
        overlap_time = 0
        data_chunks = []
        for i in range(0, num_channels - channels_per_sample + 1, channels_per_sample):
            for j in range(0, num_time_steps - chunk_time_size + 1, chunk_time_size - overlap_time):
                chunk = scaled_edf_data[i : i + channels_per_sample, j : j + chunk_time_size]
                chunk = np.expand_dims(chunk, axis=-1)
                if chunk.shape == (channels_per_sample, chunk_time_size, 1):
                    data_chunks.append(chunk)
        if data_chunks:
            preprocessed_edf_data = np.array(data_chunks)
        else:
            return None
        model_path = "RAW_MODEL.keras"
        if os.path.exists(model_path):
            loaded_model = load_model(model_path)
        else:
            return None
        predictions = None
        if loaded_model is not None and preprocessed_edf_data is not None:
            predictions = loaded_model.predict(preprocessed_edf_data)
        else:
            return None
        if predictions is not None:
            average_prediction = np.mean(predictions)
            final_classification = (average_prediction > 0.5).astype(int)
            return final_classification
        else:
            return None
    except Exception as e:
        print(f"An error occurred while processing the EDF file: {e}")
        return None

# ---- STREAMLIT UI ----
st.set_page_config(page_title="Schizophrenia Analyzer 🧠", page_icon="💡", layout="wide")
st.markdown(
"""
<style>
.stApp {
background: linear-gradient(-45deg, #1e3c72, #2a5298, #6a11cb, #2575fc, #ff6ec4, #7873f5);
background-size: 400% 400%;
animation: gradientBG 20s ease infinite;
font-family: 'Segoe UI', sans-serif;
color: #f8f9fa;
}
@keyframes gradientBG {
0% {background-position: 0% 50%;}
50% {background-position: 100% 50%;}
100% {background-position: 0% 50%;}
}
.glass-box {
background: rgba(255, 255, 255, 0.12);
border-radius: 20px;
padding: 25px;
margin: 15px 0;
backdrop-filter: blur(15px);
box-shadow: 0 8px 32px rgba(0,0,0,0.3);
transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.glass-box:hover {
transform: translateY(-3px);
box-shadow: 0 12px 40px rgba(0,0,0,0.5);
}
h1, h2, h3 {
text-shadow: 2px 2px 6px rgba(0,0,0,0.6);
font-weight: 700;
}
.stFileUploader > div {
background: rgba(255, 255, 255, 0.2);
border-radius: 15px;
padding: 12px;
border: 2px dashed rgba(255, 255, 255, 0.4);
}
div.stButton > button {
background: linear-gradient(135deg, #6a11cb, #2575fc);
color: white;
border-radius: 30px;
padding: 0.6em 2em;
font-weight: 600;
border: none;
box-shadow: 0px 4px 15px rgba(0,0,0,0.3);
transition: all 0.3s ease;
}
div.stButton > button:hover {
background: linear-gradient(135deg, #ff6ec4, #7873f5);
box-shadow: 0px 6px 20px rgba(0,0,0,0.5);
transform: scale(1.05);
}
</style>
""",
unsafe_allow_html=True,
)

st.title("🧠 Schizophrenia Disease Analyzer")
st.caption("Upload EEG/EDF files for analysis & learn about the disease")
uploaded_file = st.file_uploader("📂 Upload your `.edf` EEG file", type=["edf"])

def read_edf_to_df(tmp_path):
    f = pyedflib.EdfReader(tmp_path)
    n_signals = f.signals_in_file
    labels = f.getSignalLabels()
    first_signal = f.readSignal(0)
    sfreq = f.getSampleFrequency(0)
    time = np.arange(len(first_signal)) / sfreq
    df = pd.DataFrame({labels[0]: first_signal, "time_s": time})
    f.close()
    return df, {"n_signals": n_signals, "labels": labels, "sample_freq": sfreq}

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name
    df, meta = read_edf_to_df(tmp_path)

    st.markdown('<div class="glass-box">', unsafe_allow_html=True)
    st.success("✅ File uploaded successfully!")
    st.write("**Number of signals:**", meta["n_signals"])
    st.write("**Signal labels:**", meta["labels"])
    st.write("**Sample frequency (Hz):**", meta["sample_freq"])
    st.markdown('</div>', unsafe_allow_html=True)

    st.subheader("📈 First Signal Preview")
    st.line_chart(df.set_index("time_s")[[meta["labels"][0]]], use_container_width=True)

    # ---- ML PREDICTION ----
    st.subheader("🔮 Schizophrenia Risk Prediction")
    st.markdown('<div class="glass-box">', unsafe_allow_html=True)
    with st.spinner("Analyzing with pre-trained ML model..."):
        prediction = classify_edf_file(tmp_path)
    if prediction is not None:
        if prediction == 1:
            st.error("🚨 High risk of Schizophrenia detected by the model.")
        else:
            st.success("🟢 The model did NOT detect high risk of Schizophrenia.")
        st.write(f"Model Output: **{prediction}**")
    else:
        st.warning("⚠️ Model could not process this file. Please try another one or contact support.")
    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("👆 Please upload an EDF file to start analysis")

# ---------- DISEASE INFO ----------
st.header("📖 About Schizophrenia")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(
    """
    <div class="glass-box">
    <h3>⚠️ Symptoms</h3>
    <ul>
    <li>Hallucinations (hearing voices, seeing things)</li>
    <li>Delusions (false beliefs)</li>
    <li>Disorganized thinking & speech</li>
    <li>Social withdrawal</li>
    <li>Difficulty concentrating</li>
    </ul>
    </div>
    """,
    unsafe_allow_html=True,
)
with col2:
    st.markdown(
    """
    <div class="glass-box">
    <h3>🧬 Causes</h3>
    <ul>
    <li>Genetic predisposition</li>
    <li>Neurotransmitter imbalance (dopamine, glutamate)</li>
    <li>Brain structure abnormalities</li>
    <li>Environmental stress & trauma</li>
    <li>Prenatal complications</li>
    </ul>
    </div>
    """,
    unsafe_allow_html=True,
)
with col3:
    st.markdown(
    """
    <div class="glass-box">
    <h3>💊 Treatment</h3>
    <ul>
    <li>Antipsychotic medications</li>
    <li>Psychotherapy (CBT, supportive therapy)</li>
    <li>Social & vocational rehabilitation</li>
    <li>Family support & education</li>
    </ul>
    </div>
    """,
    unsafe_allow_html=True,
)
