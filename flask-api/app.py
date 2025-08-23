import streamlit as st
import numpy as np
import pandas as pd
import tempfile
import pyedflib
from tensorflow.keras.models import load_model

# ----------------- PAGE SETUP -----------------
st.set_page_config(page_title="Schizophrenia Analyzer 🧠", layout="wide")

st.title("🧠 Schizophrenia Disease Analyzer")
st.caption("Upload EEG/EDF files to analyze risk")

# ----------------- LOAD MODEL -----------------
@st.cache_resource
def load_keras_model():
    model = load_model("schizophrenia_model.keras")
    return model

model = load_keras_model()

# ----------------- READ EDF -----------------
def read_edf_file(tmp_path):
    f = pyedflib.EdfReader(tmp_path)
    signal = f.readSignal(0)  # just first signal
    sfreq = f.getSampleFrequency(0)
    time = np.arange(len(signal)) / sfreq
    df = pd.DataFrame({f.getLabel(0): signal, "time_s": time})
    f.close()
    return df

# ----------------- FEATURE EXTRACTOR -----------------
def extract_features(df):
    signal = df.iloc[:, 0].values
    features = [
        np.mean(signal),
        np.std(signal),
        np.min(signal),
        np.max(signal),
        np.percentile(signal, 25),
        np.percentile(signal, 75),
        np.median(signal),
        np.var(signal),
        np.ptp(signal),
        np.sum(np.square(signal)) / len(signal)
    ]
    return np.array(features).reshape(1, -1)

# ----------------- MAIN UI -----------------
uploaded_file = st.file_uploader("📂 Upload your `.edf` EEG file", type=["edf"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    df = read_edf_file(tmp_path)
    st.success("✅ File uploaded and signal extracted")
    st.line_chart(df.set_index("time_s"))

    features = extract_features(df)

    prediction = model.predict(features)
    predicted_class = np.argmax(prediction, axis=1)[0]

    st.subheader("🧠 Prediction")
    st.markdown(f"### 🧾 Predicted Class: `{predicted_class}`")
    st.markdown(f"### 🔢 Probabilities: `{prediction.tolist()[0]}`")
else:
    st.info("👆 Upload a .edf file to begin analysis")
