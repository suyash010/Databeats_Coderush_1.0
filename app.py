import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import pyedflib

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Schizophrenia Analyzer 🧠", page_icon="💡", layout="wide")

# ---------- CUSTOM STYLING ----------
st.markdown(
    """
    <style>
    /* Animated background */
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

    /* Glass card */
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

    /* Headings */
    h1, h2, h3 {
        text-shadow: 2px 2px 6px rgba(0,0,0,0.6);
        font-weight: 700;
    }

    /* File uploader */
    .stFileUploader > div {
        background: rgba(255, 255, 255, 0.2);
        border-radius: 15px;
        padding: 12px;
        border: 2px dashed rgba(255, 255, 255, 0.4);
    }

    /* Buttons */
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

# ---------- TITLE ----------
st.title("🧠 Schizophrenia Disease Analyzer")
st.caption("Upload EEG/EDF files for analysis & learn about the disease")

# ---------- FILE UPLOAD ----------
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

else:
    st.info("👆 Please upload an EDF file to start analysis")

# ---------- ANALYSIS PLACEHOLDER ----------
st.subheader("🔮 Schizophrenia Risk Prediction")
st.markdown(
    '<div class="glass-box">🧪 This is where ML-based schizophrenia detection results will appear (e.g., risk score, abnormal EEG patterns, etc.).</div>',
    unsafe_allow_html=True,
)

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

st.caption("⚕️ Note: This page is for educational purposes. Always consult a medical professional for diagnosis and treatment.")
