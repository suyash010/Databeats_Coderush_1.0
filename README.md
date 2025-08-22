# Databeats_Coderush_1.0
Coderush topic :-MD2
Project listings
Schizophrenia Detection using EEG and Deep Learning

Problem Statement
Schizophrenia is a chronic neuropsychiatric disorder affecting how a person thinks, feels, and behaves. Traditional diagnosis methods heavily rely on behavioral assessments and clinical interviews, often leading to delayed or inaccurate diagnosis. There is a pressing need for objective, quantifiable, and early-detection methods.
Electroencephalogram (EEG) signals can serve as reliable biomarkers for early identification of schizophrenia-related neurophysiological patterns. However, existing deep learning models leveraging EEG signals remain underdeveloped and underutilized in clinical environments.

Project Overview
This project presents an end-to-end deep learning pipeline for automated schizophrenia detection using EEG signals. It involves:
Data preprocessing and cleaning of raw EEG signals
Feature extraction using combined Wavelet and Fourier techniques
A CNN-based deep learning model trained on REDOD EEG dataset
Classification into 67 labels grouped into 17 unique classes
Deployment of the model via Streamlit on HuggingFace
Result storage using phpMyAdmin with XAMPP
The system predicts whether a subject shows neurophysiological traits associated with schizophrenia, aiming to aid clinicians with early diagnosis.

How is this Different from Existing Solutions?
Most models use handcrafted features or conventional ML algorithms; we use deep CNNs for robust pattern recognition.
Integration of both Wavelet and Fourier feature extraction captures both time-frequency and frequency domain representations.
Model is trained on real-world EEG datasets (REDOD), converted into CSV for streamlined processing.
The model uses naïve reweighting techniques to address class imbalance bias during training.
Deployed as a fully functional frontend using Streamlit, accessible via HuggingFace.

Unique Selling Proposition (USP)
End-to-end automated pipeline from raw EEG signal to diagnosis
Hybrid feature extraction for better accuracy
Open access web app deployment (Streamlit + HuggingFace)
Built-in class bias handling using naive reweighting
Scalable architecture with easy integration to hospital EMRs

Features
Upload EEG CSV data for inference
Predicts disease classification (67 labels → 17 classes)
Displays model confidence score
User-friendly GUI built in Streamlit
Admin panel for storing user results in MySQL database
Hosted publicly via HuggingFace Spaces

Dataset Details
Source: REDOD (Reliable EEG-based Dataset for Online Detection)
Format: EEG signals converted into structured CSV files
Labels: 67 different diagnostic identifiers
Unique Diagnostic Classes: 17 classes (e.g., Schizophrenia, Bipolar, Normal, etc.)
Sampling Rate: Varies across subjects; preprocessed for uniformity

Preprocessing
Noise removal using bandpass filtering
Normalization across channels
Windowing for signal segmentation
Feature extraction using:
Discrete Wavelet Transform (DWT)
Fast Fourier Transform (FFT)

Model Architecture
CNN with multiple Conv1D + MaxPooling layers
Dropout layers to prevent overfitting
Dense output layer with Softmax for multi-class classification
Loss function: Weighted Cross Entropy
Optimization: Adam Optimizer

Bias Handling:
Class imbalance tackled via Naive Reweighting: inverse frequency weighting of classes

Architecture Diagram
User Input (CSV)
↓
Signal Cleaning (Bandpass + Normalization)
↓
Feature Extraction (Wavelet + Fourier)
↓
CNN Model (67 labels → 17 class groups)
↓
Prediction Output + Confidence
↓
Frontend (Streamlit on HuggingFace)
↓
Database Storage (phpMyAdmin on XAMPP)

Technologies Used
Layer	Technology
Programming	Python
Deep Learning	TensorFlow / Keras
Frontend	Streamlit
Hosting	HuggingFace Spaces
Database	MySQL via phpMyAdmin
Platform	XAMPP (Localhost)
Visualization	Matplotlib, Seaborn

Screenshots / Wireframes
Interface	Description
EEG Upload Panel	Upload your EEG CSV files
Prediction Output	Shows predicted class & accuracy
Admin Dashboard	View stored results in database


Expected Outcomes
Average Accuracy: ~85–90% on validation set
Real-time predictions with EEG input
Web-based interface accessible to clinicians
Early diagnosis potential via EEG biomarkers
Structured database storage for long-term usage

Future Work
Add Transformer-based models for temporal EEG analysis
Incorporate additional clinical datasets (e.g., TUH EEG Corpus)
Add more granular diagnostic subclasses
API development for hospital EMR integration
Enable mobile interface for real-time screening



