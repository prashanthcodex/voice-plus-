# 🎙️ VocalStrong AI: Parkinson's Speech Therapy Biofeedback

An advanced, full-stack clinical speech therapy prototype engineered for Parkinson's disease acoustic rehabilitation and vocal effort biofeedback. Built using **Streamlit**, **librosa**, **sounddevice**, **Plotly**, and **Sarvam AI (`saaras:v4`)**.

---

## 🌟 Key Features

1. **Live 4-Second Raw Audio Capture (`sounddevice`)**:
   - Captures 4.0 seconds of single-channel audio directly from the patient's microphone at **16,000 Hz**.
   - Auto-detects input hardware with diagnostic fallback.
   - Also supports **Clinical Benchmark Presets** (Hypophonic Parkinsonian voice, Monopitch speech, Pitch glide) and **Audio File Upload** (WAV/MP3/M4A).

2. **Local Acoustic Voice Analysis (`librosa`)**:
   - **Root Mean Square (RMS) Decibel Tracking**: Frame-wise RMS computation converted to calibrated dB SPL with target compliance percentage and end-of-phrase loudness decay slope.
   - **Fundamental Pitch ($F_0$) via YIN Algorithm**: High-accuracy autocorrelation pitch tracking (`librosa.yin`) capturing Mean $F_0$, Pitch standard deviation ($\sigma_{F_0}$ in Hz and semitones), Dynamic Pitch Range, and Tremor/Jitter coefficient of variation.

3. **Sarvam AI (`saaras:v4`) Speech-to-Text**:
   - Integrated with the **Sarvam AI Python SDK** and REST API using the state-of-the-art `saaras:v4` model.
   - Specially optimized for Indian accents, regional phonetics, and noisy clinical/home environments.
   - Calculates words per minute (WPM), speech festination detection, and articulation metrics.

4. **Interactive Plotly Biofeedback Dashboards**:
   - **Dynamic Volume Curve vs. Target Baseline Slider**: Real-time dB trajectory with color-coded target effort zones and highlighted hypophonia deficit regions.
   - **Fundamental Pitch ($F_0$) Tracking**: YIN pitch trajectory with voiced stability envelope.

5. **Strict Rule-Based Clinical Insights (LSVT LOUD Principles)**:
   - **Severe/Mild Hypophonia Detection**: Automatic deficit calculation with actionable *"Think LOUD!"* and diaphragmatic breathing coaching.
   - **Monopitch / Reduced Prosody Warnings**: Laryngeal rigidity detection with pitch glide and inflection directives.
   - **End-of-Utterance Vocal Fade Alerts**: Trailing volume drop-off warnings to prevent sentence-end inaudibility.
   - **Vocal Tremor / Perturbation Diagnostics**: Sustained vowel stability monitoring.

6. **Clinical Session Reporting & Data Export**:
   - Instant export of clinical findings and acoustic biomarkers in **JSON** and **CSV** formats.

---

## 🚀 Quick Start Guide

### 1. Activate Environment & Run Application
```bash
# Navigate to project directory
cd C:\Users\prash\.gemini\antigravity\scratch\vocalstrong_ai

# Run with virtual environment
.venv\Scripts\streamlit.exe run app.py
```

### 2. Configure Sidebar
- **Target Intensity (dB)**: Adjust the slider (default: `70 dB`).
- **Sarvam AI API Key**: Paste your subscription key into the sidebar text field to activate live cloud transcription using `saaras:v4`.
- **Select Task**: Choose between *Sustained Vowel Phonation (/ah/)*, *Sentence Reading & Articulation*, or *Pitch Glide & Dynamic Range*.

### 3. Record & Analyze
- Click **🔴 Start 4-Second Recording** (or load a benchmark preset).
- Inspect the interactive Plotly graphs, acoustic metrics, Sarvam transcript, and personalized clinical directives.

---

## 📁 Directory Structure
```
vocalstrong_ai/
├── .venv/                   # Isolated Python 3.12 virtual environment
├── app.py                   # Main Streamlit biofeedback application
├── audio_processor.py       # sounddevice 4s capture & librosa RMS/YIN analysis
├── clinical_engine.py       # Rule-based Parkinson's speech pathology insights
├── sarvam_client.py         # Sarvam AI saaras:v4 STT client (SDK + REST)
├── sample_generator.py      # Standardized synthetic clinical benchmark generator
├── styles.py                # Modern accessible medical UI styling & CSS
├── requirements.txt         # Pinned production dependencies
├── tests/
│   └── test_vocalstrong.py  # Automated unit test suite (RMS, YIN, Rules, Sarvam)
└── README.md                # Documentation and user guide
```
