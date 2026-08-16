"""
VocalStrong AI: Parkinson's Disease Speech Therapy Prototype.
Clinical Audio Engineering Console with Center Teleprompter Stage & Adaptive Decibel Targets.
"""

import os
import io
import time
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from audio_processor import (
    record_live_audio,
    load_audio_file,
    analyze_vocal_acoustics,
    is_mic_available
)
from sarvam_client import SarvamSTTClient
from clinical_engine import ClinicalVoiceEngine
from sample_generator import generate_synthetic_samples
from styles import CUSTOM_CSS

# Page Configuration
st.set_page_config(
    page_title="VocalStrong AI - Clinical Voice Biofeedback",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject Custom Dark CSS Theme
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Curated Clinical Exercise Library with Tailored Decibel Baselines and Durations
CLINICAL_PROTOCOLS = {
    "1. Sustained Vowel Phonation (/ah/)": {
        "category": "Maximal Phonation Effort",
        "default_db": 76.0,
        "default_duration": 6,
        "engine_task": ClinicalVoiceEngine.TASK_SUSTAINED_VOWEL,
        "text": "Take a deep breath and sustain a loud, steady 'AAAAAHHHHH' sound from your belly until the timer ends.",
        "tips": "Inhale deeply through your nose • Push sound using your abdominal wall • Keep vocal pitch steady"
    },
    "2. Daily Functional Short Phrase (LSVT)": {
        "category": "Functional Speech Projection",
        "default_db": 74.0,
        "default_duration": 8,
        "engine_task": ClinicalVoiceEngine.TASK_READING,
        "text": "Good morning! Please pass the water bottle on the dining table.",
        "tips": "Think LOUD • Project voice to the back of the room • Punch the final word"
    },
    "3. Standard Diagnostic Sentence": {
        "category": "Connected Speech Articulation",
        "default_db": 72.0,
        "default_duration": 10,
        "engine_task": ClinicalVoiceEngine.TASK_READING,
        "text": "The quick brown fox jumps over the lazy dog near the flowing river bank.",
        "tips": "Maintain volume across middle syllables • Exaggerate consonant articulation (P, T, K)"
    },
    "4. Grandfather Diagnostic Passage (Paragraph)": {
        "category": "Paragraph Speech Endurance",
        "default_db": 70.0,
        "default_duration": 22,
        "engine_task": ClinicalVoiceEngine.TASK_PARAGRAPH,
        "text": "You wish to know all about my grandfather. Well, he is nearly ninety-three years old; he dresses himself in an ancient black frock coat, usually several buttons missing. A long beard clings to his chin, giving those who observe him a pronounced feeling of the utmost respect.",
        "tips": "Take quick replenishment breaths at semicolons • Do not let voice fade at sentence ends • Over-articulate"
    },
    "5. The Rainbow Passage (Paragraph)": {
        "category": "Paragraph Prosody & Endurance",
        "default_db": 70.0,
        "default_duration": 20,
        "engine_task": ClinicalVoiceEngine.TASK_PARAGRAPH,
        "text": "When the sunlight strikes raindrops in the air, they act as a prism and form a rainbow. The rainbow is a division of white light into many beautiful colors. These take the shape of a long round arch, with its path high above.",
        "tips": "Vary pitch on key words to combat monopitch • Maintain steady breath pressure throughout"
    },
    "6. Pitch Glide & Frequency Range (Siren)": {
        "category": "Laryngeal Flexibility & Range",
        "default_db": 72.0,
        "default_duration": 8,
        "engine_task": ClinicalVoiceEngine.TASK_PITCH_GLIDE,
        "text": "Start at your lowest comfortable pitch and glide your voice smoothly upward like a siren to your highest falsetto note.",
        "tips": "Stretch vocal cords smoothly • Avoid squeezing your throat • Keep sound flowing continuously"
    },
    "7. Custom Patient Line / Paragraph": {
        "category": "Personalized Practice",
        "default_db": 72.0,
        "default_duration": 12,
        "engine_task": ClinicalVoiceEngine.TASK_PARAGRAPH,
        "text": "Hello, my name is practicing vocal loudness and breath support today.",
        "tips": "Focus on consistent diaphragmatic support • Speak with intent and energy"
    }
}

# Initialize Session State
if "history" not in st.session_state:
    st.session_state.history = []

if "benchmarks" not in st.session_state:
    temp_benchmark_dir = os.path.join(os.path.expanduser("~"), ".vocalstrong_temp", "benchmarks")
    try:
        st.session_state.benchmarks = generate_synthetic_samples(temp_benchmark_dir, sample_rate=16000)
    except Exception as e:
        st.session_state.benchmarks = {}

if "current_audio_data" not in st.session_state:
    st.session_state.current_audio_data = None
if "current_audio_path" not in st.session_state:
    st.session_state.current_audio_path = None
if "current_analysis" not in st.session_state:
    st.session_state.current_analysis = None
if "current_transcript" not in st.session_state:
    st.session_state.current_transcript = None
if "is_recording" not in st.session_state:
    st.session_state.is_recording = False


# =============================================================================
# SIDEBAR CONFIGURATION
# =============================================================================
with st.sidebar:
    st.markdown("## 🎛️ Biofeedback Controls")
    
    # 1. Protocol Selection
    selected_protocol_key = st.selectbox(
        "📋 Select Speech Protocol",
        options=list(CLINICAL_PROTOCOLS.keys()),
        index=2,
        help="Select a calibrated exercise. The target intensity and capture duration will automatically adapt."
    )
    
    protocol_meta = CLINICAL_PROTOCOLS[selected_protocol_key]
    
    # Allow custom text edit if Custom selected
    if "Custom" in selected_protocol_key:
        active_reading_text = st.text_area(
            "✍️ Custom Line / Paragraph to Read:",
            value=protocol_meta["text"],
            height=110
        )
    else:
        active_reading_text = protocol_meta["text"]
    
    st.markdown("---")

    # 2. Dynamic Target Intensity Slider (Pre-calibrated to task)
    target_intensity = st.slider(
        "🎯 Target Intensity (dB SPL)",
        min_value=50.0,
        max_value=90.0,
        value=float(protocol_meta["default_db"]),
        step=1.0,
        help=f"Target loudness calibrated for {protocol_meta['category']} (Suggested: {protocol_meta['default_db']:.0f} dB)."
    )
    
    # 3. Dynamic Capture Window Slider (Pre-calibrated to task)
    rec_duration = st.slider(
        "⏱️ Capture Window (Seconds)",
        min_value=3,
        max_value=35,
        value=int(protocol_meta["default_duration"]),
        step=1,
        help=f"Recording duration configured for this task length (Suggested: {protocol_meta['default_duration']}s)."
    )
    
    # Quick duration toggles
    col_pre1, col_pre2, col_pre3, col_pre4 = st.columns(4)
    with col_pre1:
        if st.button("6s", use_container_width=True):
            rec_duration = 6
    with col_pre2:
        if st.button("10s", use_container_width=True):
            rec_duration = 10
    with col_pre3:
        if st.button("18s", use_container_width=True):
            rec_duration = 18
    with col_pre4:
        if st.button("25s", use_container_width=True):
            rec_duration = 25

    st.markdown("---")

    # 4. Sarvam AI API Key
    sarvam_api_key = st.text_input(
        "🔑 Sarvam AI API Key",
        type="password",
        placeholder="Paste subscription key...",
        help="Activates live cloud speech-to-text with the 'saaras:v4' model for high-accuracy transcription with Indian accents & background noise."
    )
    
    st.markdown("---")
    
    # 5. Audio Input Channel
    input_mode = st.radio(
        "🎙️ Audio Input Channel",
        options=["Live Microphone (sounddevice)", "Clinical Benchmark Presets", "Upload Audio File"],
        index=0
    )
    
    # Hardware Status
    mic_ok, mic_status = is_mic_available()
    if mic_ok:
        device_name = mic_status.split(':')[-1].strip() if ':' in mic_status else mic_status
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 8px; font-size: 0.8rem; color: #10B981; background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.25); border-radius: 8px; padding: 0.5rem 0.7rem; margin-top: 0.5rem;">
            <span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #10B981;"></span>
            <span><b>Mic Connected:</b> {device_name[:24]}</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 8px; font-size: 0.8rem; color: #F59E0B; background: rgba(245, 158, 11, 0.1); border: 1px solid rgba(245, 158, 11, 0.25); border-radius: 8px; padding: 0.5rem 0.7rem; margin-top: 0.5rem;">
            <span>⚠️ {mic_status}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-size: 0.76rem; color: #64748B; line-height: 1.5;'>
        <b>VocalStrong AI v1.2</b><br>
        Acoustics: <code>librosa (RMS & YIN)</code><br>
        ASR: <code>Sarvam AI (saaras:v4)</code>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# MAIN HEADER CONSOLE
# =============================================================================
st.markdown("""
<div class="console-header">
    <div class="console-tag">
        <span style="display:inline-block; width:6px; height:6px; border-radius:50%; background:#00D4B2;"></span>
        LSVT LOUD & Acoustic Biofeedback Workstation
    </div>
    <div class="console-title">🎙️ VocalStrong AI</div>
    <div class="console-subtitle">Targeted Speech Therapy & Acoustic Vocal Intensity Teleprompter for Parkinson's Disease.</div>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# CENTER STAGE: CLINICAL READING TELEPROMPTER
# =============================================================================
word_count = len(active_reading_text.split())
est_reading_sec = max(3, int(word_count * 0.45))

st.markdown(f"""
<div class="teleprompter-card">
    <div class="teleprompter-topbar">
        <div class="teleprompter-title">
            <span>📖</span> <b>Speech Teleprompter Stage:</b> {selected_protocol_key}
        </div>
        <div style="display: flex; gap: 10px; align-items: center;">
            <span style="font-size: 0.78rem; color: #94A3B8; font-family: 'JetBrains Mono', monospace;">
                {word_count} words (~{est_reading_sec}s reading)
            </span>
            <div class="target-badge-pill">
                🎯 Target: {target_intensity:.0f} dB SPL
            </div>
        </div>
    </div>
    <div style="font-size: 0.78rem; font-weight: 700; color: #00D4B2; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.2rem;">
        Text To Speak / Read Aloud:
    </div>
    <div class="prompter-text-body">
        "{active_reading_text}"
    </div>
    <div class="prompter-footer-tips">
        <span>💡 <b>Clinical Guidance:</b> {protocol_meta['tips']}</span>
        <span style="font-family: 'JetBrains Mono', monospace; color: #38BDF8;">Window: {rec_duration}s</span>
    </div>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# AUDIO RECORDING / INGESTION CONTROL BAR
# =============================================================================
col_rec_action, col_rec_info = st.columns([1.3, 0.7])

with col_rec_action:
    if input_mode == "Live Microphone (sounddevice)":
        btn_record = st.button(
            f"🔴 Start {rec_duration}-Second Vocal Recording",
            use_container_width=True,
            type="primary"
        )
        
        if btn_record:
            if not mic_ok:
                st.error(f"Microphone unavailable ({mic_status}). Select 'Clinical Benchmark Presets' in the sidebar to run testing.")
            else:
                # Live Recording with Teleprompter Active Feedback
                live_prompt_holder = st.empty()
                live_progress = st.progress(0, text=f"🎙️ RECORDING LIVE... Speak the text above now!")
                
                # Dynamic visual countdown
                start_time = time.time()
                audio_arr, wav_path, err = record_live_audio(duration=float(rec_duration), sample_rate=16000)
                
                live_progress.progress(100, text="✅ Recording complete! Running acoustic analysis...")
                time.sleep(0.3)
                live_progress.empty()
                
                if err:
                    st.error(err)
                else:
                    st.session_state.current_audio_data = audio_arr
                    st.session_state.current_audio_path = wav_path
                    st.success(f"✅ {rec_duration}-Second vocal recording analyzed successfully!")

    elif input_mode == "Clinical Benchmark Presets":
        preset_names = list(st.session_state.benchmarks.keys())
        if preset_names:
            col_bench1, col_bench2 = st.columns([2, 1])
            with col_bench1:
                selected_preset = st.selectbox("Standard Clinical Benchmarks:", preset_names, label_visibility="collapsed")
            with col_bench2:
                if st.button("⚡ Analyze Benchmark", use_container_width=True, type="primary"):
                    preset_path = st.session_state.benchmarks[selected_preset]
                    y, sr, temp_path = load_audio_file(preset_path, sample_rate=16000)
                    st.session_state.current_audio_data = y
                    st.session_state.current_audio_path = temp_path
                    st.success(f"Benchmark loaded: {selected_preset}")

    elif input_mode == "Upload Audio File":
        uploaded_file = st.file_uploader("Upload Patient Recording (WAV/MP3/M4A):", type=["wav", "mp3", "m4a", "ogg"], label_visibility="collapsed")
        if uploaded_file is not None:
            if st.button("📁 Process Uploaded File", use_container_width=True, type="primary"):
                y, sr, temp_path = load_audio_file(uploaded_file, sample_rate=16000)
                st.session_state.current_audio_data = y
                st.session_state.current_audio_path = temp_path
                st.success(f"Audio file loaded ({len(y)/sr:.1f}s)")

with col_rec_info:
    st.markdown(f"""
    <div style="background: #0B1222; border: 1px solid #1E2E4A; border-radius: 12px; padding: 0.85rem 1.1rem; height: 100%; display: flex; flex-direction: column; justify-content: center;">
        <div style="font-size: 0.75rem; color: #94A3B8; text-transform: uppercase; font-weight: 700;">Active Audio Settings</div>
        <div style="font-size: 0.95rem; font-weight: 600; color: #F8FAFC; margin-top: 0.2rem;">
            16,000 Hz Mono | <span style="color: #00D4B2;">{target_intensity:.0f} dB Target</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# PIPELINE EXECUTION & ACOUSTIC TELEMETRY HUD
# =============================================================================
if st.session_state.current_audio_data is not None and st.session_state.current_audio_path is not None:
    audio_arr = st.session_state.current_audio_data
    wav_path = st.session_state.current_audio_path
    sample_duration = float(len(audio_arr)) / 16000.0

    # Audio Playback Bar
    st.markdown("---")
    col_play1, col_play2 = st.columns([1.5, 2.5])
    with col_play1:
        st.markdown(f"**🎧 Session Playback ({sample_duration:.1f}s):**")
        st.audio(wav_path, format="audio/wav")
    with col_play2:
        st.markdown(f"""
        <div style="padding-top: 10px; font-size: 0.88rem; color: #94A3B8;">
            <b>Sampling Rate:</b> 16,000 Hz | <b>Format:</b> 16-Bit Linear PCM WAV | <b>Channel:</b> Mono
        </div>
        """, unsafe_allow_html=True)

    # Compute Acoustics (librosa RMS + YIN)
    with st.spinner("Processing acoustic spectrogram, RMS intensity & YIN fundamental pitch..."):
        acoustics = analyze_vocal_acoustics(
            y=audio_arr,
            sr=16000,
            target_db=target_intensity,
            frame_length=2048,
            hop_length=512,
            fmin=65.0,
            fmax=500.0
        )
        st.session_state.current_analysis = acoustics

    # Compute Sarvam AI Speech-to-Text (saaras:v4)
    with st.spinner("Transcribing speech with Sarvam AI saaras:v4 (Indian accent & noise resilience)..."):
        sarvam_client = SarvamSTTClient(api_key=sarvam_api_key)
        transcript_res = sarvam_client.transcribe_audio(
            audio_path=wav_path,
            language_code="unknown",
            prompt=active_reading_text
        )
        st.session_state.current_transcript = transcript_res

    # Rule-Based Clinical Insights Evaluation
    clinical_eval = ClinicalVoiceEngine.evaluate_session(
        acoustics=acoustics,
        task_type=protocol_meta["engine_task"],
        transcript_data=transcript_res,
        target_db=target_intensity
    )

    # Save to history
    st.session_state.history.append({
        "timestamp": time.strftime("%H:%M:%S"),
        "protocol": selected_protocol_key,
        "avg_db": clinical_eval["metrics"]["avg_db"],
        "target_db": target_intensity,
        "duration": round(sample_duration, 1),
        "score": clinical_eval["overall_score"],
        "grade": clinical_eval["grade"]
    })

    # =========================================================================
    # TELEMETRY HUD CARDS
    # =========================================================================
    st.markdown("### 📊 Acoustic Telemetry & Parkinsonian Biomarkers")

    avg_db_val = clinical_eval["metrics"]["avg_db"]
    delta_db = clinical_eval["db_delta"]
    delta_color = "#10B981" if delta_db >= 0 else ("#F59E0B" if delta_db >= -5.0 else "#F43F5E")
    delta_sign = "+" if delta_db >= 0 else ""

    comp_pct = clinical_eval["metrics"]["compliance_pct"]
    comp_color = "#10B981" if comp_pct >= 65 else ("#F59E0B" if comp_pct >= 35 else "#F43F5E")

    mean_f0 = clinical_eval["metrics"]["mean_f0"]
    std_f0 = clinical_eval["metrics"]["std_f0"]
    f0_range = clinical_eval["metrics"]["f0_range"]

    score_val = clinical_eval["overall_score"]
    score_color = clinical_eval["grade_color"]

    st.markdown(f"""
    <div class="hud-grid">
        <div class="hud-card">
            <div class="hud-label">Average Intensity (RMS)</div>
            <div class="hud-value" style="color: {clinical_eval['intensity_color']};">{avg_db_val:.1f}<span class="hud-unit">dB</span></div>
            <div class="hud-footer" style="color: {delta_color};">
                <b>{delta_sign}{delta_db:.1f} dB</b> vs {target_intensity:.0f} dB target
            </div>
        </div>
        <div class="hud-card">
            <div class="hud-label">Target Loudness Compliance</div>
            <div class="hud-value" style="color: {comp_color};">{comp_pct:.0f}<span class="hud-unit">%</span></div>
            <div class="hud-footer">Frames ≥ {target_intensity:.0f} dB baseline</div>
        </div>
        <div class="hud-card">
            <div class="hud-label">Fundamental Pitch (YIN F0)</div>
            <div class="hud-value" style="color: #38BDF8;">{mean_f0:.1f}<span class="hud-unit">Hz</span></div>
            <div class="hud-footer">SD: ±{std_f0:.1f} Hz | Range: {f0_range:.0f} Hz</div>
        </div>
        <div class="hud-card">
            <div class="hud-label">LSVT Effort Index ({sample_duration:.1f}s)</div>
            <div class="hud-value" style="color: {score_color};">{score_val:.0f}<span class="hud-unit">/100</span></div>
            <div class="hud-footer" style="color: {clinical_eval['intensity_color']};"><b>{clinical_eval['intensity_badge']}</b></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


    # =========================================================================
    # INTERACTIVE DARK PLOTLY GRAPHS
    # =========================================================================
    st.markdown("### 📈 Interactive Biofeedback Waveforms")

    times = acoustics["times"]
    db_series = acoustics["db_series"]
    f0_clean = acoustics["f0_clean"]
    max_time = max(sample_duration, float(np.max(times)))

    # Graph 1: Volume (dB) vs Target Intensity Baseline
    fig_volume = go.Figure()

    # Target intensity baseline
    fig_volume.add_hline(
        y=target_intensity,
        line_dash="dash",
        line_color="#F43F5E",
        line_width=2.5,
        annotation_text=f"🎯 Calibrated Target ({target_intensity:.0f} dB)",
        annotation_position="top right",
        annotation_font=dict(size=12, color="#F43F5E", family="Space Grotesk")
    )

    # Shaded Target Zone
    fig_volume.add_hrect(
        y0=target_intensity,
        y1=105.0,
        fillcolor="rgba(0, 212, 178, 0.08)",
        line_width=0,
        annotation_text="Optimal Effort Zone",
        annotation_position="bottom right",
        annotation_font=dict(size=10, color="#00D4B2")
    )

    # Shaded Hypophonia Deficit Zone
    fig_volume.add_hrect(
        y0=25.0,
        y1=target_intensity,
        fillcolor="rgba(244, 63, 94, 0.06)",
        line_width=0,
        annotation_text="Hypophonia Deficit Zone",
        annotation_position="top left",
        annotation_font=dict(size=10, color="#F43F5E")
    )

    # Actual User dB Trajectory
    fig_volume.add_trace(go.Scatter(
        x=times,
        y=db_series,
        mode="lines",
        name="Vocal Intensity (dB SPL)",
        line=dict(color="#00D4B2", width=3.5, shape="spline"),
        hovertemplate="<b>Time:</b> %{x:.2f}s<br><b>Intensity:</b> %{y:.1f} dB SPL<extra></extra>"
    ))

    # Mean Intensity Line
    fig_volume.add_hline(
        y=acoustics["avg_db"],
        line_dash="dot",
        line_color="#38BDF8",
        line_width=1.5,
        annotation_text=f"Average: {acoustics['avg_db']:.1f} dB",
        annotation_position="bottom left",
        annotation_font=dict(size=11, color="#38BDF8")
    )

    fig_volume.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0F172A",
        plot_bgcolor="#0B1222",
        title=dict(
            text=f"<b>Acoustic Intensity Trajectory (RMS dB) vs. Calibrated Target ({target_intensity:.0f} dB)</b>",
            font=dict(size=15, color="#F8FAFC", family="Space Grotesk")
        ),
        xaxis=dict(
            title="Time (seconds)",
            gridcolor="#1E2E4A",
            zeroline=False,
            range=[0, max_time]
        ),
        yaxis=dict(
            title="Intensity (dB SPL)",
            gridcolor="#1E2E4A",
            zeroline=False,
            range=[max(25.0, float(np.min(db_series)) - 5.0), max(95.0, float(np.max(db_series)) + 8.0)]
        ),
        margin=dict(l=55, r=40, t=55, b=45),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=380
    )

    st.plotly_chart(fig_volume, use_container_width=True)

    # Graph 2: Fundamental Pitch (F0) Tracking via YIN Algorithm
    fig_pitch = go.Figure()

    fig_pitch.add_trace(go.Scatter(
        x=times,
        y=f0_clean,
        mode="lines+markers",
        name="Fundamental Frequency (F0 - YIN)",
        line=dict(color="#38BDF8", width=3, shape="spline"),
        marker=dict(size=4.0, color="#0284C7", symbol="circle"),
        hovertemplate="<b>Time:</b> %{x:.2f}s<br><b>Pitch:</b> %{y:.1f} Hz<extra></extra>"
    ))

    if acoustics["mean_f0"] > 0:
        fig_pitch.add_hline(
            y=acoustics["mean_f0"],
            line_dash="dot",
            line_color="#818CF8",
            line_width=1.5,
            annotation_text=f"Mean Pitch: {acoustics['mean_f0']:.1f} Hz (±{acoustics['std_f0']:.1f} Hz)",
            annotation_position="top left",
            annotation_font=dict(size=11, color="#818CF8")
        )

    fig_pitch.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0F172A",
        plot_bgcolor="#0B1222",
        title=dict(
            text=f"<b>Fundamental Pitch Frequency (F0) Tracking via YIN Algorithm</b>",
            font=dict(size=15, color="#F8FAFC", family="Space Grotesk")
        ),
        xaxis=dict(
            title="Time (seconds)",
            gridcolor="#1E2E4A",
            zeroline=False,
            range=[0, max_time]
        ),
        yaxis=dict(
            title="Pitch Frequency (Hz)",
            gridcolor="#1E2E4A",
            zeroline=False,
            range=[50.0, max(350.0, float(np.nanmax(f0_clean) if np.any(~np.isnan(f0_clean)) else 300.0) + 30.0)]
        ),
        margin=dict(l=55, r=40, t=55, b=45),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=340
    )

    st.plotly_chart(fig_pitch, use_container_width=True)


    # =========================================================================
    # SARVAM AI TRANSCRIPTION & SPEECH ACCENT ANALYSIS
    # =========================================================================
    st.markdown("### 🇮🇳 Sarvam AI Speech-to-Text (`saaras:v4`)")

    transcript_data = st.session_state.current_transcript
    t_text = transcript_data.get("transcript", "")
    t_lang = transcript_data.get("language_code", "en-IN")
    t_mode = transcript_data.get("mode", "live")
    t_note = transcript_data.get("note", "")

    col_stt1, col_stt2 = st.columns([1.8, 1.2])

    with col_stt1:
        st.markdown(f"""
        <div class="sarvam-box">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                <span style="font-size: 0.8rem; font-weight: 700; color: #00D4B2; text-transform: uppercase; letter-spacing: 0.06em;">
                    Recognized Speech Output (saaras:v4)
                </span>
                <span style="background: #111C33; border: 1px solid #1E2E4A; color: #38BDF8; font-size: 0.74rem; font-weight: 600; padding: 0.2rem 0.6rem; border-radius: 6px;">
                    Lang: {t_lang} | {t_mode}
                </span>
            </div>
            <div class="transcript-quote">
                "{t_text if t_text else 'No intelligible speech detected'}"
            </div>
            {f"<div style='font-size: 0.8rem; color: #94A3B8; margin-top: 0.4rem;'>ℹ️ {t_note}</div>" if t_note else ""}
        </div>
        """, unsafe_allow_html=True)

    with col_stt2:
        words = t_text.split() if t_text else []
        wpm_est = (len(words) / max(sample_duration, 0.5)) * 60.0 if len(words) > 0 else 0.0
        st.markdown(f"""
        <div class="sarvam-box">
            <div style="font-size: 0.8rem; font-weight: 700; color: #64748B; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.6rem;">
                Acoustic & Accent Analytics
            </div>
            <div style="font-size: 0.9rem; color: #CBD5E1; line-height: 1.9;">
                • <b>Duration Analyzed:</b> {sample_duration:.1f} seconds<br>
                • <b>Indian Accent Model:</b> Active (saaras:v4)<br>
                • <b>Background Noise Denoising:</b> Active<br>
                • <b>Recognized Words:</b> {len(words)} words<br>
                • <b>Speech Pacing:</b> {wpm_est:.0f} WPM<br>
                • <b>Phonation Continuity:</b> {clinical_eval['metrics']['voicing_pct']:.0f}%
            </div>
        </div>
        """, unsafe_allow_html=True)


    # =========================================================================
    # STRICT RULE-BASED CLINICAL RECOMMENDATIONS
    # =========================================================================
    st.markdown("### 🩺 Rule-Based Clinical Directives (Parkinson's Speech Pathology)")

    for ins in clinical_eval["insights"]:
        status_raw = ins.get("status", "info")
        card_class = "critical" if status_raw == "warning" else ("warning" if status_raw == "caution" else ("optimal" if status_raw == "success" else "info"))
        directives_html = "".join([f"<li>{d}</li>" for d in ins.get("directives", [])])
        
        st.markdown(f"""
        <div class="clinical-report-card {card_class}">
            <div class="report-category">{ins.get('category', 'Acoustic Diagnosis')}</div>
            <div class="report-title">{ins.get('title', 'Clinical Finding')}</div>
            <div class="report-finding"><b>Observation:</b> {ins.get('finding', '')}</div>
            <div class="report-rationale"><b>Pathological Rationale:</b> {ins.get('rationale', '')}</div>
            <div class="directives-container">
                <div class="directives-heading">Actionable Speech Therapy Directives:</div>
                <ul class="directives-list">
                    {directives_html}
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)


    # =========================================================================
    # SESSION SUMMARY & CLINICAL DATA EXPORT
    # =========================================================================
    st.markdown("---")
    st.markdown("### 💾 Clinical Session Export & Audit Log")
    
    col_exp1, col_exp2 = st.columns(2)
    
    session_dict = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "protocol_task": selected_protocol_key,
        "sample_duration_seconds": round(sample_duration, 2),
        "target_intensity_db": target_intensity,
        "average_intensity_db": clinical_eval["metrics"]["avg_db"],
        "peak_intensity_db": clinical_eval["metrics"]["peak_db"],
        "compliance_pct": clinical_eval["metrics"]["compliance_pct"],
        "mean_f0_hz": clinical_eval["metrics"]["mean_f0"],
        "pitch_std_hz": clinical_eval["metrics"]["std_f0"],
        "pitch_range_hz": clinical_eval["metrics"]["f0_range"],
        "vocal_decay_slope_db_sec": clinical_eval["metrics"]["decay_slope"],
        "jitter_cv_pct": clinical_eval["metrics"]["jitter_cv"],
        "overall_effort_score": clinical_eval["overall_score"],
        "clinical_grade": clinical_eval["grade"],
        "target_text": active_reading_text,
        "transcript_text": t_text
    }
    
    with col_exp1:
        st.download_button(
            label="📥 Download Clinical Session Report (JSON)",
            data=json.dumps(session_dict, indent=2),
            file_name=f"vocalstrong_session_{int(time.time())}.json",
            mime="application/json",
            use_container_width=True
        )

    with col_exp2:
        df_export = pd.DataFrame([session_dict])
        st.download_button(
            label="📊 Download Acoustic Metrics (CSV)",
            data=df_export.to_csv(index=False).encode('utf-8'),
            file_name=f"vocalstrong_metrics_{int(time.time())}.csv",
            mime="text/csv",
            use_container_width=True
        )

else:
    # Standby state with quick prompt advice
    st.markdown(f"""
    <div style="background: #0F172A; border: 1px dashed #1E2E4A; border-radius: 14px; padding: 2.0rem; text-align: center; margin-top: 1rem;">
        <div style="font-size: 2.2rem; margin-bottom: 0.5rem;">🎙️</div>
        <div style="font-family: 'Space Grotesk', sans-serif; font-size: 1.25rem; font-weight: 700; color: #F8FAFC; margin-bottom: 0.3rem;">
            Ready for Vocal Biofeedback
        </div>
        <div style="font-size: 0.92rem; color: #94A3B8; max-width: 580px; margin: 0 auto 1.2rem auto; line-height: 1.6;">
            Look at the <b>Speech Teleprompter</b> in the center above, take a deep breath, and click <b>🔴 Start {rec_duration}-Second Vocal Recording</b> to speak and receive real-time volume and pitch feedback.
        </div>
    </div>
    """, unsafe_allow_html=True)
