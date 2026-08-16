"""
Audio Processing Module for VocalStrong AI.
Handles sounddevice live 4s capture at 16000Hz,
librosa RMS-based decibel (dB) intensity tracking,
and YIN algorithm fundamental pitch (F0) analysis.
"""

import os
import io
import time
import logging
from typing import Dict, Any, Tuple, Optional

import numpy as np
import soundfile as sf
import librosa

logger = logging.getLogger(__name__)

# Try importing sounddevice with safety check
SOUNDDEVICE_AVAILABLE = False
try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except Exception as e:
    logger.warning(f"Sounddevice not available or driver missing: {e}")
    SOUNDDEVICE_AVAILABLE = False


# Acoustic calibration constants for SPL estimation
# Standard digital RMS to conversational SPL mapping offset
SPL_CALIBRATION_OFFSET = 96.0  # Maps full-scale 0 dBFS sine to 96 dB SPL
MIN_DB_FLOOR = 25.0           # Ambient noise floor baseline


def is_mic_available() -> Tuple[bool, str]:
    """Check if microphone input hardware is accessible."""
    if not SOUNDDEVICE_AVAILABLE:
        return False, "sounddevice library could not initialize audio backend."
    try:
        devices = sd.query_devices()
        input_devices = [d for d in devices if d.get('max_input_channels', 0) > 0]
        if not input_devices:
            return False, "No audio input recording devices found."
        default_in = sd.query_devices(kind='input')
        return True, f"Found input device: {default_in.get('name', 'Default Microphone')}"
    except Exception as e:
        return False, f"Audio hardware query error: {str(e)}"


def record_live_audio(
    duration: float = 4.0,
    sample_rate: int = 16000,
    output_path: Optional[str] = None
) -> Tuple[Optional[np.ndarray], Optional[str], Optional[str]]:
    """
    Capture a raw audio sample from the microphone using sounddevice.
    
    Args:
        duration: Capture duration in seconds (default 4.0s).
        sample_rate: Sampling frequency in Hz (default 16000Hz).
        output_path: Optional path to save WAV file.
        
    Returns:
        (audio_array, wav_path, error_message)
    """
    available, msg = is_mic_available()
    if not available:
        return None, None, f"Microphone unavailable: {msg}"

    try:
        num_frames = int(duration * sample_rate)
        logger.info(f"Starting live audio capture: {duration}s at {sample_rate}Hz...")
        
        # Record 1-channel 16kHz float32 audio
        audio_data = sd.rec(
            frames=num_frames,
            samplerate=sample_rate,
            channels=1,
            dtype='float32'
        )
        sd.wait()  # Wait for recording completion
        
        # Flatten to 1D array
        audio_1d = audio_data.flatten()

        # Generate temporary WAV file if path not specified
        if output_path is None:
            temp_dir = os.path.join(os.path.expanduser("~"), ".vocalstrong_temp")
            os.makedirs(temp_dir, exist_ok=True)
            output_path = os.path.join(temp_dir, f"recording_{int(time.time()*1000)}.wav")

        sf.write(output_path, audio_1d, sample_rate, subtype='PCM_16')
        return audio_1d, output_path, None

    except Exception as e:
        logger.exception("Error during live sounddevice capture")
        return None, None, f"Audio capture failed: {str(e)}"


def load_audio_file(file_path_or_bytes, sample_rate: int = 16000) -> Tuple[np.ndarray, int, str]:
    """
    Load an audio file into 16kHz mono float32 numpy array.
    """
    y, sr = librosa.load(file_path_or_bytes, sr=sample_rate, mono=True)
    
    # Save a temporary WAV copy for Sarvam STT if given bytes
    temp_dir = os.path.join(os.path.expanduser("~"), ".vocalstrong_temp")
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, f"session_{int(time.time()*1000)}.wav")
    sf.write(temp_path, y, sr, subtype='PCM_16')
    
    return y, sr, temp_path


def analyze_vocal_acoustics(
    y: np.ndarray,
    sr: int = 16000,
    target_db: float = 70.0,
    frame_length: int = 2048,
    hop_length: int = 512,
    fmin: float = 65.0,
    fmax: float = 500.0
) -> Dict[str, Any]:
    """
    Perform local acoustic voice analysis using librosa.
    Computes:
    - Root Mean Square (RMS) frame energy & calibrated decibels (dB)
    - YIN algorithm fundamental pitch frequency (F0) tracking
    - Parkinsonian acoustic biomarkers (monopitch index, loudness decay, voicing ratio)
    
    Args:
        y: Audio signal array (1D float).
        sr: Sample rate (Hz).
        target_db: User's target intensity threshold (dB).
        frame_length: Window size for STFT / RMS.
        hop_length: Hop length for frame analysis.
        fmin: Minimum fundamental frequency for YIN (Hz).
        fmax: Maximum fundamental frequency for YIN (Hz).
        
    Returns:
        Comprehensive dictionary of acoustic metrics and time-series arrays.
    """
    if len(y) == 0:
        raise ValueError("Empty audio signal received for acoustic analysis.")

    # 1. Duration and Time grid
    duration = float(len(y)) / sr
    times = librosa.times_like(librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0], sr=sr, hop_length=hop_length)

    # 2. RMS Energy & Decibels (dB)
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    
    # Calibrate digital amplitude to approximate SPL decibels
    # dB = 20 * log10(rms + eps) + offset
    eps = 1e-6
    db_series = 20.0 * np.log10(np.maximum(rms, eps)) + SPL_CALIBRATION_OFFSET
    db_series = np.clip(db_series, MIN_DB_FLOOR, 105.0)

    # Voice Activity / Silence Threshold
    silence_db_threshold = max(MIN_DB_FLOOR + 10.0, float(np.percentile(db_series, 20)))
    voiced_mask = db_series > silence_db_threshold

    avg_db = float(np.mean(db_series[voiced_mask])) if np.any(voiced_mask) else float(np.mean(db_series))
    peak_db = float(np.max(db_series))
    min_db = float(np.min(db_series))
    db_std = float(np.std(db_series[voiced_mask])) if np.any(voiced_mask) else float(np.std(db_series))
    
    # Percentage of voiced frames meeting or exceeding target intensity
    if np.any(voiced_mask):
        target_compliance_pct = float(np.mean(db_series[voiced_mask] >= target_db) * 100.0)
    else:
        target_compliance_pct = float(np.mean(db_series >= target_db) * 100.0)

    # Loudness decay slope (measuring trailing vocal fade / hypophonia fatigue)
    if len(db_series) > 5 and np.any(voiced_mask):
        voiced_idx = np.where(voiced_mask)[0]
        if len(voiced_idx) > 3:
            slope, _ = np.polyfit(times[voiced_idx], db_series[voiced_idx], 1)
            loudness_decay_slope = float(slope)  # dB per second
        else:
            loudness_decay_slope = 0.0
    else:
        loudness_decay_slope = 0.0

    # 3. Fundamental Pitch (F0) via YIN algorithm
    try:
        f0_raw = librosa.yin(
            y=y,
            fmin=fmin,
            fmax=fmax,
            sr=sr,
            frame_length=frame_length,
            hop_length=hop_length
        )
    except Exception as e:
        logger.warning(f"librosa.yin computation fallback: {e}")
        # pyin fallback if yin encountered an edge-case
        f0_raw, _, _ = librosa.pyin(y=y, fmin=fmin, fmax=fmax, sr=sr, frame_length=frame_length, hop_length=hop_length)
        f0_raw = np.nan_to_num(f0_raw, nan=0.0)

    # Align array lengths
    min_len = min(len(times), len(db_series), len(f0_raw))
    times = times[:min_len]
    db_series = db_series[:min_len]
    f0_raw = f0_raw[:min_len]
    voiced_mask = voiced_mask[:min_len]

    # Clean pitch curve: Mask out unvoiced/silent frames or boundary outliers
    f0_clean = np.copy(f0_raw)
    pitch_valid_mask = voiced_mask & (f0_clean >= (fmin + 2)) & (f0_clean <= (fmax - 5))
    f0_clean[~pitch_valid_mask] = np.nan

    valid_pitches = f0_clean[pitch_valid_mask]
    
    if len(valid_pitches) > 0:
        mean_f0 = float(np.mean(valid_pitches))
        median_f0 = float(np.median(valid_pitches))
        std_f0 = float(np.std(valid_pitches))
        min_f0 = float(np.min(valid_pitches))
        max_f0 = float(np.max(valid_pitches))
        f0_range = max_f0 - min_f0

        # Pitch variation in semitones (ST) = 12 * log2(F0 / mean_F0)
        semitones = 12.0 * np.log2(valid_pitches / np.maximum(mean_f0, 1.0))
        semitone_std = float(np.std(semitones))

        # Jitter / Coefficient of Variation (% CV)
        jitter_cv = float((std_f0 / mean_f0) * 100.0) if mean_f0 > 0 else 0.0
    else:
        mean_f0 = 0.0
        median_f0 = 0.0
        std_f0 = 0.0
        min_f0 = 0.0
        max_f0 = 0.0
        f0_range = 0.0
        semitone_std = 0.0
        jitter_cv = 0.0

    # Voicing percentage (% of total duration actively voiced)
    voicing_pct = float(np.mean(pitch_valid_mask) * 100.0) if len(pitch_valid_mask) > 0 else 0.0

    return {
        "duration": duration,
        "sample_rate": sr,
        "times": times,
        "db_series": db_series,
        "avg_db": avg_db,
        "peak_db": peak_db,
        "min_db": min_db,
        "db_std": db_std,
        "target_db": target_db,
        "target_compliance_pct": target_compliance_pct,
        "loudness_decay_slope": loudness_decay_slope,
        "f0_raw": f0_raw,
        "f0_clean": f0_clean,
        "pitch_valid_mask": pitch_valid_mask,
        "mean_f0": mean_f0,
        "median_f0": median_f0,
        "std_f0": std_f0,
        "min_f0": min_f0,
        "max_f0": max_f0,
        "f0_range": f0_range,
        "semitone_std": semitone_std,
        "jitter_cv": jitter_cv,
        "voicing_pct": voicing_pct
    }
