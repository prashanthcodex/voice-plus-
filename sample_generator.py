"""
Clinical Audio Benchmark Generator for VocalStrong AI.
Generates realistic 4-second 16000Hz speech synthesis benchmarks:
1. Hypophonic Parkinsonian Voice (Low dB, restricted pitch, vocal fade)
2. Target Loud / Projected Voice (Healthy volume, strong breath support)
3. Monopitch Connected Speech (Flat prosody, minimal F0 variance)
4. Pitch Glide & Siren (Wide dynamic vocal range)
"""

import os
import numpy as np
import soundfile as sf
from typing import Dict, Tuple


def generate_synthetic_samples(output_dir: str, sample_rate: int = 16000) -> Dict[str, str]:
    """
    Generate standard 4.0s audio benchmark WAV files for therapy testing.
    """
    os.makedirs(output_dir, exist_ok=True)
    duration = 4.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    samples = {}

    # 1. Hypophonic Parkinsonian Phonation (Low amplitude ~52 dB, slight tremor, end fade)
    f0_base = 135.0  # Hz
    tremor_freq = 5.5  # 5.5 Hz vocal tremor
    f0_mod = f0_base + 3.0 * np.sin(2 * np.pi * tremor_freq * t)
    phase = 2 * np.pi * np.cumsum(f0_mod) / sample_rate
    
    # Harmonically rich glottal wave
    glottal_hypo = (
        0.5 * np.sin(phase) +
        0.3 * np.sin(2 * phase) +
        0.15 * np.sin(3 * phase) +
        0.05 * np.sin(4 * phase)
    )
    # Formant filter envelope simulation
    formant_mod = (1 + 0.3 * np.sin(2 * np.pi * 500 * t / sample_rate))
    # Low volume envelope with end decay
    env_hypo = (0.015 * (1.0 - 0.4 * (t / duration))) * (0.8 + 0.2 * np.sin(2 * np.pi * tremor_freq * t))
    audio_hypo = glottal_hypo * formant_mod * env_hypo + 0.001 * np.random.normal(0, 1, len(t))
    audio_hypo = np.clip(audio_hypo, -1.0, 1.0).astype(np.float32)
    
    path_hypo = os.path.join(output_dir, "benchmark_hypophonic_parkinsons.wav")
    sf.write(path_hypo, audio_hypo, sample_rate, subtype='PCM_16')
    samples["Hypophonic Voice (Parkinson's ~52 dB)"] = path_hypo

    # 2. Target Loud / Healthy Projected Phonation (~76 dB, robust breath support)
    f0_healthy = 150.0
    phase_healthy = 2 * np.pi * f0_healthy * t
    glottal_healthy = (
        0.6 * np.sin(phase_healthy) +
        0.4 * np.sin(2 * phase_healthy) +
        0.25 * np.sin(3 * phase_healthy) +
        0.15 * np.sin(4 * phase_healthy)
    )
    # Strong sustained envelope
    attack = np.minimum(t / 0.2, 1.0)
    env_healthy = 0.12 * attack  # High amplitude
    audio_healthy = glottal_healthy * env_healthy + 0.0005 * np.random.normal(0, 1, len(t))
    audio_healthy = np.clip(audio_healthy, -1.0, 1.0).astype(np.float32)

    path_healthy = os.path.join(output_dir, "benchmark_healthy_projected.wav")
    sf.write(path_healthy, audio_healthy, sample_rate, subtype='PCM_16')
    samples["Projected Voice (Target Loud ~76 dB)"] = path_healthy

    # 3. Monopitch Speech Simulation (Flat pitch ~125 Hz, moderate dB ~65 dB)
    f0_mono = 125.0 + 1.2 * np.sin(2 * np.pi * 0.8 * t)  # Very narrow variation
    phase_mono = 2 * np.pi * np.cumsum(f0_mono) / sample_rate
    glottal_mono = (
        0.5 * np.sin(phase_mono) +
        0.35 * np.sin(2 * phase_mono) +
        0.2 * np.sin(3 * phase_mono)
    )
    # Syllable bursts
    syllable_env = 0.04 * (0.6 + 0.4 * np.sin(2 * np.pi * 3.5 * t)**2)
    audio_mono = glottal_mono * syllable_env + 0.0008 * np.random.normal(0, 1, len(t))
    audio_mono = np.clip(audio_mono, -1.0, 1.0).astype(np.float32)

    path_mono = os.path.join(output_dir, "benchmark_monopitch_reading.wav")
    sf.write(path_mono, audio_mono, sample_rate, subtype='PCM_16')
    samples["Monopitch Reading Sample (Flat F0 ~125 Hz)"] = path_mono

    # 4. Pitch Glide / Dynamic Flexibility (Sweep 110 Hz -> 320 Hz)
    f0_sweep = 110.0 + (320.0 - 110.0) * (t / duration)**1.5
    phase_sweep = 2 * np.pi * np.cumsum(f0_sweep) / sample_rate
    glottal_glide = 0.5 * np.sin(phase_sweep) + 0.3 * np.sin(2 * phase_sweep)
    env_glide = 0.06 * np.sin(np.pi * t / duration)
    audio_glide = glottal_glide * env_glide + 0.0005 * np.random.normal(0, 1, len(t))
    audio_glide = np.clip(audio_glide, -1.0, 1.0).astype(np.float32)

    path_glide = os.path.join(output_dir, "benchmark_pitch_glide.wav")
    sf.write(path_glide, audio_glide, sample_rate, subtype='PCM_16')
    samples["Pitch Glide Exercise (110 Hz -> 320 Hz)"] = path_glide

    return samples
