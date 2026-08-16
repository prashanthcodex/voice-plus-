"""
Automated unit tests for VocalStrong AI.
Verifies audio processing (RMS, YIN algorithm),
clinical rule engine, and Sarvam AI client.
"""

import unittest
import numpy as np
import os
import tempfile

from audio_processor import analyze_vocal_acoustics
from clinical_engine import ClinicalVoiceEngine
from sarvam_client import SarvamSTTClient
from sample_generator import generate_synthetic_samples


class TestVocalStrongAI(unittest.TestCase):

    def setUp(self):
        self.sr = 16000
        self.duration = 4.0
        t = np.linspace(0, self.duration, int(self.sr * self.duration), endpoint=False)
        # Synthetic pure tone + harmonic at 150 Hz
        self.synth_audio = (
            0.1 * np.sin(2 * np.pi * 150.0 * t) +
            0.05 * np.sin(2 * np.pi * 300.0 * t)
        ).astype(np.float32)

    def test_audio_acoustics_analysis(self):
        """Test RMS decibels and YIN pitch extraction."""
        res = analyze_vocal_acoustics(self.synth_audio, sr=self.sr, target_db=70.0)
        
        self.assertIn("avg_db", res)
        self.assertIn("mean_f0", res)
        self.assertIn("target_compliance_pct", res)
        self.assertIn("f0_clean", res)
        
        # Fundamental frequency should be approximately 150 Hz
        self.assertTrue(140.0 <= res["mean_f0"] <= 160.0, f"Expected F0 ~150Hz, got {res['mean_f0']}")
        self.assertTrue(res["avg_db"] > 40.0, f"Expected reasonable dB SPL, got {res['avg_db']}")

    def test_hypophonia_clinical_rule(self):
        """Test hypophonia detection when volume is below target."""
        # Low volume signal
        low_synth = (0.005 * np.sin(2 * np.pi * 140.0 * np.linspace(0, 4, 64000))).astype(np.float32)
        acoustics = analyze_vocal_acoustics(low_synth, sr=self.sr, target_db=75.0)
        
        eval_res = ClinicalVoiceEngine.evaluate_session(
            acoustics=acoustics,
            task_type=ClinicalVoiceEngine.TASK_SUSTAINED_VOWEL,
            target_db=75.0
        )
        
        self.assertIn("Severe Hypophonia", eval_res["intensity_status"])
        self.assertIn("CRITICAL DEFICIT", eval_res["intensity_badge"])
        self.assertTrue(any("Hypophonia" in ins["title"] for ins in eval_res["insights"]))

    def test_monopitch_clinical_rule(self):
        """Test monopitch detection during reading task when pitch variance is restricted."""
        # Flat pitch signal
        flat_synth = (0.08 * np.sin(2 * np.pi * 130.0 * np.linspace(0, 4, 64000))).astype(np.float32)
        acoustics = analyze_vocal_acoustics(flat_synth, sr=self.sr, target_db=65.0)
        
        eval_res = ClinicalVoiceEngine.evaluate_session(
            acoustics=acoustics,
            task_type=ClinicalVoiceEngine.TASK_READING,
            target_db=65.0
        )
        
        # Should flag Monopitch
        monopitch_flagged = any("Monopitch" in ins["title"] for ins in eval_res["insights"])
        self.assertTrue(monopitch_flagged, "Expected monopitch warning for flat pitch audio")

    def test_sarvam_client_mock_and_config(self):
        """Test Sarvam client initializes properly and handles mock mode gracefully."""
        client_no_key = SarvamSTTClient(api_key="")
        self.assertFalse(client_no_key.is_configured)
        
        mock_res = client_no_key.transcribe_audio("nonexistent.wav", prompt="Hello test")
        self.assertTrue(mock_res["success"])
        self.assertEqual(mock_res["mode"], "demo_mock")

    def test_sample_generator(self):
        """Test synthetic benchmark audio file generator."""
        with tempfile.TemporaryDirectory() as tmpdir:
            benchmarks = generate_synthetic_samples(tmpdir, sample_rate=16000)
            self.assertEqual(len(benchmarks), 4)
            for name, path in benchmarks.items():
                self.assertTrue(os.path.exists(path))
                self.assertGreater(os.path.getsize(path), 1000)


if __name__ == "__main__":
    unittest.main()
