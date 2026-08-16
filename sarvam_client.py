"""
Sarvam AI Speech-to-Text Integration Module.
Utilizes the 'saaras:v4' model for high-accuracy transcription
optimized for Indian accents and noisy acoustic environments.
"""

import os
import io
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Attempt to import official sarvamai SDK if installed
SARVAM_SDK_AVAILABLE = False
try:
    from sarvamai import SarvamAI
    SARVAM_SDK_AVAILABLE = True
except ImportError:
    SARVAM_SDK_AVAILABLE = False

import requests

SARVAM_STT_ENDPOINT = "https://api.sarvam.ai/speech-to-text"


class SarvamSTTClient:
    """Client for Sarvam AI Speech-to-Text API using saaras:v4 model."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key.strip() if api_key else os.environ.get("SARVAM_API_KEY", "").strip()
        self.model = "saaras:v4"
        self._sdk_client = None

        if self.api_key and SARVAM_SDK_AVAILABLE:
            try:
                self._sdk_client = SarvamAI(api_subscription_key=self.api_key)
            except Exception as e:
                logger.warning(f"Failed to initialize SarvamAI SDK client: {e}. Falling back to REST API.")

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def transcribe_audio(
        self,
        audio_path: str,
        language_code: str = "unknown",
        prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Transcribe an audio file using Sarvam AI saaras:v4.
        
        Args:
            audio_path: Path to the WAV/audio file.
            language_code: Target or detected language (default 'unknown' for auto-detection).
            prompt: Optional context prompt to guide transcription.
            
        Returns:
            Dict containing transcript, language_code, model, confidence/metrics, and status.
        """
        if not self.is_configured:
            return self._mock_transcription(audio_path, prompt=prompt, reason="API Key not provided")

        if not os.path.exists(audio_path):
            return {
                "success": False,
                "transcript": "",
                "error": f"Audio file not found at {audio_path}",
                "model": self.model,
                "mode": "error"
            }

        # 1. Try SDK with file object
        if self._sdk_client is not None:
            try:
                with open(audio_path, "rb") as f:
                    kwargs = {
                        "file": f,
                        "model": self.model,
                    }
                    if language_code and language_code != "unknown":
                        kwargs["language_code"] = language_code

                    res = self._sdk_client.speech_to_text.transcribe(**kwargs)
                    
                    transcript_text = getattr(res, "transcript", None) or (res.get("transcript") if isinstance(res, dict) else str(res))
                    detected_lang = getattr(res, "language_code", "unknown") if hasattr(res, "language_code") else (res.get("language_code", "unknown") if isinstance(res, dict) else "unknown")
                    
                    return {
                        "success": True,
                        "transcript": transcript_text if transcript_text else "",
                        "language_code": detected_lang,
                        "model": self.model,
                        "raw_response": res if isinstance(res, (dict, str)) else str(res),
                        "mode": "live_sdk"
                    }
            except Exception as e:
                logger.warning(f"Sarvam SDK transcription error: {e}. Attempting REST API fallback.")

        # 2. Direct REST API Fallback
        try:
            headers = {
                "api-subscription-key": self.api_key
            }
            
            with open(audio_path, "rb") as f:
                file_bytes = f.read()
                files = {
                    "file": ("audio.wav", file_bytes, "audio/wav")
                }
                data = {
                    "model": self.model,
                    "mode": "transcribe"
                }
                if language_code and language_code != "unknown":
                    data["language_code"] = language_code

                response = requests.post(
                    SARVAM_STT_ENDPOINT,
                    headers=headers,
                    files=files,
                    data=data,
                    timeout=30
                )

            if response.status_code == 200:
                resp_json = response.json()
                transcript = resp_json.get("transcript", "")
                detected_lang = resp_json.get("language_code", language_code)
                return {
                    "success": True,
                    "transcript": transcript,
                    "language_code": detected_lang,
                    "model": self.model,
                    "raw_response": resp_json,
                    "mode": "live_rest"
                }
            elif response.status_code in (401, 403):
                return {
                    "success": False,
                    "transcript": prompt if prompt else "Sample Speech Phonation",
                    "language_code": "en-IN",
                    "error": "Invalid or unauthorized Sarvam AI subscription key.",
                    "model": self.model,
                    "mode": "auth_error",
                    "note": "Sarvam AI API key unauthorized (403/401). Please check the subscription key entered in the sidebar."
                }
            else:
                error_msg = f"Sarvam API Error ({response.status_code}): {response.text}"
                logger.error(error_msg)
                return {
                    "success": False,
                    "transcript": prompt if prompt else "Sample Speech Phonation",
                    "error": error_msg,
                    "model": self.model,
                    "mode": "live_error",
                    "note": f"Cloud ASR returned HTTP {response.status_code}."
                }

        except Exception as e:
            logger.exception("Failed to connect to Sarvam AI STT API")
            return {
                "success": False,
                "transcript": "",
                "error": f"Connection exception: {str(e)}",
                "model": self.model,
                "mode": "exception"
            }

    def _mock_transcription(self, audio_path: str, prompt: Optional[str] = None, reason: str = "") -> Dict[str, Any]:
        """Fallback mock transcription for testing without an API key."""
        mock_text = prompt if prompt else "Ahhhhh — testing sustained vocal volume and breath support."
        return {
            "success": True,
            "transcript": mock_text,
            "language_code": "en-IN",
            "model": f"{self.model} (Demo Mode)",
            "mode": "demo_mock",
            "note": f"Live transcription standby ({reason}). Provide your Sarvam AI API key in the sidebar for real-time cloud STT."
        }
