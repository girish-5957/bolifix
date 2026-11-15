"""Speech playback for BOLIFIX clarification questions.

TTS speaks a question; it does NOT verify the accuracy of the transcript.
This module never writes generated speech to storage.
"""
from __future__ import annotations

import os

from backend.clarification import suggest_clarification

LANGUAGES = {"hi-IN", "en-IN"}  # Start with two tested prompts; extend with native validation.

HINDI_QUESTIONS = {
    "identifier": "कृपया आखिरी ऑर्डर या रेफरेंस आईडी एक-एक अक्षर बताकर पुष्टि करें।",
    "number": "कृपया अंतिम संख्या या रकम दोबारा बताइए।",
    "negation_or_correction": "कृपया बताइए कि आप क्या करना चाहते हैं और क्या नहीं करना चाहते।",
}


def clarification_question(transcript: str, language: str) -> str | None:
    """Generate question from high-impact signals (not a correctness verdict)."""
    if language not in LANGUAGES:
        raise ValueError("Unsupported TTS language for this starter")
    result = suggest_clarification(transcript)
    if not result["needs_clarification"]:
        return None
    if language == "hi-IN":
        for kind in ("identifier", "number", "negation_or_correction"):
            if kind in result["signals"]:
                return HINDI_QUESTIONS[kind]
    return result["suggested_question"]


def synthesize_timbre(text: str, language: str) -> bytes:
    """Call Gnani's official SDK only for a user-requested synthesis."""
    key = os.getenv("GNANI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GNANI_API_KEY is missing")
    # Lazy import keeps offline tests and mock mode independent of the SDK.
    from gnani.tts import GnaniTTSClient, AudioConfig

    client = GnaniTTSClient(api_key=key)
    return client.synthesize(
        text,
        model="timbre-v2.5",
        voice="Nalini" if language == "hi-IN" else "Kaveri",
        language=language,
        speed=1.0,
        audio_config=AudioConfig(sample_rate=16000, encoding="linear_pcm", container="wav"),
    )
