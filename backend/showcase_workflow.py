"""Local-only showcase helpers for independent ASR and speaker-confirmed receipts.

Not a training pipeline, model accuracy report, production order or audit certificate.
No API requests or persistent storage are made by these helpers.
"""
from __future__ import annotations

import io
import json
from hashlib import sha256


def make_receipt(original: str, first_reply: str, second_reply: str,
                 intended_value: str, original_audio: bytes,
                 whisper_hypothesis: str = "") -> dict:
    """Create a downloadable session-only summary without saving audio or secrets."""
    if not intended_value or not intended_value.strip():
        raise ValueError("The speaker must explicitly confirm an intended value.")
    if not original or not original.strip():
        raise ValueError("An original transcript is required.")
    if not original_audio:
        raise ValueError("Original audio is required to bind this session receipt.")
    return {
        "project": "BOLIFIX",
        "schema_version": "showcase-recovery-v1",
        "record_type": "speaker_confirmed_intent_not_asr_ground_truth",
        "original_transcript_prisma": original,
        "independent_whisper_hypothesis": whisper_hypothesis or None,
        "first_reply_prisma": first_reply or None,
        "second_reply_prisma": second_reply or None,
        "speaker_confirmed_intended_value": intended_value.strip(),
        "original_audio_sha256": sha256(original_audio).hexdigest(),
        "audio_included": False,
        "speaker_confirmation": "explicitly_confirmed_in_current_session",
        "training_data_authorized": False,
        "claim": "A user confirmed intended meaning, not exact spoken words.",
    }


def receipt_json(receipt: dict) -> str:
    """JSON: structured text for optional local download, not a database write."""
    return json.dumps(receipt, ensure_ascii=False, indent=2)


def run_local_whisper(audio_bytes: bytes, model) -> str:
    """Use a preloaded local Whisper model; never calls Hugging Face or Gnani.

    Model must be supplied by the caller and loaded with local_files_only=True.
    """
    if not audio_bytes or len(audio_bytes) > 10 * 1024 * 1024:
        raise ValueError("Audio must be nonempty and less than 10 MB.")
    segments, _info = model.transcribe(
        io.BytesIO(audio_bytes), language="hi", task="transcribe",
        beam_size=5, temperature=0, vad_filter=True,
        condition_on_previous_text=False,
    )
    return " ".join(s.text.strip() for s in segments).strip()
