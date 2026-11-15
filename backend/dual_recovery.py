"""Bounded, permission-neutral dual-ASR reply decisions.

This module is deliberately pure: no model inference, cloud calls, database
access, audio storage or claims of exact transcript verification. The caller
must obtain explicit permission before sending any reply audio to Prisma.
"""
from __future__ import annotations

from backend.dual_asr import compare_hypotheses
from backend.recovery import number_candidates

MAX_ATTEMPTS = 2

FOLLOWUP_HI = "कृपया केवल अंतिम सही संख्या एक बार बोलिए, बाकी कुछ नहीं।"
FOLLOWUP_EN = "Please say only the final intended number once, without other words."


def next_dual_recovery_step(prisma_transcript: str, whisper_transcript: str,
                            reply_transcript: str, attempt: int) -> dict:
    """Return a proposed UI state, not a verified recognition or training label.

    Every call recomputes the route from the original hypothesis pair. The
    client must bind replies to that pair; changed hypotheses invalidate prior
    session confirmations.
    """
    if attempt not in (1, 2):
        raise ValueError("attempt must be 1 or 2")
    comparison = compare_hypotheses(prisma_transcript, whisper_transcript)
    route = comparison["clarification_plan"]["route"]
    values = number_candidates(reply_transcript) if route == "confirm_final_number" else []

    if route in {"no_prompt", "review_script_only", "manual_listen"}:
        state = "manual_listening_required" if route == "manual_listen" else "no_voice_clarification"
        candidate = None
        message = "No automatic reply-based conclusion. Listen to the original audio if the information matters."
    elif route == "confirm_final_number" and len(values) == 1:
        state = "candidate_confirmation"
        candidate = values[0]
        message = "One possible number was recognized. The speaker must verify it before use."
    elif route == "confirm_final_number" and attempt == 1:
        state = "ask_followup"
        candidate = None
        message = "The reply did not isolate one number; one final focused voice attempt is permitted."
    else:
        state = "manual_verification"
        candidate = None
        message = (
            "The answer must be verified directly by the speaker; "
            "a second ASR prediction cannot resolve names, identifiers or negated actions."
        )
    return {
        "state": state,
        "route": route,
        "attempt": attempt,
        "max_voice_attempts": MAX_ATTEMPTS,
        "candidate": candidate,
        "message": message,
        "followup_question_hi": FOLLOWUP_HI if state == "ask_followup" else None,
        "followup_question_en": FOLLOWUP_EN if state == "ask_followup" else None,
        "requires_speaker_confirmation": state != "no_voice_clarification",
        "training_ready": False,
        "verified_original_audio": False,
        "record_saved": False,
        "note": (
            "Only a speaker-confirmed intended value can be established here. "
            "The exact words in the original recording remain unverified. "
            "No data or model parameters were changed."
        ),
    }
