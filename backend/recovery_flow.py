"""A bounded, deterministic clarification flow for the local BOLIFIX prototype.

This module makes no provider requests, writes no data, and does not claim
to know whether a recognizer's output matches the original speech.
"""
from __future__ import annotations

from backend.recovery import review_reply

MAX_VOICE_ATTEMPTS = 2

FOLLOWUP_QUESTIONS = {
    "hi-IN": "कृपया सिर्फ अंतिम संख्या एक बार बताइए। अगर कठिनाई हो, तो उसे लिखकर पुष्टि करें।",
    "en-IN": "Please say only the final number once. If that is difficult, type it to confirm.",
}


def next_recovery_step(original_transcript: str, reply_transcript: str, attempt: int) -> dict:
    """Decide what to do after a recognized clarification reply.

    `attempt` is 1 or 2. A response may be a candidate for *manual* confirmation,
    another short question, or a manual-entry fallback. It is never a training label.
    """
    if attempt not in (1, 2):
        raise ValueError("attempt must be 1 or 2")
    review = review_reply(original_transcript, reply_transcript)
    status = review["status"]

    if status == "candidate_needs_confirmation":
        state = "candidate_confirmation"
        message = "A possible final number was recognized. Confirm or correct it directly."
    elif status == "no_trigger":
        state = "no_trigger"
        message = "No configured trigger; this is not proof that the transcript is accurate."
    elif status in {"ambiguous_reply", "no_numeric_candidate"} and attempt < MAX_VOICE_ATTEMPTS:
        state = "ask_followup"
        message = "The reply did not supply one clear number. Ask one more focused question."
    else:
        state = "manual_verification"
        message = (
            "No reliable single value was obtained, or an exact identifier/action needs "
            "direct review. Ask the speaker to type, spell, or verify the intended value."
        )

    return {
        "state": state,
        "attempt": attempt,
        "max_voice_attempts": MAX_VOICE_ATTEMPTS,
        "signal_type": review["signal_type"],
        "candidate": review["candidate"] if state == "candidate_confirmation" else None,
        "message": message,
        "followup_question_en": FOLLOWUP_QUESTIONS["en-IN"] if state == "ask_followup" else None,
        "followup_question_hi": FOLLOWUP_QUESTIONS["hi-IN"] if state == "ask_followup" else None,
        "requires_speaker_confirmation": True,
        "training_ready": False,
    }
