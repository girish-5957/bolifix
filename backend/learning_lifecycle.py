"""Read-only learning lifecycle for a single Showcase session.

Crucially, this module does NOT persist audio, grant training permission,
run model updates, access external applications, or infer ASR correctness.
"""
from __future__ import annotations

LEARNING_STAGES = (
    ("Private evidence capture", "Research tool", "The separately run local research backend can retain permitted recovery audio; the Showcase cannot."),
    ("Independent word-for-word review", "Research tool", "Two reviewers must listen to original audio and agree on a reference before training eligibility."),
    ("Dataset validation", "Research tool", "Explicit training permission, integrity checks and speaker separation are required."),
    ("Open ASR fine-tuning", "Not yet built", "An approved dataset and a training implementation are still needed."),
    ("Independent model evaluation", "Partial", "Evaluation tools exist, but one recording cannot establish improvement."),
    ("Collaborative cross-app learning", "Future research", "No deployed cross-app or federated model learning is present."),
    ("Real-time phone/voice-agent connector", "Not yet built", "Short recording capture works; a true call-streaming integration is planned."),
)


def session_learning_status(*, audio_present: bool, transcript_present: bool,
                            reply_present: bool, confirmed_intent: bool) -> dict:
    """Return the observed session journey; do not upgrade intent to transcript truth."""
    have_audio = bool(audio_present)
    have_transcript = have_audio and bool(transcript_present)
    have_reply = have_transcript and bool(reply_present)
    have_intent = have_reply and bool(confirmed_intent)
    return {
        "scope": "browser_session_only",
        "speaker_confirmed": have_intent,
        "saved_to_research_db": False,
        "training_label_verified": False,
        "model_weights_updated": False,
        "live_phone_call_connected": False,
        "runtime": [
            {"label":"Speech supplied", "detail":"A recorded or uploaded clip is available for this session.", "complete": have_audio},
            {"label":"Prisma transcription", "detail":"A first machine hypothesis was returned after permission.", "complete": have_transcript},
            {"label":"Clarification reply", "detail":"A follow-up response was transcribed for this session.", "complete": have_reply},
            {"label":"Speaker-confirmed intent", "detail":"The intended value was explicitly verified for this interaction only.", "complete": have_intent},
        ],
    }
