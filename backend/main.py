"""BOLIFIX phase-1 local prototype.

Not suitable for hosting on the public Internet without authentication,
tenant authorization, encryption and production data protection.
"""
from __future__ import annotations

import hashlib
from contextlib import asynccontextmanager
import io
import os
import sqlite3
import uuid
import wave
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field
from backend.clarification import suggest_clarification
from backend.recovery import review_reply
from backend.recovery_flow import next_recovery_step
from backend import tts as tts_service
from backend import evidence as evidence_service
from backend import asr_review as review_service
from backend import quality_gate as quality_service
from backend import dataset_validation as dataset_service
from backend.dual_asr import compare_hypotheses
from backend.dual_recovery import next_dual_recovery_step

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

API_URL = "https://api.vachana.ai/stt/v3"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".aac", ".m4a"}
DB_PATH = ROOT / os.getenv("BOLIFIX_DB_PATH", "storage/bolifix.sqlite3")
AUDIO_DIR = ROOT / os.getenv("BOLIFIX_AUDIO_DIR", "storage/recordings")
MOCK_STT = os.getenv("BOLIFIX_MOCK_STT", "false").lower() == "true"

def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS corrections (
                id TEXT PRIMARY KEY,
                app_id TEXT NOT NULL,
                model_name TEXT NOT NULL,
                language_hint TEXT NOT NULL,
                transcript_raw TEXT NOT NULL,
                speaker_reference TEXT NOT NULL,
                reference_kind TEXT NOT NULL,
                review_status TEXT NOT NULL,
                consent_scope TEXT NOT NULL,
                audio_sha256 TEXT NOT NULL,
                audio_path TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
    evidence_service.ensure_recovery_table(DB_PATH)
    review_service.ensure_review_table(DB_PATH)
    quality_service.ensure_quality_tables(DB_PATH)


@asynccontextmanager
async def lifespan(application: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="BOLIFIX", version="0.1.0",
    description="First milestone: transcribe and collect explicitly permissioned corrections",
    lifespan=lifespan,
)


def get_audio(filename: str, data: bytes) -> tuple[bytes, str]:
    ext = Path(filename or "").suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported audio file extension")
    if not data or len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="Audio must be nonempty and under 10 MB")
    if ext == ".wav":
        try:
            with wave.open(io.BytesIO(data), "rb") as wav:
                duration = wav.getnframes() / wav.getframerate()
            if duration > 30:
                raise HTTPException(status_code=400, detail="WAV recording exceeds Prisma REST 30-second limit")
        except (wave.Error, EOFError, ZeroDivisionError):
            raise HTTPException(status_code=400, detail="Invalid WAV audio")
    return data, ext


def transcribe_via_gnani(audio_bytes: bytes, filename: str, language_code: str, output_format: str) -> dict:
    if MOCK_STT:
        return {
            "transcript": "यह एक MOCK डेमो ट्रांसक्रिप्ट है।",
            "model": "MOCK — not Gnani",
            "request_id": "mock-" + uuid.uuid4().hex[:10],
            "detected_language_code": language_code,
            "mock": True,
        }
    key = os.getenv("GNANI_API_KEY", "").strip()
    if not key:
        raise HTTPException(status_code=503, detail="GNANI_API_KEY is not set in the server .env")
    try:
        response = requests.post(
            API_URL,
            headers={"X-API-Key-ID": key},
            data={"model": "prisma-v3.0", "language_code": language_code, "format": output_format},
            files={"audio_file": (filename, io.BytesIO(audio_bytes))},
            timeout=65,
        )
        response.raise_for_status()
        result = response.json()
    except requests.Timeout:
        raise HTTPException(status_code=504, detail="Gnani transcription timed out")
    except requests.RequestException as exc:
        status = getattr(exc.response, "status_code", None)
        raise HTTPException(status_code=502, detail=f"Gnani request failed (upstream HTTP {status or 'unavailable'})")
    except ValueError:
        raise HTTPException(status_code=502, detail="Gnani returned a non-JSON response")
    if not result.get("success", False) or not isinstance(result.get("transcript"), str):
        raise HTTPException(status_code=502, detail="Gnani returned an unsuccessful transcription result")
    return {
        "transcript": result["transcript"],
        "model": result.get("model", "prisma-v3.0"),
        "request_id": result.get("request_id"),
        "detected_language_code": result.get("detected_language_code"),
        "mock": False,
    }


@app.get("/health")
def health():
    return {"status": "ok", "mock_mode": MOCK_STT, "configured": bool(os.getenv("GNANI_API_KEY")),
            "tts_mock_mode": os.getenv("BOLIFIX_MOCK_TTS", "true").lower() == "true"}


class AnalyzeTranscriptRequest(BaseModel):
    transcript: str = Field(min_length=1, max_length=2000)


@app.post("/api/v1/clarification-suggestion")
def clarification_suggestion(payload: AnalyzeTranscriptRequest):
    # Deterministic, no Gnani call; this is a flag, not an error verdict.
    return suggest_clarification(payload.transcript)


class DualAsrTextRequest(BaseModel):
    prisma_transcript: str = Field(min_length=1, max_length=2000)
    whisper_transcript: str = Field(min_length=1, max_length=2000)


@app.post("/api/v1/dual-asr/compare-text")
def dual_asr_compare_text(payload: DualAsrTextRequest):
    """Compare two TEXT hypotheses only; no audio, Gnani, Whisper or database call."""
    return compare_hypotheses(payload.prisma_transcript, payload.whisper_transcript)


class DualAsrClarificationAudioRequest(DualAsrTextRequest):
    language: str = Field(default="hi-IN", pattern=r"^(hi-IN|en-IN)$")


@app.post("/api/v1/dual-asr/clarification-audio")
def dual_asr_clarification_audio(payload: DualAsrClarificationAudioRequest):
    """Optional targeted question; synthesize ONLY on explicit user action.

    The server derives a fixed question from both hypotheses instead of allowing
    arbitrary caller-supplied speech. Neither the comparison nor this endpoint
    changes the stored original audio, transcripts or training permissions.
    """
    comparison = compare_hypotheses(payload.prisma_transcript, payload.whisper_transcript)
    route = comparison["clarification_plan"]
    if not route["can_speak"]:
        return {
            "needs_clarification": False, "mock": True, "question": None,
            "route": route["route"],
            "message": "No automatic TTS: review the original audio or script difference first.",
        }
    question = route["question_hi"] if payload.language == "hi-IN" else route["question_en"]
    if os.getenv("BOLIFIX_MOCK_TTS", "true").lower() == "true":
        return {
            "needs_clarification": True, "mock": True, "question": question,
            "route": route["route"],
            "message": "MOCK TTS: no audio generated; no Gnani credits used.",
        }
    if not os.getenv("GNANI_API_KEY", "").strip():
        raise HTTPException(status_code=503, detail="GNANI_API_KEY not configured")
    try:
        speech = tts_service.synthesize_timbre(question, payload.language)
    except Exception:
        raise HTTPException(status_code=502, detail="Gnani Timbre failed to generate the clarification question")
    if not isinstance(speech, bytes) or not speech.startswith(b"RIFF") or b"WAVE" not in speech[:16]:
        raise HTTPException(status_code=502, detail="Gnani Timbre returned invalid WAV audio")
    if len(speech) > 5 * 1024 * 1024:
        raise HTTPException(status_code=502, detail="Gnani Timbre audio exceeds expected limit")
    return Response(
        content=speech,
        media_type="audio/wav",
        headers={"Cache-Control": "no-store", "X-BOLIFIX-Mock": "false"},
    )


class DualAsrReplyRequest(DualAsrTextRequest):
    reply_transcript: str = Field(min_length=1, max_length=2000)
    attempt: int = Field(ge=1, le=2)


@app.post("/api/v1/dual-asr/recovery-next")
def dual_asr_recovery_next(payload: DualAsrReplyRequest):
    """Offline reply decision: no model inference, storage or speaker verification."""
    return next_dual_recovery_step(
        payload.prisma_transcript, payload.whisper_transcript,
        payload.reply_transcript, payload.attempt,
    )


class DualAsrFollowupAudioRequest(DualAsrReplyRequest):
    language: str = Field(default="hi-IN", pattern=r"^(hi-IN|en-IN)$")


@app.post("/api/v1/dual-asr/followup-audio")
def dual_asr_followup_audio(payload: DualAsrFollowupAudioRequest):
    """Optional second targeted question; only invoked by explicit user click."""
    decision = next_dual_recovery_step(
        payload.prisma_transcript, payload.whisper_transcript,
        payload.reply_transcript, payload.attempt,
    )
    if decision["state"] != "ask_followup":
        return {"needs_followup": False, "mock": True, "question": None,
                "message": "No follow-up allowed. No provider call was made."}
    question = (decision["followup_question_hi"] if payload.language == "hi-IN"
                else decision["followup_question_en"])
    if os.getenv("BOLIFIX_MOCK_TTS", "true").lower() == "true":
        return {"needs_followup": True, "mock": True, "question": question,
                "message": "MOCK TTS: no audio generated; no Gnani credits used."}
    if not os.getenv("GNANI_API_KEY", "").strip():
        raise HTTPException(status_code=503, detail="GNANI_API_KEY not configured")
    try:
        audio = tts_service.synthesize_timbre(question, payload.language)
    except Exception:
        raise HTTPException(status_code=502, detail="Gnani Timbre failed to produce the follow-up")
    if not isinstance(audio, bytes) or not audio.startswith(b"RIFF") or b"WAVE" not in audio[:16]:
        raise HTTPException(status_code=502, detail="Gnani Timbre returned invalid WAV audio")
    if len(audio) > 5 * 1024 * 1024:
        raise HTTPException(status_code=502, detail="Gnani Timbre audio exceeds expected limit")
    return Response(content=audio, media_type="audio/wav",
                    headers={"Cache-Control": "no-store", "X-BOLIFIX-Mock": "false"})


class ClarificationAudioRequest(BaseModel):
    transcript: str = Field(min_length=1, max_length=2000)
    language: str = Field(default="hi-IN", pattern=r"^(hi-IN|en-IN)$")


class ClarificationReviewRequest(BaseModel):
    original_transcript: str = Field(min_length=1, max_length=2000)
    reply_transcript: str = Field(min_length=1, max_length=2000)


@app.post("/api/v1/clarification-review")
def clarification_review(payload: ClarificationReviewRequest):
    """Legacy Step 3 reply reviewer, preserved for compatibility."""
    return review_reply(payload.original_transcript, payload.reply_transcript)


class RecoveryNextRequest(ClarificationReviewRequest):
    attempt: int = Field(ge=1, le=2)


class RecoveryFollowupAudioRequest(RecoveryNextRequest):
    language: str = Field(default="hi-IN", pattern=r"^(hi-IN|en-IN)$")


@app.post("/api/v1/recovery-next")
def recovery_next(payload: RecoveryNextRequest):
    """Bounded local decision; does not spend credits or store recordings."""
    return next_recovery_step(
        payload.original_transcript, payload.reply_transcript, payload.attempt
    )


@app.post("/api/v1/recovery-followup-audio")
def recovery_followup_audio(payload: RecoveryFollowupAudioRequest):
    """Speak a *fixed* follow-up question only when a second turn is warranted.

    A user click initiates synthesis. The caller cannot supply arbitrary TTS text.
    """
    decision = next_recovery_step(
        payload.original_transcript, payload.reply_transcript, payload.attempt
    )
    if decision["state"] != "ask_followup":
        return {
            "needs_followup": False,
            "question": None,
            "mock": True,
            "message": "No follow-up question available. No TTS call made.",
        }
    question = (decision["followup_question_hi"] if payload.language == "hi-IN"
                else decision["followup_question_en"])
    if os.getenv("BOLIFIX_MOCK_TTS", "true").lower() == "true":
        return {
            "needs_followup": True,
            "question": question,
            "mock": True,
            "message": "MOCK TTS: no audio generated; no Gnani credits used.",
        }
    if not os.getenv("GNANI_API_KEY", "").strip():
        raise HTTPException(status_code=503, detail="GNANI_API_KEY not configured")
    try:
        speech = tts_service.synthesize_timbre(question, payload.language)
    except Exception:
        raise HTTPException(
            status_code=502,
            detail="Gnani Timbre request failed. Check the backend log and TTS access.",
        )
    if not isinstance(speech, bytes) or not speech.startswith(b"RIFF") or b"WAVE" not in speech[:16]:
        raise HTTPException(status_code=502, detail="Gnani Timbre did not return a valid WAV file")
    if len(speech) > 5 * 1024 * 1024:
        raise HTTPException(status_code=502, detail="Gnani Timbre response unexpectedly large")
    return Response(
        content=speech,
        media_type="audio/wav",
        headers={"Cache-Control": "no-store", "X-BOLIFIX-Mock": "false"},
    )


@app.post("/api/v1/clarification-audio")
def clarification_audio(payload: ClarificationAudioRequest):
    """Generate a spoken question only when a user explicitly requests playback.

    Returns JSON in mock mode, WAV bytes in real mode. Does not retain audio.
    """
    question = tts_service.clarification_question(payload.transcript, payload.language)
    if question is None:
        return {"needs_clarification": False, "question": None, "mock": True,
                "message": "No predefined high-impact clue; no TTS call made."}
    if os.getenv("BOLIFIX_MOCK_TTS", "true").lower() == "true":
        return {"needs_clarification": True, "question": question, "mock": True,
                "message": "MOCK TTS: no audio generated; no Gnani credits used."}
    if not os.getenv("GNANI_API_KEY", "").strip():
        raise HTTPException(status_code=503, detail="GNANI_API_KEY not configured")
    try:
        speech = tts_service.synthesize_timbre(question, payload.language)
    except Exception:
        # Do not leak API credentials or external exception content.
        raise HTTPException(status_code=502, detail="Gnani Timbre request failed. Check the backend log and your TTS access.")
    if not isinstance(speech, bytes) or not speech.startswith(b"RIFF") or b"WAVE" not in speech[:16]:
        raise HTTPException(status_code=502, detail="Gnani Timbre did not return a valid WAV file")
    if len(speech) > 5 * 1024 * 1024:
        raise HTTPException(status_code=502, detail="Gnani Timbre response unexpectedly large")
    return Response(content=speech, media_type="audio/wav",
                    headers={"Cache-Control": "no-store", "X-BOLIFIX-Mock": "false"})


@app.post("/api/v1/transcribe")
def transcribe(
    audio: UploadFile = File(...),
    language_code: str = Form("indic-auto"),
    output_format: str = Form("codemix"),
    permission_to_process: bool = Form(False),
):
    if not permission_to_process:
        raise HTTPException(status_code=400, detail="Audio-processing permission is required")
    if language_code not in {
        "indic-auto", "hi-IN", "en-IN", "hi-IN,en-IN", "bn-IN", "mr-IN", "gu-IN",
        "pa-IN", "kn-IN", "te-IN", "ta-IN", "ml-IN",
    }:
        raise HTTPException(status_code=400, detail="Unsupported language hint in starter")
    if output_format not in {"verbatim", "transcribe", "codemix", "transliterate"}:
        raise HTTPException(status_code=400, detail="Unsupported transcription format")
    blob, _ = get_audio(audio.filename or "audio.wav", audio.file.read(MAX_UPLOAD_BYTES + 1))
    # No audio or transcript is stored by this endpoint.
    return transcribe_via_gnani(blob, audio.filename or "audio.wav", language_code, output_format)


@app.post("/api/v1/corrections", status_code=201)
def save_correction(
    audio: UploadFile = File(...),
    app_id: str = Form("demo-app"),
    transcript_raw: str = Form(...),
    speaker_reference: str = Form(...),
    language_hint: str = Form("indic-auto"),
    model_name: str = Form("prisma-v3.0"),
    consent_scope: str = Form(...),
    speaker_confirmed_exact_words: bool = Form(False),
):
    if consent_scope not in {"private", "research_eligible"}:
        raise HTTPException(status_code=400, detail="Explicit storage consent is required")
    if not speaker_confirmed_exact_words:
        raise HTTPException(status_code=400, detail="Speaker confirmation is required")
    if model_name.startswith("MOCK") and consent_scope == "research_eligible":
        raise HTTPException(status_code=400, detail="Mock transcriptions cannot enter the research-eligible pool")
    if not 1 <= len(speaker_reference.strip()) <= 2000:
        raise HTTPException(status_code=400, detail="Correction must be 1–2000 characters")
    if not 1 <= len(transcript_raw) <= 2000:
        raise HTTPException(status_code=400, detail="Transcript must be 1–2000 characters")
    if not 1 <= len(app_id) <= 80 or not all(c.isalnum() or c in "_-" for c in app_id):
        raise HTTPException(status_code=400, detail="app_id must use letters, digits, - or _")
    blob, ext = get_audio(audio.filename or "audio.wav", audio.file.read(MAX_UPLOAD_BYTES + 1))
    init_db()
    identifier = str(uuid.uuid4())
    path = AUDIO_DIR / f"{identifier}{ext}"
    path.write_bytes(blob)
    try:
        with sqlite3.connect(DB_PATH) as con:
            con.execute("""
                INSERT INTO corrections
                (id, app_id, model_name, language_hint, transcript_raw,
                 speaker_reference, reference_kind, review_status, consent_scope,
                 audio_sha256, audio_path, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                identifier, app_id, model_name, language_hint, transcript_raw,
                speaker_reference.strip(), "speaker_self_report", "needs_independent_review",
                consent_scope, hashlib.sha256(blob).hexdigest(), str(path),
                datetime.now(timezone.utc).isoformat(),
            ))
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return {"id": identifier, "status": "saved_locally_for_review", "training_ready": False}


@app.get("/api/v1/corrections")
def list_corrections():
    # Local developer tool only: add authentication and tenant scoping before public deployment.
    init_db()
    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute("""
            SELECT id, app_id, model_name, language_hint, transcript_raw,
                   speaker_reference, reference_kind, review_status, consent_scope,
                   created_at
            FROM corrections ORDER BY created_at DESC LIMIT 100
        """).fetchall()
    return {"items": [dict(r) for r in rows]}


@app.delete("/api/v1/corrections/{correction_id}")
def delete_correction(correction_id: str):
    init_db()
    with sqlite3.connect(DB_PATH) as con:
        row = con.execute("SELECT audio_path FROM corrections WHERE id = ?", (correction_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Correction not found")
        con.execute("DELETE FROM corrections WHERE id = ?", (correction_id,))
    Path(row[0]).unlink(missing_ok=True)
    return {"deleted": True, "id": correction_id}


@app.post("/api/v1/recovery-evidence", status_code=201)
def create_recovery_evidence(
    original_audio: UploadFile = File(...),
    first_reply_audio: UploadFile = File(...),
    second_reply_audio: UploadFile | None = File(None),
    app_id: str = Form("demo-app"),
    original_transcript: str = Form(...),
    first_reply_transcript: str = Form(...),
    second_reply_transcript: str = Form(""),
    confirmed_intent: str = Form(...),
    language_hint: str = Form("indic-auto"),
    source_model: str = Form("prisma-v3.0"),
    speaker_confirmed_intent: bool = Form(False),
    explicit_storage_consent: bool = Form(False),
    original_mock: bool = Form(False),
    first_reply_mock: bool = Form(False),
    second_reply_mock: bool = Form(False),
):
    """Store a *private* speaker-confirmed intent trace; never ASR ground truth.

    This local-only endpoint has no login or tenant isolation yet. Do not host
    on the public internet or connect actual customer audio.
    """
    if not speaker_confirmed_intent or not explicit_storage_consent:
        raise HTTPException(status_code=400, detail="Speaker confirmation and separate storage consent are required")
    if original_mock or first_reply_mock or second_reply_mock or source_model.upper().startswith("MOCK"):
        raise HTTPException(status_code=400, detail="Mock recognition is not eligible for recovery evidence")
    if not 1 <= len(app_id) <= 80 or not all(c.isalnum() or c in "_-" for c in app_id):
        raise HTTPException(status_code=400, detail="Invalid application ID")
    fields = (original_transcript, first_reply_transcript, confirmed_intent)
    if any(not 1 <= len(value.strip()) <= 2000 for value in fields):
        raise HTTPException(status_code=400, detail="Transcripts and confirmed intent must contain 1–2000 characters")
    if not 1 <= len(language_hint) <= 40 or not 1 <= len(source_model) <= 80:
        raise HTTPException(status_code=400, detail="Invalid language or source model")
    if second_reply_audio is None and second_reply_transcript.strip():
        raise HTTPException(status_code=400, detail="Second reply transcript requires its audio")
    if second_reply_audio is not None and not second_reply_transcript.strip():
        raise HTTPException(status_code=400, detail="Second reply audio requires its transcript")
    if len(second_reply_transcript) > 2000:
        raise HTTPException(status_code=400, detail="Second reply transcript too long")
    first_blob, first_ext = get_audio(
        original_audio.filename or "original.wav", original_audio.file.read(MAX_UPLOAD_BYTES + 1))
    reply_blob, reply_ext = get_audio(
        first_reply_audio.filename or "reply.wav", first_reply_audio.file.read(MAX_UPLOAD_BYTES + 1))
    optional_second = None
    if second_reply_audio is not None:
        second_blob, second_ext = get_audio(
            second_reply_audio.filename or "reply2.wav", second_reply_audio.file.read(MAX_UPLOAD_BYTES + 1))
        optional_second = (second_blob, second_ext)
    identifier = evidence_service.save_recovery(
        db_path=DB_PATH, audio_dir=AUDIO_DIR / "recovery_evidence",
        app_id=app_id, source_model=source_model, language_hint=language_hint,
        original_transcript=original_transcript.strip(),
        first_reply_transcript=first_reply_transcript.strip(),
        second_reply_transcript=second_reply_transcript.strip() if optional_second else None,
        confirmed_intent=confirmed_intent.strip(),
        original_audio=(first_blob, first_ext), first_audio=(reply_blob, reply_ext),
        second_audio=optional_second,
    )
    return {
        "id": identifier,
        "status": "saved_privately_for_review",
        "evidence_kind": "speaker_confirmed_intent_only",
        "training_ready": False,
        "audio_count": 3 if optional_second else 2,
    }


@app.get("/api/v1/recovery-evidence")
def list_recovery_evidence():
    # Developer-only local endpoint. Must add auth and per-app authorization before hosting.
    return {"items": evidence_service.list_recoveries(DB_PATH)}


@app.delete("/api/v1/recovery-evidence/{evidence_id}")
def delete_recovery_evidence(evidence_id: str):
    try:
        deleted = evidence_service.delete_recovery(DB_PATH, evidence_id, AUDIO_DIR / "recovery_evidence")
    except ValueError:
        raise HTTPException(status_code=409, detail="Storage path issue: needs developer investigation")
    if not deleted:
        raise HTTPException(status_code=404, detail="Recovery evidence not found")
    return {"deleted": True, "id": evidence_id}


# Phase 2 Step 6: local-only human review of the ORIGINAL recording.
# No research sharing, export, or fine-tuning is enabled by these endpoints.

class AsrReviewRequest(BaseModel):
    evidence_id: str = Field(min_length=1, max_length=80)
    reviewer_code: str = Field(min_length=1, max_length=40)
    verbatim_reference: str = Field(min_length=1, max_length=2000)
    error_category: str = Field(min_length=1, max_length=40)
    reviewer_note: str = Field(default="", max_length=500)
    confirmed_listened: bool = False
    confirmed_verbatim: bool = False


@app.get("/api/v1/recovery-evidence/{evidence_id}/original-audio")
def listen_original_evidence(evidence_id: str):
    """Local reviewer playback. Do not expose this API without authentication."""
    init_db()
    try:
        audio = review_service.get_original_audio(DB_PATH, AUDIO_DIR / "recovery_evidence", evidence_id)
    except (ValueError, OSError):
        raise HTTPException(status_code=409, detail="Stored audio is missing, altered, or outside the expected directory")
    if audio is None:
        raise HTTPException(status_code=404, detail="Recovery evidence not found")
    content, media_type = audio
    return Response(content=content, media_type=media_type, headers={
        "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff",
    })


@app.post("/api/v1/asr-reviews", status_code=201)
def create_asr_review(payload: AsrReviewRequest):
    """A reviewer supplies a separately listened-to *verbatim* transcript.

    No automatic transcription, training label approval, or export occurs.
    """
    init_db()
    try:
        # Must check stored media presence and integrity before attaching a label.
        clip = review_service.get_original_audio(DB_PATH, AUDIO_DIR / "recovery_evidence", payload.evidence_id)
        if clip is None:
            raise HTTPException(status_code=404, detail="Recovery evidence not found")
        return review_service.create_review(
            db_path=DB_PATH, evidence_id=payload.evidence_id, reviewer_code=payload.reviewer_code,
            verbatim_reference=payload.verbatim_reference, error_category=payload.error_category,
            reviewer_note=payload.reviewer_note, confirmed_listened=payload.confirmed_listened,
            confirmed_verbatim=payload.confirmed_verbatim,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except LookupError:
        raise HTTPException(status_code=404, detail="Recovery evidence not found")
    except OSError:
        raise HTTPException(status_code=409, detail="Stored audio is unavailable")


@app.get("/api/v1/asr-reviews")
def list_asr_reviews():
    return {"items": review_service.list_reviews(DB_PATH), "training_ready_count": 0}


@app.delete("/api/v1/asr-reviews/{evidence_id}")
def remove_asr_review(evidence_id: str):
    if not review_service.delete_review(DB_PATH, evidence_id):
        raise HTTPException(status_code=404, detail="ASR review not found")
    return {"deleted": True, "evidence_id": evidence_id}


# Phase 2 Step 7: independent original-audio review and LOCAL-ONLY pilot manifest.
# Reviewer aliases are not authenticated; this is not safe for real customer calls.

class IndependentAsrReviewRequest(BaseModel):
    evidence_id: str = Field(min_length=1, max_length=80)
    reviewer_code: str = Field(min_length=1, max_length=40)
    verbatim_reference: str = Field(default="", max_length=2000)
    reviewer_note: str = Field(default="", max_length=500)
    confirmed_listened: bool = False
    confirmed_verbatim: bool = False


@app.post("/api/v1/asr-second-reviews", status_code=201)
def create_independent_asr_review(payload: IndependentAsrReviewRequest):
    init_db()
    try:
        return quality_service.submit_second_review(
            db_path=DB_PATH, audio_dir=AUDIO_DIR / "recovery_evidence",
            evidence_id=payload.evidence_id, reviewer_code=payload.reviewer_code,
            verbatim_reference=payload.verbatim_reference, reviewer_note=payload.reviewer_note,
            confirmed_listened=payload.confirmed_listened, confirmed_verbatim=payload.confirmed_verbatim,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/v1/asr-second-reviews")
def list_independent_asr_reviews():
    return {"items": quality_service.list_second_reviews(DB_PATH)}


@app.delete("/api/v1/asr-second-reviews/{evidence_id}")
def remove_independent_asr_review(evidence_id: str):
    if not quality_service.delete_second_review(DB_PATH, evidence_id):
        raise HTTPException(status_code=404, detail="Independent ASR review not found")
    return {"deleted": True, "evidence_id": evidence_id,
            "permission_cleared": True}


class LocalTrainingPermissionRequest(BaseModel):
    evidence_id: str = Field(min_length=1, max_length=80)
    attested_by: str = Field(min_length=1, max_length=40)
    speaker_authorized: bool = False
    privacy_screened: bool = False
    local_training_authorized: bool = False


@app.post("/api/v1/asr-local-permissions", status_code=201)
def allow_local_asr_training(payload: LocalTrainingPermissionRequest):
    init_db()
    try:
        return quality_service.grant_local_permission(
            db_path=DB_PATH, audio_dir=AUDIO_DIR / "recovery_evidence",
            evidence_id=payload.evidence_id, attested_by=payload.attested_by,
            speaker_authorized=payload.speaker_authorized,
            privacy_screened=payload.privacy_screened,
            local_training_authorized=payload.local_training_authorized,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/v1/asr-local-permissions")
def list_asr_local_permissions():
    return {"items": quality_service.list_local_permissions(DB_PATH)}


@app.delete("/api/v1/asr-local-permissions/{evidence_id}")
def remove_asr_local_permission(evidence_id: str):
    if not quality_service.revoke_local_permission(DB_PATH, evidence_id):
        raise HTTPException(status_code=404, detail="Local-training permission not found")
    return {"revoked": True, "evidence_id": evidence_id,
            "notice": "Regenerate manifests; delete any previously downloaded local copies."}


class LocalManifestRequest(BaseModel):
    confirm_local_pilot_only: bool = False


@app.post("/api/v1/asr-local-dataset-manifest")
def build_local_asr_manifest(payload: LocalManifestRequest):
    """Return an IN-MEMORY JSONL index; never copies audio or calls Gnani.

    Must not be hosted publicly. The manifest is only a preparation artifact,
    not a legal permission record or a production training approval.
    """
    if not payload.confirm_local_pilot_only:
        raise HTTPException(status_code=400, detail="Confirm this is a private, local-only pilot")
    try:
        items = quality_service.candidate_manifest(
            db_path=DB_PATH, audio_dir=AUDIO_DIR / "recovery_evidence", project_root=ROOT)
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    if not items:
        raise HTTPException(status_code=409, detail="No authorized, matching, intact ASR candidates found")
    return Response(
        content=quality_service.manifest_jsonl(items).encode("utf-8"),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff",
                 "Content-Disposition": 'attachment; filename="bolifix-local-candidates.jsonl"'},
    )


# Phase 3, Step 1: in-memory checks and reproducible local speaker splits.
# These development-only routes do not make recordings training-ready.
class LocalSpeakerAssignment(BaseModel):
    evidence_id: str = Field(min_length=1, max_length=100)
    speaker_code: str = Field(pattern=r"^spk-[A-Za-z0-9_-]{1,24}$")


class ValidateLocalSplitRequest(BaseModel):
    speaker_assignments: list[LocalSpeakerAssignment] = Field(default_factory=list, max_length=500)
    seed: int = Field(default=42, ge=0, le=4294967295)
    confirm_local_pilot_only: bool = False


def _current_local_candidates():
    try:
        return quality_service.candidate_manifest(
            db_path=DB_PATH, audio_dir=AUDIO_DIR / "recovery_evidence", project_root=ROOT
        )
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/api/v1/asr-split-candidates")
def list_split_candidates():
    """List currently authorized original-audio IDs for manual pseudonymous grouping."""
    rows = _current_local_candidates()
    return {
        "items": [
            {"evidence_id": r["evidence_id"], "language_hint": r["language_hint"]}
            for r in rows
        ],
        "count": len(rows),
        "training_ready": False,
        "notice": "Assign stable spk-XX pseudonyms; the app cannot verify real speaker identity.",
    }


def _validate_split(payload: ValidateLocalSplitRequest) -> tuple[list[dict], dict]:
    if not payload.confirm_local_pilot_only:
        raise HTTPException(status_code=400, detail="Confirm this is a private local-only pilot")
    rows = _current_local_candidates()
    report = dataset_service.validate_candidates(
        candidates=rows,
        speaker_assignments=[r.model_dump() for r in payload.speaker_assignments],
        project_root=ROOT,
        seed=payload.seed,
    )
    return rows, report


@app.post("/api/v1/asr-validate-split")
def check_local_split(payload: ValidateLocalSplitRequest):
    """Read-only analysis: errors prevent export; warnings remain visible."""
    _, report = _validate_split(payload)
    return report


@app.post("/api/v1/asr-split-manifest")
def build_local_split_manifest(payload: ValidateLocalSplitRequest):
    """Only export a checked, speaker-disjoint local JSONL index (never audio bytes)."""
    rows, report = _validate_split(payload)
    if not report["ready_for_local_split"]:
        reasons = "; ".join(sorted({item["code"] for item in report["issues"]}))
        raise HTTPException(status_code=409, detail=f"Pilot split blocked: {reasons}")
    entries = dataset_service.split_manifest(rows, report)
    return Response(
        content=quality_service.manifest_jsonl(entries).encode("utf-8"),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff",
                 "Content-Disposition": 'attachment; filename="bolifix-speaker-splits-local.jsonl"'},
    )
