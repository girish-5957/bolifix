"""Phase 3 Step 3 — isolated local ASR evaluation demos.

Two deliberately separate workflows:
(1) synthetic TEXT pairs demonstrate evaluation math without audio or model;
(2) optional single currently approved original clip may be transcribed locally
    as an exploratory smoke test. This is NEVER a held-out benchmark.

No API calls, uploads, training, database updates, or manifest/split bypass.
The Step 2 benchmark continues to require a real speaker-disjoint split.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import wave
from typing import Callable

from backend import asr_baseline, quality_gate
from backend.asr_review import compare_transcripts

# These are entirely invented TEXT PAIRS — not model outputs, real speakers,
# human-reviewed audio, or train/test records.
SYNTHETIC_TEXT_CASES = (
    ("मुझे पंद्रह पैकेट चाहिए", "मुझे पाँच पैकेट चाहिए"),
    ("कल भुवनेश्वर जाना है", "कल भुवनेश्वर जाना है"),
    ("order number twenty five", "order number twenty"),
)


def synthetic_demo() -> dict:
    """Exercise the existing evaluator on invented text, never touch any audio."""
    rows = [
        {"evidence_id": f"SYNTHETIC-TEXT-{i}", "verbatim_transcript": truth}
        for i, (truth, _) in enumerate(SYNTHETIC_TEXT_CASES, 1)
    ]
    predictions = {
        f"SYNTHETIC-TEXT-{i}": guess
        for i, (_, guess) in enumerate(SYNTHETIC_TEXT_CASES, 1)
    }
    report = asr_baseline.evaluate_transcripts(rows, predictions)
    report.update({
        "mode": "synthetic_text_math_only",
        "real_recordings_used": 0,
        "model_inference_performed": False,
        "is_benchmark": False,
        "note": "Made-up text pairs demonstrate error-rate calculations only; these are NOT speech-recognition scores.",
    })
    return report


def approved_clips(*, project_root: Path, db_path: Path, audio_dir: Path) -> list[dict]:
    """Get CURRENTLY approved original clips from the existing Step 7 gate.

    The gate validates two real reviews, permission, privacy and SHA-256.
    No downloaded JSONL manifest is trusted to authorize this demo.
    """
    return quality_gate.candidate_manifest(
        project_root=project_root, db_path=db_path, audio_dir=audio_dir,
    )


def _checked_original(row: dict, *, project_root: Path) -> Path:
    """Re-check a single original file for exploratory local use."""
    root = project_root.resolve()
    rel = Path(row["audio_path"])
    path = (root / rel).resolve()
    if rel.is_absolute() or root not in path.parents or not path.is_file():
        raise ValueError("Original audio missing or unsafe")
    if "recovery_evidence" not in rel.parts or not path.stem.endswith("-original") or path.suffix.lower() != ".wav":
        raise ValueError("Only approved original WAV evidence can be used")
    if hashlib.sha256(path.read_bytes()).hexdigest() != row["audio_sha256"]:
        raise ValueError("Original WAV checksum changed; abort inference")
    try:
        with wave.open(str(path), "rb") as f:
            rate, channels, width, frames = f.getframerate(), f.getnchannels(), f.getsampwidth(), f.getnframes()
            if (f.getcomptype() != "NONE" or rate != 16000 or channels != 1 or width != 2
                or not .25 <= frames / rate <= 30):
                raise ValueError("Clip must be 16kHz mono 16-bit PCM WAV, 0.25–30 sec")
            pcm = f.readframes(frames)
            if len(pcm) != frames * channels * width:
                raise ValueError("Truncated WAV")
    except (OSError, EOFError, wave.Error) as exc:
        raise ValueError("Unreadable approved original WAV") from exc
    return path


def choose_approved_clip(*, project_root: Path, db_path: Path, audio_dir: Path,
                         index: int = 1) -> dict:
    """Inspect ONE approved clip, without pretending to create a train/test split."""
    if not isinstance(index, int) or isinstance(index, bool) or index < 1:
        raise ValueError("Approved clip index must be at least 1")
    rows = approved_clips(project_root=project_root, db_path=db_path, audio_dir=audio_dir)
    if not rows:
        raise ValueError("No Step 7 authorized recordings; use synthetic demo instead")
    if index > len(rows):
        raise ValueError(f"Only {len(rows)} approved clip(s) currently exist")
    selected = rows[index - 1]
    _checked_original(selected, project_root=project_root)
    return {"row": selected, "approved_count": len(rows)}


def exploratory_inference(*, project_root: Path, db_path: Path, audio_dir: Path,
                          selected_evidence_id: str, transcribe_one: Callable[[Path], str]) -> dict:
    """Re-check authorization and original audio immediately before one model call.

    Does NOT call evaluate_transcripts() on a fake test split, save results,
    grant new permissions, or update the model/SQLite database.
    """
    if not selected_evidence_id:
        raise ValueError("Select an approved evidence ID")
    rows = approved_clips(project_root=project_root, db_path=db_path, audio_dir=audio_dir)
    matching = [row for row in rows if row["evidence_id"] == selected_evidence_id]
    if len(matching) != 1:
        raise ValueError("Clip is no longer authorized; aborting local inference")
    selected = matching[0]
    audio = _checked_original(selected, project_root=project_root)
    hypothesis = transcribe_one(audio)
    if not isinstance(hypothesis, str):
        raise ValueError("Local ASR adapter must return text")
    prediction = hypothesis.strip()
    comparison = compare_transcripts(selected["verbatim_transcript"], prediction)
    return {
        "mode": "single_authorized_clip_smoke_test",
        "is_benchmark": False,
        "training_performed": False,
        "real_recordings_used": 1,
        "model_inference_performed": True,
        "prediction": prediction,
        "reference": selected["verbatim_transcript"],
        "metrics": comparison,
        "note": "ONE approved recording only: useful to test local inference, NOT representative WER/CER or speaker-independent evaluation.",
    }
