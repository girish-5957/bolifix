"""Local-only, human-attested ASR review of the ORIGINAL recovery recording.

Speaker-confirmed intent is *not* reused as a word-for-word transcript. A single
review does not grant research/training permission or count as training-ready.
"""
from __future__ import annotations

import hashlib
import sqlite3
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

CATEGORIES = {
    "no_observed_error", "word_substitution", "word_omission", "extra_word",
    "number_or_amount", "name_or_place", "script_or_format",
    "multiple_or_other", "uncertain",
}


def ensure_review_table(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as con:
        con.execute("PRAGMA foreign_keys = ON")
        con.execute("""
            CREATE TABLE IF NOT EXISTS asr_reviews (
                evidence_id TEXT PRIMARY KEY,
                reviewer_code TEXT NOT NULL,
                verbatim_reference TEXT NOT NULL,
                error_category TEXT NOT NULL,
                reviewer_note TEXT NOT NULL,
                reference_kind TEXT NOT NULL,
                review_status TEXT NOT NULL,
                consent_scope TEXT NOT NULL,
                original_audio_sha256 TEXT NOT NULL,
                substitutions INTEGER NOT NULL,
                deletions INTEGER NOT NULL,
                insertions INTEGER NOT NULL,
                reference_words INTEGER NOT NULL,
                wer REAL NOT NULL,
                cer REAL NOT NULL,
                training_ready INTEGER NOT NULL DEFAULT 0 CHECK(training_ready = 0),
                created_at TEXT NOT NULL,
                FOREIGN KEY(evidence_id) REFERENCES recovery_evidence(id) ON DELETE CASCADE
            )
        """)


def normalize_tokens(text: str) -> list[str]:
    """Unicode-aware: keep Devanagari combining marks with their word."""
    chars = []
    for char in unicodedata.normalize("NFC", text).casefold():
        if unicodedata.category(char)[0] in {"L", "M", "N"}:
            chars.append(char)
        else:
            chars.append(" ")
    return "".join(chars).split()


def word_edits(reference: list[str], hypothesis: list[str]) -> tuple[int, int, int]:
    """S, D, I counts; deterministic minimal-edit alignment (a single path)."""
    previous = [(j, 0, 0, j) for j in range(len(hypothesis) + 1)]
    for i, true_word in enumerate(reference, 1):
        current = [(i, 0, i, 0)]
        for j, predicted_word in enumerate(hypothesis, 1):
            if true_word == predicted_word:
                current.append(previous[j - 1]); continue
            p = previous[j - 1]
            sub = (p[0] + 1, p[1] + 1, p[2], p[3])
            p = previous[j]
            delete = (p[0] + 1, p[1], p[2] + 1, p[3])
            p = current[-1]
            insert = (p[0] + 1, p[1], p[2], p[3] + 1)
            # Prefer substitution on ties; stable metrics, not definitive blame.
            current.append(min((sub, delete, insert), key=lambda v: v[0]))
        previous = current
    _, s, d, i = previous[-1]
    return s, d, i


def edit_distance(reference: str, hypothesis: str) -> int:
    """Memory-bounded Levenshtein for a descriptive character-error ratio."""
    prev = list(range(len(hypothesis) + 1))
    for i, char in enumerate(reference, 1):
        cur = [i]
        for j, other in enumerate(hypothesis, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (char != other)))
        prev = cur
    return prev[-1]


def compare_transcripts(reference: str, hypothesis: str) -> dict:
    truth = normalize_tokens(reference)
    guess = normalize_tokens(hypothesis)
    if not truth:
        raise ValueError("Reference must include audible words, not only punctuation")
    s, d, i = word_edits(truth, guess)
    # Remove spaces for CER so punctuation and spacing don't inflate the metric.
    char_truth = "".join(truth)
    char_guess = "".join(guess)
    return {
        "substitutions": s, "deletions": d, "insertions": i,
        "reference_words": len(truth),
        "wer": round((s + d + i) / len(truth), 4),
        "cer": round(edit_distance(char_truth, char_guess) / len(char_truth), 4),
        "normalized_exact_match": truth == guess,
    }


def get_original_audio(db_path: Path, audio_dir: Path, evidence_id: str) -> tuple[bytes, str] | None:
    """Only open the original stored evidence clip; fail closed on path/hash drift."""
    with sqlite3.connect(db_path) as con:
        record = con.execute(
            "SELECT original_audio_path, original_sha256 FROM recovery_evidence WHERE id = ?",
            (evidence_id,),
        ).fetchone()
    if record is None:
        return None
    path = Path(record[0]).resolve()
    if audio_dir.resolve() not in path.parents:
        raise ValueError("Unexpected stored audio path")
    if not path.is_file():
        raise ValueError("Original audio is missing from disk")
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != record[1]:
        raise ValueError("Original audio checksum mismatch")
    formats = {
        ".wav": "audio/wav", ".mp3": "audio/mpeg", ".ogg": "audio/ogg",
        ".flac": "audio/flac", ".aac": "audio/aac", ".m4a": "audio/mp4",
    }
    if path.suffix.lower() not in formats:
        raise ValueError("Unexpected saved audio format")
    return content, formats[path.suffix.lower()]


def create_review(
    *, db_path: Path, evidence_id: str, reviewer_code: str,
    verbatim_reference: str, error_category: str, reviewer_note: str,
    confirmed_listened: bool, confirmed_verbatim: bool,
) -> dict:
    ensure_review_table(db_path)
    if not confirmed_listened:
        raise ValueError("You must listen to the original recording before reviewing it")
    if error_category not in CATEGORIES:
        raise ValueError("Unsupported error classification")
    if not 1 <= len(reviewer_code) <= 40 or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in reviewer_code):
        raise ValueError("Reviewer code must be 1–40 letters/digits/-/_")
    verbatim_reference = verbatim_reference.strip()
    if not 1 <= len(verbatim_reference) <= 2000:
        raise ValueError("Verbatim reference must contain 1–2000 characters")
    if len(reviewer_note) > 500:
        raise ValueError("Reviewer note must be at most 500 characters")
    if error_category == "uncertain" and confirmed_verbatim:
        raise ValueError("An uncertain review cannot be marked word-for-word verified")
    if error_category != "uncertain" and not confirmed_verbatim:
        raise ValueError("Confirm the word-for-word transcript or choose uncertain")
    with sqlite3.connect(db_path) as con:
        con.execute("PRAGMA foreign_keys = ON")
        existing = con.execute("SELECT 1 FROM asr_reviews WHERE evidence_id = ?", (evidence_id,)).fetchone()
        if existing:
            raise FileExistsError("A review exists for this recording; delete it before submitting a revision")
        row = con.execute("""
            SELECT original_transcript, original_sha256, consent_scope, source_model
            FROM recovery_evidence WHERE id = ?
        """, (evidence_id,)).fetchone()
        if row is None:
            raise LookupError("Recovery recording not found")
        model_transcript, audio_hash, consent_scope, model = row
        if consent_scope != "private" or model.upper().startswith("MOCK"):
            raise ValueError("Only stored real, private recordings can be reviewed")
        metrics = compare_transcripts(verbatim_reference, model_transcript)
        # A reviewer may flag formatting differences or uncertain cases; all
        # metrics are descriptive and do not replace human listening.
        status = "needs_relisten" if error_category == "uncertain" else "awaiting_independent_quality_check"
        try:
            con.execute("""
                INSERT INTO asr_reviews (
                    evidence_id, reviewer_code, verbatim_reference, error_category,
                    reviewer_note, reference_kind, review_status, consent_scope,
                    original_audio_sha256, substitutions, deletions, insertions,
                    reference_words, wer, cer, training_ready, created_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                evidence_id, reviewer_code, verbatim_reference, error_category,
                reviewer_note, "human_listened_original_audio" if confirmed_verbatim else "reviewer_draft_uncertain",
                status, consent_scope, audio_hash,
                metrics["substitutions"], metrics["deletions"], metrics["insertions"],
                metrics["reference_words"], metrics["wer"], metrics["cer"], 0,
                datetime.now(timezone.utc).isoformat(),
            ))
        except sqlite3.IntegrityError as e:
            raise FileExistsError("Review already exists, or source evidence was removed") from e
    return {
        "evidence_id": evidence_id,
        "review_status": status,
        "reference_kind": "human_listened_original_audio" if confirmed_verbatim else "reviewer_draft_uncertain",
        "training_ready": False,
        "research_export_allowed": False,
        "metrics": metrics,
    }


def list_reviews(db_path: Path) -> list[dict]:
    ensure_review_table(db_path)
    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute("""
            SELECT evidence_id, reviewer_code, verbatim_reference, error_category,
                   reviewer_note, reference_kind, review_status, consent_scope,
                   substitutions, deletions, insertions, reference_words, wer, cer,
                   created_at
            FROM asr_reviews ORDER BY created_at DESC LIMIT 100
        """).fetchall()
    return [{**dict(row), "training_ready": False, "research_export_allowed": False} for row in rows]


def delete_review(db_path: Path, evidence_id: str) -> bool:
    ensure_review_table(db_path)
    with sqlite3.connect(db_path) as con:
        con.execute("PRAGMA foreign_keys = ON")  # Cascade to Step 7 quality and consent rows.
        deleted = con.execute("DELETE FROM asr_reviews WHERE evidence_id = ?", (evidence_id,))
        return deleted.rowcount == 1
