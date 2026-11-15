"""Private recovery *intent* evidence for a local-only learning prototype.

A speaker confirming an intended value is NOT a word-for-word audio label.
Such evidence is never considered training ready or shared between tenants.
"""
from __future__ import annotations

import hashlib
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path


def ensure_recovery_table(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS recovery_evidence (
                id TEXT PRIMARY KEY,
                app_id TEXT NOT NULL,
                source_model TEXT NOT NULL,
                language_hint TEXT NOT NULL,
                original_transcript TEXT NOT NULL,
                first_reply_transcript TEXT NOT NULL,
                second_reply_transcript TEXT,
                confirmed_intent TEXT NOT NULL,
                verification_kind TEXT NOT NULL,
                review_status TEXT NOT NULL,
                consent_scope TEXT NOT NULL,
                original_audio_path TEXT NOT NULL,
                original_sha256 TEXT NOT NULL,
                first_audio_path TEXT NOT NULL,
                first_sha256 TEXT NOT NULL,
                second_audio_path TEXT,
                second_sha256 TEXT,
                created_at TEXT NOT NULL
            )
        """)


def save_recovery(
    *, db_path: Path, audio_dir: Path,
    app_id: str, source_model: str, language_hint: str,
    original_transcript: str, first_reply_transcript: str,
    second_reply_transcript: str | None, confirmed_intent: str,
    original_audio: tuple[bytes, str], first_audio: tuple[bytes, str],
    second_audio: tuple[bytes, str] | None,
) -> str:
    """Persist an opt-in private record; keep media outside SQLite.

    Caller MUST validate permissions and audio formats/lengths beforehand.
    """
    ensure_recovery_table(db_path)
    audio_dir.mkdir(parents=True, exist_ok=True)
    identifier = str(uuid.uuid4())
    files = [
        ("original", original_audio),
        ("reply1", first_audio),
        *(([("reply2", second_audio)]) if second_audio is not None else []),
    ]
    stored: dict[str, tuple[Path, str]] = {}
    created: list[Path] = []
    try:
        for suffix, (content, extension) in files:
            path = audio_dir / f"{identifier}-{suffix}{extension}"
            # Unique names, no user-provided filename or raw path accepted.
            with path.open("xb") as out:
                out.write(content)
            created.append(path)
            stored[suffix] = (path, hashlib.sha256(content).hexdigest())
        with sqlite3.connect(db_path) as con:
            con.execute("""
                INSERT INTO recovery_evidence (
                    id, app_id, source_model, language_hint,
                    original_transcript, first_reply_transcript,
                    second_reply_transcript, confirmed_intent,
                    verification_kind, review_status, consent_scope,
                    original_audio_path, original_sha256,
                    first_audio_path, first_sha256, second_audio_path, second_sha256,
                    created_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                identifier, app_id, source_model, language_hint,
                original_transcript, first_reply_transcript,
                second_reply_transcript, confirmed_intent,
                "speaker_confirmed_intent_only", "needs_independent_review", "private",
                str(stored["original"][0]), stored["original"][1],
                str(stored["reply1"][0]), stored["reply1"][1],
                str(stored["reply2"][0]) if "reply2" in stored else None,
                stored["reply2"][1] if "reply2" in stored else None,
                datetime.now(timezone.utc).isoformat(),
            ))
    except Exception:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    return identifier


def list_recoveries(db_path: Path) -> list[dict]:
    ensure_recovery_table(db_path)
    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute("""
            SELECT id, app_id, source_model, language_hint,
                   original_transcript, first_reply_transcript, second_reply_transcript,
                   confirmed_intent, verification_kind, review_status,
                   consent_scope, created_at
            FROM recovery_evidence ORDER BY created_at DESC LIMIT 100
        """).fetchall()
        return [{**dict(row), "training_ready": False} for row in rows]


def delete_recovery(db_path: Path, identifier: str, audio_dir: Path) -> bool:
    """Remove record and its media, with paths restricted to our local directory."""
    ensure_recovery_table(db_path)
    with sqlite3.connect(db_path) as con:
        # Clear any linked Step 6 review with the parent evidence.
        con.execute("PRAGMA foreign_keys = ON")
        row = con.execute("""
            SELECT original_audio_path, first_audio_path, second_audio_path
            FROM recovery_evidence WHERE id = ?
        """, (identifier,)).fetchone()
        if row is None:
            return False
        # Don't delete any file outside the dedicated local evidence store.
        base = audio_dir.resolve()
        paths = [Path(p).resolve() for p in row if p]
        if any(base not in p.parents for p in paths):
            raise ValueError("Record has an unexpected audio path; manual investigation required")
        con.execute("DELETE FROM recovery_evidence WHERE id = ?", (identifier,))
    for path in paths:
        path.unlink(missing_ok=True)
    return True
