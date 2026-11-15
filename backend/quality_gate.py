"""Step 7: local-only independent listening review and *candidate* dataset manifest.

No model is trained here. Two matching transcriptions, a local-use permission
attestation, a privacy check and intact stored audio are mandatory. Reviewer
aliases are NOT authenticated identities; deployment requires real RBAC.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from backend.asr_review import get_original_audio, normalize_tokens, ensure_review_table
from backend.evidence import ensure_recovery_table


ALIAS_CHARS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" )


def _alias(value: str) -> str:
    value = value.strip()
    if not 1 <= len(value) <= 40 or any(ch not in ALIAS_CHARS for ch in value):
        raise ValueError("Reviewer alias must contain 1–40 letters/digits/-/_")
    return value


def ensure_quality_tables(db_path: Path) -> None:
    ensure_recovery_table(db_path)
    ensure_review_table(db_path)
    with sqlite3.connect(db_path) as con:
        con.execute("PRAGMA foreign_keys = ON")
        con.execute("""
            CREATE TABLE IF NOT EXISTS independent_asr_reviews (
                evidence_id TEXT PRIMARY KEY,
                reviewer_code TEXT NOT NULL,
                verbatim_reference TEXT NOT NULL,
                reviewer_note TEXT NOT NULL,
                confirmed_verbatim INTEGER NOT NULL CHECK(confirmed_verbatim IN (0,1)),
                status TEXT NOT NULL CHECK(status IN ('agreed','disagreed','uncertain')),
                original_audio_sha256 TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(evidence_id) REFERENCES asr_reviews(evidence_id) ON DELETE CASCADE
            )
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS local_training_permissions (
                evidence_id TEXT PRIMARY KEY,
                attested_by TEXT NOT NULL,
                scope TEXT NOT NULL CHECK(scope = 'local_model_training_only'),
                speaker_authorized INTEGER NOT NULL CHECK(speaker_authorized = 1),
                privacy_screened INTEGER NOT NULL CHECK(privacy_screened = 1),
                original_audio_sha256 TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(evidence_id) REFERENCES independent_asr_reviews(evidence_id) ON DELETE CASCADE
            )
        """)


def submit_second_review(
    *, db_path: Path, audio_dir: Path, evidence_id: str,
    reviewer_code: str, verbatim_reference: str, reviewer_note: str,
    confirmed_listened: bool, confirmed_verbatim: bool,
) -> dict:
    ensure_quality_tables(db_path)
    reviewer_code = _alias(reviewer_code)
    if not confirmed_listened:
        raise ValueError("Second reviewer must listen to the ORIGINAL recording")
    if len(reviewer_note) > 500:
        raise ValueError("Reviewer note must be at most 500 characters")
    verbatim_reference = verbatim_reference.strip()
    if len(verbatim_reference) > 2000:
        raise ValueError("Reference is too long")
    if confirmed_verbatim and not normalize_tokens(verbatim_reference):
        raise ValueError("Verified reference must contain audible words")
    if not confirmed_verbatim and verbatim_reference:
        raise ValueError("Uncertain second review must leave the reference blank")
    audio = get_original_audio(db_path, audio_dir, evidence_id)
    if audio is None:
        raise LookupError("Original recovery evidence not found")
    with sqlite3.connect(db_path) as con:
        con.execute("PRAGMA foreign_keys = ON")
        row = con.execute("""
            SELECT a.reviewer_code, a.verbatim_reference, a.reference_kind,
                   a.review_status, a.original_audio_sha256,
                   e.original_sha256, e.consent_scope, e.source_model
            FROM asr_reviews a JOIN recovery_evidence e ON e.id = a.evidence_id
            WHERE a.evidence_id = ?
        """, (evidence_id,)).fetchone()
        if row is None:
            raise LookupError("First ASR review must be saved before an independent review")
        (first_reviewer, first_text, ref_kind, first_status,
         first_hash, original_hash, scope, source_model) = row
        if reviewer_code.casefold() == first_reviewer.casefold():
            raise ValueError("Second reviewer must use a DIFFERENT reviewer alias")
        if ref_kind != "human_listened_original_audio" or first_status != "awaiting_independent_quality_check":
            raise ValueError("First review is uncertain; resolve it before independent approval")
        if scope != "private" or source_model.upper().startswith("MOCK"):
            raise ValueError("Only real private audio is eligible")
        if first_hash != original_hash or hashlib.sha256(audio[0]).hexdigest() != original_hash:
            raise ValueError("Original audio or first-review checksum changed")
        agreed = confirmed_verbatim and normalize_tokens(first_text) == normalize_tokens(verbatim_reference)
        status = "uncertain" if not confirmed_verbatim else "agreed" if agreed else "disagreed"
        try:
            con.execute("""
                INSERT INTO independent_asr_reviews (
                    evidence_id, reviewer_code, verbatim_reference, reviewer_note,
                    confirmed_verbatim, status, original_audio_sha256, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (evidence_id, reviewer_code, verbatim_reference, reviewer_note.strip(),
                  int(confirmed_verbatim), status, original_hash,
                  datetime.now(timezone.utc).isoformat()))
        except sqlite3.IntegrityError as exc:
            raise FileExistsError("Independent review already exists; delete it before revising") from exc
    return {
        "evidence_id": evidence_id, "status": status, "matched_first_reference": bool(agreed),
        "eligible_for_local_pilot": False, "training_ready": False,
        "message": ("Two transcripts agree after simple normalization; separate training permission is still required."
                    if agreed else "Do not export: independent review is uncertain or disagrees."),
    }


def list_second_reviews(db_path: Path) -> list[dict]:
    ensure_quality_tables(db_path)
    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute("""
            SELECT evidence_id, reviewer_code, verbatim_reference, reviewer_note,
                   confirmed_verbatim, status, created_at
            FROM independent_asr_reviews ORDER BY created_at DESC LIMIT 100
        """).fetchall()
        return [{**dict(r), "training_ready": False} for r in rows]


def delete_second_review(db_path: Path, evidence_id: str) -> bool:
    ensure_quality_tables(db_path)
    with sqlite3.connect(db_path) as con:
        con.execute("PRAGMA foreign_keys = ON")
        result = con.execute("DELETE FROM independent_asr_reviews WHERE evidence_id = ?", (evidence_id,))
        return result.rowcount == 1


def grant_local_permission(
    *, db_path: Path, audio_dir: Path, evidence_id: str,
    attested_by: str, speaker_authorized: bool, privacy_screened: bool,
    local_training_authorized: bool,
) -> dict:
    ensure_quality_tables(db_path)
    attested_by = _alias(attested_by)
    if not (speaker_authorized and privacy_screened and local_training_authorized):
        raise ValueError("Speaker authorization, privacy screening and explicit local-training consent are ALL required")
    audio = get_original_audio(db_path, audio_dir, evidence_id)
    if audio is None:
        raise LookupError("Recovery evidence not found")
    with sqlite3.connect(db_path) as con:
        con.execute("PRAGMA foreign_keys = ON")
        row = con.execute("""
            SELECT q.status, q.original_audio_sha256, a.verbatim_reference,
                   e.original_sha256, e.consent_scope, e.source_model
            FROM independent_asr_reviews q
            JOIN asr_reviews a ON a.evidence_id = q.evidence_id
            JOIN recovery_evidence e ON e.id = q.evidence_id
            WHERE q.evidence_id = ?
        """, (evidence_id,)).fetchone()
        if row is None:
            raise LookupError("Independent review must be completed first")
        status, second_hash, first_ref, original_hash, scope, source_model = row
        if status != "agreed" or not normalize_tokens(first_ref):
            raise ValueError("Both independent word-for-word reviews must agree")
        if scope != "private" or source_model.upper().startswith("MOCK"):
            raise ValueError("Only real privately stored recordings are eligible")
        if second_hash != original_hash or hashlib.sha256(audio[0]).hexdigest() != original_hash:
            raise ValueError("Audio checksum does not match the saved review")
        try:
            con.execute("""
                INSERT INTO local_training_permissions (
                    evidence_id, attested_by, scope, speaker_authorized,
                    privacy_screened, original_audio_sha256, created_at
                ) VALUES (?, ?, 'local_model_training_only', 1, 1, ?, ?)
            """, (evidence_id, attested_by, original_hash,
                  datetime.now(timezone.utc).isoformat()))
        except sqlite3.IntegrityError as exc:
            raise FileExistsError("Local-training permission already recorded") from exc
    return {"evidence_id": evidence_id, "scope": "local_model_training_only",
            "eligible_for_local_pilot": True, "training_ready": False,
            "notice": "Local prototype candidate only; no model training or external sharing"}


def revoke_local_permission(db_path: Path, evidence_id: str) -> bool:
    ensure_quality_tables(db_path)
    with sqlite3.connect(db_path) as con:
        con.execute("PRAGMA foreign_keys = ON")
        result = con.execute("DELETE FROM local_training_permissions WHERE evidence_id = ?", (evidence_id,))
        return result.rowcount == 1


def list_local_permissions(db_path: Path) -> list[dict]:
    ensure_quality_tables(db_path)
    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute("""
            SELECT evidence_id, attested_by, scope, created_at
            FROM local_training_permissions ORDER BY created_at DESC LIMIT 100
        """).fetchall()
        return [{**dict(r), "eligible_for_local_pilot": True,
                 "training_ready": False} for r in rows]


def candidate_manifest(*, db_path: Path, audio_dir: Path, project_root: Path) -> list[dict]:
    """In-memory manifest only. Do not copy media, train models or send data online.

    Fail *entire* export on checksum drift (rather than silently omitting bad media).
    """
    ensure_quality_tables(db_path)
    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute("""
            SELECT e.id, e.original_audio_path, e.original_sha256,
                   e.original_transcript, e.language_hint, e.source_model,
                   e.consent_scope, a.verbatim_reference AS first_ref,
                   a.reference_kind, a.review_status, a.original_audio_sha256 AS first_hash,
                   q.verbatim_reference AS second_ref, q.status AS second_status,
                   q.original_audio_sha256 AS second_hash,
                   p.original_audio_sha256 AS permission_hash, p.scope AS permission_scope
            FROM local_training_permissions p
            JOIN independent_asr_reviews q ON q.evidence_id = p.evidence_id
            JOIN asr_reviews a ON a.evidence_id = p.evidence_id
            JOIN recovery_evidence e ON e.id = p.evidence_id
            ORDER BY e.created_at, e.id
        """).fetchall()
    results = []
    for row in rows:
        row = dict(row)
        if (row["consent_scope"] != "private" or row["source_model"].upper().startswith("MOCK")
            or row["reference_kind"] != "human_listened_original_audio"
            or row["review_status"] != "awaiting_independent_quality_check"
            or row["second_status"] != "agreed"
            or row["permission_scope"] != "local_model_training_only"
            or not normalize_tokens(row["first_ref"])
            or normalize_tokens(row["first_ref"]) != normalize_tokens(row["second_ref"]) 
            or len({row["original_sha256"], row["first_hash"], row["second_hash"], row["permission_hash"]}) != 1):
            raise ValueError("Saved approval or review data is inconsistent; dataset was NOT exported")
        clip = get_original_audio(db_path, audio_dir, row["id"])
        if clip is None:
            raise ValueError("Approved source audio is missing")
        if hashlib.sha256(clip[0]).hexdigest() != row["original_sha256"]:
            raise ValueError("Audio changed after local-training permission")
        audio_path = Path(row["original_audio_path"]).resolve()
        try:
            relative_path = audio_path.relative_to(project_root.resolve()).as_posix()
        except ValueError as exc:
            raise ValueError("Audio is outside the local project") from exc
        results.append({
            "evidence_id": row["id"],
            "audio_path": relative_path,
            "audio_sha256": row["original_sha256"],
            "verbatim_transcript": row["first_ref"],
            "model_hypothesis": row["original_transcript"],
            "language_hint": row["language_hint"],
            "source_model": row["source_model"],
            "approval": "two_reviewers_agree_and_local_training_authorized",
            "allowed_use": "local_pilot_only",
            "split": "unassigned_require_speaker_disjoint_split",
        })
    return results


def manifest_jsonl(items: list[dict]) -> str:
    return "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in items)
