"""Phase 3 Step 2: private, read-only evaluation of a *held-out* ASR baseline.

The speaker-split manifest is untrusted. Every attempt rechecks *current* Step 7
permission, reviews, audio checksums and Step 1 speaker-disjoint split decisions.
No Gnani calls, model training, uploads, or cloud audio inference here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

from backend import dataset_validation, quality_gate
from backend.asr_review import compare_transcripts, normalize_tokens, edit_distance

MAX_MANIFEST_BYTES = 2_000_000
MAX_CLIPS = 500


def load_manifest(path: Path) -> list[dict]:
    """Read a small Step 1 JSONL manifest; reject duplicate/invalid entries."""
    if not path.is_file() or path.stat().st_size > MAX_MANIFEST_BYTES:
        raise ValueError("Manifest missing or too large (2 MB maximum)")
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except UnicodeError as exc:
        raise ValueError("Manifest must be UTF-8 JSONL") from exc
    if not lines or len(lines) > MAX_CLIPS or any(not line.strip() for line in lines):
        raise ValueError("Manifest must contain 1–500 nonblank JSONL records")
    try:
        rows = [json.loads(line) for line in lines]
    except json.JSONDecodeError as exc:
        raise ValueError("Manifest contains invalid JSON") from exc
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("Each JSONL line must be a JSON object")
    ids = [row.get("evidence_id") for row in rows]
    if any(not isinstance(eid, str) or not eid for eid in ids) or len(set(ids)) != len(ids):
        raise ValueError("Manifest contains missing or duplicate recording IDs")
    return rows


def validate_manifest(
    *, rows: list[dict], project_root: Path, db_path: Path, audio_dir: Path,
) -> dict:
    """Rebuild current approved rows and splits; never trust a downloaded manifest alone."""
    if not rows or len(rows) > MAX_CLIPS:
        raise ValueError("Provide 1–500 manifest records")
    if any(not isinstance(row, dict) or not isinstance(row.get("evidence_id"), str)
           for row in rows):
        raise ValueError("Each manifest row must have a recording ID")
    seed_values = [row.get("split_seed") for row in rows]
    if any(not isinstance(value, int) or isinstance(value, bool) or
           not 0 <= value <= 4294967295 for value in seed_values):
        raise ValueError("Invalid split seed")
    if len(set(seed_values)) != 1:
        raise ValueError("All records must share one frozen split seed")
    seed = seed_values[0]
    current = quality_gate.candidate_manifest(
        db_path=db_path, audio_dir=audio_dir, project_root=project_root
    )
    submitted = {row.get("evidence_id") for row in rows}
    expected_ids = {row["evidence_id"] for row in current}
    if submitted != expected_ids or len(rows) != len(expected_ids):
        raise ValueError("Manifest is outdated or incomplete; regenerate Step 1 export after permission changes")
    assignments = [
        {"evidence_id": row.get("evidence_id"), "speaker_code": row.get("speaker_group")}
        for row in rows
    ]
    report = dataset_validation.validate_candidates(
        candidates=current, speaker_assignments=assignments,
        project_root=project_root, seed=seed,
    )
    if not report["ready_for_local_split"]:
        codes = ", ".join(sorted({item["code"] for item in report["issues"]}))
        raise ValueError(f"Current local dataset fails speaker/audio checks: {codes}")
    expected = {
        row["evidence_id"]: row
        for row in dataset_validation.split_manifest(current, report)
    }
    for row in rows:
        true = expected[row["evidence_id"]]
        # Includes original transcript, permission markers, audio fingerprint,
        # speaker code, split protocol and split name. Extra fields are denied.
        if row != true:
            raise ValueError("Manifest was changed or is stale; download a fresh Step 1 split manifest")
    tests = sorted((row for row in rows if row["split"] == "test"), key=lambda r: r["evidence_id"])
    if not tests:
        raise ValueError("No held-out test recordings found")
    return {
        "test_rows": tests,
        "clip_count": len(rows),
        "test_clip_count": len(tests),
        "speaker_count": report["speaker_count"],
        "test_speaker_count": report["split_speaker_counts"]["test"],
        "seed": seed,
        "warnings": report["warnings"],
        "manifest_sha256": hashlib.sha256(
            quality_gate.manifest_jsonl(sorted(rows, key=lambda r: (r["split"], r["evidence_id"]))).encode("utf-8")
        ).hexdigest(),
        "note": "Small pilot: results describe these clips only, not general ASR accuracy.",
    }


def evaluate_transcripts(test_rows: list[dict], predictions: dict[str, str]) -> dict:
    """Micro-averaged WER/CER with no rounding before accumulation."""
    ids = {row["evidence_id"] for row in test_rows}
    if set(predictions) != ids or len(ids) != len(test_rows):
        raise ValueError("Predictions must match the held-out test recording IDs exactly")
    totals = {"substitutions": 0, "deletions": 0, "insertions": 0, "reference_words": 0,
              "character_edits": 0, "reference_chars": 0}
    results = []
    for row in test_rows:
        eid = row["evidence_id"]
        hypothesis = predictions[eid]
        if not isinstance(hypothesis, str):
            raise ValueError("Every prediction must be text")
        reference = row["verbatim_transcript"]
        metrics = compare_transcripts(reference, hypothesis)
        normal_ref = "".join(normalize_tokens(reference))
        normal_pred = "".join(normalize_tokens(hypothesis))
        changes = edit_distance(normal_ref, normal_pred)
        for name in ("substitutions", "deletions", "insertions", "reference_words"):
            totals[name] += metrics[name]
        totals["character_edits"] += changes
        totals["reference_chars"] += len(normal_ref)
        results.append({"evidence_id": eid, "metrics": metrics})
    wer = ((totals["substitutions"] + totals["deletions"] + totals["insertions"])
           / totals["reference_words"])
    cer = totals["character_edits"] / totals["reference_chars"]
    return {
        "test_clip_count": len(results),
        "micro_wer": round(wer, 6), "micro_cer": round(cer, 6),
        "totals": totals, "per_clip": results,
        "training_performed": False,
        "interpretation": "Pilot scores only; do not compare models without the same frozen test set and normalization.",
    }


def predict_with_model(test_rows: list[dict], *, project_root: Path,
                       transcribe_one: Callable[[Path], str]) -> dict[str, str]:
    """Injectable inference adapter. Recheck exact original WAV hashes at use time."""
    root = project_root.resolve()
    out = {}
    for row in test_rows:
        relative = Path(row["audio_path"])
        audio = (root / relative).resolve()
        if relative.is_absolute() or root not in audio.parents:
            raise ValueError("Unsafe original-audio path")
        # Only the original WAV can be evaluated, never a clarification reply.
        if "recovery_evidence" not in relative.parts or not audio.stem.endswith("-original"):
            raise ValueError("Not an approved original-audio recording")
        if hashlib.sha256(audio.read_bytes()).hexdigest() != row["audio_sha256"]:
            raise ValueError("Original audio modified since manifest approval")
        result = transcribe_one(audio)
        if not isinstance(result, str):
            raise ValueError("The speech model must return text")
        out[row["evidence_id"]] = result.strip()
    return out
