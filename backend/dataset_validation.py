"""Phase 3, Step 1: read-only dataset checks and reproducible speaker splits.

This module never trains a model, uploads data, writes splits to disk, or
establishes a real speaker's identity. The separate Step 7 consent gate remains
mandatory upstream; labels here are operator-supplied *pseudonyms* only.
"""
from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from pathlib import Path
import re
import wave

MIN_PILOT_SPEAKERS = 3
MAX_PILOT_CANDIDATES = 500
SPEAKER_CODE_RE = re.compile(r"^spk-[A-Za-z0-9_-]{1,24}$")


def _notice(code: str, message: str, evidence_ids: list[str] | None = None) -> dict:
    result = {"code": code, "message": message}
    if evidence_ids:
        result["evidence_ids"] = sorted(evidence_ids)
    return result


def _wave_properties(path: Path) -> tuple[dict, str]:
    """Inspect uncompressed PCM WAV, and fingerprint decoded PCM independently of WAV headers."""
    try:
        with wave.open(str(path), "rb") as audio:
            if audio.getcomptype() != "NONE":
                raise ValueError("Compressed WAV isn't supported")
            rate = audio.getframerate()
            channels = audio.getnchannels()
            width = audio.getsampwidth()
            frames = audio.getnframes()
            if rate <= 0 or frames <= 0:
                raise ValueError("Empty or invalid WAV")
            pcm = audio.readframes(frames)
            if len(pcm) != frames * channels * width:
                raise ValueError("Truncated PCM WAV")
        pcm_hash = hashlib.sha256(
            f"{rate}|{channels}|{width}|".encode("ascii") + pcm
        ).hexdigest()
        return {
            "duration_seconds": frames / rate,
            "sample_rate": rate,
            "channels": channels,
            "sample_width_bytes": width,
        }, pcm_hash
    except (OSError, EOFError, wave.Error, ValueError) as exc:
        raise ValueError("Invalid/unreadable PCM WAV") from exc


def _speaker_splits(codes: set[str], seed: int) -> dict[str, str]:
    """Deterministic by seed and complete set of speaker codes; speakers never straddle splits."""
    order = sorted(codes, key=lambda c: (hashlib.sha256(f"{seed}|{c}".encode()).hexdigest(), c))
    count = len(order)
    # This method is only used with >=3 speakers.
    n_val = max(1, round(count * .15))
    n_test = max(1, round(count * .15))
    if n_val + n_test >= count:
        n_val = 1
        n_test = 1
    n_train = count - n_val - n_test
    labels = ["train"] * n_train + ["validation"] * n_val + ["test"] * n_test
    return dict(zip(order, labels))


def validate_candidates(
    *, candidates: list[dict], speaker_assignments: list[dict],
    project_root: Path, seed: int = 42,
) -> dict:
    """Validate Step 7 candidates and create an in-memory split *plan*.

    Returns structured blocking issues and nonblocking warnings. Export MUST
    verify this report is ready. No speaker inference or audio modification.
    """
    if not isinstance(seed, int) or isinstance(seed, bool) or not (0 <= seed <= 4294967295):
        raise ValueError("Seed must be a 32-bit nonnegative integer")
    issues: list[dict] = []
    warnings: list[dict] = []
    if not candidates:
        issues.append(_notice("no_candidates", "No Step 7 authorized recordings are available"))
    if len(candidates) > MAX_PILOT_CANDIDATES:
        issues.append(_notice("too_many_candidates", "Local pilot supports at most 500 clips per validation"))
    ids = [item.get("evidence_id") for item in candidates]
    if len(ids) != len(set(ids)):
        issues.append(_notice("duplicate_candidate_id", "Manifest contains repeated evidence identifiers"))
    assigned: dict[str, str] = {}
    for row in speaker_assignments:
        eid = row.get("evidence_id")
        code = row.get("speaker_code")
        if not isinstance(eid, str) or not eid.strip():
            issues.append(_notice("bad_evidence_id", "Each assignment needs an evidence identifier"))
            continue
        if not isinstance(code, str) or not SPEAKER_CODE_RE.fullmatch(code):
            issues.append(_notice("bad_speaker_code", "Use pseudonyms such as spk-01; never names or contact details", [eid]))
            continue
        if eid in assigned:
            issues.append(_notice("repeated_assignment", "A recording was assigned more than once", [eid]))
        assigned[eid] = code
    missing = sorted(set(ids) - set(assigned))
    extra = sorted(set(assigned) - set(ids))
    if missing:
        issues.append(_notice("missing_speaker", "Assign one pseudonymous speaker code to every clip", missing))
    if extra:
        issues.append(_notice("unknown_recording", "Assignments refer to clips not currently authorized", extra))
    speaker_codes = set(assigned[eid] for eid in ids if eid in assigned)
    if len(speaker_codes) < MIN_PILOT_SPEAKERS:
        issues.append(_notice("insufficient_speakers", "At least 3 distinct known speakers are required for a nonempty speaker-disjoint train/validation/test split"))
    if len(speaker_codes) < 5:
        warnings.append(_notice("small_speaker_sample", "Fewer than 5 speakers: this is only a split demonstration, not credible ASR evaluation"))
    if len(candidates) < 30:
        warnings.append(_notice("small_clip_sample", "Fewer than 30 clips: do not interpret these splits as a reliable model benchmark"))

    checksums: dict[str, list[str]] = defaultdict(list)
    pcm_fingerprints: dict[str, list[str]] = defaultdict(list)
    texts: dict[tuple[str, str], list[str]] = defaultdict(list)
    audio_properties = {}
    root = project_root.resolve()
    for row in candidates:
        eid = row.get("evidence_id")
        if not isinstance(eid, str):
            continue
        if row.get("approval") != "two_reviewers_agree_and_local_training_authorized" or row.get("allowed_use") != "local_pilot_only":
            issues.append(_notice("not_authorized", "Step 7 authorization is missing", [eid]))
        path_value = row.get("audio_path", "")
        try:
            relative = Path(path_value)
            path = (root / relative).resolve()
            if relative.is_absolute() or root not in path.parents:
                raise ValueError("path escaped project root")
            # Keep this tool limited to Step 5 original audio, not a reply.
            if "recovery_evidence" not in relative.parts or "-original" not in relative.stem:
                raise ValueError("not an original evidence clip")
            content = path.read_bytes()
        except (OSError, ValueError, TypeError):
            issues.append(_notice("missing_or_unsafe_audio", "Original recording is missing or outside the approved local evidence store", [eid]))
            continue
        calculated = hashlib.sha256(content).hexdigest()
        if calculated != row.get("audio_sha256"):
            issues.append(_notice("checksum_mismatch", "Original audio fingerprint changed", [eid]))
            continue
        checksums[calculated].append(eid)
        if path.suffix.lower() != ".wav":
            issues.append(_notice("wav_required", "Pilot manifest requires original PCM WAV; no conversion is performed", [eid]))
            continue
        try:
            props, pcm_hash = _wave_properties(path)
        except ValueError:
            issues.append(_notice("invalid_wav", "Recording is not a readable PCM WAV", [eid]))
            continue
        audio_properties[eid] = props
        pcm_fingerprints[pcm_hash].append(eid)
        if props["sample_rate"] != 16000 or props["channels"] != 1 or props["sample_width_bytes"] != 2:
            issues.append(_notice("audio_format", "Pilot export requires 16 kHz, mono, 16-bit PCM WAV; convert and re-verify in a future step", [eid]))
        if props["duration_seconds"] < .25 or props["duration_seconds"] > 30.0:
            issues.append(_notice("audio_duration", "Pilot clips must be between 0.25 and 30 seconds", [eid]))
        transcript = row.get("verbatim_transcript", "")
        if not isinstance(transcript, str) or not transcript.strip():
            issues.append(_notice("missing_transcript", "Human-verified transcript is empty", [eid]))
        elif len(transcript) > 2000:
            issues.append(_notice("long_transcript", "Transcript is unexpectedly long", [eid]))
        if eid in assigned and isinstance(transcript, str):
            texts[(assigned[eid], " ".join(transcript.casefold().split()))].append(eid)

    for group in checksums.values():
        if len(group) > 1:
            issues.append(_notice("duplicate_audio", "Identical original audio appears in multiple records", group))
    for group in pcm_fingerprints.values():
        if len(group) > 1:
            issues.append(_notice("duplicate_pcm", "Identical PCM samples appear in multiple WAV containers", group))
    for (speaker, _), group in texts.items():
        if len(group) > 1:
            warnings.append(_notice("repeated_text_same_speaker", "Same speaker code has multiple clips with the same reference text; check for near-duplicates", group))

    split_for_speaker = _speaker_splits(speaker_codes, seed) if len(speaker_codes) >= MIN_PILOT_SPEAKERS else {}
    plan = []
    for row in sorted(candidates, key=lambda item: item.get("evidence_id", "")):
        eid = row.get("evidence_id")
        code = assigned.get(eid)
        if code and code in split_for_speaker:
            plan.append({"evidence_id": eid, "speaker_code": code, "split": split_for_speaker[code]})
    split_counts = dict(Counter(item["split"] for item in plan))
    speakers_per_split = dict(Counter(split_for_speaker.values()))
    return {
        "ready_for_local_split": not issues,
        "training_ready": False,
        "scope": "local_pilot_only",
        "seed": seed,
        "clip_count": len(candidates),
        "speaker_count": len(speaker_codes),
        "split_clip_counts": {s: split_counts.get(s, 0) for s in ("train", "validation", "test")},
        "split_speaker_counts": {s: speakers_per_split.get(s, 0) for s in ("train", "validation", "test")},
        "issues": issues,
        "warnings": warnings,
        "plan": plan,
        "note": "Speaker codes are self-reported pseudonyms, not identity proof; a tiny pilot split is not an ASR evaluation.",
    }


def split_manifest(candidates: list[dict], report: dict) -> list[dict]:
    """Build a reproducible JSONL-ready manifest after all current checks pass."""
    if not report.get("ready_for_local_split"):
        raise ValueError("Dataset validation did not pass; cannot export speaker splits")
    mapping = {item["evidence_id"]: item for item in report["plan"]}
    result = []
    for row in candidates:
        plan = mapping[row["evidence_id"]]
        item = dict(row)
        item.update({
            "split": plan["split"],
            "speaker_group": plan["speaker_code"],
            "split_seed": report["seed"],
            "split_protocol": "speaker_disjoint_local_pilot_v1",
            "training_ready": False,
        })
        result.append(item)
    return sorted(result, key=lambda row: (row["split"], row["evidence_id"]))
