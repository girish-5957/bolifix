"""Step 8: read-only dataset validation and reproducible local speaker splits."""
import hashlib
import io
import json
import sqlite3
import struct
import wave
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend import dataset_validation as splits
from backend import main


def pcm_wav(marker=1, frames=12000, rate=16000, channels=1, width=2):
    out = io.BytesIO()
    with wave.open(out, "wb") as f:
        f.setnchannels(channels)
        f.setsampwidth(width)
        f.setframerate(rate)
        f.writeframes(bytes([marker % 250 + 1]) * frames * channels * width)
    return out.getvalue()


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "ROOT", tmp_path)
    monkeypatch.setattr(main, "DB_PATH", tmp_path / "storage" / "bolifix.sqlite3")
    monkeypatch.setattr(main, "AUDIO_DIR", tmp_path / "storage" / "recordings")
    monkeypatch.setattr(main, "MOCK_STT", True)
    with TestClient(main.app) as cli:
        yield cli


def approved(client, marker=1, *, orig=None, filename="original.wav"):
    raw = orig if orig is not None else pcm_wav(marker)
    r = client.post("/api/v1/recovery-evidence", data={
        "app_id": "demo-app", "original_transcript": "मुझे पाँच पैकेट चाहिए",
        "first_reply_transcript": "पंद्रह", "confirmed_intent": "15",
        "source_model": "prisma-v3.0", "language_hint": "hi-IN",
        "speaker_confirmed_intent": "true", "explicit_storage_consent": "true",
        "original_mock": "false", "first_reply_mock": "false",
    }, files={
        "original_audio": (filename, raw),
        "first_reply_audio": ("reply.wav", pcm_wav(70)),
    })
    assert r.status_code == 201, r.text
    eid = r.json()["id"]
    r = client.post("/api/v1/asr-reviews", json={
        "evidence_id": eid, "reviewer_code": "reviewer-1", "verbatim_reference": "मुझे पंद्रह पैकेट चाहिए",
        "error_category": "number_or_amount", "reviewer_note": "",
        "confirmed_listened": True, "confirmed_verbatim": True,
    })
    assert r.status_code == 201, r.text
    r = client.post("/api/v1/asr-second-reviews", json={
        "evidence_id": eid, "reviewer_code": "reviewer-2", "verbatim_reference": "मुझे पंद्रह पैकेट चाहिए",
        "reviewer_note": "", "confirmed_listened": True, "confirmed_verbatim": True,
    })
    assert r.status_code == 201, r.text
    r = client.post("/api/v1/asr-local-permissions", json={
        "evidence_id": eid, "attested_by": "local-operator", "speaker_authorized": True,
        "privacy_screened": True, "local_training_authorized": True,
    })
    assert r.status_code == 201, r.text
    return eid


def body(*ids, speaker_codes=None, seed=42, ack=True):
    codes = speaker_codes or [f"spk-{i+1:02}" for i in range(len(ids))]
    return {"speaker_assignments": [
        {"evidence_id": eid, "speaker_code": code} for eid, code in zip(ids, codes)
    ], "seed": seed, "confirm_local_pilot_only": ack}


def call(client, route, payload):
    result = client.post(f"/api/v1/{route}", json=payload)
    return result


def group(client, count=3):
    return [approved(client, marker=i + 1) for i in range(count)]


def issue_codes(report):
    return {e["code"] for e in report["issues"]}


def test_empty_candidates_returns_explanation_without_audio_or_model_request(client):
    r = call(client, "asr-validate-split", body())
    assert r.status_code == 200
    assert {"no_candidates", "insufficient_speakers"} <= issue_codes(r.json())
    assert r.json()["training_ready"] is False
    assert call(client, "asr-split-manifest", body()).status_code == 409


def test_candidates_endpoint_exposes_metadata_only(client):
    identity = approved(client)
    r = client.get("/api/v1/asr-split-candidates")
    assert r.status_code == 200
    assert r.json()["items"] == [{"evidence_id": identity, "language_hint": "hi-IN"}]
    assert "verbatim_transcript" not in r.text
    assert r.json()["training_ready"] is False


def test_one_speaker_blocked_and_warnings_visible(client):
    eid = approved(client)
    r = call(client, "asr-validate-split", body(eid))
    assert r.status_code == 200
    assert "insufficient_speakers" in issue_codes(r.json())
    assert "small_clip_sample" in {i["code"] for i in r.json()["warnings"]}
    assert call(client, "asr-split-manifest", body(eid)).status_code == 409


def test_three_speakers_disjoint_reproducible_manifest(client, tmp_path):
    ids = group(client)
    payload = body(*ids)
    a = call(client, "asr-validate-split", payload)
    b = call(client, "asr-validate-split", payload)
    assert a.status_code == 200
    assert a.json() == b.json()
    report = a.json()
    assert report["ready_for_local_split"] is True
    assert report["split_speaker_counts"] == {"train": 1, "validation": 1, "test": 1}
    assert report["split_clip_counts"] == {"train": 1, "validation": 1, "test": 1}
    assert report["training_ready"] is False
    before = set(tmp_path.rglob("*.jsonl"))
    result = call(client, "asr-split-manifest", payload)
    assert result.status_code == 200, result.text
    assert "no-store" in result.headers["cache-control"]
    assert set(tmp_path.rglob("*.jsonl")) == before
    rows = [json.loads(line) for line in result.text.splitlines()]
    assert {row["split"] for row in rows} == {"train", "validation", "test"}
    assert len({row["speaker_group"] for row in rows}) == 3
    assert all(row["training_ready"] is False and row["split_seed"] == 42 for row in rows)
    assert all(row["audio_path"].endswith("-original.wav") for row in rows)
    assert "RIFF" not in result.text


def test_multiple_clips_same_speaker_in_same_split(client):
    ids = group(client, count=4)
    payload = body(*ids, speaker_codes=["spk-01", "spk-01", "spk-02", "spk-03"])
    result = call(client, "asr-split-manifest", payload)
    assert result.status_code == 200, result.text
    entries = [json.loads(line) for line in result.text.splitlines()]
    by_id = {r["evidence_id"]: r for r in entries}
    assert by_id[ids[0]]["split"] == by_id[ids[1]]["split"]
    assert len({r["speaker_group"] for r in entries}) == 3


def test_assignment_order_does_not_change_split(client):
    ids = group(client)
    obj = body(*ids)
    r1 = call(client, "asr-validate-split", obj).json()
    obj["speaker_assignments"].reverse()
    r2 = call(client, "asr-validate-split", obj).json()
    assert r1["plan"] == r2["plan"]


@pytest.mark.parametrize("route", ["asr-validate-split", "asr-split-manifest"])
def test_local_only_ack_required(client, route):
    ids = group(client)
    r = call(client, route, body(*ids, ack=False))
    assert r.status_code == 400


def test_bad_speaker_code_is_rejected(client):
    eid = approved(client)
    r = call(client, "asr-validate-split", body(eid, speaker_codes=["person@example.com"]))
    assert r.status_code == 422


def test_missing_speaker_code_blocks_export(client):
    ids = group(client)
    payload = body(*ids[:2])
    report = call(client, "asr-validate-split", payload).json()
    assert "missing_speaker" in issue_codes(report)
    assert call(client, "asr-split-manifest", payload).status_code == 409


def test_unknown_id_blocks_export(client):
    ids = group(client)
    payload = body(*ids, "unknown")
    report = call(client, "asr-validate-split", payload).json()
    assert "unknown_recording" in issue_codes(report)


def test_repeated_assignment_blocks_export(client):
    ids = group(client)
    payload = body(*ids)
    payload["speaker_assignments"].append(payload["speaker_assignments"][0].copy())
    report = call(client, "asr-validate-split", payload).json()
    assert "repeated_assignment" in issue_codes(report)
    assert call(client, "asr-split-manifest", payload).status_code == 409


def test_two_clips_with_identical_bytes_are_duplicate(client):
    common = pcm_wav(7)
    ids = [approved(client, marker=i, orig=(common if i < 2 else pcm_wav(8))) for i in range(3)]
    report = call(client, "asr-validate-split", body(*ids)).json()
    assert "duplicate_audio" in issue_codes(report)
    assert call(client, "asr-split-manifest", body(*ids)).status_code == 409


def test_identical_pcm_with_different_wave_headers_is_duplicate(client):
    original = pcm_wav(12)
    # Append a harmless RIFF chunk; WAV bytes differ but audio samples are identical.
    extra = b"JUNK" + struct.pack("<I", 4) + b"test"
    changed = bytearray(original)
    struct.pack_into("<I", changed, 4, struct.unpack_from("<I", original, 4)[0] + len(extra))
    changed.extend(extra)
    assert original != changed
    ids = [approved(client, orig=original), approved(client, orig=bytes(changed)), approved(client, orig=pcm_wav(13))]
    report = call(client, "asr-validate-split", body(*ids)).json()
    assert "duplicate_pcm" in issue_codes(report)


@pytest.mark.parametrize("file,expected", [
    (pcm_wav(10, channels=2), "audio_format"),
    (pcm_wav(10, rate=8000), "audio_format"),
    (pcm_wav(10, width=1), "audio_format"),
    (pcm_wav(10, frames=1000), "audio_duration"),
])
def test_audio_specs_block_export(client, file, expected):
    ids = [approved(client, marker=1, orig=file), approved(client, marker=2), approved(client, marker=3)]
    report = call(client, "asr-validate-split", body(*ids)).json()
    assert expected in issue_codes(report)
    assert report["ready_for_local_split"] is False


def test_corrupted_audio_fails_closed_upstream(client, tmp_path):
    ids = group(client)
    original = next((tmp_path / "storage" / "recordings" / "recovery_evidence").glob("*-original.wav"))
    original.write_bytes(b"not the authorized audio")
    assert call(client, "asr-validate-split", body(*ids)).status_code == 409
    assert call(client, "asr-split-manifest", body(*ids)).status_code == 409


def test_revoking_permission_removes_record_before_new_export(client):
    ids = group(client)
    payload = body(*ids)
    assert call(client, "asr-split-manifest", payload).status_code == 200
    assert client.delete(f"/api/v1/asr-local-permissions/{ids[0]}").status_code == 200
    fresh = client.get("/api/v1/asr-split-candidates").json()
    assert fresh["count"] == 2
    report = call(client, "asr-validate-split", payload).json()
    assert "unknown_recording" in issue_codes(report)
    assert call(client, "asr-split-manifest", payload).status_code == 409


def test_direct_bad_audio_path_cannot_escape_project_root(tmp_path):
    outside = tmp_path.parent / "outside.wav"
    outside.write_bytes(pcm_wav())
    item = {
        "evidence_id": "demo", "audio_path": "../outside.wav",
        "audio_sha256": hashlib.sha256(outside.read_bytes()).hexdigest(),
        "verbatim_transcript": "hello", "approval": "two_reviewers_agree_and_local_training_authorized",
        "allowed_use": "local_pilot_only",
    }
    r = splits.validate_candidates(candidates=[item], speaker_assignments=[{"evidence_id": "demo", "speaker_code": "spk-01"}], project_root=tmp_path)
    assert "missing_or_unsafe_audio" in issue_codes(r)


def test_manual_speaker_aliases_are_not_a_proof_of_identity(client):
    ids = group(client)
    report = call(client, "asr-validate-split", body(*ids)).json()
    assert "not identity proof" in report["note"]
    assert report["training_ready"] is False
