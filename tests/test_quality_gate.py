"""Step 7: independent review, per-record local-training permission, and safe export.

No Gnani requests, no public sharing, no model training.
"""
import hashlib
import io
import json
import sqlite3
import wave

import pytest
from fastapi.testclient import TestClient

from backend import main
from backend import quality_gate


def wav_bytes(frames=12000):
    memory = io.BytesIO()
    with wave.open(memory, "wb") as recording:
        recording.setnchannels(1)
        recording.setsampwidth(2)
        recording.setframerate(16000)
        recording.writeframes(b"\0\0" * frames)
    return memory.getvalue()


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "ROOT", tmp_path)
    monkeypatch.setattr(main, "DB_PATH", tmp_path / "storage" / "bolifix.sqlite3")
    monkeypatch.setattr(main, "AUDIO_DIR", tmp_path / "storage" / "recordings")
    monkeypatch.setattr(main, "MOCK_STT", True)
    with TestClient(main.app) as cli:
        yield cli


def evidence(client):
    response = client.post("/api/v1/recovery-evidence", data={
        "app_id": "demo-app", "original_transcript": "मुझे पाँच पैकेट चाहिए",
        "first_reply_transcript": "पंद्रह", "confirmed_intent": "15",
        "source_model": "prisma-v3.0", "language_hint": "hi-IN",
        "speaker_confirmed_intent": "true", "explicit_storage_consent": "true",
        "original_mock": "false", "first_reply_mock": "false",
    }, files={
        "original_audio": ("original.wav", wav_bytes(12000)),
        "first_reply_audio": ("reply.wav", wav_bytes(6000)),
    })
    assert response.status_code == 201, response.text
    return response.json()["id"]


def first_review(client, identifier, **kwargs):
    payload = {
        "evidence_id": identifier, "reviewer_code": "reviewer-one",
        "verbatim_reference": "मुझे पंद्रह पैकेट चाहिए",
        "error_category": "number_or_amount", "reviewer_note": "",
        "confirmed_listened": True, "confirmed_verbatim": True,
    }
    payload.update(kwargs)
    return client.post("/api/v1/asr-reviews", json=payload)


def second_review(client, identifier, **kwargs):
    payload = {
        "evidence_id": identifier, "reviewer_code": "reviewer-two",
        "verbatim_reference": "मुझे पंद्रह पैकेट चाहिए",
        "reviewer_note": "Listened independently to original audio",
        "confirmed_listened": True, "confirmed_verbatim": True,
    }
    payload.update(kwargs)
    return client.post("/api/v1/asr-second-reviews", json=payload)


def permission(client, identifier, **kwargs):
    payload = {
        "evidence_id": identifier, "attested_by": "local-operator",
        "speaker_authorized": True, "privacy_screened": True,
        "local_training_authorized": True,
    }
    payload.update(kwargs)
    return client.post("/api/v1/asr-local-permissions", json=payload)


def manifest(client, agreed=True):
    return client.post("/api/v1/asr-local-dataset-manifest", json={
        "confirm_local_pilot_only": agreed,
    })


def test_absent_first_review_blocks_second(client):
    identity = evidence(client)
    r = second_review(client, identity)
    assert r.status_code == 404
    assert "First ASR review" in r.text


def test_same_reviewer_alias_is_rejected_case_insensitively(client):
    identity = evidence(client)
    assert first_review(client, identity).status_code == 201
    r = second_review(client, identity, reviewer_code="REVIEWER-ONE")
    assert r.status_code == 400
    assert "DIFFERENT" in r.text
    assert client.get("/api/v1/asr-second-reviews").json()["items"] == []


def test_original_audio_listening_is_required(client):
    identity = evidence(client)
    assert first_review(client, identity).status_code == 201
    r = second_review(client, identity, confirmed_listened=False)
    assert r.status_code == 400
    assert "ORIGINAL" in r.text


def test_first_uncertain_review_blocks_second(client):
    identity = evidence(client)
    assert first_review(
        client, identity, error_category="uncertain", confirmed_verbatim=False
    ).status_code == 201
    r = second_review(client, identity)
    assert r.status_code == 400
    assert "uncertain" in r.text


def test_second_uncertain_blocks_permission(client):
    identity = evidence(client)
    assert first_review(client, identity).status_code == 201
    r = second_review(client, identity, verbatim_reference="", confirmed_verbatim=False)
    assert r.status_code == 201
    assert r.json()["status"] == "uncertain"
    assert permission(client, identity).status_code == 400
    assert manifest(client).status_code == 409


def test_uncertain_second_review_must_leave_text_blank(client):
    identity = evidence(client)
    assert first_review(client, identity).status_code == 201
    assert second_review(client, identity, confirmed_verbatim=False).status_code == 400


def test_mismatch_never_exports_even_with_training_permission(client):
    identity = evidence(client)
    assert first_review(client, identity).status_code == 201
    r = second_review(client, identity, verbatim_reference="मुझे पच्चीस पैकेट चाहिए")
    assert r.status_code == 201
    assert r.json()["status"] == "disagreed"
    assert permission(client, identity).status_code == 400
    assert manifest(client).status_code == 409


def test_matching_review_is_not_training_approval(client):
    identity = evidence(client)
    assert first_review(client, identity).status_code == 201
    r = second_review(client, identity, verbatim_reference="मुझे पंद्रह पैकेट चाहिए!")
    assert r.status_code == 201
    assert r.json()["status"] == "agreed"
    assert r.json()["eligible_for_local_pilot"] is False
    assert r.json()["training_ready"] is False
    assert manifest(client).status_code == 409
    assert client.get("/api/v1/asr-local-permissions").json()["items"] == []


@pytest.mark.parametrize("flag", ["speaker_authorized", "privacy_screened", "local_training_authorized"])
def test_each_permission_attestation_is_mandatory(client, flag):
    identity = evidence(client)
    assert first_review(client, identity).status_code == 201
    assert second_review(client, identity).status_code == 201
    assert permission(client, identity, **{flag: False}).status_code == 400
    assert manifest(client).status_code == 409


def test_manifest_is_private_in_memory_index_of_original_audio_only(client, tmp_path):
    identity = evidence(client)
    assert first_review(client, identity).status_code == 201
    assert second_review(client, identity).status_code == 201
    saved = permission(client, identity)
    assert saved.status_code == 201, saved.text
    assert saved.json()["scope"] == "local_model_training_only"
    assert saved.json()["training_ready"] is False
    assert manifest(client, agreed=False).status_code == 400
    r = manifest(client)
    assert r.status_code == 200, r.text
    assert r.headers["cache-control"] == "no-store"
    assert "application/x-ndjson" in r.headers["content-type"]
    rows = [json.loads(line) for line in r.text.splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["evidence_id"] == identity
    assert row["verbatim_transcript"] == "मुझे पंद्रह पैकेट चाहिए"
    assert row["model_hypothesis"] == "मुझे पाँच पैकेट चाहिए"
    assert row["allowed_use"] == "local_pilot_only"
    assert row["split"].startswith("unassigned")
    assert row["audio_path"].startswith("storage/recordings/recovery_evidence/")
    assert row["audio_path"].endswith("-original.wav")
    assert row["audio_sha256"] == hashlib.sha256(wav_bytes(12000)).hexdigest()
    assert (tmp_path / row["audio_path"]).read_bytes() == wav_bytes(12000)
    assert not (tmp_path / "storage" / "datasets").exists()


def test_permission_can_be_revoked(client):
    identity = evidence(client)
    assert first_review(client, identity).status_code == 201
    assert second_review(client, identity).status_code == 201
    assert permission(client, identity).status_code == 201
    assert manifest(client).status_code == 200
    revoked = client.delete(f"/api/v1/asr-local-permissions/{identity}")
    assert revoked.status_code == 200
    assert revoked.json()["revoked"] is True
    assert manifest(client).status_code == 409
    assert client.get("/api/v1/asr-local-permissions").json()["items"] == []


def test_deleting_second_review_cascades_permission(client):
    identity = evidence(client)
    assert first_review(client, identity).status_code == 201
    assert second_review(client, identity).status_code == 201
    assert permission(client, identity).status_code == 201
    assert client.delete(f"/api/v1/asr-second-reviews/{identity}").status_code == 200
    assert manifest(client).status_code == 409
    assert client.get("/api/v1/asr-local-permissions").json()["items"] == []


def test_deleting_first_review_cascades_second_and_permission(client):
    identity = evidence(client)
    assert first_review(client, identity).status_code == 201
    assert second_review(client, identity).status_code == 201
    assert permission(client, identity).status_code == 201
    assert client.delete(f"/api/v1/asr-reviews/{identity}").status_code == 200
    assert client.get("/api/v1/asr-second-reviews").json()["items"] == []
    assert client.get("/api/v1/asr-local-permissions").json()["items"] == []
    assert manifest(client).status_code == 409


def test_deleting_parent_evidence_cascades_everything_and_removes_audio(client, tmp_path):
    identity = evidence(client)
    assert first_review(client, identity).status_code == 201
    assert second_review(client, identity).status_code == 201
    assert permission(client, identity).status_code == 201
    assert client.delete(f"/api/v1/recovery-evidence/{identity}").status_code == 200
    assert client.get("/api/v1/asr-second-reviews").json()["items"] == []
    assert client.get("/api/v1/asr-local-permissions").json()["items"] == []
    assert not list((tmp_path / "storage" / "recordings" / "recovery_evidence").glob("*.wav"))
    assert manifest(client).status_code == 409


def test_integrity_check_before_second_review(client, tmp_path):
    identity = evidence(client)
    assert first_review(client, identity).status_code == 201
    original = next((tmp_path / "storage" / "recordings" / "recovery_evidence").glob("*-original.wav"))
    original.write_bytes(b"altered audio")
    assert second_review(client, identity).status_code == 400


def test_integrity_check_before_manifest_export(client, tmp_path):
    identity = evidence(client)
    assert first_review(client, identity).status_code == 201
    assert second_review(client, identity).status_code == 201
    assert permission(client, identity).status_code == 201
    original = next((tmp_path / "storage" / "recordings" / "recovery_evidence").glob("*-original.wav"))
    original.write_bytes(b"altered audio")
    r = manifest(client)
    assert r.status_code == 409
    assert "checksum" in r.text.lower()


def test_duplicate_independent_and_permission_records_rejected(client):
    identity = evidence(client)
    assert first_review(client, identity).status_code == 201
    assert second_review(client, identity).status_code == 201
    assert second_review(client, identity).status_code == 409
    assert permission(client, identity).status_code == 201
    assert permission(client, identity).status_code == 409


def test_review_alias_not_free_form_html_or_personal_data(client):
    identity = evidence(client)
    assert first_review(client, identity).status_code == 201
    assert second_review(client, identity, reviewer_code="<script>").status_code == 400


def test_cannot_escape_audio_directory_with_tampered_database_path(client, tmp_path):
    identity = evidence(client)
    assert first_review(client, identity).status_code == 201
    with sqlite3.connect(main.DB_PATH) as conn:
        conn.execute("UPDATE recovery_evidence SET original_audio_path = ? WHERE id = ?",
                     (str(tmp_path / "outside.wav"), identity))
    assert second_review(client, identity).status_code == 400
