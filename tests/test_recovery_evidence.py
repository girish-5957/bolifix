"""No Gnani calls: test private recovery traces and their consent boundaries."""
import io
import sqlite3
import wave
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from backend import main


def wav_bytes(frames=16000):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as sound:
        sound.setnchannels(1)
        sound.setsampwidth(2)
        sound.setframerate(16000)
        sound.writeframes(b"\0\0" * frames)
    return buffer.getvalue()


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "DB_PATH", tmp_path / "bolifix.sqlite3")
    monkeypatch.setattr(main, "AUDIO_DIR", tmp_path / "recordings")
    monkeypatch.setattr(main, "MOCK_STT", True)
    with TestClient(main.app) as cli:
        yield cli


def submit(client, *, second=False, **overrides):
    data = {
        "app_id": "test-app",
        "original_transcript": "मुझे पाँच नहीं पंद्रह चाहिए",
        "first_reply_transcript": "पाँच नहीं पंद्रह",
        "confirmed_intent": "15",
        "source_model": "prisma-v3.0",
        "language_hint": "hi-IN",
        "speaker_confirmed_intent": "true",
        "explicit_storage_consent": "true",
        "original_mock": "false",
        "first_reply_mock": "false",
        "second_reply_mock": "false",
        "second_reply_transcript": "15" if second else "",
    }
    data.update(overrides)
    files = {
        "original_audio": ("original.wav", wav_bytes()),
        "first_reply_audio": ("first.wav", wav_bytes(12000)),
    }
    if second:
        files["second_reply_audio"] = ("second.wav", wav_bytes(8000))
    return client.post("/api/v1/recovery-evidence", data=data, files=files)


def test_two_reply_files_are_preserved(client, tmp_path):
    resp = submit(client, second=True)
    assert resp.status_code == 201, resp.text
    saved = resp.json()
    assert saved["audio_count"] == 3
    assert saved["training_ready"] is False
    assert saved["evidence_kind"] == "speaker_confirmed_intent_only"
    items = client.get("/api/v1/recovery-evidence").json()["items"]
    assert len(items) == 1
    item = items[0]
    assert item["id"] == saved["id"]
    assert item["first_reply_transcript"] == "पाँच नहीं पंद्रह"
    assert item["second_reply_transcript"] == "15"
    assert item["confirmed_intent"] == "15"
    assert item["verification_kind"] == "speaker_confirmed_intent_only"
    assert item["review_status"] == "needs_independent_review"
    assert item["consent_scope"] == "private"
    assert "audio_path" not in item
    files = list((tmp_path / "recordings" / "recovery_evidence").glob("*.wav"))
    assert len(files) == 3
    assert all(p.stat().st_size > 40 for p in files)


def test_one_reply_trace_saves_two_files(client, tmp_path):
    resp = submit(client)
    assert resp.status_code == 201
    assert resp.json()["audio_count"] == 2
    assert len(list((tmp_path / "recordings" / "recovery_evidence").glob("*.wav"))) == 2
    item = client.get("/api/v1/recovery-evidence").json()["items"][0]
    assert item["second_reply_transcript"] is None


def test_retention_requires_separate_explicit_permission(client, tmp_path):
    assert submit(client, explicit_storage_consent="false").status_code == 400
    assert submit(client, speaker_confirmed_intent="false").status_code == 400
    assert client.get("/api/v1/recovery-evidence").json()["items"] == []
    assert not list((tmp_path / "recordings").rglob("*.wav"))


def test_reject_mock_recognition(client):
    assert submit(client, original_mock="true").status_code == 400
    assert submit(client, first_reply_mock="true").status_code == 400
    assert submit(client, source_model="MOCK - not Gnani").status_code == 400
    assert client.get("/api/v1/recovery-evidence").json()["items"] == []


def test_cannot_mismatch_second_reply_text_and_audio(client, tmp_path):
    response = submit(client, second_reply_transcript="20")
    assert response.status_code == 400
    response = submit(client, second=True, second_reply_transcript="")
    assert response.status_code == 400
    assert not list((tmp_path / "recordings").rglob("*.wav"))


def test_reject_bad_id_and_empty_intent(client):
    assert submit(client, app_id="bad/name").status_code == 400
    assert submit(client, confirmed_intent="  ").status_code == 400


def test_reject_invalid_audio_without_persisting(client, tmp_path):
    data = {
        "original_transcript": "original", "first_reply_transcript": "first",
        "confirmed_intent": "15", "speaker_confirmed_intent": "true",
        "explicit_storage_consent": "true",
    }
    resp = client.post("/api/v1/recovery-evidence", data=data, files={
        "original_audio": ("x.wav", b"not-a-wave"),
        "first_reply_audio": ("y.wav", wav_bytes()),
    })
    assert resp.status_code == 400
    assert client.get("/api/v1/recovery-evidence").json()["items"] == []
    assert not list((tmp_path / "recordings").rglob("*.wav"))


def test_delete_removes_all_audio_and_db_row(client, tmp_path):
    resp = submit(client, second=True)
    rec_id = resp.json()["id"]
    store = tmp_path / "recordings" / "recovery_evidence"
    assert len(list(store.glob("*.wav"))) == 3
    assert client.delete(f"/api/v1/recovery-evidence/{rec_id}").status_code == 200
    assert client.get("/api/v1/recovery-evidence").json()["items"] == []
    assert list(store.glob("*.wav")) == []
    assert client.delete(f"/api/v1/recovery-evidence/{rec_id}").status_code == 404


def test_does_not_touch_existing_corrections(client):
    response = client.post(
        "/api/v1/corrections",
        data={"app_id": "test-app", "transcript_raw": "wrong", "speaker_reference": "right",
              "language_hint": "hi-IN", "model_name": "prisma-v3.0",
              "consent_scope": "private", "speaker_confirmed_exact_words": "true"},
        files={"audio": ("recording.wav", wav_bytes())},
    )
    assert response.status_code == 201
    rec = submit(client)
    assert rec.status_code == 201
    assert len(client.get("/api/v1/corrections").json()["items"]) == 1
    assert len(client.get("/api/v1/recovery-evidence").json()["items"]) == 1
    assert client.delete("/api/v1/recovery-evidence/" + rec.json()["id"]).status_code == 200
    assert len(client.get("/api/v1/corrections").json()["items"]) == 1


def test_upgrade_creates_new_table_preserving_existing_database(client):
    with sqlite3.connect(main.DB_PATH) as con:
        names = {row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "corrections" in names
    assert "recovery_evidence" in names
