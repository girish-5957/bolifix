"""Tests that work without a live Gnani API key."""
import io
import wave

import pytest
from fastapi.testclient import TestClient

from backend import main


def wav_bytes(seconds=1):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\0\0" * (16000 * seconds))
    return buf.getvalue()


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "DB_PATH", tmp_path / "test.sqlite3")
    monkeypatch.setattr(main, "AUDIO_DIR", tmp_path / "audios")
    monkeypatch.setattr(main, "MOCK_STT", True)
    with TestClient(main.app) as c:
        yield c


def test_health(client):
    result = client.get("/health").json()
    assert result["status"] == "ok"
    assert result["mock_mode"] is True


def test_audio_requires_permission(client):
    resp = client.post("/api/v1/transcribe", files={"audio": ("x.wav", wav_bytes())})
    assert resp.status_code == 400


def test_mock_transcription_not_stored(client):
    resp = client.post(
        "/api/v1/transcribe",
        files={"audio": ("x.wav", wav_bytes())},
        data={"permission_to_process": "true", "language_code": "hi-IN,en-IN", "output_format": "codemix"},
    )
    assert resp.status_code == 200
    assert resp.json()["mock"] is True
    assert client.get("/api/v1/corrections").json()["items"] == []


def test_save_and_delete_local_correction(client):
    payload = {
        "app_id": "test-app", "transcript_raw": "wrong name", "speaker_reference": "right name",
        "language_hint": "hi-IN", "model_name": "prisma-v3.0",
        "consent_scope": "private", "speaker_confirmed_exact_words": "true",
    }
    resp = client.post("/api/v1/corrections", data=payload, files={"audio": ("x.wav", wav_bytes())})
    assert resp.status_code == 201, resp.text
    record = resp.json()
    assert record["training_ready"] is False
    assert len(client.get("/api/v1/corrections").json()["items"]) == 1
    assert client.delete("/api/v1/corrections/" + record["id"]).status_code == 200
    assert client.get("/api/v1/corrections").json()["items"] == []


def test_invalid_scope_not_saved(client):
    resp = client.post("/api/v1/corrections", data={
        "app_id": "app", "transcript_raw": "x", "speaker_reference": "y",
        "consent_scope": "not_a_scope", "speaker_confirmed_exact_words": "true",
    }, files={"audio": ("x.wav", wav_bytes())})
    assert resp.status_code == 400
    assert client.get("/api/v1/corrections").json()["items"] == []
