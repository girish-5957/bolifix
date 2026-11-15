"""Tests for Phase 2 Step 2 — no real Gnani calls or credits."""
import io
import wave
from fastapi.testclient import TestClient

from backend import main
from backend.tts import clarification_question


def test_hindi_number_question():
    result = clarification_question("मुझे पाँच नहीं पंद्रह packets चाहिए", "hi-IN")
    assert "संख्या" in result


def test_english_number_question():
    result = clarification_question("I want fifteen not fifty", "en-IN")
    assert "number" in result.lower()


def test_no_question_for_unflagged_speech():
    assert clarification_question("नमस्ते आपका स्वागत है", "hi-IN") is None


def test_identifier_priority():
    assert "आईडी" in clarification_question("Order ID AB 629", "hi-IN")


def test_mock_returns_no_audio(monkeypatch):
    monkeypatch.setenv("BOLIFIX_MOCK_TTS", "true")
    with TestClient(main.app) as client:
        r = client.post("/api/v1/clarification-audio", json={"transcript": "मुझे पंद्रह नहीं पचपन चाहिए", "language": "hi-IN"})
    assert r.status_code == 200
    assert r.json()["mock"] is True
    assert "audio" not in r.headers.get("content-type", "")


def test_no_trigger_does_not_call_provider(monkeypatch):
    monkeypatch.setenv("BOLIFIX_MOCK_TTS", "false")
    def fail(*args):
        raise AssertionError("Should not call TTS")
    monkeypatch.setattr(main.tts_service, "synthesize_timbre", fail)
    with TestClient(main.app) as client:
        r = client.post("/api/v1/clarification-audio", json={"transcript": "Namaste", "language": "en-IN"})
    assert r.status_code == 200 and r.json()["needs_clarification"] is False


def test_invalid_language_rejected():
    with TestClient(main.app) as client:
        r = client.post("/api/v1/clarification-audio", json={"transcript": "पंद्रह", "language": "xx-XX"})
    assert r.status_code == 422


def test_synthesized_wav_passes_through(monkeypatch):
    monkeypatch.setenv("BOLIFIX_MOCK_TTS", "false")
    monkeypatch.setenv("GNANI_API_KEY", "dummy-test-key")
    stream = io.BytesIO()
    with wave.open(stream, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 80)
    expected_wav = stream.getvalue()
    monkeypatch.setattr(main.tts_service, "synthesize_timbre", lambda text, language: expected_wav)
    with TestClient(main.app) as client:
        r = client.post("/api/v1/clarification-audio", json={"transcript": "पंद्रह नहीं पचपन", "language": "hi-IN"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("audio/wav")
    assert r.content == expected_wav


def test_missing_key_does_not_call_synthesizer(monkeypatch):
    monkeypatch.setenv("BOLIFIX_MOCK_TTS", "false")
    monkeypatch.delenv("GNANI_API_KEY", raising=False)
    with TestClient(main.app) as client:
        r = client.post("/api/v1/clarification-audio", json={"transcript": "पंद्रह नहीं पचपन", "language": "hi-IN"})
    assert r.status_code == 503
