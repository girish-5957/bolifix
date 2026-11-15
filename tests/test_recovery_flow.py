"""Offline Phase 2 Step 4 tests. No network / Gnani charges."""
import io
import wave

import pytest
from fastapi.testclient import TestClient
from backend import main
from backend.recovery_flow import next_recovery_step


ORIGINAL = "मुझे पाँच नहीं पंद्रह packets चाहिए"


def test_clear_reply_requires_confirmation_not_training():
    r = next_recovery_step(ORIGINAL, "पंद्रह", 1)
    assert r["state"] == "candidate_confirmation"
    assert r["candidate"] == "15"
    assert r["requires_speaker_confirmation"] is True
    assert r["training_ready"] is False


def test_ambiguous_first_reply_asks_one_more():
    r = next_recovery_step(ORIGINAL, "पाँच नहीं पंद्रह", 1)
    assert r["state"] == "ask_followup"
    assert "अंतिम संख्या" in r["followup_question_hi"]
    assert r["attempt"] == 1


def test_second_ambiguous_reply_stops_voice_loop():
    r = next_recovery_step(ORIGINAL, "पाँच नहीं पंद्रह", 2)
    assert r["state"] == "manual_verification"
    assert r["followup_question_en"] is None


def test_second_reply_can_resolve_to_unverified_candidate():
    r = next_recovery_step(ORIGINAL, "पंद्रह", 2)
    assert r["state"] == "candidate_confirmation"
    assert r["candidate"] == "15"


def test_identifier_requires_direct_review():
    r = next_recovery_step("Order ID AB 629", "AB six two nine", 1)
    assert r["state"] == "manual_verification"
    assert r["candidate"] is None


def test_negated_action_does_not_guess():
    r = next_recovery_step("मेरा order cancel मत करना", "cancel mat karna", 1)
    assert r["state"] == "manual_verification"


def test_no_trigger_does_not_claim_verified():
    r = next_recovery_step("Namaste", "Hello", 1)
    assert r["state"] == "no_trigger"
    assert r["training_ready"] is False


@pytest.mark.parametrize("attempt", [0, 3, -1])
def test_invalid_attempt_in_python(attempt):
    with pytest.raises(ValueError):
        next_recovery_step(ORIGINAL, "पंद्रह", attempt)


def test_api_bounded_response():
    with TestClient(main.app) as client:
        r = client.post("/api/v1/recovery-next", json={
            "original_transcript": ORIGINAL,
            "reply_transcript": "पाँच नहीं पंद्रह",
            "attempt": 1,
        })
    assert r.status_code == 200
    assert r.json()["state"] == "ask_followup"


def test_api_rejects_out_of_range_attempt():
    with TestClient(main.app) as client:
        r = client.post("/api/v1/recovery-next", json={
            "original_transcript": ORIGINAL, "reply_transcript": "पंद्रह",
            "attempt": 3,
        })
    assert r.status_code == 422


def test_followup_mock_never_calls_tts(monkeypatch):
    monkeypatch.setenv("BOLIFIX_MOCK_TTS", "true")
    monkeypatch.setattr(main.tts_service, "synthesize_timbre",
                        lambda *_: (_ for _ in ()).throw(AssertionError("No TTS allowed")))
    with TestClient(main.app) as client:
        r = client.post("/api/v1/recovery-followup-audio", json={
            "original_transcript": ORIGINAL,
            "reply_transcript": "पाँच नहीं पंद्रह",
            "attempt": 1, "language": "hi-IN",
        })
    assert r.status_code == 200
    assert r.json()["mock"] is True
    assert "audio" not in r.headers.get("content-type", "")


def test_no_followup_means_no_speech_or_provider_call(monkeypatch):
    monkeypatch.setenv("BOLIFIX_MOCK_TTS", "false")
    monkeypatch.setattr(main.tts_service, "synthesize_timbre",
                        lambda *_: (_ for _ in ()).throw(AssertionError("No TTS allowed")))
    with TestClient(main.app) as client:
        r = client.post("/api/v1/recovery-followup-audio", json={
            "original_transcript": ORIGINAL, "reply_transcript": "पंद्रह",
            "attempt": 1, "language": "en-IN",
        })
    assert r.status_code == 200
    assert r.json()["needs_followup"] is False


def test_second_turn_cannot_trigger_more_audio(monkeypatch):
    monkeypatch.setenv("BOLIFIX_MOCK_TTS", "false")
    monkeypatch.setattr(main.tts_service, "synthesize_timbre",
                        lambda *_: (_ for _ in ()).throw(AssertionError("No TTS allowed")))
    with TestClient(main.app) as client:
        r = client.post("/api/v1/recovery-followup-audio", json={
            "original_transcript": ORIGINAL, "reply_transcript": "पाँच नहीं पंद्रह",
            "attempt": 2, "language": "hi-IN",
        })
    assert r.status_code == 200
    assert r.json()["needs_followup"] is False


def test_invalid_audio_language_rejected():
    with TestClient(main.app) as client:
        r = client.post("/api/v1/recovery-followup-audio", json={
            "original_transcript": ORIGINAL, "reply_transcript": "पाँच नहीं पंद्रह",
            "attempt": 1, "language": "fr-FR",
        })
    assert r.status_code == 422


def test_real_followup_wav_passes_through_with_mocked_provider(monkeypatch):
    monkeypatch.setenv("BOLIFIX_MOCK_TTS", "false")
    monkeypatch.setenv("GNANI_API_KEY", "dummy-local-testing-key")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\\x00\\x00" * 20)
    expected = buf.getvalue()
    monkeypatch.setattr(main.tts_service, "synthesize_timbre",
                        lambda text, language: expected)
    with TestClient(main.app) as client:
        r = client.post("/api/v1/recovery-followup-audio", json={
            "original_transcript": ORIGINAL,
            "reply_transcript": "पाँच नहीं पंद्रह",
            "attempt": 1, "language": "hi-IN",
        })
    assert r.status_code == 200
    assert r.content == expected
    assert r.headers["content-type"].startswith("audio/wav")
