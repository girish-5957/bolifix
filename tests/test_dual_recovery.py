"""Phase 3 Step 6: tests operate offline, no provider requests or real recordings."""
from __future__ import annotations

import io
import wave

import pytest
from fastapi.testclient import TestClient

from backend import main
from backend.dual_recovery import next_dual_recovery_step

P5 = "मुझे पाँच पैकेट चाहिए"
P15 = "मुझे पंद्रह पैकेट चाहिए"


def test_number_difference_with_clear_reply_proposes_only_a_candidate():
    d = next_dual_recovery_step(P5, P15, "पंद्रह", 1)
    assert d["state"] == "candidate_confirmation"
    assert d["candidate"] == "15"
    assert d["requires_speaker_confirmation"]
    assert d["training_ready"] is False
    assert d["verified_original_audio"] is False
    assert d["record_saved"] is False


@pytest.mark.parametrize("reply", ["पाँच नहीं पंद्रह", "कौन सी संख्या", "5 15"])
def test_ambiguous_first_reply_permits_one_followup(reply):
    d = next_dual_recovery_step(P5, P15, reply, 1)
    assert d["state"] == "ask_followup"
    assert "अंतिम" in d["followup_question_hi"]
    assert d["max_voice_attempts"] == 2
    assert d["candidate"] is None


@pytest.mark.parametrize("reply", ["पाँच नहीं पंद्रह", "कौन सी संख्या"])
def test_second_ambiguous_reply_becomes_manual_review(reply):
    d = next_dual_recovery_step(P5, P15, reply, 2)
    assert d["state"] == "manual_verification"
    assert d["followup_question_hi"] is None
    assert d["candidate"] is None


def test_clear_second_reply_is_still_not_verified():
    d = next_dual_recovery_step(P5, P15, "15", 2)
    assert d["state"] == "candidate_confirmation"
    assert d["candidate"] == "15"
    assert d["training_ready"] is False


def test_non_number_name_disagreement_never_picks_model_as_correct():
    d = next_dual_recovery_step("जयपुर", "रायपुर", "जयपुर", 1)
    assert d["state"] == "manual_verification"
    assert d["candidate"] is None


def test_negation_disagreement_never_asserts_resolved_action():
    d = next_dual_recovery_step("पैकेट नहीं भेजना", "पैकेट भेजना", "नहीं भेजना", 1)
    assert d["state"] == "manual_verification"
    assert d["candidate"] is None


def test_only_cross_script_spelling_change_does_not_start_voice():
    d = next_dual_recovery_step("Panch nhi pandra packet chaiye", "पाँच नहीं पंद्रह पैकेट चाहिए", "15", 1)
    assert d["state"] == "no_voice_clarification"
    assert d["candidate"] is None


def test_unknown_name_in_two_scripts_requires_original_audio_review():
    d = next_dual_recovery_step("भंजनगर", "bhanjanagar", "भंजनगर", 1)
    assert d["state"] == "manual_listening_required"
    assert d["candidate"] is None


def test_no_trigger_no_voice():
    d = next_dual_recovery_step("कल मौसम अच्छा है", "कल मौसम अच्छा है", "अच्छा", 1)
    assert d["state"] == "no_voice_clarification"


@pytest.mark.parametrize("attempt", [0, -1, 3, 100])
def test_bad_attempts_rejected(attempt):
    with pytest.raises(ValueError):
        next_dual_recovery_step(P5, P15, "15", attempt)


def test_api_returns_offline_decision_without_model_calls(monkeypatch):
    monkeypatch.setattr(main.tts_service, "synthesize_timbre", lambda *_: (_ for _ in ()).throw(AssertionError("No TTS")))
    monkeypatch.setattr(main, "transcribe_via_gnani", lambda *_: (_ for _ in ()).throw(AssertionError("No STT")))
    with TestClient(main.app) as client:
        r = client.post("/api/v1/dual-asr/recovery-next", json={
            "prisma_transcript": P5, "whisper_transcript": P15,
            "reply_transcript": "15", "attempt": 1,
        })
    assert r.status_code == 200
    assert r.json()["candidate"] == "15"
    assert r.json()["record_saved"] is False


def test_api_rejects_empty_and_out_of_range_text():
    with TestClient(main.app) as client:
        for payload in [
            {"prisma_transcript": "", "whisper_transcript": P15, "reply_transcript": "15", "attempt": 1},
            {"prisma_transcript": P5, "whisper_transcript": P15, "reply_transcript": " ", "attempt": 3},
            {"prisma_transcript": P5, "whisper_transcript": P15, "reply_transcript": "x" * 2001, "attempt": 1},
        ]:
            assert client.post("/api/v1/dual-asr/recovery-next", json=payload).status_code == 422


def test_followup_in_mock_mode_never_calls_tts(monkeypatch):
    monkeypatch.setenv("BOLIFIX_MOCK_TTS", "true")
    monkeypatch.setattr(main.tts_service, "synthesize_timbre", lambda *_: (_ for _ in ()).throw(AssertionError("No TTS")))
    with TestClient(main.app) as client:
        r = client.post("/api/v1/dual-asr/followup-audio", json={
            "prisma_transcript": P5, "whisper_transcript": P15,
            "reply_transcript": "पाँच नहीं पंद्रह", "attempt": 1, "language": "hi-IN",
        })
    assert r.status_code == 200
    assert r.json()["mock"] is True
    assert r.json()["needs_followup"] is True
    assert "ऑडियो" not in r.headers.get("content-type", "")


@pytest.mark.parametrize("reply,attempt", [("15", 1), ("5 15", 2)])
def test_no_followup_if_not_appropriate(monkeypatch, reply, attempt):
    monkeypatch.setenv("BOLIFIX_MOCK_TTS", "false")
    monkeypatch.setattr(main.tts_service, "synthesize_timbre", lambda *_: (_ for _ in ()).throw(AssertionError("No TTS")))
    with TestClient(main.app) as client:
        r = client.post("/api/v1/dual-asr/followup-audio", json={
            "prisma_transcript": P5, "whisper_transcript": P15,
            "reply_transcript": reply, "attempt": attempt, "language": "en-IN",
        })
    assert r.status_code == 200
    assert r.json()["needs_followup"] is False


def test_followup_rejects_invalid_language(monkeypatch):
    monkeypatch.setenv("BOLIFIX_MOCK_TTS", "true")
    with TestClient(main.app) as client:
        r = client.post("/api/v1/dual-asr/followup-audio", json={
            "prisma_transcript": P5, "whisper_transcript": P15,
            "reply_transcript": "5 15", "attempt": 1, "language": "de-DE",
        })
    assert r.status_code == 422


def test_valid_real_followup_audio_with_stubbed_provider(monkeypatch):
    monkeypatch.setenv("BOLIFIX_MOCK_TTS", "false")
    monkeypatch.setenv("GNANI_API_KEY", "dummy-test-only")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 20)
    expected = buffer.getvalue()
    monkeypatch.setattr(main.tts_service, "synthesize_timbre", lambda text, language: expected)
    with TestClient(main.app) as client:
        r = client.post("/api/v1/dual-asr/followup-audio", json={
            "prisma_transcript": P5, "whisper_transcript": P15,
            "reply_transcript": "5 15", "attempt": 1, "language": "en-IN",
        })
    assert r.status_code == 200
    assert r.content == expected
    assert r.headers["content-type"].startswith("audio/wav")
    assert r.headers["cache-control"] == "no-store"


def test_reject_invalid_timbre_bytes(monkeypatch):
    monkeypatch.setenv("BOLIFIX_MOCK_TTS", "false")
    monkeypatch.setenv("GNANI_API_KEY", "dummy-test-only")
    monkeypatch.setattr(main.tts_service, "synthesize_timbre", lambda text, language: b"not a wav")
    with TestClient(main.app) as client:
        r = client.post("/api/v1/dual-asr/followup-audio", json={
            "prisma_transcript": P5, "whisper_transcript": P15,
            "reply_transcript": "5 15", "attempt": 1,
        })
    assert r.status_code == 502


def test_streamlit_has_explicit_consent_and_reset_guards():
    from pathlib import Path
    app_text = (Path(__file__).resolve().parents[1] / "frontend" / "app.py").read_text(encoding="utf-8")
    assert "hybrid_first_permission" in app_text
    assert "hybrid_second_permission" in app_text
    assert "hybrid_bound_sig" in app_text
    assert "hybrid_first_fingerprint" in app_text
    assert "hybrid_verified_intent" in app_text
    assert "Session only: NOT an exact original-audio reference" in app_text
