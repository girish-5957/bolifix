"""Step 5: safe, limited Hindi/romanized Hindi interpretation and optional TTS.

No tests call Gnani, Whisper or download model weights.
"""
from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from backend import dual_asr, main


@pytest.mark.parametrize("a,b", [
    ("Panch nhi pandra packet chaiye", "पाँच नहीं पंद्रह पैकेट चाहिए"),
    ("mujhe 15 packet chahiye", "मुझे पंद्रह पैकेट चाहिए"),
    ("मुझे १५ पैकेट चाहिए", "mujhe pandrah packets chaiye"),
])
def test_limited_cross_script_equivalence(a, b):
    result = dual_asr.compare_hypotheses(a, b)
    assert result["limited_equivalent"] is True
    assert result["limited_interpretation"] == "limited_cross_script_match"
    assert result["clarification_plan"]["route"] == "review_script_only"
    assert result["clarification_plan"]["can_speak"] is False
    assert result["is_accuracy_score"] is False
    assert result["is_training_label"] is False
    # The v1 status is intentionally preserved for existing clients.
    assert result["status"] == "script_review_needed"


def test_key_number_disagreement_is_not_normalized_away():
    result = dual_asr.compare_hypotheses("mujhe paanch chahiye", "मुझे पंद्रह चाहिए")
    assert result["prisma_numbers"] == ["5"]
    assert result["whisper_numbers"] == ["15"]
    assert result["limited_equivalent"] is False
    assert result["clarification_plan"]["route"] == "confirm_final_number"
    assert "अंतिम" in result["clarification_plan"]["question_hi"]
    assert result["potential_high_impact_difference"] is True


def test_missing_number_is_high_priority_not_automatically_resolved():
    result = dual_asr.compare_hypotheses(
        "मुझे पाँच नहीं पंद्रह पैकेट चाहिए", "आज नहीं पन्द्रा पेकर चाहिये",
    )
    assert result["prisma_numbers"] == ["5", "15"]
    assert result["whisper_numbers"] == ["15"]
    assert result["clarification_plan"]["route"] == "confirm_final_number"
    assert result["clarification_plan"]["verification_required"] is True
    assert result["clarification_plan"]["maximum_automatic_retries"] == 0


def test_negation_mismatch_requires_specific_check():
    result = dual_asr.compare_hypotheses("ये पैकेट नहीं भेजना", "ये पैकेट भेजना")
    assert result["clarification_plan"]["route"] == "confirm_action_and_negation"
    assert "क्या नहीं" in result["clarification_plan"]["question_hi"]


def test_unknown_place_names_are_not_assumed_equivalent():
    result = dual_asr.compare_hypotheses("भंजनगर", "bhanjanagar")
    assert result["limited_equivalent"] is False
    assert result["limited_interpretation"] == "unresolved_cross_script_difference"
    assert result["clarification_plan"]["route"] == "manual_listen"
    assert result["clarification_plan"]["question_hi"] is None


def test_same_risky_numbers_still_suggest_verification():
    result = dual_asr.compare_hypotheses("पंद्रह पैकेट", "पंद्रह पैकेट")
    assert result["status"] == "no_difference_after_limited_normalization"
    assert result["clarification_plan"]["route"] == "confirm_final_number"
    assert result["clarification_plan"]["can_speak"] is True
    assert not result["is_training_label"]


def test_no_risk_no_disagreement_no_question():
    result = dual_asr.compare_hypotheses("कल मौसम अच्छा है", "कल मौसम अच्छा है")
    assert result["clarification_plan"]["route"] == "no_prompt"
    assert result["clarification_plan"]["can_speak"] is False


def test_unknown_words_in_same_script_trigger_repeat_not_ground_truth():
    result = dual_asr.compare_hypotheses("मुझे रायपुर जाना", "मुझे जयपुर जाना")
    assert result["clarification_plan"]["route"] == "repeat_key_word"
    assert result["clarification_plan"]["can_speak"] is True


def test_text_endpoint_is_offline_and_has_plan(monkeypatch):
    def forbid_tts(*a, **kw):
        raise AssertionError("Text comparison must not call Timbre")
    monkeypatch.setattr(main.tts_service, "synthesize_timbre", forbid_tts)
    with TestClient(main.app) as client:
        response = client.post("/api/v1/dual-asr/compare-text", json={
            "prisma_transcript": "मुझे 5 चाहिए", "whisper_transcript": "मुझे 15 चाहिए",
        })
    assert response.status_code == 200
    assert response.json()["clarification_plan"]["route"] == "confirm_final_number"


def test_tts_endpoint_does_not_speak_for_script_only(monkeypatch):
    monkeypatch.setenv("BOLIFIX_MOCK_TTS", "false")
    def forbid_tts(*a, **kw):
        raise AssertionError("Script-only review must not call Timbre")
    monkeypatch.setattr(main.tts_service, "synthesize_timbre", forbid_tts)
    with TestClient(main.app) as client:
        response = client.post("/api/v1/dual-asr/clarification-audio", json={
            "prisma_transcript": "panch nhi pandra packet chaiye",
            "whisper_transcript": "पाँच नहीं पंद्रह पैकेट चाहिए", "language": "hi-IN",
        })
    assert response.status_code == 200
    assert response.json()["needs_clarification"] is False
    assert response.json()["route"] == "review_script_only"


def test_mock_audio_uses_no_external_service(monkeypatch):
    monkeypatch.setenv("BOLIFIX_MOCK_TTS", "true")
    def forbid_tts(*a, **kw):
        raise AssertionError("MOCK TTS must not call real Gnani")
    monkeypatch.setattr(main.tts_service, "synthesize_timbre", forbid_tts)
    with TestClient(main.app) as client:
        response = client.post("/api/v1/dual-asr/clarification-audio", json={
            "prisma_transcript": "मुझे पाँच चाहिए",
            "whisper_transcript": "मुझे पंद्रह चाहिए",
            "language": "hi-IN",
        })
    assert response.status_code == 200
    assert response.json()["mock"] is True
    assert response.json()["route"] == "confirm_final_number"
    assert "credits" in response.json()["message"]


def test_real_tts_requires_key(monkeypatch):
    monkeypatch.setenv("BOLIFIX_MOCK_TTS", "false")
    monkeypatch.delenv("GNANI_API_KEY", raising=False)
    with TestClient(main.app) as client:
        response = client.post("/api/v1/dual-asr/clarification-audio", json={
            "prisma_transcript": "पाँच", "whisper_transcript": "पंद्रह",
        })
    assert response.status_code == 503


def test_real_tts_returns_wav_only_after_api_click(monkeypatch):
    monkeypatch.setenv("BOLIFIX_MOCK_TTS", "false")
    monkeypatch.setenv("GNANI_API_KEY", "test-key-not-real")
    spoken = []
    def fake_tts(text, language):
        spoken.append((text, language))
        return b"RIFF" + b"\0" * 4 + b"WAVE" + b"fake"
    monkeypatch.setattr(main.tts_service, "synthesize_timbre", fake_tts)
    with TestClient(main.app) as client:
        response = client.post("/api/v1/dual-asr/clarification-audio", json={
            "prisma_transcript": "मुझे 5 चाहिए", "whisper_transcript": "मुझे 15 चाहिए",
            "language": "en-IN",
        })
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/wav")
    assert response.headers["cache-control"] == "no-store"
    assert len(spoken) == 1
    assert spoken[0][1] == "en-IN"
    assert "final intended number" in spoken[0][0]


def test_tts_validation_and_fail_closed(monkeypatch):
    monkeypatch.setenv("BOLIFIX_MOCK_TTS", "false")
    monkeypatch.setenv("GNANI_API_KEY", "test-key-not-real")
    monkeypatch.setattr(main.tts_service, "synthesize_timbre", lambda *args: b"invalid")
    with TestClient(main.app) as client:
        invalid = client.post("/api/v1/dual-asr/clarification-audio", json={
            "prisma_transcript": "5", "whisper_transcript": "15", "language": "not-a-language",
        })
        bad_audio = client.post("/api/v1/dual-asr/clarification-audio", json={
            "prisma_transcript": "5", "whisper_transcript": "15", "language": "hi-IN",
        })
    assert invalid.status_code == 422
    assert bad_audio.status_code == 502


def test_no_tts_for_agreed_low_impact_case(monkeypatch):
    monkeypatch.setenv("BOLIFIX_MOCK_TTS", "false")
    monkeypatch.setattr(main.tts_service, "synthesize_timbre", lambda *args: 1/0)
    with TestClient(main.app) as client:
        response = client.post("/api/v1/dual-asr/clarification-audio", json={
            "prisma_transcript": "कल मौसम अच्छा है", "whisper_transcript": "कल मौसम अच्छा है",
        })
    assert response.status_code == 200
    assert response.json()["mock"] is True
    assert response.json()["needs_clarification"] is False
