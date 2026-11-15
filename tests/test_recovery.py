"""Step 3: clarification reply review. Offline; no Gnani credits."""
from fastapi.testclient import TestClient
from backend.main import app
from backend.recovery import number_candidates, review_reply


def test_single_numeric_candidate():
    result = review_reply("मुझे पाँच नहीं पंद्रह packets चाहिए", "पंद्रह")
    assert result["status"] == "candidate_needs_confirmation"
    assert result["candidate"] == "15"
    assert result["requires_speaker_confirmation"] is True
    assert result["training_ready"] is False


def test_english_number_candidate():
    assert number_candidates("fifteen") == ["15"]
    assert number_candidates("१५") == ["15"]
    assert number_candidates("55") == ["55"]


def test_multiple_numbers_never_guessed():
    result = review_reply("Mujhe paanch nahi pandrah chahiye", "five nahi fifteen")
    assert result["status"] == "ambiguous_reply"
    assert result["candidate"] is None


def test_no_number_never_guessed():
    result = review_reply("मुझे पाँच नहीं पंद्रह चाहिए", "जी सही है")
    assert result["status"] == "no_numeric_candidate"
    assert result["candidate"] is None


def test_identifier_requires_manual_verification():
    result = review_reply("Order ID AB six two nine", "AB 629")
    assert result["signal_type"] == "identifier"
    assert result["candidate"] is None
    assert result["training_ready"] is False


def test_negation_does_not_claim_understanding():
    result = review_reply("मेरा order cancel मत करना", "cancel mat karna")
    assert result["status"] == "manual_review"
    assert result["candidate"] is None


def test_no_trigger_is_not_correctness_proof():
    result = review_reply("आज मौसम अच्छा है", "जी")
    assert result["status"] == "no_trigger"
    assert result["training_ready"] is False


def test_endpoint_returns_unverified_candidate():
    with TestClient(app) as client:
        r = client.post("/api/v1/clarification-review", json={
            "original_transcript": "मुझे पाँच नहीं पंद्रह चाहिए",
            "reply_transcript": "पंद्रह",
        })
    assert r.status_code == 200
    assert r.json()["candidate"] == "15"
    assert r.json()["training_ready"] is False


def test_endpoint_rejects_empty_reply():
    with TestClient(app) as client:
        r = client.post("/api/v1/clarification-review", json={
            "original_transcript": "मुझे पंद्रह चाहिए", "reply_transcript": "",
        })
    assert r.status_code == 422


def test_compound_number_is_not_simplified_to_wrong_value():
    assert number_candidates("fifteen thousand rupees") == []
    assert number_candidates("पंद्रह हजार रुपये") == []
