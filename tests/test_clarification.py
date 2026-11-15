"""Phase-2 rules and endpoint tests; no API credits used."""
from fastapi.testclient import TestClient
from backend.clarification import suggest_clarification
from backend.main import app


def test_number_and_negation_are_flagged():
    result = suggest_clarification("Mujhe paanch nahi pandrah packets chahiye")
    assert result["needs_clarification"] is True
    assert "number" in result["signals"]
    assert "negation_or_correction" in result["signals"]


def test_hindi_numbers_and_negation_are_flagged():
    result = suggest_clarification("पाँच नहीं पंद्रह पैकेट चाहिए")
    assert "number" in result["signals"]
    assert "negation_or_correction" in result["signals"]


def test_identifier_gets_specific_question():
    result = suggest_clarification("Order ID AB six two nine hai, AB nine two six nahi")
    assert "identifier" in result["signals"]
    assert "character by character" in result["suggested_question"]


def test_low_risk_does_not_imply_correctness():
    result = suggest_clarification("Aaj mausam achha hai")
    assert result["needs_clarification"] is False
    assert result["suggested_question"] is None
    assert "does not prove" in result["explanation"]


def test_endpoint_returns_suggestion():
    with TestClient(app) as client:
        result = client.post("/api/v1/clarification-suggestion", json={"transcript": "पंद्रह नहीं पचपन"})
    assert result.status_code == 200
    assert result.json()["needs_clarification"]


def test_empty_transcript_rejected():
    with TestClient(app) as client:
        result = client.post("/api/v1/clarification-suggestion", json={"transcript": ""})
    assert result.status_code == 422
