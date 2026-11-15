"""Step 4: private, rule-based ASR-hypothesis comparison."""
import sqlite3

import pytest
from fastapi.testclient import TestClient

from backend import dual_asr, main, demo_evaluation
from scripts import dual_asr as cli


def test_equal_text_is_not_evidence_of_correctness():
    r = dual_asr.compare_hypotheses("मुझे पाँच चाहिए", "मुझे पांच चाहिए")
    assert r["status"] == "no_difference_after_limited_normalization"
    assert r["potential_high_impact_difference"] is False
    assert r["is_accuracy_score"] is False
    assert r["is_training_label"] is False
    assert r["human_review_recommended"] is True  # even agreement on a consequential number


def test_number_spelled_vs_digits_are_equivalent():
    r = dual_asr.compare_hypotheses("पंद्रह", "15")
    assert "number_mentions_differ" not in r["signals"]
    assert r["prisma_numbers"] == ["15"]
    assert r["whisper_numbers"] == ["15"]
    assert r["status"] == "no_difference_after_limited_normalization"


def test_panch_and_pandrah_never_collapse_to_same_number():
    r = dual_asr.compare_hypotheses("मुझे पांच नहीं पंद्रह चाहिए", "मुझे पांच चाहिए")
    assert r["status"] == "priority_human_review"
    assert r["potential_high_impact_difference"] is True
    assert r["prisma_numbers"] == ["5", "15"]
    assert r["whisper_numbers"] == ["5"]
    assert "number_mentions_differ" in r["signals"]


def test_small_model_example_detects_missing_number():
    r = dual_asr.compare_hypotheses("मुझे पाँच नहीं पंद्रह पैकेट चाहिए", "आज नहीं पन्द्रा पेकर चाहिये")
    assert "number_mentions_differ" in r["signals"]
    assert r["prisma_numbers"] == ["5", "15"]
    assert r["whisper_numbers"] == ["15"]
    assert "negation_presence_differs" not in r["signals"]
    assert r["is_training_label"] is False


def test_negation_disagreement_is_high_priority():
    r = dual_asr.compare_hypotheses("ये पैकेट नहीं भेजना", "ये पैकेट भेजना")
    assert "negation_presence_differs" in r["signals"]
    assert r["status"] == "priority_human_review"


def test_romanized_negation_and_devanagari_detected():
    r = dual_asr.compare_hypotheses("nahi", "नहीं")
    assert "negation_presence_differs" not in r["signals"]
    assert "writing_systems_differ" in r["signals"]


def test_unrelated_words_flag_review_without_claiming_wrong_model():
    r = dual_asr.compare_hypotheses("कल भुवनेश्वर जाना", "कल बालेश्वर जाना")
    assert r["status"] == "human_review_needed"
    assert r["human_review_recommended"] is True
    assert r["suggested_question"]
    assert r["is_accuracy_score"] is False


def test_different_scripts_do_not_establish_winner():
    r = dual_asr.compare_hypotheses("मुझे पैकेट चाहिए", "mujhe packet chahiye")
    assert "writing_systems_differ" in r["signals"]
    assert r["status"] == "script_review_needed"
    assert r["potential_high_impact_difference"] is False


def test_devanagari_digit_equivalent_to_ascii():
    r = dual_asr.compare_hypotheses("मुझे १५ चाहिए", "मुझे 15 चाहिए")
    assert r["status"] == "no_difference_after_limited_normalization"


@pytest.mark.parametrize("a,b", [(None, "abc"), (" ", "abc"), ("a" * 2001, "abc"), ("abc", None)])
def test_invalid_hypotheses_rejected(a, b):
    with pytest.raises(ValueError):
        dual_asr.compare_hypotheses(a, b)


def test_text_api_has_no_model_or_api_dependency(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "ROOT", tmp_path)
    monkeypatch.setattr(main, "DB_PATH", tmp_path / "storage" / "test.sqlite3")
    monkeypatch.setattr(main, "AUDIO_DIR", tmp_path / "storage" / "recordings")
    with TestClient(main.app) as client:
        r = client.post("/api/v1/dual-asr/compare-text", json={
            "prisma_transcript": "पांच नहीं पंद्रह",
            "whisper_transcript": "आज नहीं पंद्रह",
        })
        assert r.status_code == 200
        assert r.json()["status"] == "priority_human_review"
        assert not r.json()["is_training_label"]
        bad = client.post("/api/v1/dual-asr/compare-text", json={"prisma_transcript": ""})
        assert bad.status_code == 422


def test_cli_requires_consent_and_does_not_accidentally_load_model():
    with pytest.raises(SystemExit) as exc:
        cli.run(["--run-whisper"])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        cli.run(["--single-approved", "--confirm-local-only", "--run-whisper"])
    assert exc.value.code == 2  # no model download flag
    with pytest.raises(SystemExit) as exc:
        cli.run(["--single-approved", "--confirm-local-only", "--show-text"])
    assert exc.value.code == 2


def test_cli_check_only_reads_existing_prisma_record_without_inference(tmp_path, monkeypatch, capsys):
    db = tmp_path / "storage" / "bolifix.sqlite3"
    db.parent.mkdir(parents=True)
    with sqlite3.connect(db) as con:
        con.execute("CREATE TABLE recovery_evidence(id TEXT, original_transcript TEXT, source_model TEXT)")
        con.execute("INSERT INTO recovery_evidence VALUES (?,?,?)", ("a1", "पांच नहीं पंद्रह", "prisma-v3.0"))
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(demo_evaluation, "choose_approved_clip", lambda **kwargs: {
        "approved_count": 1, "row": {"evidence_id": "a1"},
    })
    assert cli.run(["--single-approved", "--confirm-local-only"]) == 0
    out = capsys.readouterr().out
    assert "no new Gnani request" in out
    assert "CHECK ONLY" in out
