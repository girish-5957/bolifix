"""Step 6: 0 Gnani credits. Test review boundaries and original-audio provenance."""
import io
import sqlite3
import wave
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend import asr_review, main


def wav_bytes(frames=12000):
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\0\0" * frames)
    return output.getvalue()


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "DB_PATH", tmp_path / "bolifix.sqlite3")
    monkeypatch.setattr(main, "AUDIO_DIR", tmp_path / "recordings")
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
        "original_audio": ("original.wav", wav_bytes()),
        "first_reply_audio": ("reply.wav", wav_bytes(6000)),
    })
    assert response.status_code == 201, response.text
    return response.json()["id"]


def review(client, identifier, **overrides):
    body = {
        "evidence_id": identifier, "reviewer_code": "local-reviewer",
        "verbatim_reference": "मुझे पंद्रह पैकेट चाहिए",
        "error_category": "number_or_amount",
        "reviewer_note": "I listened to original clip twice.",
        "confirmed_listened": True,
        "confirmed_verbatim": True,
    }
    body.update(overrides)
    return client.post("/api/v1/asr-reviews", json=body)


def test_review_is_private_and_never_training_ready(client):
    identifier = evidence(client)
    saved = review(client, identifier)
    assert saved.status_code == 201, saved.text
    result = saved.json()
    assert result["training_ready"] is False
    assert result["research_export_allowed"] is False
    assert result["review_status"] == "awaiting_independent_quality_check"
    assert result["metrics"]["substitutions"] == 1
    assert result["metrics"]["deletions"] == 0
    assert result["metrics"]["insertions"] == 0
    assert result["metrics"]["wer"] == .25
    items = client.get("/api/v1/asr-reviews").json()
    assert items["training_ready_count"] == 0
    row = items["items"][0]
    assert row["evidence_id"] == identifier
    assert row["reference_kind"] == "human_listened_original_audio"
    assert row["consent_scope"] == "private"
    assert row["training_ready"] is False
    assert "original_audio_path" not in row
    assert "original_audio_sha256" not in row


def test_original_audio_endpoint_returns_original_not_clarification(client):
    identifier = evidence(client)
    response = client.get(f"/api/v1/recovery-evidence/{identifier}/original-audio")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/wav")
    assert response.headers["cache-control"] == "no-store"
    assert response.content == wav_bytes()
    assert response.content != wav_bytes(6000)


def test_no_review_without_explicit_listening_and_verbatim_attestation(client):
    identifier = evidence(client)
    assert review(client, identifier, confirmed_listened=False).status_code == 400
    assert review(client, identifier, confirmed_verbatim=False).status_code == 400
    assert client.get("/api/v1/asr-reviews").json()["items"] == []


def test_uncertainty_remains_a_draft_and_is_not_verified(client):
    identifier = evidence(client)
    assert review(client, identifier, error_category="uncertain", confirmed_verbatim=True).status_code == 400
    saved = review(client, identifier, error_category="uncertain", confirmed_verbatim=False)
    assert saved.status_code == 201, saved.text
    assert saved.json()["review_status"] == "needs_relisten"
    assert saved.json()["reference_kind"] == "reviewer_draft_uncertain"
    assert saved.json()["training_ready"] is False


def test_classification_and_reviewer_input_are_validated(client):
    identifier = evidence(client)
    assert review(client, identifier, error_category="fake").status_code == 400
    assert review(client, identifier, reviewer_code="../../secret").status_code == 400
    assert review(client, identifier, verbatim_reference=" ").status_code == 400
    assert review(client, identifier, verbatim_reference="!?.,").status_code == 400
    assert review(client, identifier, reviewer_note="x" * 501).status_code == 422
    assert client.get("/api/v1/asr-reviews").json()["items"] == []


def test_duplicate_review_does_not_silently_change_label(client):
    identifier = evidence(client)
    assert review(client, identifier).status_code == 201
    assert review(client, identifier, verbatim_reference="दूसरा").status_code == 409
    items = client.get("/api/v1/asr-reviews").json()["items"]
    assert len(items) == 1
    assert items[0]["verbatim_reference"] == "मुझे पंद्रह पैकेट चाहिए"


def test_delete_review_preserves_original_audio_and_allows_re_review(client):
    identifier = evidence(client)
    assert review(client, identifier).status_code == 201
    assert client.delete(f"/api/v1/asr-reviews/{identifier}").status_code == 200
    assert client.delete(f"/api/v1/asr-reviews/{identifier}").status_code == 404
    assert client.get(f"/api/v1/recovery-evidence/{identifier}/original-audio").status_code == 200
    assert len(client.get("/api/v1/recovery-evidence").json()["items"]) == 1
    assert review(client, identifier, error_category="script_or_format").status_code == 201


def test_parent_evidence_deletion_cascades_review_and_audio(client, tmp_path):
    identifier = evidence(client)
    assert review(client, identifier).status_code == 201
    assert client.delete(f"/api/v1/recovery-evidence/{identifier}").status_code == 200
    assert client.get("/api/v1/asr-reviews").json()["items"] == []
    assert client.get(f"/api/v1/recovery-evidence/{identifier}/original-audio").status_code == 404
    assert not list((tmp_path / "recordings" / "recovery_evidence").glob("*.wav"))


def test_missing_evidence_never_creates_orphaned_review(client):
    assert review(client, "not-found").status_code == 404
    assert client.get("/api/v1/asr-reviews").json()["items"] == []


def test_audio_tamper_blocks_review_and_playback(client, tmp_path):
    identifier = evidence(client)
    with sqlite3.connect(main.DB_PATH) as con:
        audio_path = Path(con.execute(
            "SELECT original_audio_path FROM recovery_evidence WHERE id=?", (identifier,),
        ).fetchone()[0])
    audio_path.write_bytes(b"changed-data")
    assert client.get(f"/api/v1/recovery-evidence/{identifier}/original-audio").status_code == 409
    assert review(client, identifier).status_code == 400
    assert client.get("/api/v1/asr-reviews").json()["items"] == []


def test_out_of_directory_audio_path_is_blocked(client, tmp_path):
    identifier = evidence(client)
    outsider = tmp_path / "not-your-audio.wav"
    outsider.write_bytes(wav_bytes())
    with sqlite3.connect(main.DB_PATH) as con:
        con.execute("UPDATE recovery_evidence SET original_audio_path=? WHERE id=?", (str(outsider), identifier))
    assert client.get(f"/api/v1/recovery-evidence/{identifier}/original-audio").status_code == 409
    assert review(client, identifier).status_code == 400


def test_unicode_devanagari_marks_are_preserved_in_tokenization():
    assert asr_review.normalize_tokens("हिन्दी में पाँच, १५ बोलें!") == ["हिन्दी", "में", "पाँच", "१५", "बोलें"]
    assert asr_review.compare_transcripts("हिन्दी में", "हिन्दी में!")["wer"] == 0.0


def test_word_insertions_deletions_and_substitutions_are_separated():
    assert asr_review.word_edits(["a", "b"], ["a", "c"]) == (1, 0, 0)
    assert asr_review.word_edits(["a", "b"], ["a"]) == (0, 1, 0)
    assert asr_review.word_edits(["a"], ["a", "b"]) == (0, 0, 1)
    assert asr_review.compare_transcripts("HELLO, world!", "hello world")["normalized_exact_match"] is True


def test_empty_audio_record_prevents_review(client):
    identifier = evidence(client)
    with sqlite3.connect(main.DB_PATH) as con:
        audio_path = Path(con.execute(
            "SELECT original_audio_path FROM recovery_evidence WHERE id=?", (identifier,),
        ).fetchone()[0])
    audio_path.unlink()
    assert review(client, identifier).status_code == 400


def test_all_tables_exist_after_upgrade(client):
    with sqlite3.connect(main.DB_PATH) as con:
        tables = {row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"corrections", "recovery_evidence", "asr_reviews"}.issubset(tables)
