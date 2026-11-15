"""Phase 3 Step 3: text-demo guard + approved-one-clip read-only checks."""
import hashlib
import io
from pathlib import Path
import wave

import pytest
from fastapi.testclient import TestClient

from backend import demo_evaluation, main, quality_gate
from scripts.demo_evaluation import run


def sample_wav(channels=1, rate=16000):
    output = io.BytesIO()
    with wave.open(output, "wb") as obj:
        obj.setnchannels(channels)
        obj.setsampwidth(2)
        obj.setframerate(rate)
        obj.writeframes(b"\0\1" * rate * channels)
    return output.getvalue()


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "ROOT", tmp_path)
    monkeypatch.setattr(main, "DB_PATH", tmp_path / "storage" / "bolifix.sqlite3")
    monkeypatch.setattr(main, "AUDIO_DIR", tmp_path / "storage" / "recordings")
    monkeypatch.setattr(main, "MOCK_STT", True)
    with TestClient(main.app) as api:
        yield api


def approved(client):
    created = client.post("/api/v1/recovery-evidence", data={
        "app_id": "demo-app", "original_transcript": "मुझे पाँच पैकेट चाहिए",
        "first_reply_transcript": "पंद्रह", "confirmed_intent": "15",
        "source_model": "prisma-v3.0", "language_hint": "hi-IN",
        "speaker_confirmed_intent": "true", "explicit_storage_consent": "true",
        "original_mock": "false", "first_reply_mock": "false",
    }, files={"original_audio": ("original.wav", sample_wav()),
              "first_reply_audio": ("reply.wav", sample_wav())})
    assert created.status_code == 201, created.text
    eid = created.json()["id"]
    r = client.post("/api/v1/asr-reviews", json={
        "evidence_id": eid, "reviewer_code": "person-one", "verbatim_reference": "मुझे पंद्रह पैकेट चाहिए",
        "error_category": "number_or_amount", "reviewer_note": "",
        "confirmed_listened": True, "confirmed_verbatim": True,
    })
    assert r.status_code == 201, r.text
    r = client.post("/api/v1/asr-second-reviews", json={
        "evidence_id": eid, "reviewer_code": "person-two", "verbatim_reference": "मुझे पंद्रह पैकेट चाहिए",
        "reviewer_note": "", "confirmed_listened": True, "confirmed_verbatim": True,
    })
    assert r.status_code == 201, r.text
    r = client.post("/api/v1/asr-local-permissions", json={
        "evidence_id": eid, "attested_by": "authorized", "speaker_authorized": True,
        "privacy_screened": True, "local_training_authorized": True,
    })
    assert r.status_code == 201, r.text
    return eid


def paths(tmp_path):
    return {"project_root": tmp_path,
            "db_path": tmp_path / "storage" / "bolifix.sqlite3",
            "audio_dir": tmp_path / "storage" / "recordings" / "recovery_evidence"}


def test_text_synthetic_scoring_is_explicitly_not_model_accuracy():
    result = demo_evaluation.synthetic_demo()
    assert result["real_recordings_used"] == 0
    assert result["model_inference_performed"] is False
    assert result["is_benchmark"] is False
    assert result["test_clip_count"] == 3
    assert result["per_clip"][0]["metrics"]["wer"] == 0.25
    assert result["per_clip"][1]["metrics"]["wer"] == 0
    assert result["micro_wer"] > 0


def test_default_cli_no_real_data_or_dependency(capsys):
    assert run([]) == 0
    out = capsys.readouterr().out
    assert "TEXT-ONLY DEMO" in out
    assert "NOT AN ASR BENCHMARK" in out
    assert "Gnani call" in out


def test_cli_blocks_unconsented_and_accidental_inference():
    with pytest.raises(SystemExit) as err:
        run(["--run-whisper"])
    assert err.value.code == 2
    with pytest.raises(SystemExit) as err:
        run(["--single-approved"])
    assert err.value.code == 2


def test_cli_blocks_show_text_without_inference():
    with pytest.raises(SystemExit) as err:
        run(["--show-text"])
    assert err.value.code == 2


def test_select_one_approved_clip_without_any_split(client, tmp_path):
    eid = approved(client)
    got = demo_evaluation.choose_approved_clip(**paths(tmp_path))
    assert got["row"]["evidence_id"] == eid
    assert got["approved_count"] == 1
    assert got["row"]["split"] == "unassigned_require_speaker_disjoint_split"


def test_one_real_clip_adapter_is_called_once_on_original_audio(client, tmp_path):
    eid = approved(client)
    seen = []

    def adapter(path):
        seen.append(path)
        assert path.stem.endswith("-original")
        assert b"RIFF" == path.read_bytes()[:4]
        return "मुझे पंद्रह पैकेट चाहिए"

    report = demo_evaluation.exploratory_inference(
        **paths(tmp_path), selected_evidence_id=eid, transcribe_one=adapter)
    assert len(seen) == 1
    assert report["metrics"]["wer"] == 0
    assert report["is_benchmark"] is False
    assert report["training_performed"] is False
    assert report["real_recordings_used"] == 1


def test_revocation_blocks_existing_one_clip(client, tmp_path):
    eid = approved(client)
    assert client.delete(f"/api/v1/asr-local-permissions/{eid}").status_code == 200
    with pytest.raises(ValueError, match="no longer authorized"):
        demo_evaluation.exploratory_inference(
            **paths(tmp_path), selected_evidence_id=eid,
            transcribe_one=lambda p: "should never run")


def test_changed_audio_blocks_inference(client, tmp_path):
    eid = approved(client)
    row = demo_evaluation.choose_approved_clip(**paths(tmp_path))["row"]
    original = tmp_path / row["audio_path"]
    original.write_bytes(b"tampered file")
    with pytest.raises(ValueError):
        demo_evaluation.exploratory_inference(
            **paths(tmp_path), selected_evidence_id=eid,
            transcribe_one=lambda p: "should never run")


def test_missing_approval_blocks_single_clip(client, tmp_path):
    with pytest.raises(ValueError, match="No Step 7 authorized"):
        demo_evaluation.choose_approved_clip(**paths(tmp_path))


def test_out_of_range_index_blocks(client, tmp_path):
    approved(client)
    with pytest.raises(ValueError, match="Only 1"):
        demo_evaluation.choose_approved_clip(**paths(tmp_path), index=2)
    with pytest.raises(ValueError):
        demo_evaluation.choose_approved_clip(**paths(tmp_path), index=0)


def test_bad_transcribe_return_rejected(client, tmp_path):
    eid = approved(client)
    with pytest.raises(ValueError, match="return text"):
        demo_evaluation.exploratory_inference(
            **paths(tmp_path), selected_evidence_id=eid, transcribe_one=lambda p: None)


def test_bad_reference_audio_format_rejected(client, tmp_path):
    eid = approved(client)
    row = demo_evaluation.choose_approved_clip(**paths(tmp_path))["row"]
    # Mimic a correctly hashed but unsuitable WAV coming from a stale external record.
    rel = tmp_path / row["audio_path"]
    rel.write_bytes(sample_wav(channels=2))
    forged = dict(row, audio_sha256=hashlib.sha256(rel.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="16kHz mono"):
        demo_evaluation._checked_original(forged, project_root=tmp_path)


def test_absolute_audio_path_rejected(client, tmp_path):
    approved(client)
    row = demo_evaluation.choose_approved_clip(**paths(tmp_path))["row"]
    forged = dict(row, audio_path=str((tmp_path / row["audio_path"]).resolve()))
    with pytest.raises(ValueError, match="unsafe"):
        demo_evaluation._checked_original(forged, project_root=tmp_path)
