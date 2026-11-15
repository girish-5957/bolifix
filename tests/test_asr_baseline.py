"""Phase 3 Step 2: offline local baseline guard and scoring tests."""
import hashlib
import io
import json
import wave
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend import asr_baseline, main


def wave_bytes(value):
    out = io.BytesIO()
    with wave.open(out, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(16000)
        f.writeframes(bytes([value]) * 16000 * 2)
    return out.getvalue()


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "ROOT", tmp_path)
    monkeypatch.setattr(main, "DB_PATH", tmp_path / "storage" / "bolifix.sqlite3")
    monkeypatch.setattr(main, "AUDIO_DIR", tmp_path / "storage" / "recordings")
    monkeypatch.setattr(main, "MOCK_STT", True)
    with TestClient(main.app) as api:
        yield api


def create_approved(client, value):
    res = client.post("/api/v1/recovery-evidence", data={
        "app_id": "demo-app", "original_transcript": "मुझे पाँच पैकेट चाहिए",
        "first_reply_transcript": "पंद्रह", "confirmed_intent": "15",
        "source_model": "prisma-v3.0", "language_hint": "hi-IN",
        "speaker_confirmed_intent": "true", "explicit_storage_consent": "true",
        "original_mock": "false", "first_reply_mock": "false",
    }, files={"original_audio": ("original.wav", wave_bytes(value)),
              "first_reply_audio": ("reply.wav", wave_bytes(55))})
    assert res.status_code == 201, res.text
    eid = res.json()["id"]
    res = client.post("/api/v1/asr-reviews", json={
        "evidence_id": eid, "reviewer_code": "reviewer-1",
        "verbatim_reference": "मुझे पंद्रह पैकेट चाहिए", "error_category": "number_or_amount",
        "reviewer_note": "", "confirmed_listened": True, "confirmed_verbatim": True,
    })
    assert res.status_code == 201, res.text
    res = client.post("/api/v1/asr-second-reviews", json={
        "evidence_id": eid, "reviewer_code": "reviewer-2",
        "verbatim_reference": "मुझे पंद्रह पैकेट चाहिए", "reviewer_note": "",
        "confirmed_listened": True, "confirmed_verbatim": True,
    })
    assert res.status_code == 201, res.text
    res = client.post("/api/v1/asr-local-permissions", json={
        "evidence_id": eid, "attested_by": "demo", "speaker_authorized": True,
        "privacy_screened": True, "local_training_authorized": True,
    })
    assert res.status_code == 201, res.text
    return eid


def approved_manifest(client):
    ids = [create_approved(client, i + 1) for i in range(3)]
    res = client.post("/api/v1/asr-split-manifest", json={
        "speaker_assignments": [{"evidence_id": eid, "speaker_code": f"spk-{i+1:02}"}
                                for i, eid in enumerate(ids)],
        "seed": 42, "confirm_local_pilot_only": True,
    })
    assert res.status_code == 200, res.text
    return [json.loads(line) for line in res.text.splitlines()]


def checked(rows, tmp_path):
    return asr_baseline.validate_manifest(
        rows=rows, project_root=tmp_path,
        db_path=tmp_path / "storage" / "bolifix.sqlite3",
        audio_dir=tmp_path / "storage" / "recordings" / "recovery_evidence")


def test_manifest_load_and_check(client, tmp_path):
    rows = approved_manifest(client)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf8")
    read_rows = asr_baseline.load_manifest(manifest)
    report = checked(read_rows, tmp_path)
    assert report["test_clip_count"] == 1
    assert report["clip_count"] == 3
    assert report["test_speaker_count"] == 1
    assert len(report["manifest_sha256"]) == 64


def test_readiness_does_not_need_model(client, tmp_path):
    rows = approved_manifest(client)
    report = checked(rows, tmp_path)
    assert report["note"].startswith("Small pilot")
    assert "tiny" not in report


@pytest.mark.parametrize("change", [
    lambda rows: rows[0].update(verbatim_transcript="wrong"),
    lambda rows: rows[0].update(audio_sha256="0" * 64),
    lambda rows: rows[0].update(split="train" if rows[0]["split"] != "train" else "test"),
    lambda rows: rows[0].update(speaker_group="not-an-approved-speaker-format"),
    lambda rows: rows[0].update(approval="unapproved"),
    lambda rows: rows[0].update(training_ready=True),
])
def test_tampered_manifest_rejected(client, tmp_path, change):
    rows = approved_manifest(client)
    change(rows)
    with pytest.raises(ValueError):
        checked(rows, tmp_path)


def test_missing_or_duplicate_manifest_rows_rejected(client, tmp_path):
    rows = approved_manifest(client)
    with pytest.raises(ValueError):
        checked(rows[:-1], tmp_path)
    with pytest.raises(ValueError):
        checked(rows + [dict(rows[0])], tmp_path)


def test_revoked_permission_blocks_old_manifest(client, tmp_path):
    rows = approved_manifest(client)
    assert client.delete(f"/api/v1/asr-local-permissions/{rows[0]['evidence_id']}").status_code == 200
    with pytest.raises(ValueError, match="outdated|incomplete"):
        checked(rows, tmp_path)


def test_changed_audio_blocks_old_manifest(client, tmp_path):
    rows = approved_manifest(client)
    path = tmp_path / rows[0]["audio_path"]
    path.write_bytes(b"changed original")
    with pytest.raises(ValueError):
        checked(rows, tmp_path)


def test_model_only_gets_heldout_audio(client, tmp_path):
    rows = approved_manifest(client)
    test_rows = checked(rows, tmp_path)["test_rows"]
    seen = []

    def fake_model(path):
        seen.append(path)
        return "मुझे पंद्रह पैकेट चाहिए"

    predictions = asr_baseline.predict_with_model(
        test_rows, project_root=tmp_path, transcribe_one=fake_model)
    assert len(seen) == 1
    assert seen[0].name.endswith("-original.wav")
    metrics = asr_baseline.evaluate_transcripts(test_rows, predictions)
    assert metrics["micro_wer"] == 0
    assert metrics["micro_cer"] == 0
    assert metrics["training_performed"] is False


def test_metric_micro_average_and_number_error():
    rows = [
        {"evidence_id": "a", "verbatim_transcript": "मुझे पंद्रह पैकेट चाहिए"},
        {"evidence_id": "b", "verbatim_transcript": "पाँच"},
    ]
    out = asr_baseline.evaluate_transcripts(rows, {"a": "मुझे पाँच पैकेट चाहिए", "b": "पाँच"})
    assert out["totals"]["substitutions"] == 1
    assert out["totals"]["reference_words"] == 5
    assert out["micro_wer"] == 0.2
    assert out["micro_cer"] > 0
    assert out["test_clip_count"] == 2


def test_empty_prediction_is_scored_as_deletion():
    out = asr_baseline.evaluate_transcripts(
        [{"evidence_id": "test", "verbatim_transcript": "हां जी"}], {"test": ""})
    assert out["micro_wer"] == 1
    assert out["totals"]["deletions"] == 2


def test_prediction_missing_id_rejected():
    with pytest.raises(ValueError, match="exactly"):
        asr_baseline.evaluate_transcripts(
            [{"evidence_id": "foo", "verbatim_transcript": "नमस्ते"}], {})


def test_rejects_changed_clip_before_running_model(client, tmp_path):
    rows = approved_manifest(client)
    test_rows = checked(rows, tmp_path)["test_rows"]
    (tmp_path / test_rows[0]["audio_path"]).write_bytes(b"changed")
    called = []
    with pytest.raises(ValueError, match="modified"):
        asr_baseline.predict_with_model(
            test_rows, project_root=tmp_path, transcribe_one=lambda p: called.append(p))
    assert called == []


@pytest.mark.parametrize("text", ["", "not-json\n", '{"evidence_id":"1"}\n{"evidence_id":"1"}\n'])
def test_invalid_manifest_jsonl_rejected(tmp_path, text):
    source = tmp_path / "bad.jsonl"
    source.write_text(text, encoding="utf8")
    with pytest.raises(ValueError):
        asr_baseline.load_manifest(source)


def test_does_not_allow_audio_path_escape(client, tmp_path):
    rows = approved_manifest(client)
    check = checked(rows, tmp_path)
    test = check["test_rows"]
    test[0]["audio_path"] = "../somewhere.wav"
    with pytest.raises(ValueError, match="Unsafe"):
        asr_baseline.predict_with_model(test, project_root=tmp_path, transcribe_one=lambda p: "")


def test_cli_check_only_prints_no_download(client, tmp_path, monkeypatch, capsys):
    from scripts import asr_baseline as cli
    rows = approved_manifest(client)
    manifest = tmp_path / "splits.jsonl"
    manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf8")
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.delenv("BOLIFIX_DB_PATH", raising=False)
    monkeypatch.delenv("BOLIFIX_AUDIO_DIR", raising=False)
    assert cli.run(["--manifest", str(manifest)]) == 0
    output = capsys.readouterr().out
    assert "CHECK ONLY" in output
    assert "held-out test" in output
    assert "faster-whisper" in output


def test_cli_requires_explicit_consent_before_loading_model(client, tmp_path, monkeypatch):
    from scripts import asr_baseline as cli
    rows = approved_manifest(client)
    manifest = tmp_path / "splits.jsonl"
    manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf8")
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.delenv("BOLIFIX_DB_PATH", raising=False)
    monkeypatch.delenv("BOLIFIX_AUDIO_DIR", raising=False)
    with pytest.raises(SystemExit) as exit_info:
        cli.run(["--manifest", str(manifest), "--run"])
    assert exit_info.value.code == 2
