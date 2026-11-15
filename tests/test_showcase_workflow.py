from types import SimpleNamespace
import json
import pytest
from backend.showcase_workflow import make_receipt, receipt_json, run_local_whisper


def test_intent_receipt_preserves_provenance_and_does_not_claim_training():
    r = make_receipt("पाँच नहीं पंद्रह", "पंद्रह", "", "15", b"testWAV", "पन्द्रा")
    assert r["record_type"] == "speaker_confirmed_intent_not_asr_ground_truth"
    assert not r["training_data_authorized"] and not r["audio_included"]
    assert r["speaker_confirmed_intended_value"] == "15"
    assert r["original_audio_sha256"] != ""
    assert json.loads(receipt_json(r))["original_transcript_prisma"] == "पाँच नहीं पंद्रह"


def test_receipt_rejects_missing_confirmation():
    with pytest.raises(ValueError):
        make_receipt("hello", "", "", "", b"audio")


def test_receipt_rejects_missing_source():
    with pytest.raises(ValueError):
        make_receipt("hello", "", "", "ok", b"")


class FakeWhisper:
    def __init__(self): self.params = None
    def transcribe(self, audio, **kwargs):
        self.params = kwargs
        assert audio.read() == b"wav-bytes"
        return [SimpleNamespace(text=" पंद्रह ")], SimpleNamespace(language="hi")


def test_local_second_opinion_uses_bytes_without_api_or_disk():
    model = FakeWhisper()
    assert run_local_whisper(b"wav-bytes", model) == "पंद्रह"
    assert model.params["language"] == "hi" and model.params["vad_filter"] is True


def test_local_second_opinion_size_guard():
    with pytest.raises(ValueError):
        run_local_whisper(b"", FakeWhisper())


def test_showcase_has_local_only_guard_and_download_receipt():
    from pathlib import Path
    text=(Path(__file__).resolve().parents[1]/"showcase_app.py").read_text(encoding="utf-8")
    assert 'if not is_public_demo():' in text
    assert 'local_files_only=True' in text
    assert 'st.download_button(' in text
    assert 'voice_whisper_sig' in text
