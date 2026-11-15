"""Clear product copy and honest learning status, not an artificial model result."""
from pathlib import Path

APP = (Path(__file__).resolve().parents[1] / "showcase_app.py").read_text(encoding="utf-8")

def test_homepage_states_both_loops_in_plain_language():
    for sentence in (
        "Recover the meaning now.", "Learn for the next conversation.",
        "BOLIFIX helps Indian-language voice applications",
        "It asks a focused question and lets the speaker confirm.",
        "human-reviewed recordings", "Adapt after evaluation",
    ):
        assert sentence in APP

def test_no_false_claim_of_live_learning():
    assert "model retraining and cross-app rollout are not yet live" in APP
    assert "Working short-recording experience" in APP
    assert "separate permission" in APP
    assert "15 packets confirmed" not in APP

def test_unchanged_audio_permission_and_voice_controls():
    assert 'disabled=not (src and consent and active_key())' in APP
    assert 'st.audio_input(' in APP
    assert 'st.file_uploader(' in APP
    assert 'Transcribe with Prisma' in APP
    assert 'Generate spoken question with Timbre' in APP
