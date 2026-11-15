from pathlib import Path
from backend.learning_lifecycle import session_learning_status, LEARNING_STAGES


def test_never_claims_automatic_model_learning():
    data = session_learning_status(audio_present=True, transcript_present=True,
                                   reply_present=True, confirmed_intent=True)
    assert data["speaker_confirmed"] is True
    assert data["scope"] == "browser_session_only"
    assert not data["saved_to_research_db"]
    assert not data["training_label_verified"]
    assert not data["model_weights_updated"]
    assert not data["live_phone_call_connected"]


def test_a_confirmed_value_requires_actual_previous_steps():
    fake = session_learning_status(audio_present=False, transcript_present=True,
                                   reply_present=True, confirmed_intent=True)
    assert all(not s["complete"] for s in fake["runtime"])
    no_reply = session_learning_status(audio_present=True, transcript_present=True,
                                      reply_present=False, confirmed_intent=True)
    assert no_reply["runtime"][1]["complete"]
    assert not no_reply["speaker_confirmed"]


def test_learning_stages_are_transparently_labeled():
    assert any(s[0] == "Open ASR fine-tuning" and s[1] == "Not yet built" for s in LEARNING_STAGES)
    assert any(s[0] == "Collaborative cross-app learning" and s[1] == "Future research" for s in LEARNING_STAGES)


def test_learning_ui_connected_to_real_session_and_no_mutating_calls():
    text = (Path(__file__).resolve().parents[1] / "showcase_app.py").read_text(encoding="utf-8")
    assert '"Learning loop"' in text
    assert 'session_learning_status(' in text
    assert 'st.session_state.get("voice_confirmed")' in text
    assert 'st.session_state.get("voice_transcript")' in text
    assert '"After a call · approved evidence"' in text
    assert "No automatic retraining or cross-app sharing" in text
