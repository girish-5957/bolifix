"""Frontend contract: readable real product, not a simulated transcript."""
from pathlib import Path
import ast
APP=(Path(__file__).resolve().parents[1]/"showcase_app.py").read_text(encoding="utf-8")

def test_homepage_explains_product_and_future_separately():
    assert "Recover the meaning now." in APP
    assert "One missed number can change an order." in APP
    assert "Learn from verified corrections" in APP
    assert "Working short-recording experience" in APP
    assert "model retraining and cross-app rollout are not yet live" in APP

def test_voice_capture_is_first_class_and_consent_preserved():
    assert 'input_mode = st.radio(' in APP
    assert "Record with microphone" in APP and "Upload an audio file" in APP
    assert 'microphone = st.audio_input(' in APP
    assert 'upload = st.file_uploader(' in APP
    assert 'src = microphone or upload' in APP
    assert 'with st.expander("Language & transcription preferences"' in APP
    assert "permission to send this audio" in APP.lower()

def test_existing_live_paths_preserved():
    for phrase in ('call_prisma(', 'call_timbre(', 'voice_confirmed', 'voice_receipt_download',
                   'voice_local_whisper', 'BOLIFIX_PUBLIC_DEMO', 'if public and not gate:'):
        assert phrase in APP

def test_syntax(): ast.parse(APP)

def test_homepage_has_no_fabricated_call_result():
    assert "bf-demo-panel" not in APP
    assert "15 packets confirmed" not in APP
    assert "Illustrative example · not a live AI result" in APP

def test_gnani_cannot_run_automatically():
    assert "disabled=not (src and consent and active_key())" in APP
    assert 'if st.button("Generate spoken question with Timbre"' in APP
    assert 'if public and not gate:' in APP
