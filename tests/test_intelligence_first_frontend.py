"""Product experience communicates scope while retaining independent learning details."""
from pathlib import Path
import ast
APP=(Path(__file__).resolve().parents[1]/"showcase_app.py").read_text(encoding="utf-8")

def test_two_clear_loops_on_homepage():
    assert 'In this conversation' in APP
    assert 'Across future conversations' in APP
    assert 'Prisma recognizes speech. BOLIFIX flags a possible problem. Timbre asks a short question.' in APP
    assert 'With separate consent, reviewers check the original audio.' in APP
    assert 'train and evaluate an independent speech model' in APP

def test_research_stages_remain_detailed_in_learning_tab():
    assert 'st.radio("Follow the intelligence lifecycle"' in APP
    assert 'learning_story[path]' in APP
    assert 'session_learning_status(' in APP
    assert 'session_slot.markdown' in APP
    assert 'No automatic retraining or cross-app sharing' in APP
    assert "Prisma's own model can only be updated by its provider" in APP

def test_live_functions_and_speaker_permission_remain():
    for text in ('st.audio_input(', 'call_prisma(', 'call_timbre(', 'Transcribe with Prisma',
                 'Generate spoken question with Timbre',
                 'I personally verify this is the speaker\'s intended value',
                 'permission to send this audio', 'Download private recovery receipt'):
        assert text.lower() in APP.lower()
    assert 'st.container(border=True)' in APP

def test_app_syntax(): ast.parse(APP)
