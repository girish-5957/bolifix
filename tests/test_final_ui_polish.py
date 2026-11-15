"""Check legibility, responsible claims, and the editable live workflow."""
from pathlib import Path
APP = (Path(__file__).resolve().parents[1] / "showcase_app.py").read_text(encoding="utf-8")

def test_primary_button_nested_text_is_readable_when_enabled_or_disabled():
    assert '[data-testid="stButton"] button[kind="primary"]:not(:disabled) *' in APP
    assert 'color:#ffffff!important' in APP
    assert '[data-testid="stButton"] button[kind="primary"]:disabled *' in APP
    assert 'color:#415d54!important' in APP

def test_language_scope_is_clear_not_unverified():
    assert 'Designed for Indian-language voice services' in APP
    assert 'Live demo tested primarily with Hindi and Hinglish' in APP
    assert 'other language options require validation' in APP

def test_examples_do_not_block_main_live_controls():
    assert 'st.expander("See an example: why a single number matters", expanded=False)' in APP
    assert 'Voice Recovery Studio' in APP
    assert 'Record with microphone' in APP
    assert 'st.button("Transcribe with Prisma", type="primary", disabled=not (src and consent and active_key())' in APP
