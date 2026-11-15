"""Submission-polish checks; no Gnani API requests or stored private audio."""
from pathlib import Path

from backend.dual_asr import compare_hypotheses


APP = (Path(__file__).resolve().parents[1] / 'showcase_app.py').read_text(encoding='utf-8')


def test_guided_progress_is_based_on_session_actions():
    assert 'def render_voice_progress(audio_present: bool)' in APP
    assert 'st.session_state.get("voice_transcript")' in APP
    assert 'st.session_state.get("voice_reply")' in APP
    assert 'st.session_state.get("voice_confirmed")' in APP


def test_scenarios_are_explicitly_synthetic_and_user_initiated():
    assert 'COMPARISON_SCENARIOS' in APP
    assert 'load_comparison_scenario' in APP
    assert 'on_click=load_comparison_scenario' in APP
    assert 'synthetic' in APP.lower()
    assert 'no API calls' in APP


def test_quantity_and_negation_are_flagged():
    quantity = compare_hypotheses('मुझे पाँच पैकेट चाहिए', 'मुझे पंद्रह पैकेट चाहिए')
    assert quantity['potential_high_impact_difference']
    assert quantity['clarification_plan']['route'] == 'confirm_final_number'
    negation = compare_hypotheses('पंद्रह पैकेट नहीं चाहिए', 'पंद्रह पैकेट चाहिए')
    assert negation['potential_high_impact_difference']


def test_scripts_can_match_without_fabricating_ground_truth():
    report = compare_hypotheses('Panch nhi pandra packet chaiye', 'पाँच नहीं पंद्रह पैकेट चाहिए')
    assert report['limited_equivalent']
    assert report['clarification_plan']['route'] == 'review_script_only'


def test_header_fix_and_private_operations_remain_present():
    assert 'header[data-testid="stHeader"]' in APP
    assert 'if public and not gate:' in APP
    assert 'permission to send this audio' in APP.lower()
    assert 'permission to send this reply' in APP.lower()
    assert 'training_data_authorized' not in APP or 'not a training label' in APP.lower()
