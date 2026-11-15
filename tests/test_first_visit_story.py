"""First-time visitor explanation, grounded in what is built."""
from pathlib import Path
import ast

APP = (Path(__file__).resolve().parents[1] / "showcase_app.py").read_text(encoding="utf-8")

def test_story_explains_customer_problem_and_intervention():
    for phrase in (
        "recover misunderstood instructions before acting",
        "One missed number can change an order.", "Possible recognition mistake",
        "BOLIFIX asks for clarity", "The speaker decides",
        "See an example: why a single number matters",
    ):
        assert phrase in APP

def test_demo_is_separate_from_model_learning_and_fabricated_results():
    for phrase in (
        "Illustrative example · not a live AI result", "separate permission",
        "reviewers check the original audio", "train and evaluate an independent speech model",
        "model retraining and cross-app rollout are not yet live",
        "Working short-recording experience",
    ):
        assert phrase in APP
    assert "BOLIFIX_PUBLIC_DEMO" in APP

def test_no_functional_regression_in_voice_path():
    ast.parse(APP)
    for phrase in (
        "st.audio_input(", "st.file_uploader(", "call_prisma(", "call_timbre(",
        "voice_confirmed", "disabled=not (src and consent and active_key())",
        "run_local_whisper", "BOLIFIX_DEMO_ACCESS_CODE",
    ):
        assert phrase in APP
