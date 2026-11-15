"""Offline quality gates for the public-facing showcase."""
from pathlib import Path


def test_showcase_contains_real_calls_not_fake_results():
    text=(Path(__file__).resolve().parents[1]/"showcase_app.py").read_text(encoding="utf-8")
    assert 'https://api.vachana.ai/stt/v3' in text
    assert 'GnaniTTSClient' in text
    assert 'simulated' not in text.lower().split('def call_prisma')[1].split('def call_timbre')[0]


def test_showcase_has_no_database_and_requires_permission():
    text=(Path(__file__).resolve().parents[1]/"showcase_app.py").read_text(encoding="utf-8")
    assert 'sqlite3' not in text and 'storage/recordings' not in text
    assert 'permission to send this audio' in text.lower()
    assert 'permission to send this reply' in text.lower()


def test_public_mode_requires_key_or_access_control():
    text=(Path(__file__).resolve().parents[1]/"showcase_app.py").read_text(encoding="utf-8")
    assert 'BOLIFIX_PUBLIC_DEMO' in text
    assert 'if public and not gate:' in text
    assert 'BOLIFIX_DEMO_ACCESS_CODE' in text
