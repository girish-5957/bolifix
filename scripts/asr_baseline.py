"""Run BOLIFIX Phase 3 Step 2's offline/local, independent Whisper baseline.

Usage from the project directory:
  python -m scripts.asr_baseline --manifest path/to/splits.jsonl
  python -m scripts.asr_baseline --manifest path/to/splits.jsonl --run --confirm-local-only --allow-model-download

Default = safety check only, with zero network/model/Gnani requests.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from backend import asr_baseline

ROOT = Path(__file__).resolve().parent.parent


def run(argv=None):
    parser = argparse.ArgumentParser(description="BOLIFIX private held-out Whisper baseline")
    parser.add_argument("--manifest", required=True, type=Path, help="Previously downloaded Step 1 speaker splits JSONL")
    parser.add_argument("--run", action="store_true", help="Actually run local Whisper inference (default is validation only)")
    parser.add_argument("--confirm-local-only", action="store_true", help="Confirm all recordings are privately permissioned")
    parser.add_argument("--allow-model-download", action="store_true", help="Permit public model weights download/cache if not already present")
    parser.add_argument("--model", choices=("tiny", "base"), default="tiny", help="Multilingual independent Whisper model")
    parser.add_argument("--language", default="auto", choices=("auto", "hi", "en", "bn", "mr", "ta", "te", "kn", "gu", "pa"))
    args = parser.parse_args(argv)
    load_dotenv(ROOT / ".env")
    db = ROOT / os.getenv("BOLIFIX_DB_PATH", "storage/bolifix.sqlite3")
    audio_dir = ROOT / os.getenv("BOLIFIX_AUDIO_DIR", "storage/recordings") / "recovery_evidence"
    try:
        rows = asr_baseline.load_manifest(args.manifest)
        check = asr_baseline.validate_manifest(rows=rows, project_root=ROOT, db_path=db,
                                               audio_dir=audio_dir)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"BLOCKED: {exc}\n")
    print(f"Permission/audio/split verification: PASS. {check['clip_count']} total clips; "
          f"{check['test_clip_count']} held-out test clip(s); {check['test_speaker_count']} test speaker(s).")
    print(f"Manifest SHA-256: {check['manifest_sha256']}")
    for notice in check["warnings"]:
        print(f"CAUTION — {notice['code']}: {notice['message']}")
    print("Small or unrepresentative tests cannot establish general model accuracy.")
    if not args.run:
        print("CHECK ONLY: no model loaded, no audio transcribed, no download, no Gnani credits.")
        print("To run, install faster-whisper and add --run --confirm-local-only --allow-model-download.")
        return 0
    if not args.confirm_local_only:
        parser.exit(2, "BLOCKED: --confirm-local-only is required to transcribe approved recordings.\n")
    if not args.allow_model_download:
        parser.exit(2, "BLOCKED: add --allow-model-download to acknowledge the model may download public weights.\n")
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        parser.exit(2, "Install optional local inference package: python -m pip install faster-whisper\n")
    print(f"Loading local multilingual Whisper '{args.model}' (CPU/int8). "
          "The first run may download public model weights; recordings stay local.")
    model = WhisperModel(args.model, device="cpu", compute_type="int8")

    def transcribe(path: Path) -> str:
        segments, _ = model.transcribe(str(path), task="transcribe", language=None if args.language == "auto" else args.language,
                                       beam_size=1, temperature=0, vad_filter=False,
                                       condition_on_previous_text=False)
        return " ".join(seg.text.strip() for seg in segments).strip()

    try:
        predictions = asr_baseline.predict_with_model(
            check["test_rows"], project_root=ROOT, transcribe_one=transcribe)
        result = asr_baseline.evaluate_transcripts(check["test_rows"], predictions)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"EVALUATION BLOCKED: {exc}\n")
    print(f"Held-out test micro-WER: {result['micro_wer']:.4f}")
    print(f"Held-out test micro-CER: {result['micro_cer']:.4f}")
    print(f"Test clips: {result['test_clip_count']} | model={args.model}, language={args.language}, beam=1, CPU/int8")
    print("No training, no Gnani API calls. No transcript or report files saved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
