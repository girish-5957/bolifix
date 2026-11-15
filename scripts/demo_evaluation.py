"""BOLIFIX Phase 3 Step 3: no-data text demo + optional one-clip local ASR smoke test.

Zero-dependency default:
    python -m scripts.demo_evaluation

Inspect ONLY the current Step 7 authorized list (no Whisper):
    python -m scripts.demo_evaluation --single-approved --confirm-local-only

OPTIONAL, explicitly authorized single-clip local Whisper experiment:
    python -m scripts.demo_evaluation --single-approved --confirm-local-only --run-whisper --allow-model-download --model tiny --language hi

IMPORTANT: none of these is the locked, three-speaker, speaker-disjoint benchmark.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from backend import demo_evaluation

ROOT = Path(__file__).resolve().parent.parent


def run(argv=None):
    parser = argparse.ArgumentParser(description="BOLIFIX evaluation demo; does NOT bypass speaker-split benchmark")
    parser.add_argument("--single-approved", action="store_true", help="Check one currently approved original clip (no split generated)")
    parser.add_argument("--approved-index", type=int, default=1, help="1-based index in currently approved list, default=1")
    parser.add_argument("--confirm-local-only", action="store_true", help="You attest this is permitted local analysis of the original clip")
    parser.add_argument("--run-whisper", action="store_true", help="Run optional local Whisper on the one approved clip")
    parser.add_argument("--allow-model-download", action="store_true", help="Allow fetching public Whisper weights if absent")
    parser.add_argument("--show-text", action="store_true", help="Display private reference/prediction in the local terminal")
    parser.add_argument("--model", choices=["tiny", "base", "small"], default="tiny")
    parser.add_argument("--language", choices=("auto", "hi", "en", "bn", "mr", "ta", "te", "kn", "gu", "pa"), default="hi")
    args = parser.parse_args(argv)
    if args.run_whisper and not args.single_approved:
        parser.exit(2, "BLOCKED: --run-whisper requires --single-approved and explicit local-only confirmation.\n")
    if args.show_text and not args.run_whisper:
        parser.exit(2, "BLOCKED: --show-text requires --run-whisper.\n")
    if not args.single_approved:
        report = demo_evaluation.synthetic_demo()
        print("BOLIFIX TEXT-ONLY DEMO — 3 made-up reference/prediction pairs")
        print("No audio, model inference, Gnani call, download, database access or training.")
        for i, result in enumerate(report["per_clip"], 1):
            metrics = result["metrics"]
            print(f"  Text case {i}: WER={metrics['wer']:.4f}; CER={metrics['cer']:.4f}; "
                  f"S={metrics['substitutions']}, D={metrics['deletions']}, I={metrics['insertions']}")
        print(f"Synthetic-text pooled WER={report['micro_wer']:.4f}; CER={report['micro_cer']:.4f}")
        print("NOT AN ASR BENCHMARK: results are illustrative text-edit measurements, not model accuracy.")
        return 0
    if not args.confirm_local_only:
        parser.exit(2, "BLOCKED: add --confirm-local-only only when you have permission for this local experiment.\n")
    load_dotenv(ROOT / ".env")
    db = ROOT / os.getenv("BOLIFIX_DB_PATH", "storage/bolifix.sqlite3")
    audio = ROOT / os.getenv("BOLIFIX_AUDIO_DIR", "storage/recordings") / "recovery_evidence"
    try:
        selected = demo_evaluation.choose_approved_clip(
            project_root=ROOT, db_path=db, audio_dir=audio, index=args.approved_index,
        )
    except (OSError, ValueError, LookupError) as exc:
        parser.exit(2, f"BLOCKED: {exc}\n")
    row = selected["row"]
    print(f"Currently authorized private clips: {selected['approved_count']}. Selected clip index: {args.approved_index}.")
    print("Original WAV checksum, review agreement and local permission: verified.")
    print("No train/validation/test split created. This recording cannot establish model accuracy.")
    if not args.run_whisper:
        print("CHECK ONLY: no Whisper downloaded/loaded, no audio transcribed and no Gnani credits used.")
        return 0
    if not args.allow_model_download:
        parser.exit(2, "BLOCKED: add --allow-model-download to permit downloading public model weights.\n")
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        parser.exit(2, "Install optional local model: python -m pip install faster-whisper\n")
    print(f"Loading multilingual Whisper {args.model}, CPU/int8. First run may download public weights.")
    model = WhisperModel(args.model, device="cpu", compute_type="int8")

    def transcribe(path: Path) -> str:
        segments, info = model.transcribe(
            str(path), task="transcribe", language=None if args.language == "auto" else args.language,
            beam_size=5, temperature=0, vad_filter=True, condition_on_previous_text=False,
        )
        print("Language used:", info.language)
        print("Original audio duration:", round(info.duration, 2), "seconds")
        print("Duration after VAD:", round(info.duration_after_vad, 2), "seconds")
        return " ".join(s.text.strip() for s in segments).strip()

    try:
        result = demo_evaluation.exploratory_inference(
            project_root=ROOT, db_path=db, audio_dir=audio,
            selected_evidence_id=row["evidence_id"], transcribe_one=transcribe,
        )
    except (OSError, ValueError, LookupError) as exc:
        parser.exit(2, f"INFERENCE BLOCKED: {exc}\n")
    m = result["metrics"]
    print(f"ONE-CLIP exploratory text comparison: WER={m['wer']:.4f}, CER={m['cer']:.4f}.")
    if args.show_text:
        print("Human reference (PRIVATE):", result["reference"])
        print("Whisper prediction (PRIVATE):", result["prediction"])
    print("NOT A BENCHMARK. Not speaker-disjoint; no training, file saved, cloud ASR, or Gnani usage.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
