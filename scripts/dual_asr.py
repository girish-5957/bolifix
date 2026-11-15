"""Phase 3 Step 4: side-by-side Prisma vs Whisper on ONE already approved clip.

This is a PRIVATE, read-only exploratory comparison. Not a benchmark or training.
Does not call Gnani, save audio, change SQLite, or download without explicit flag.
"""
from __future__ import annotations

import argparse
import os
import sqlite3
from pathlib import Path

from dotenv import load_dotenv

from backend import demo_evaluation
from backend.dual_asr import compare_hypotheses

ROOT = Path(__file__).resolve().parent.parent


def run(argv=None):
    parser = argparse.ArgumentParser(description="BOLIFIX dual-ASR review (local, not a benchmark)")
    parser.add_argument("--single-approved", action="store_true", help="Select a currently authorized original recording")
    parser.add_argument("--approved-index", type=int, default=1, help="1-based index of an authorized original recording")
    parser.add_argument("--confirm-local-only", action="store_true", help="Confirm permitted local exploratory processing")
    parser.add_argument("--run-whisper", action="store_true", help="Perform local inference; otherwise just check the clip")
    parser.add_argument("--allow-model-download", action="store_true", help="Allow public model weights to download if missing")
    parser.add_argument("--show-text", action="store_true", help="Print private ASR text in this local terminal")
    parser.add_argument("--model", choices=("tiny", "base", "small"), default="small")
    parser.add_argument("--language", choices=("hi", "en", "bn", "mr", "ta", "te", "kn", "gu", "pa", "auto"), default="hi")
    args = parser.parse_args(argv)
    if not args.single_approved or not args.confirm_local_only:
        parser.exit(2, "BLOCKED: --single-approved and --confirm-local-only are required.\n")
    if args.show_text and not args.run_whisper:
        parser.exit(2, "BLOCKED: --show-text requires --run-whisper.\n")
    if args.run_whisper and not args.allow_model_download:
        parser.exit(2, "BLOCKED: add --allow-model-download to permit loading the optional public model.\n")
    load_dotenv(ROOT / ".env")
    db = ROOT / os.getenv("BOLIFIX_DB_PATH", "storage/bolifix.sqlite3")
    audio_dir = ROOT / os.getenv("BOLIFIX_AUDIO_DIR", "storage/recordings") / "recovery_evidence"
    try:
        selected = demo_evaluation.choose_approved_clip(
            project_root=ROOT, db_path=db, audio_dir=audio_dir, index=args.approved_index
        )
        row = selected["row"]
        with sqlite3.connect(db) as con:
            record = con.execute(
                "SELECT original_transcript, source_model FROM recovery_evidence WHERE id = ?",
                (row["evidence_id"],),
            ).fetchone()
        if not record or not str(record[1]).lower().startswith("prisma"):
            raise ValueError("Original record is not labeled as a real Prisma transcription")
        prisma_text = record[0]
    except (ValueError, LookupError, OSError, sqlite3.Error) as exc:
        parser.exit(2, f"BLOCKED: {exc}\n")
    print(f"Approved original clips: {selected['approved_count']}; selected index: {args.approved_index}.")
    print("Current Step 7 permission, reviewer agreement and SHA-256 original-audio integrity checked.")
    print("Original saved Prisma transcript found; no new Gnani request or credits required.")
    if not args.run_whisper:
        print("CHECK ONLY: no model loaded, no second inference, no data written.")
        return 0

    # Set a D:-drive-friendly default on Windows, before importing the Hub model package.
    os.environ.setdefault("HF_HOME", str(ROOT.parent / "model-cache"))
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        parser.exit(2, "Install faster-whisper and compatible PyAV first.\n")
    print(f"Loading local Whisper {args.model}, CPU/int8; first use may download weights.")
    model = WhisperModel(args.model, device="cpu", compute_type="int8")

    def transcribe(path: Path) -> str:
        segments, info = model.transcribe(
            str(path), task="transcribe",
            language=None if args.language == "auto" else args.language,
            beam_size=5, temperature=0, vad_filter=True,
            condition_on_previous_text=False,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
        print(f"Whisper language: {info.language}; audio duration: {info.duration:.2f} sec.")
        return text

    try:
        output = demo_evaluation.exploratory_inference(
            project_root=ROOT, db_path=db, audio_dir=audio_dir,
            selected_evidence_id=row["evidence_id"], transcribe_one=transcribe,
        )
        review = compare_hypotheses(prisma_text, output["prediction"] or "(no speech recognized)")
    except (OSError, ValueError, LookupError) as exc:
        parser.exit(2, f"BLOCKED: {exc}\n")
    print("DUAL-ASR status:", review["status"])
    print("Potential differences:", ", ".join(review["signals"]) or "none after limited normalization")
    print("Number mentions — Prisma:", review["prisma_numbers"], "Whisper:", review["whisper_numbers"])
    if review["suggested_question"]:
        print("Suggested clarification:", review["suggested_question"])
    if args.show_text:
        print("Saved Prisma hypothesis (PRIVATE):", prisma_text)
        print("Local Whisper hypothesis (PRIVATE):", output["prediction"])
    print("NOT A BENCHMARK. No correction, training label, database write, or model-quality verdict produced.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
