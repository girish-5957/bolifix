"""Conservative review of a speech clarification reply.

A second ASR transcription is *not* a reference transcript: it may be wrong too.
The output is only a candidate for the speaker's explicit confirmation.
No recordings, transcripts or labels are stored by this module.
"""
from __future__ import annotations

import re
import unicodedata
from backend.clarification import suggest_clarification

# Narrow vocabulary on purpose; unknown, compound, or conflicting values need
# direct human verification rather than a guessed numeric value.
NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20,
    "fifty": 50, "fifty-five": 55,
    "paanch": 5, "panch": 5, "pandrah": 15, "pandra": 15,
    "pachpan": 55, "pachaas": 50,
    "पाँच": 5, "पांच": 5, "पंद्रह": 15, "पन्दरह": 15,
    "पचपन": 55, "पचास": 50, "दस": 10, "बीस": 20,
}
# Hindi words contain Unicode combining marks, which Python's regex \w can split.
# Tokenize by Unicode letter/mark/number categories to preserve full Hindi words.
DIGITS_PATTERN = re.compile(r"^[0-9०-९]+$")
COMPLEX_OR_APPROXIMATE = {
    "hundred", "thousand", "lakh", "crore", "point", "decimal",
    "around", "about", "approximately", "hundreds",
    "सौ", "हजार", "हज़ार", "लाख", "करोड़", "दशमलव", "लगभग",
}


def word_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    current = ""
    for ch in text.lower():
        if unicodedata.category(ch)[0] in {"L", "M", "N"} or ch == "-":
            current += ch
        elif current:
            tokens.append(current.strip("-"))
            current = ""
    if current:
        tokens.append(current.strip("-"))
    return [token for token in tokens if token]


def number_candidates(text: str) -> list[str]:
    """Extract only simple isolated numbers; preserve ambiguity."""
    results: list[str] = []
    tokens = word_tokens(text)
    # Never mistake "fifteen thousand" for the final number 15.
    if any(token in COMPLEX_OR_APPROXIMATE for token in tokens):
        return []
    for token in tokens:
        if DIGITS_PATTERN.fullmatch(token):
            value = str(int(token.translate(str.maketrans("०१२३४५६७८९", "0123456789"))))
        elif token in NUMBER_WORDS:
            value = str(NUMBER_WORDS[token])
        else:
            continue
        results.append(value)
    return results


def review_reply(original_transcript: str, reply_transcript: str) -> dict:
    """Review a new reply for a *possible* value to confirm with the speaker.

    Never infers that the original transcript is incorrect and never marks the
    reply as verified just because a recognizer produced it.
    """
    analysis = suggest_clarification(original_transcript)
    reply = reply_transcript.strip()
    if not analysis["needs_clarification"]:
        return {
            "status": "no_trigger", "signal_type": "none", "candidate": None,
            "message": "The original transcript has no configured risk trigger. This does not prove it is correct.",
            "requires_speaker_confirmation": True, "training_ready": False,
        }
    signals = analysis["signals"]
    if "identifier" in signals:
        return {
            "status": "manual_review", "signal_type": "identifier", "candidate": None,
            "message": "Please verify the exact identifier with the speaker, preferably character by character or by typing it. Speech recognition alone cannot establish an exact ID.",
            "requires_speaker_confirmation": True, "training_ready": False,
        }
    if "number" in signals:
        values = number_candidates(reply)
        if len(values) == 1:
            return {
                "status": "candidate_needs_confirmation", "signal_type": "number",
                "candidate": values[0],
                "message": "One possible numeric value was found in the reply. Verify it with the speaker before using it; the reply transcription could also be wrong.",
                "requires_speaker_confirmation": True, "training_ready": False,
            }
        return {
            "status": "ambiguous_reply" if values else "no_numeric_candidate",
            "signal_type": "number", "candidate": None,
            "message": "Could not isolate one reliable numeric candidate. Ask the speaker to state one final number, or type it manually.",
            "requires_speaker_confirmation": True, "training_ready": False,
        }
    return {
        "status": "manual_review", "signal_type": "negation_or_correction", "candidate": None,
        "message": "Clarify what action the speaker wants and does not want; automatic semantic resolution is not implemented.",
        "requires_speaker_confirmation": True, "training_ready": False,
    }
