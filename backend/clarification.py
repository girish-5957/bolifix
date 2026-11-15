"""Explainable, conservative clarification suggestions for Phase 2.

A rule can flag a risky transcript, but cannot verify whether ASR is right.
No audio is processed or stored in this module.
"""
from __future__ import annotations

import re


ID_PATTERN = re.compile(
    r"\b(?:order\s*(?:id|number|no)|booking\s*(?:id|number|no)|"
    r"reference\s*(?:id|number|no)|id|otp|pin)\b|"
    r"ऑर्डर\s*(?:आईडी|नंबर)|आईडी|ओटीपी|पिन",
    flags=re.IGNORECASE,
)
NUMBER_PATTERN = re.compile(
    r"(?<!\w)[0-9०-९]+(?!\w)|"
    r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|"
    r"fifteen|fifty|twenty|thirty|forty|sixty|hundred|thousand|"
    r"paanch|panch|pandrah|pachpan|pachaas)\b|"
    r"पाँच|पांच|पंद्रह|पन्दरह|पचपन|पचास|पाँच|दस|बीस|पच्‍चीस",
    flags=re.IGNORECASE,
)
NEGATION_PATTERN = re.compile(
    r"\b(?:not|no|don't|dont|nahi|nahin|mat)\b|नहीं|नही|मत|ना\b",
    flags=re.IGNORECASE,
)


def suggest_clarification(transcript: str) -> dict:
    """Return risk signals and one suggested question, not an error verdict."""
    transcript = transcript.strip()
    flags: list[str] = []
    if ID_PATTERN.search(transcript):
        flags.append("identifier")
    if NUMBER_PATTERN.search(transcript):
        flags.append("number")
    if NEGATION_PATTERN.search(transcript):
        flags.append("negation_or_correction")

    if "identifier" in flags:
        question = (
            "Could you confirm the final order/reference ID, character by character, "
            "including any correction?"
        )
    elif "number" in flags:
        question = "Please confirm the final number or amount you want us to use."
    elif "negation_or_correction" in flags:
        question = "Please confirm what you want us to do and what you do NOT want us to do."
    else:
        question = None

    return {
        "needs_clarification": bool(flags),
        "signals": flags,
        "suggested_question": question,
        "explanation": (
            "Potentially consequential terms were detected. This does not prove "
            "that the transcript is wrong."
            if flags else
            "No predefined risk signal was detected. This does not prove the transcript is correct."
        ),
        "approach": "keyword_rules_v1",
    }
