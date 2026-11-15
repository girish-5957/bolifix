"""Phase 3 Step 4: conservative comparison of two ASR hypotheses.

No model calls, database access, audio processing or storage here. Text disagreement
is a *review signal*, not ground truth or a model-quality judgement.
"""
from __future__ import annotations

import re
import unicodedata

from backend.clarification import suggest_clarification

# Deliberately small transparent mappings. Do NOT collapse 5 and 15, or
# silently translate full Romanized Hindi into Devanagari.
NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "ten": "10", "fifteen": "15", "twenty": "20", "fifty": "50",
    "panch": "5", "paanch": "5", "panchh": "5",
    "pandra": "15", "pandrah": "15", "pandraah": "15",
    "पाँच": "5", "पांच": "5", "पाच": "5", "पंच": "5",
    "पंद्रह": "15", "पन्द्रह": "15", "पन्दरा": "15", "पन्द्रा": "15",
    "दस": "10", "पचास": "50", "बीस": "20", "पच्चीस": "25",
}
NEGATION_WORDS = {"नहीं", "नही", "नहिं", "nahi", "nahin", "nhi", "not", "no", "मत", "mat"}
# Match Devanagari combining marks as part of one lexical token.
TOKEN_RE = re.compile(r"[\u0900-\u097f]+|[A-Za-z]+(?:['’][A-Za-z]+)?|[0-9०-९]+")
DEVANAGARI_RE = re.compile(r"[\u0900-\u097f]")
LATIN_RE = re.compile(r"[A-Za-z]")


def _ascii_digits(token: str) -> str:
    return "".join(str(unicodedata.digit(char)) if char.isdigit() else char for char in token)


def analysis_tokens(transcript: str) -> list[str]:
    if not isinstance(transcript, str) or not transcript.strip() or len(transcript) > 2000:
        raise ValueError("Transcript must contain 1–2000 characters")
    tokens = TOKEN_RE.findall(unicodedata.normalize("NFKC", transcript).casefold())
    result = []
    for token in tokens:
        standardized = _ascii_digits(token)
        if standardized.isdecimal():
            result.append("num:" + str(int(standardized)))
        elif standardized in NUMBER_WORDS:
            result.append("num:" + NUMBER_WORDS[standardized])
        elif standardized in NEGATION_WORDS:
            result.append("negation")
        else:
            result.append(standardized)
    return result


# Only a tiny set of *reviewed demonstration words*, not a transliterator.
# Unknown place names and words are intentionally never auto-equated.
CROSS_SCRIPT_WORDS = {
    "mujhe": "मुझे", "mujhey": "मुझे",
    "chahiye": "चाहिए", "chaiye": "चाहिए", "chahie": "चाहिए",
    "चाहिये": "चाहिए",
    "packet": "पैकेट", "packets": "पैकेट", "पैकेट्स": "पैकेट",
    "peket": "पैकेट",
    "hai": "है", "he": "है",
}


def limited_comparison_tokens(transcript: str) -> list[str]:
    """Map *known* number and word variants for display, never for training.

    No phonetic similarity, fuzzy matching, or general transliteration is used.
    """
    return [CROSS_SCRIPT_WORDS.get(token, token) for token in analysis_tokens(transcript)]


def choose_clarification_route(
    *,
    number_difference: bool,
    negation_difference: bool,
    equal_after_limited_mapping: bool,
    scripts_differ: bool,
    transcript_difference: bool,
    risk_signals: list[str],
) -> dict:
    """Propose a safe question. A model disagreement is *never* an error verdict."""
    if number_difference:
        code = "confirm_final_number"
        hi = "कृपया केवल अंतिम सही संख्या स्पष्ट रूप से एक बार बताइए।"
        en = "Please say only the final intended number once, clearly."
    elif negation_difference:
        code = "confirm_action_and_negation"
        hi = "कृपया साफ़ बताइए कि क्या करना है और क्या नहीं करना है।"
        en = "Please confirm what should happen and what must not happen."
    elif transcript_difference and scripts_differ and equal_after_limited_mapping:
        # Romanized Hindi and Devanagari may look different but match in our
        # limited dictionary. Do not spend a TTS call for spelling alone.
        code, hi, en = "review_script_only", None, None
    elif transcript_difference and scripts_differ:
        # We do not know how to transliterate arbitrary names; an automated
        # spoken prompt would risk suggesting that a specific word is wrong.
        code, hi, en = "manual_listen", None, None
    elif transcript_difference:
        code = "repeat_key_word"
        hi = "कृपया महत्वपूर्ण नाम या शब्द दोबारा धीरे-धीरे बोलिए।"
        en = "Please repeat the important name or word slowly."
    elif "identifier" in risk_signals:
        code = "confirm_identifier"
        hi = "कृपया अंतिम आईडी के अक्षर और अंक एक-एक करके बताइए।"
        en = "Please confirm the final ID, character by character."
    elif "number" in risk_signals:
        code = "confirm_final_number"
        hi = "कृपया केवल अंतिम सही संख्या स्पष्ट रूप से एक बार बताइए।"
        en = "Please say only the final intended number once, clearly."
    elif "negation_or_correction" in risk_signals:
        code = "confirm_action_and_negation"
        hi = "कृपया साफ़ बताइए कि क्या करना है और क्या नहीं करना है।"
        en = "Please confirm what should happen and what must not happen."
    else:
        code, hi, en = "no_prompt", None, None
    return {
        "route": code,
        "question_hi": hi,
        "question_en": en,
        "can_speak": hi is not None and en is not None,
        "verification_required": code != "no_prompt",
        "maximum_automatic_retries": 0,
        "note": (
            "A proposed question does not establish which recognizer is correct. "
            "Speaker intent confirmation is not a verified original-audio transcript. "
            "This demo does not start a reply session or save a record."
        ),
    }


def _script_category(text: str) -> str:
    latin, dev = bool(LATIN_RE.search(text)), bool(DEVANAGARI_RE.search(text))
    return "mixed" if latin and dev else "devanagari" if dev else "latin" if latin else "digits_only"


def compare_hypotheses(prisma_transcript: str, whisper_transcript: str) -> dict:
    """Flag potential differences without electing either ASR as correct."""
    p = analysis_tokens(prisma_transcript)
    w = analysis_tokens(whisper_transcript)
    p_numbers = [t[4:] for t in p if t.startswith("num:")]
    w_numbers = [t[4:] for t in w if t.startswith("num:")]
    p_neg, w_neg = "negation" in p, "negation" in w
    script_p, script_w = _script_category(prisma_transcript), _script_category(whisper_transcript)
    scripts_differ = script_p != script_w and script_p != "digits_only" and script_w != "digits_only"
    equal = p == w
    signals = []
    if p_numbers != w_numbers:
        signals.append("number_mentions_differ")
    if p_neg != w_neg:
        signals.append("negation_presence_differs")
    if scripts_differ:
        signals.append("writing_systems_differ")
    if not equal:
        signals.append("transcripts_differ")
    critical = "number_mentions_differ" in signals or "negation_presence_differs" in signals
    if equal:
        status = "no_difference_after_limited_normalization"
        question = None
    elif critical:
        status = "priority_human_review"
        question = (
            "Please confirm every number mentioned and the final intended number."
            if "number_mentions_differ" in signals else
            "Please confirm what you want to do and what you do NOT want to do."
        )
    elif scripts_differ:
        status = "script_review_needed"
        question = None
    else:
        status = "human_review_needed"
        question = "Please repeat or spell the important word, name, or instruction."
    # A risk signal in one transcript alone is still useful, even when models agree.
    prisma_risks = suggest_clarification(prisma_transcript)["signals"]
    whisper_risks = suggest_clarification(whisper_transcript)["signals"]
    limited_p = limited_comparison_tokens(prisma_transcript)
    limited_w = limited_comparison_tokens(whisper_transcript)
    limited_equal = limited_p == limited_w
    # Retain the original v1 status/signals so existing consumers keep working.
    # Add an explicitly LIMITED interpretation, not a full Romanization engine.
    if limited_equal and scripts_differ:
        interpretation = "limited_cross_script_match"
    elif critical:
        interpretation = "critical_content_disagreement"
    elif scripts_differ:
        interpretation = "unresolved_cross_script_difference"
    elif equal:
        interpretation = "no_difference_after_limited_mapping"
    else:
        interpretation = "wording_difference_needs_review"
    risk_all = list(dict.fromkeys(prisma_risks + whisper_risks))
    routing = choose_clarification_route(
        number_difference="number_mentions_differ" in signals,
        negation_difference="negation_presence_differs" in signals,
        equal_after_limited_mapping=limited_equal,
        scripts_differ=scripts_differ,
        transcript_difference=not equal,
        risk_signals=risk_all,
    )
    return {
        "status": status,
        "signals": signals,
        "potential_high_impact_difference": critical,
        "human_review_recommended": not equal or bool(prisma_risks),
        "suggested_question": question,
        "prisma_numbers": p_numbers,
        "whisper_numbers": w_numbers,
        "prisma_script": script_p,
        "whisper_script": script_w,
        "prisma_risk_signals": prisma_risks,
        "whisper_risk_signals": whisper_risks,
        "method": "limited_rule_based_dual_asr_v2",
        "limited_equivalent": limited_equal,
        "limited_interpretation": interpretation,
        "prisma_limited_tokens": limited_p,
        "whisper_limited_tokens": limited_w,
        "clarification_plan": routing,
        "is_accuracy_score": False,
        "is_training_label": False,
        "note": (
            "Disagreement or agreement cannot establish which recognizer is correct. "
            "Mixed writing systems and incomplete number-word dictionaries may cause false alarms. "
            "Only human listening to the original recording can verify exact words. "
            "The limited cross-script mapping is NOT a WER/CER normalization rule or training label."
        ),
    }
