# BOLIFIX — Submission text

## Project title
BOLIFIX — Indian-Language Voice Recovery and Verification Layer

## One-sentence pitch
BOLIFIX helps Indian-language voice applications recover misunderstood instructions through speaker-confirmed clarification and lays the groundwork to improve future recognition using separately consented, human-reviewed corrections.

## Problem and motivation
In multilingual Indian voice workflows, quantity corrections, region-specific names, code-mixed speech and negations can change business outcomes. Speech-recognition text can appear reasonable even when a critical detail is wrong. A one-shot transcript without confirmation is risky for delivery, support, ordering and other voice workflows.

## Solution
BOLIFIX uses Gnani **Prisma v3.0** to transcribe a permissioned audio recording, transparent rules to flag possible high-impact information, Gnani **Timbre v2.5** to speak a targeted question, and Prisma again to capture the speaker's clarification. A bounded dialogue flow proposes an intended value and requires explicit human confirmation. A separate experimental module compares Prisma with local Whisper Small and identifies potential disagreement; it never treats agreement as proof or automatically generates training labels.

## Target users
Voice-agent developers and teams handling Indian-language customer-service and order workflows, especially when small recognition mistakes may lead to consequential actions.

## Models and technologies
- **Gnani Prisma v3.0** — speech-to-text (actual API integration)
- **Gnani Timbre v2.5** — text-to-speech for clarification (actual SDK integration)
- **Whisper Small / faster-whisper** — independent local research hypothesis; not required for the public showcase
- **Python, FastAPI, Streamlit, SQLite** — application, local research API, interface and permissioned records

## What is demonstrably working
Prisma transcription; Hindi/Hinglish flagging; user-triggered Timbre audio; reply transcription; speaker intent confirmation; offline dual-model text comparison; a separate local audio-review workflow with consent and training eligibility gates. The public showcase deliberately uses session-only processing and saves no uploaded recordings.

## Honesty about scope
This is a prototype, not a production voice-agent service. We have **not** fine-tuned Whisper or demonstrated improved generalization. Comparative ASR scores from one clip are not a benchmark. User-confirmed intent is not a verified original word-for-word transcript. Security and tenancy protections are incomplete in the local research backend, so that backend is not deployed publicly.

## The original two-loop innovation
**Recover in this conversation:** Gnani Prisma transcribes, BOLIFIX flags supported risk patterns, Timbre voices a focused follow-up, and the speaker confirms the intended instruction. The present Showcase handles short recordings, not ongoing phone calls.

**Learn across future conversations:** The separate local backend supports private opt-in evidence, application IDs, deletion, independent audio review, and dataset validation. Training an independent ASR model, demonstrating improvement on unseen speakers, and distributing validated model versions across authorized apps remain future experiments. Gnani Prisma itself is not retrained by BOLIFIX.

## Future direction
Expand consented, independently verified datasets; add more Indian language varieties; validate speaker-disjoint evaluation; fine-tune an independent open-weight ASR model; test whether verified corrections improve recognition without harming other speech categories.

## Links to fill before submission
- GitHub repository: [paste after creating]
- Public showcase (optional): [paste after deployment]
- LinkedIn demo post: [paste after publishing]
- X demo post (if used): [paste after publishing]
- Demo video: [paste actual link]

## Contribution
Describe only the work actually completed by each individual/team member. If you used AI-assisted coding, accurately discuss your own decisions, integration, testing, and verified contributions.