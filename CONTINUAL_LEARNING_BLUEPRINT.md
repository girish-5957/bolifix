# BOLIFIX — original call-to-learning architecture

**Core idea:** Recover uncertainty during a real conversation, then make future speech systems better using *separately consented, reviewed* errors. Do not pretend that the ASR model learns automatically from a single live call.

## Two learning speeds

**Immediate, on-call adaptation (implemented for short recorded audio):** Prisma produces a hypothesis; BOLIFIX may identify a critical ambiguous number/name/negation; a question can be spoken by Timbre; a new utterance is transcribed and the person confirms the intended value. The current interaction uses that value. Neither Prisma's nor Whisper's weights change.

**Slow, evidence-driven model improvement (infrastructure partially implemented):** Each organization/application would register an authorized speech event, retain the exact original audio only with separate retention permission, record the original model hypothesis, and store any clarification as **intent evidence**, not ground truth. Human reviewers listen to the original audio, independently write and agree on the verbatim text, and check permissions, provenance, quality and speaker separation. Only then can examples enter a versioned candidate dataset. A separate training job can fine-tune an independent open-weight model, which must be compared with its unchanged baseline on independent, held-out speakers and high-impact fields. Deployment must support rollback.

## Future real-time connection

An authenticated integration will ingest **authorized** audio frames from an actual voice/telephony provider or voice-agent SDK. Gnani's streaming STT documentation and voice-agent interfaces must be implemented and tested; the current Showcase processes **short recorded/uploaded clips** and does not establish real phone calls. A protected event pipeline would preserve per-organization boundaries; no caller audio should be retained by default.

## Current implementation — precise status

| Component | Status and file |
|---|---|
| Prisma clip recognition | Working prototype, `showcase_app.py` and `backend/main.py` |
| Risk/dual-ASR comparison | Rule-based prototype, `backend/dual_asr.py` |
| Targeted spoken clarification and confirmation | Working prototype, `backend/tts.py`, `backend/recovery_flow.py`, `backend/dual_recovery.py` |
| Current-call confirmed-intent memory | Session-only, `showcase_app.py`; no future-call persistence |
| Private opt-in evidence storage | Separate **local research** app, `backend/evidence.py`; **not** auto-connected to Showcase |
| Independent listening and training authorization | Local research tools, `backend/asr_review.py` and `backend/quality_gate.py` |
| Dataset validation/speaker splits | Research tools, `backend/dataset_validation.py` |
| Independent Whisper baseline and exploratory metrics | Research tools, `backend/asr_baseline.py`; 1 approved clip cannot validate model quality |
| Actual fine-tuning of Whisper and controlled deployment | **Not implemented** |
| Model updates based on live real phone calls | **Not implemented** |
| Collaborative cross-app/federated learning | **Not implemented** |
| Production call ingestion, authentication and privacy isolation | **Not implemented** |

## Safe system boundaries

- Transcribing an authorized voice clip is different from keeping that recording. Keeping a recording is different from using it to train a model. Sharing across applications is a separate additional permission.
- A user confirming **15 packets** is *not* evidence that the audio contains the precise words “pandra packet”. The original audio needs independent transcription review to train ASR.
- No raw recordings, API keys, personal information, training permissions or private SQLite databases belong in a public GitHub repository or a demo video.
- Do not automate or record real third-party phone calls without all required permissions and applicable legal compliance.
- Promotion gates: at least three truly different speakers for split validation, sufficient approved examples for credible metrics, untouched test audio, critical number/name/negation regression checks, and rollback.

## Build plan after submission (not claims of current implementation)

1. **Real call adapter:** authenticated telephony/voice-agent connector, streaming PCM ASR, consent and retention events, per-organization session IDs, error handling and latency metrics.
2. **Private evidence service:** safe transport from explicitly opted-in app sessions to encrypted, tenant-separated storage with retention and revocation/deletion controls.
3. **Learning orchestrator:** reviewer queue, separate word-for-word labels, independent review, data minimization, immutable dataset versions.
4. **Trainable ASR service:** reproducible Whisper fine-tuning, evaluated against a frozen baseline on speaker- and app-disjoint test sets; no reliance on Prisma labels unless licensing permits.
5. **Collaborative research:** authorized mock App A and mock App B with strict isolation; compare independent learning versus pooled, appropriately permissioned training; federated learning remains a separate architecture requiring privacy safeguards.
6. **Measured rollout:** promote improvements only if unseen-speaker performance, critical entities, dialect coverage, latency and privacy checks pass; retain a rollback path.

## Competition demonstration

Demo **today**: actual recorded speech -> Prisma result -> targeted Timbre question -> speaker reply -> explicit confirmed intent, optional local Whisper, downloadable session receipt. Then open the **Learning loop** tab to show what happened in the current browser session and which private research tools are available. Introduce ongoing cross-call improvement explicitly as the *research roadmap*, not a working trained model.