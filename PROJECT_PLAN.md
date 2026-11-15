# BOLIFIX — Product & engineering plan

## Product statement

**BOLIFIX** is a provider-agnostic recovery and continual-learning layer for Indian-language speech recognition. It listens to an authorized application’s speech-transcription events, identifies **potential** errors, obtains additional evidence from a user/reviewer, resolves the current interaction if possible, and converts **permissioned and reviewed** audio-reference pairs into experiments for improving independently trainable ASR models.

**Not promised:** omniscient error detection, instant training after every call, automatic weight updates to proprietary Gnani Prisma, storage of all users' voices, instant reliable coverage of every dialect, or a global first-of-its-kind claim.

## Technical choices

- **Foundation / local prototype:** Python 3.11, FastAPI, `requests` for the documented Prisma REST API, Streamlit, SQLite, Pytest.
- **Live speech phase:** Gnani Prisma v3.0 WebSocket. Streaming protocol and client are documented at `https://docs.gnani.ai/api/STT/stt-websocket` and `https://docs.gnani.ai/python-sdk/speech-to-text`. Mic frames must follow documented 16-bit mono PCM framing, not arbitrary MP3 chunks.
- **Spoken confirmations:** Gnani Timbre v2.5 initially. Choose matching voice/language (the voice `Poorvi` is listed for Hinglish), with short questions rather than long read-backs. See `https://docs.gnani.ai/api/TTS/available-voices`.
- **Fast learning:** use speaker-confirmed intent in the current session, context-aware vocabulary changes only after testing. Prisma v3.0 has `keyterms`, `topic`, `previous_text`; these do **not** modify its model weights.
- **Slow learning:** fine-tune a trainable open-weight ASR baseline (initial candidate Whisper-small; compare IndicConformer). Use PyTorch, Hugging Face, or NeMo based on model. Verify license per selected checkpoint.
- **Production services later:** React + TypeScript, PostgreSQL, object storage, worker queue, MLflow/model registry, Docker and authentication/authorization.

## State machine for runtime repair

1. `RECEIVED`: a recording/stream is available under processing permission.
2. `TRANSCRIBED`: record the model/version and transcript. Do not assume it is true.
3. `LIKELY_CLEAR`: continue the host application's normal workflow with its own safeguards.
4. `NEEDS_CLARIFICATION`: triggered by explicit correction, incompatible candidate transcriptions, a contradictory task state, or critical-field uncertainty. These are *signals*, not proof.
5. `QUESTION_ASKED`: request one targeted detail using TTS/display/DTMF as appropriate.
6. `RESOLVED`: speaker supplies an unambiguous correction (or an authorized human reviewer verifies it). Use corrected state immediately.
7. `UNRESOLVED`: if ambiguity persists, stop an affected high-impact simulated action and refer to manual handling.
8. `REVIEW_PENDING`: optionally store authorized audio and reference; **not automatically training-ready**.

The system must allow: user says the original transcript was correct; user cancels verification; user refuses recording storage; ambiguity cannot be resolved.

## Planned database design

Initial prototype already has `corrections`. Its records are local and single-machine only.

Future PostgreSQL schema:

| Table | Key fields | Purpose |
|---|---|---|
| `organizations` | id, name, region, retention_policy | Data-owner boundaries |
| `applications` | id, org_id, label, allowed_languages | Source app and authorization |
| `sessions` | id, app_id, provider, model_version, created_at | Group related utterances |
| `utterances` | id, session_id, encrypted_audio_ref, duration_ms, locale_hints | Transcribed speech and provenance |
| `hypotheses` | id, utterance_id, provider, model, candidate_text, timing | Do not replace raw model outputs |
| `clarifications` | id, utterance_id, prompt, response, resolution_type | How uncertainty was addressed |
| `references` | id, utterance_id, reference_text, labeler_type, review_status | Potential ground-truth labels |
| `permissions` | id, utterance_id, scope, granted_at, revoked_at | Processing, storage, sharing and deletion |
| `datasets` | id, version, language, license, provenance | Reproducible train/val/test exports |
| `experiments` | id, baseline, candidate, dataset_version, metrics | Track improvements/regressions |
| `deployments` | id, model_version, evaluation_id, rollout_percentage | Controlled rollout and rollback |

Use a private scope by default. Cross-application use requires separate authorization, legal basis, valid licensing and quality review. A verified **intended action** is not automatically a verified **verbatim transcript**.

## ML development methodology

**A. Establish baselines.** Evaluate pristine open-weight checkpoint on held-out Hindi/Hinglish plus one additional supported language. Collect WER/CER, field-specific accuracy, latency and breakdown by conditions.

**B. Obtain training-quality references.** Use explicitly authorized, word-for-word audio-text pairs. Reject synthetic transcript labels falsely treated as human verification. Remove or minimize identifying content, and track acquisition license, data origin, speaker id (pseudonymous), locale and review decisions.

**C. Split correctly.** Split by speaker AND source application where possible. Never let near-duplicate or transformed copies of the same recording appear in both train and test. Keep a locked untouched evaluation set for final results.

**D. Fine-tune candidate model.** Begin with a small, measurable experiment using balanced new and old examples. Update only trainable open models. Version training configuration, seed, data hashes, optimizer and checkpoint. Run on a suitable GPU; separate ML environment is recommended for Windows/WSL2.

**E. Promotion gate.** Compare new vs old on locked test sets. Reject changes that degrade important languages, named entities, numerical expressions, negation, or latency beyond pre-agreed limits. Report uncertainty and sample counts; don't claim improvement from one example.

**F. Test cross-app transfer.** Collect authorized data from mock App A and App B. Train candidate using App A labels, evaluate only on App B unseen-speaker test set. Compare to unchanged baseline and app-specific adaptation. This tests our main network-value hypothesis rather than assuming it works.

## Product success criteria

| Phase | Definition of done |
|---|---|
| 1 | App sends a short consented recording through real Gnani Prisma; shows real result; optional correction is stored locally only with permission; tests pass |
| 2 | A speaker can correct a disputed field in a simulated task; the corrected value changes the simulated outcome; unresolved cases do not silently proceed |
| 3 | Training data can be explicitly reviewed/exported/deleted, with auditable source and consent status |
| 4 | At least one proposed adaptation is compared with a control and held-out evaluation, including harms introduced |
| 5 | Open-weight ASR candidate can be trained and scored reproducibly from labeled examples |
| 6 | Two isolated mock apps demonstrate or disprove positive cross-app transfer, using authorized test data |
| 7 | Production pilot satisfies independent security/privacy review and monitoring requirements |

## Major architectural risks

- **Feedback ambiguity:** "yes" verifies an interpretation, not necessarily the exact sound. Keep meaning and transcript feedback distinct.
- **Confirmation fatigue:** repeated full-sentence playback is too slow. Prefer adaptive targeted questions and multimodal alternatives (spell, DTMF, UI).
- **Detection blind spots:** confidence is not always trustworthy/available. The system must admit unknown errors.
- **Model contamination:** training on noisy or incorrect labels can worsen accuracy. Review and held-out evaluation are required.
- **Catastrophic forgetting:** a new accent/language may improve while an old one worsens. Balance data and test by segment.
- **Data governance:** processing for a task is distinct from retaining/reusing voice recordings for model training.
- **Proprietary dependency:** Prisma is not currently a model we can directly fine-tune through the public REST API. An open-weight track provides genuine learning.
- **Cost/latency:** compare number of extra clarification turns, time-to-resolution, GPU training cost and impact on user task completion.
- **Vendor/version changes:** encapsulate each speech provider behind an adapter. Record exact versions for repeatability.

## Immediate next task (Phase 2)

Implement `ClarificationSession` in the backend with the state machine above, one synthetic delivery-location workflow and Timbre confirmation audio. The first evaluation dataset should include correct names, similar-sounding names, explicit corrections, code-switching, and an unresolved case. All downstream actions must be simulated and reversible.