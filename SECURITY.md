# Security and responsible use

BOLIFIX is an independent research prototype, not a production-secure platform. Do **not** expose the local research FastAPI/SQLite evidence backend on the public internet without separate authentication, authorization, isolation, encryption and deletion protections.

- Never commit `.env`, access keys, private recordings, recovery transcripts with personal information, databases, cached model files, or local secrets.
- Only process speech recordings with permission; research/training requires **separate explicit permission**.
- A confirmed intended value is **not** a word-for-word training reference. Ground truth requires review of the original audio.
- Unknown and unsupported speech cases must remain uncertain. Two recognizers agreeing does not establish correctness.
- Do not use Gnani API outputs to train another model unless the applicable terms permit it.
- Public Showcase deployments must disable unrestricted provider-funded audio processing and prohibit arbitrary access to any private evidence workflow.

For a suspected vulnerability, contact the repository maintainer privately rather than posting recordings or secrets in a public issue.
