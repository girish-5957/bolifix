# Deploy the BOLIFIX Showcase safely

## 1. What you're deploying

`showcase_app.py` is an independent Streamlit frontend with direct Gnani Prisma/Timbre integration. It doesn't use the local FastAPI backend or SQLite and doesn't save audio. The full private research UI is in `frontend/app.py` and should **not** be publicly hosted.

## 2. Run locally first

```powershell
python -m pip install -r requirements.txt
python -m streamlit run showcase_app.py
```

Check that the site loads, compare two *sample text hypotheses* (zero credits), then run one **genuine** Prisma transcription and (optionally) one Timbre response with a non-sensitive recording. Use microphone permission and do not display an API key in the recording.

## 3. Prepare GitHub

The repo must include `showcase_app.py`, `backend/`, `requirements.txt`, `.streamlit/config.toml`, tests and README. **Never upload** `.env`, `.streamlit/secrets.toml`, `storage/bolifix.sqlite3`, recordings, personal information or `model-cache/`.

Use `git status` and `git ls-files` to verify tracked files before pushing.

## 4. Streamlit Community Cloud

1. Open https://share.streamlit.io and sign in using GitHub.
2. Click **Create app**, choose your GitHub repository and `main` branch.
3. Set **Main file path** to `showcase_app.py`.
4. Select a supported Python version (3.12 is reasonable for the showcase; it has fewer dependency surprises than an untested version).
5. Under **Advanced settings → Secrets**, paste:

```toml
BOLIFIX_PUBLIC_DEMO = "true"
```

**Recommended public mode:** Keep `GNANI_API_KEY` absent. Visitors can explore the text comparison and, with their own Gnani key, use live audio. This avoids exposing your programme credits to unrestricted anonymous users.

**Optional restricted live demo:** If you want to provision your own Gnani key privately, set both:

```toml
BOLIFIX_PUBLIC_DEMO = "true"
BOLIFIX_DEMO_ACCESS_CODE = "make-a-strong-random-secret"
GNANI_API_KEY = "your-private-gnani-key"
```

Then disclose the private access code only to authorized reviewers through an appropriate private channel, **not** in a public README or demo video. Even with a shared code, public hosting has usage and abuse risks, so monitor API credits and revoke the key if needed.

6. Deploy, verify the comparison screen works, and check the app logs for any errors. Never paste your actual secrets into a chat, screenshot, GitHub or public documentation.

## 5. What if Community Cloud fails?

Publishing a public working video and a clean GitHub repo is more important than improvising a public database service right before the deadline. Run `showcase_app.py` locally, screen-record the real Prisma/Timbre demo, post the authentic video, and submit the URLs per challenge instructions. Label the public app as a partial demo if live API isn't available to anonymous visitors.

## Glossary

- **GitHub repository:** A tracked folder of code accessible through a Git host.
- **Git commit:** A saved snapshot of the project files.
- **Git push:** Uploading commits from your computer to GitHub.
- **Streamlit Community Cloud:** A hosting service that runs a Python Streamlit app from a GitHub repository.
- **Environment variable:** A configuration value outside your source code.
- **Secret:** A sensitive credential that must not be committed to a public repository.
- **Entrypoint:** The script the hosting service runs to start the application.