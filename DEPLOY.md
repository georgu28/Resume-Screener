# Deploying to Streamlit Community Cloud

Step-by-step guide to putting this app online. Free, no card required, and it
redeploys automatically on every push to `main`.

- **Repo:** https://github.com/georgu28/Resume-Screener
- **Branch:** `main`
- **Entry point:** `app.py`
- **Python:** 3.11–3.14 all work. Pick the newest the Streamlit Cloud dropdown
  offers (3.14 if listed). Every dependency — torch, faiss-cpu, scikit-learn,
  numpy, scipy — ships prebuilt wheels through 3.14, so nothing compiles from
  source. 3.11 is only the conservative floor (it's what `.devcontainer` uses).

## 0. Push first

The latest fixes are committed locally but not yet pushed. Streamlit Cloud
deploys from GitHub, so push before you start:

```bash
git push origin main
```

Everything the app needs at runtime is already in the repo: the trained model
(`models/resume_clf.joblib`, 32 MB) is committed, so **no training happens on
deploy**, and the raw dataset is not needed at runtime. The MiniLM embedding
model (~90 MB) downloads from Hugging Face on first launch — that's expected.

## 1. Create the app

1. Go to **https://share.streamlit.io** and **sign in with GitHub** (authorize
   access to the `georgu28/Resume-Screener` repo when prompted).
2. Click **Create app** → **Deploy a public app from GitHub**.
3. Fill in:
   - **Repository:** `georgu28/Resume-Screener`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Open **Advanced settings** and set the **Python version** to the newest
   offered (3.14 if available; any of 3.11–3.14 works).

Don't click Deploy yet — add the secret first (step 2), so the AI Fit Analysis
tab works on the first boot.

## 2. Add the Anthropic API key as a Secret

In **Advanced settings → Secrets**, paste this (TOML format), with your real key:

```toml
ANTHROPIC_API_KEY = "sk-ant-..."
```

The app reads the key from `st.secrets` and falls back to an env var
(`app.py::anthropic_key()`), so this is all it needs. Without the key the app
still runs — the Fit Analysis tab just shows the retrieved job requirements
instead of the Claude-generated write-up.

> You can also add or change this later under the app's **⋮ → Settings →
> Secrets**. Editing secrets reboots the app.

## 3. Deploy

Click **Deploy**. The first build takes several minutes (it installs the torch
stack and downloads MiniLM on first run). When it's live you'll get a public URL
like `https://resume-screener-<hash>.streamlit.app`.

## 4. Smoke-test the live app

- Upload a PDF (e.g. one from `pdfs/`).
- **Analysis Results** tab: job-match similarity bars + resume category.
- **Fit Analysis (AI)** tab: pick a role → **Generate fit analysis** → Claude
  output appears (this is the check that the secret is wired correctly).

---

## Resource limits (read this if the build or app fails)

Streamlit Community Cloud's free tier gives roughly **1 GB RAM**. This app loads
PyTorch + sentence-transformers + FAISS, so it can be tight.

### The torch install is huge by default

Installing `torch` from PyPI pulls the **CUDA build** plus a stack of
`nvidia-*` packages (multiple GB) — pointless here since Streamlit Cloud has no
GPU, and it can blow the build's disk/time budget. To force the smaller
**CPU-only** wheel, add these two lines at the **very top** of
`requirements.txt` (match the version to whatever `torch` resolves to; it was
`2.13.0` at last install):

```
--extra-index-url https://download.pytorch.org/whl/cpu
torch==2.13.0+cpu
```

Commit and push — the next build uses the CPU wheel, which is far smaller and
lighter on RAM.

### If it still OOMs

The model + two transformer stacks can exceed 1 GB at peak. Options, cheapest
first:
1. Confirm the CPU torch wheel above is in effect (check the build logs — you
   should not see `nvidia-*` packages installing).
2. Move to **Hugging Face Spaces** (Streamlit SDK, **16 GB** free RAM) — same
   repo, same `app.py`, set `ANTHROPIC_API_KEY` as a Space secret. This is the
   recommended fallback and has plenty of headroom for the torch stack.

## Cost note

The Anthropic API is pay-per-use and independent of hosting. `rag.py` uses
`MODEL = "claude-sonnet-5"`; swap it to `claude-haiku-4-5` for a cheaper, faster
analysis. One Anthropic key works for every model.

## Redeploys

Streamlit Cloud watches `main` — every push auto-redeploys. To force a rebuild
(e.g. after changing secrets or Python version), use **⋮ → Reboot app**.
