# Handoff

Context for the next agent working on this repo.

## What this project is

A Streamlit ML app that (1) **classifies** resumes into job categories, (2) **ranks**
them against open roles by semantic similarity, and (3) generates a **grounded fit
analysis** via RAG. It exists to back a specific résumé project with real, reproducible
code.

## Résumé claims → status

The owner's résumé bullet (TensorFlow was intentionally dropped):

> - Built an NLP pipeline over 1,000+ labeled resumes with tokenization and TF-IDF vectorization for fit scoring at 92% accuracy.
> - Ranked resumes with scikit-learn models and RAG retrieval, deployed via Streamlit.

| Claim | Status |
|---|---|
| 1,000+ labeled resumes | ✅ 12,085 distinct (Resume-Atlas dataset) |
| tokenization + TF-IDF | ✅ |
| 92% accuracy | ✅ **as top-3 (94.0%)**; top-1 is 82.0%. Use the top-3 framing — it's honest. |
| scikit-learn models | ✅ calibrated LinearSVC |
| RAG retrieval | ✅ FAISS index + Claude generation (`resume_screener/rag.py`) |
| deployed via Streamlit | ⚠️ **NOT deployed yet — this is the main remaining task** |

## Architecture

```
app.py                    # Streamlit entry point
train.py                  # trains the classifier, writes models/
resume_screener/          # library package
  parser.py               #   PDF text + section extraction (robust to multi-word headers)
  classifier.py           #   ResumeClassifier — loads models/resume_clf.joblib
  semantic.py             #   SemanticMatcher — MiniLM embeddings + cosine
  rag.py                  #   RagEngine — FAISS retrieval + Claude fit analysis
  config.py, utils.py
scripts/download_data.py  # fetch Resume-Atlas from Hugging Face (no auth)
tests/test_components.py
models/                   # resume_clf.joblib (32 MB) + metrics.json (committed)
data/                     # raw dataset (git-ignored)
pdfs/                     # sample resumes + job descriptions
```

**Run everything from the repo root** — imports use the `resume_screener` package and
relative paths (`pdfs/…`) assume CWD is the root.

## Models & data

- **Classifier**: `TfidfVectorizer(ngram_range=(1,2), max_features=30000, sublinear_tf=True)`
  → `CalibratedClassifierCV(LinearSVC())`. Held-out: **top-1 0.820, top-3 0.940, macro-F1 0.821**
  (see `models/metrics.json`). Retrain with `python train.py`.
- **Dataset**: [ahmedheakl/resume-atlas](https://huggingface.co/datasets/ahmedheakl/resume-atlas)
  (13k rows, 12,085 unique, 43 categories). `scripts/download_data.py` fetches it to
  `data/resume_atlas.csv`. The original repo dataset was a trap — 962 rows but only 166
  unique (deduped), which is why accuracy couldn't be trusted before.
- **Semantic / RAG embeddings**: `sentence-transformers` `all-MiniLM-L6-v2` (PyTorch, 384-dim).
- **RAG LLM**: Anthropic. `MODEL = "claude-sonnet-5"` in `rag.py` — one-line swap to
  `claude-haiku-4-5` (cheaper) or `claude-opus-5`. **One Anthropic key works for all models.**

## Secrets

- `ANTHROPIC_API_KEY` lives in `.env` (git-ignored; `.env.example` documents it; `app.py`
  calls `load_dotenv()`). On Streamlit Cloud, set it as a Secret instead.
- The old OpenAI key was committed in pre-session history and is being revoked by the owner;
  OpenAI is not used anywhere in the code. (A Claude key was briefly committed **locally**
  this session and removed by rewriting local history **before any push** — nothing leaked
  to the remote.)

## Environment (important)

- **Windows 11.** Two Python installs: **`py -3.11` has the full stack** (torch,
  sentence-transformers, faiss, anthropic) and is the app runtime; **`python`/3.13 does NOT
  have torch** — don't use it for anything touching `semantic.py`/`rag.py`.
- Owner plans to move to **WSL/Ubuntu** (recommended: one clean 3.11 venv; matches all
  deploy targets; devcontainer already targets `python:3.11-bullseye`).
- Windows papercuts seen: Git Bash maps the scratchpad to `/tmp/...` but Python needs the
  `C:\…` path; CRLF warnings on commit (harmless); the cp1252 console can't print emoji —
  use `PYTHONIOENCODING=utf-8` when a script prints model output.

## How to run

```bash
pip install -r requirements.txt
python train.py            # downloads data + trains (~2 min); writes models/
streamlit run app.py       # the app
python tests/test_components.py   # smoke tests (run from root)
```

## Remaining work

1. **Deploy to Streamlit Community Cloud** (owner's choice): connect the GitHub repo, main
   file `app.py`, add `ANTHROPIC_API_KEY` as a Secret. Watch the ~1 GB RAM free-tier limit —
   the torch stack may be tight; Hugging Face Spaces (16 GB free) is the fallback.
2. Optional: surface **batch resume-ranking** (many resumes vs one role) in the UI to back
   "ranked resumes" more literally — `SemanticMatcher.batch_compare_resumes` already exists.
3. Minor: `config.py` `PATHS['dataset']` still points at the removed `UpdatedResumeDataSet.csv`
   (harmless/unused — clean up if touching config).

## Conventions

- **No "Co-Authored-By: Claude" or Claude attribution in commit messages.**
- Granular, conventional-commit-style commits.
