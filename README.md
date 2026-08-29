# MDST Resume Screener

A machine-learning resume screener that (1) classifies resumes into job categories and
(2) ranks them against open roles by semantic similarity — served through a Streamlit app.

## What it does

- **Category classification** — a TF-IDF + calibrated LinearSVC model predicts a resume's
  job category, trained on **12,000+ distinct labeled resumes across 43 categories**.
- **Semantic job matching** — sentence-transformer embeddings + cosine similarity rank an
  uploaded resume against the open roles.
- **PDF parsing** — extracts text and resume sections (with a full-text fallback for
  unusual layouts).
- **Interactive app** — Streamlit UI with cached models, so analysis is fast after the
  first load.

## Model performance

Held-out evaluation (20% test split, deduplicated, TF-IDF fit on train only):

| Metric | Score |
|---|---|
| Distinct resumes | 12,085 |
| Categories | 43 |
| Top-1 accuracy | **82.0%** |
| Top-3 accuracy | **94.0%** |
| Macro F1 | 0.82 |

Model: `TfidfVectorizer(ngram_range=(1,2), max_features=30000, sublinear_tf=True)` →
`CalibratedClassifierCV(LinearSVC())`. Reproduce with `python train.py`.

## Setup

```bash
git clone https://github.com/georgu28/MDST-Resume-Screener.git
cd MDST-Resume-Screener
pip install -r requirements.txt

# Download the dataset (~54 MB, from Hugging Face — no credentials needed) and train.
# This writes models/resume_clf.joblib, which the app loads.
python train.py
```

## Run the app

```bash
streamlit run app.py
```

Upload a resume PDF to see its best-matching role, per-role similarity scores, and its
predicted category with confidence.

## Command-line usage

Run examples from the repo root (so the `resume_screener` package resolves):

```python
from resume_screener.classifier import ResumeClassifier
clf = ResumeClassifier()                       # loads models/resume_clf.joblib
print(clf.predict_pdf("pdfs/jakes-resume.pdf"))
print(clf.get_prediction_probabilities("pdfs/jakes-resume.pdf"))

from resume_screener.semantic import SemanticMatcher
m = SemanticMatcher()
resume = m.embed_resume("pdfs/jakes-resume.pdf")
job = m.embed_job("pdfs/java.pdf")
print(m.cosine(resume, job))                   # cosine similarity in [0, 1]
```

## Project structure

```
app.py                       # Streamlit entry point (run this)
train.py                     # Train/evaluate the classifier -> models/resume_clf.joblib
resume_screener/             # Library package
  parser.py                  #   PDF text + section extraction
  classifier.py              #   ResumeClassifier — loads the trained sklearn pipeline
  semantic.py                #   SemanticMatcher — embeddings + cosine similarity
  rag.py                     #   RagEngine — FAISS retrieval + Claude fit analysis
  config.py                  #   Section titles + regex patterns
  utils.py                   #   Text helpers
scripts/download_data.py     # Fetch the Resume-Atlas dataset from Hugging Face
tests/test_components.py     # Component smoke tests
models/                      # Trained model + metrics.json (committed)
data/                        # Raw dataset (git-ignored; fetched by download_data.py)
pdfs/                        # Sample resumes and job descriptions
```

## Dataset

[Resume-Atlas](https://huggingface.co/datasets/ahmedheakl/resume-atlas) — 13k+ labeled
resumes over 43 categories, downloaded automatically by `scripts/download_data.py`.

## Acknowledgments

Michigan Data Science Team (MDST) · sentence-transformers · scikit-learn · Streamlit.
