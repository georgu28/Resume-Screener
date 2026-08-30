from streamlit_pdf_viewer import pdf_viewer
import streamlit as st
import tempfile
import os
import json
from resume_screener.parser import read_pdf
from resume_screener.semantic import SemanticMatcher
from resume_screener.classifier import ResumeClassifier
from resume_screener.rag import RagEngine
import logging

try:
    from dotenv import load_dotenv
    load_dotenv()  # load ANTHROPIC_API_KEY from a local .env if present
except ImportError:
    pass

_METRICS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "metrics.json")


def load_metrics() -> dict:
    """Load the held-out metrics written by train.py (empty dict if unavailable)."""
    try:
        with open(_METRICS_PATH) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Open roles to score every uploaded resume against. Each value is the job
# description text itself (chunked for RAG retrieval and embedded whole for the
# semantic match) — edit these freely to change the roles, no PDF needed.
JOB_DESCRIPTIONS = {
    "Backend Engineer": (
        "Backend Engineer\n"
        "Design, build, and operate scalable server-side services and APIs.\n"
        "Build and maintain REST and gRPC APIs and microservices.\n"
        "Design relational and NoSQL data models; optimize queries and indexes.\n"
        "Write clean, well-tested Python (or Go/Java) backend services.\n"
        "Own reliability, latency, observability, and on-call for production systems.\n"
        "Requirements: strong Python and SQL; PostgreSQL, Redis; REST API design;\n"
        "Docker containers; CI/CD pipelines; AWS or GCP; unit and integration testing;\n"
        "distributed systems, caching, message queues, and event-driven architecture."
    ),
    "Frontend Engineer": (
        "Frontend Engineer\n"
        "Build responsive, accessible web interfaces and design systems.\n"
        "Develop UI in React with TypeScript, HTML, and modern CSS.\n"
        "Implement state management, data fetching, and component libraries.\n"
        "Ensure accessibility (WCAG), cross-browser support, and performance.\n"
        "Collaborate with designers to ship pixel-accurate, interactive experiences.\n"
        "Requirements: JavaScript/TypeScript, React, HTML, CSS; responsive design;\n"
        "web accessibility; frontend testing (Jest, Playwright); build tooling (Vite,\n"
        "webpack); REST/GraphQL API integration; Core Web Vitals and performance."
    ),
    "Full Stack Engineer": (
        "Full Stack Engineer\n"
        "Own features end to end, from database to user interface.\n"
        "Build backend APIs in Python or Node and frontends in React/TypeScript.\n"
        "Model data, write services, and design the UI that consumes them.\n"
        "Ship, deploy, and monitor features across the whole stack.\n"
        "Requirements: Python or Node.js plus React and TypeScript; SQL databases;\n"
        "REST API design; Docker and CI/CD; cloud deployment (AWS/GCP); testing across\n"
        "frontend and backend; comfort moving fluidly between client and server."
    ),
    "Machine Learning Engineer": (
        "Machine Learning Engineer\n"
        "Build, train, evaluate, and deploy machine learning models in production.\n"
        "Develop NLP and predictive models with Python, scikit-learn, and PyTorch.\n"
        "Engineer features, build data pipelines, and run rigorous evaluation.\n"
        "Deploy models and embeddings as services; monitor drift and performance.\n"
        "Work with LLMs, retrieval, vector search, and RAG systems.\n"
        "Requirements: strong Python; scikit-learn, PyTorch or TensorFlow; NLP,\n"
        "embeddings, transformers; TF-IDF and classical ML; model evaluation metrics;\n"
        "data pipelines and MLOps; Docker; deploying ML models to the cloud."
    ),
    "Product Manager": (
        "Product Manager\n"
        "Define product strategy, roadmap, and priorities for a product area.\n"
        "Conduct user research and translate insight into requirements and specs.\n"
        "Prioritize a backlog and align engineering, design, and stakeholders.\n"
        "Define success metrics; run experiments and A/B tests; analyze results.\n"
        "Communicate roadmap and trade-offs clearly across the organization.\n"
        "Requirements: product roadmap ownership; user research; writing specs and\n"
        "PRDs; data-driven prioritization and metrics; stakeholder management;\n"
        "cross-functional leadership; familiarity with agile delivery."
    ),
    "Product Designer": (
        "Product Designer\n"
        "Design intuitive, accessible end-to-end product experiences.\n"
        "Create wireframes, high-fidelity mockups, and interactive prototypes in Figma.\n"
        "Run user research and usability testing to validate designs.\n"
        "Own interaction design, visual design, and contributions to the design system.\n"
        "Partner with engineers to ship polished, accessible interfaces.\n"
        "Requirements: UX and UI design; Figma; wireframing and prototyping; user\n"
        "research and usability testing; design systems; interaction and visual design;\n"
        "accessibility; strong portfolio of shipped product work."
    ),
}

# A cosine similarity this high is effectively a perfect match; used only to
# scale the progress bars so small real-world differences stay visible.
_BAR_SCALE = 0.7


# ---------------------------------------------------------------------------
# Cached heavy resources. Without these, every upload reloaded the classifier
# and the sentence-transformer from scratch (several seconds each).
# @st.cache_resource keeps a single instance alive across reruns and uploads.
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading classifier (first run only)...")
def load_classifier() -> ResumeClassifier:
    return ResumeClassifier()


@st.cache_resource(show_spinner="Loading semantic model (first run only)...")
def load_matcher() -> SemanticMatcher:
    return SemanticMatcher()


@st.cache_resource(show_spinner="Embedding job descriptions (first run only)...")
def load_job_embeddings() -> dict:
    """Embed each job description exactly once and reuse across uploads."""
    matcher = load_matcher()
    return {
        title: matcher.embed_job_text(text)
        for title, text in JOB_DESCRIPTIONS.items()
    }


@st.cache_resource(show_spinner="Building retrieval index (first run only)...")
def load_rag() -> RagEngine:
    """Build the RAG index once, reusing the semantic matcher's embedder."""
    return RagEngine(JOB_DESCRIPTIONS, embedder=load_matcher().model)


def anthropic_key() -> str:
    """Resolve the Anthropic key from env or Streamlit secrets (env wins)."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return os.environ["ANTHROPIC_API_KEY"]
    try:
        key = st.secrets.get("ANTHROPIC_API_KEY")  # raises if no secrets file
    except Exception:
        key = None
    if key:
        os.environ["ANTHROPIC_API_KEY"] = key  # so the anthropic SDK picks it up
    return key or ""


# ---------------------------------------------------------------------------
# Presentation helpers. The design system (Data-Dense Dashboard, blue #1E40AF +
# amber accent, Fira Sans/Fira Code, semantic status colors) lives in THEME_CSS;
# these functions render the custom score bars, badges, and KPI cards on top of
# Streamlit's native widgets.
# ---------------------------------------------------------------------------

THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600&family=Fira+Sans:wght@300;400;500;600;700&display=swap');

:root {
    --rs-primary: #1E40AF;
    --rs-primary-2: #3B82F6;
    --rs-accent: #D97706;
    --rs-success: #059669;
    --rs-bg: #F8FAFC;
    --rs-surface: #FFFFFF;
    --rs-surface-2: #FBFCFE;
    --rs-text: #0F172A;
    --rs-muted: #64748B;
    --rs-border: #E2E8F0;
    --rs-divider: #F1F5F9;
    --rs-track: #EEF2F7;
    --rs-hover: #F1F5F9;
    --rs-hero-grad: linear-gradient(180deg, #FFFFFF 0%, #FBFCFE 100%);
    --rs-highlight-bg: linear-gradient(180deg, #EFF6FF, #F8FBFF);
    --rs-highlight-border: #BFDBFE;
    --rs-badge-bg: #ECFDF5;
    --rs-badge-border: #A7F3D0;
    --rs-badge-text: #059669;
    --rs-btn-bg: #1E40AF;
    --rs-btn-hover: #1B3A9E;
    --rs-on-primary: #FFFFFF;
    --rs-radius: 14px;
    --rs-shadow: 0 1px 2px rgba(15,23,42,.04), 0 8px 24px -12px rgba(15,23,42,.12);
}

/* --- Base typography ------------------------------------------------------ */
html, body, .stApp, [data-testid="stAppViewContainer"],
[data-testid="stSidebar"], .stMarkdown, .stMarkdown p, button, input, textarea, select {
    font-family: 'Fira Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}
.stApp { background: var(--rs-bg); color: var(--rs-text); }

/* Tighten the default top padding so the hero sits higher. */
[data-testid="stAppViewContainer"] .main .block-container,
[data-testid="stMainBlockContainer"] {
    padding-top: 2.2rem;
    max-width: 1120px;
}
[data-testid="stHeader"] { background: transparent; }

h1, h2, h3, h4 { color: var(--rs-text); letter-spacing: -0.01em; font-weight: 600; }

/* --- Hero header ---------------------------------------------------------- */
.rs-hero {
    display: flex; align-items: center; gap: 16px;
    padding: 22px 24px; margin-bottom: 8px;
    background: var(--rs-hero-grad);
    border: 1px solid var(--rs-border); border-radius: var(--rs-radius);
    box-shadow: var(--rs-shadow);
}
.rs-hero-mark {
    flex: 0 0 auto; width: 48px; height: 48px; border-radius: 12px;
    display: grid; place-items: center; color: #fff;
    background: linear-gradient(135deg, var(--rs-primary-2), var(--rs-primary));
    box-shadow: 0 6px 16px -6px rgba(30,64,175,.55);
}
.rs-hero-mark svg { width: 26px; height: 26px; }
.rs-hero-title { margin: 0; font-size: 1.55rem; font-weight: 700; line-height: 1.15; }
.rs-hero-sub { margin: 3px 0 0; color: var(--rs-muted) !important; font-size: .95rem; }

/* --- Section labels ------------------------------------------------------- */
.rs-section {
    display: flex; align-items: center; gap: 9px;
    font-size: .78rem; font-weight: 600; letter-spacing: .06em; text-transform: uppercase;
    color: var(--rs-muted); margin: 6px 0 14px;
}
.rs-section svg { width: 16px; height: 16px; color: var(--rs-primary-2); }

/* --- Score rows (job match / category) ------------------------------------ */
.rs-card {
    background: var(--rs-surface); border: 1px solid var(--rs-border);
    border-radius: var(--rs-radius); padding: 18px 20px; box-shadow: var(--rs-shadow);
}
.rs-score-row { padding: 11px 0; border-bottom: 1px solid var(--rs-divider); }
.rs-score-row:last-child { border-bottom: none; }
.rs-score-head {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 8px; gap: 12px;
}
.rs-score-label { font-weight: 500; font-size: .96rem; color: var(--rs-text); }
.rs-score-val {
    font-family: 'Fira Code', ui-monospace, monospace; font-size: .9rem;
    font-weight: 500; color: var(--rs-muted); font-variant-numeric: tabular-nums;
}
.rs-track { height: 8px; background: var(--rs-track); border-radius: 999px; overflow: hidden; }
.rs-fill {
    height: 100%; border-radius: 999px;
    background: linear-gradient(90deg, var(--rs-primary-2), var(--rs-primary));
    transition: width .5s cubic-bezier(.22,.61,.36,1);
}
.rs-fill-alt { background: linear-gradient(90deg, #93C5FD, var(--rs-primary-2)); }
.rs-best .rs-score-label { font-weight: 600; }
.rs-badge {
    display: inline-flex; align-items: center; gap: 5px;
    font-size: .68rem; font-weight: 600; letter-spacing: .03em; text-transform: uppercase;
    padding: 2px 9px; border-radius: 999px; margin-left: 9px; vertical-align: middle;
}
.rs-badge-best { color: var(--rs-badge-text); background: var(--rs-badge-bg); border: 1px solid var(--rs-badge-border); }
.rs-badge-best svg { width: 11px; height: 11px; }

/* --- Best-match callout --------------------------------------------------- */
.rs-highlight {
    display: flex; align-items: center; gap: 12px;
    padding: 14px 18px; margin-bottom: 14px;
    background: var(--rs-highlight-bg);
    border: 1px solid var(--rs-highlight-border); border-radius: 12px;
}
.rs-highlight svg { width: 22px; height: 22px; color: var(--rs-primary); flex: 0 0 auto; }
.rs-highlight-role { font-weight: 600; color: var(--rs-primary); }
.rs-highlight-meta { color: var(--rs-muted) !important; font-size: .9rem; }

.rs-caption { color: var(--rs-muted) !important; font-size: .82rem; margin-top: 12px; line-height: 1.5; }

/* --- Sidebar -------------------------------------------------------------- */
[data-testid="stSidebar"] { background: var(--rs-surface); border-right: 1px solid var(--rs-border); }
[data-testid="stSidebar"] .block-container { padding-top: 1.6rem; }
.rs-side-block { margin-bottom: 22px; }
.rs-side-h {
    display: flex; align-items: center; gap: 8px;
    font-size: .74rem; font-weight: 600; letter-spacing: .06em; text-transform: uppercase;
    color: var(--rs-muted); margin-bottom: 10px;
}
.rs-side-h svg { width: 15px; height: 15px; color: var(--rs-primary-2); }
.rs-side-item {
    display: flex; align-items: center; gap: 9px; padding: 6px 0;
    font-size: .9rem; color: var(--rs-text);
}
.rs-side-item svg { width: 15px; height: 15px; color: var(--rs-muted); flex: 0 0 auto; }
.rs-feature { font-size: .88rem; color: var(--rs-text); line-height: 1.45; margin: 3px 0; }
.rs-feature b { font-weight: 600; }
.rs-kpi-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
.rs-kpi {
    background: var(--rs-bg); border: 1px solid var(--rs-border);
    border-radius: 11px; padding: 12px 13px;
}
.rs-kpi-val {
    font-family: 'Fira Code', ui-monospace, monospace; font-size: 1.15rem;
    font-weight: 600; color: var(--rs-primary); font-variant-numeric: tabular-nums;
}
.rs-kpi-label { font-size: .72rem; color: var(--rs-muted); margin-top: 2px; }
.rs-side-note { font-size: .82rem; color: var(--rs-muted); line-height: 1.5; }

/* --- Tabs ----------------------------------------------------------------- */
.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid var(--rs-border); }
.stTabs [data-baseweb="tab"] {
    height: 44px; padding: 0 16px; border-radius: 10px 10px 0 0;
    font-weight: 500; color: var(--rs-muted); transition: color .2s, background .2s;
}
.stTabs [data-baseweb="tab"]:hover { color: var(--rs-primary); background: var(--rs-hover); }
.stTabs [aria-selected="true"] { color: var(--rs-primary) !important; font-weight: 600; }
.stTabs [data-baseweb="tab-highlight"] { background: var(--rs-primary); height: 3px; }

/* --- Buttons -------------------------------------------------------------- */
.stButton > button, [data-testid="stBaseButton-primary"], [data-testid="stBaseButton-secondary"] {
    border-radius: 10px; font-weight: 600; transition: transform .15s, box-shadow .2s, background .2s;
}
[data-testid="stBaseButton-primary"] {
    background: var(--rs-btn-bg); border: 1px solid var(--rs-btn-bg); color: var(--rs-on-primary);
    box-shadow: 0 6px 16px -8px rgba(30,64,175,.6);
}
[data-testid="stBaseButton-primary"]:hover { background: var(--rs-btn-hover); transform: translateY(-1px); }
.stButton > button:active { transform: translateY(0); }

/* --- File uploader -------------------------------------------------------- */
[data-testid="stFileUploaderDropzone"] {
    background: var(--rs-surface) !important; border: 1.5px dashed var(--rs-border); border-radius: var(--rs-radius);
    transition: border-color .2s, background .2s;
}
[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--rs-primary-2) !important; background: var(--rs-surface-2) !important; }

/* --- Native Streamlit surfaces (kept in sync with the theme, esp. dark) --- */
/* Main content + header sit on the app background so no light gray shows through. */
[data-testid="stAppViewContainer"], [data-testid="stMain"],
[data-testid="stMainBlockContainer"], [data-testid="stBottom"] { background: var(--rs-bg); }
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li { color: var(--rs-text); }
[data-testid="stWidgetLabel"] label, .stSelectbox label, [data-testid="stFileUploader"] label { color: var(--rs-text) !important; }
code { background: var(--rs-hover); color: var(--rs-text); border-radius: 5px; padding: .08em .35em; }
/* Select box (closed control + dropdown popover rendered in a portal) */
.stSelectbox [data-baseweb="select"] > div, .stSelectbox [data-baseweb="select"] div[role="button"] {
    background: var(--rs-surface) !important; border-color: var(--rs-border) !important; color: var(--rs-text) !important;
}
.stSelectbox [data-baseweb="select"] svg { fill: var(--rs-muted); }
[data-baseweb="popover"] [role="listbox"], [data-baseweb="menu"], [data-baseweb="menu"] ul {
    background: var(--rs-surface) !important; border: 1px solid var(--rs-border);
}
[data-baseweb="popover"] [role="option"], [data-baseweb="menu"] li { color: var(--rs-text) !important; }
[data-baseweb="popover"] [role="option"]:hover, [data-baseweb="menu"] li:hover { background: var(--rs-hover) !important; }
/* Text inputs / text area */
.stTextArea textarea, .stTextInput input {
    background: var(--rs-surface) !important; color: var(--rs-text) !important; border-color: var(--rs-border) !important;
}
/* Secondary buttons (e.g. the uploader's "Browse files") */
[data-testid="stBaseButton-secondary"] {
    background: var(--rs-surface) !important; color: var(--rs-text) !important; border: 1px solid var(--rs-border) !important;
}
[data-testid="stBaseButton-secondary"]:hover { border-color: var(--rs-primary-2) !important; color: var(--rs-primary) !important; }
/* Expander */
[data-testid="stExpander"] { border: 1px solid var(--rs-border) !important; border-radius: 12px; background: var(--rs-surface) !important; }
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary * { color: var(--rs-text) !important; }
/* File uploader helper text (only after its surface is forced dark above) */
[data-testid="stFileUploaderDropzone"] span, [data-testid="stFileUploaderDropzone"] div { color: var(--rs-text) !important; }
[data-testid="stFileUploaderDropzone"] small { color: var(--rs-muted) !important; }

/* --- Alerts (subtle, on-brand) ------------------------------------------- */
[data-testid="stAlert"] { border-radius: 12px; background: var(--rs-surface) !important; border: 1px solid var(--rs-border); }
[data-testid="stAlert"] p, [data-testid="stAlert"] div, [data-testid="stAlert"] span { color: var(--rs-text) !important; }
[data-testid="stAlert"] code { background: var(--rs-hover); }

/* Focus visibility for keyboard users. */
:is(button, a, input, [tabindex]):focus-visible {
    outline: 2px solid var(--rs-primary-2); outline-offset: 2px;
}

@media (prefers-reduced-motion: reduce) {
    .rs-fill, .stButton > button, [data-testid="stBaseButton-primary"] { transition: none; }
}
</style>
"""

# Dark palette. Injected AFTER THEME_CSS so its :root redefinitions win; every
# rule above reads var(--rs-*), so flipping these tokens re-themes the whole app.
DARK_CSS = """
<style>
:root {
    --rs-primary: #60A5FA;
    --rs-primary-2: #3B82F6;
    --rs-accent: #F59E0B;
    --rs-success: #34D399;
    --rs-bg: #0B1220;
    --rs-surface: #131C2E;
    --rs-surface-2: #16213A;
    --rs-text: #E6EBF4;
    --rs-muted: #93A2BC;
    --rs-border: #26324C;
    --rs-divider: #1E2942;
    --rs-track: #1C2740;
    --rs-hover: #1A2338;
    --rs-hero-grad: linear-gradient(180deg, #16213A 0%, #131C2E 100%);
    --rs-highlight-bg: linear-gradient(180deg, #12203B, #0F1A30);
    --rs-highlight-border: #274063;
    --rs-badge-bg: rgba(52,211,153,.12);
    --rs-badge-border: rgba(52,211,153,.38);
    --rs-badge-text: #34D399;
    --rs-btn-bg: #2563EB;
    --rs-btn-hover: #1D4ED8;
    --rs-on-primary: #FFFFFF;
    --rs-shadow: 0 1px 2px rgba(0,0,0,.35), 0 14px 34px -16px rgba(0,0,0,.65);
}
/* Best-effort override of Streamlit's own theme tokens so any native widget we
   didn't target explicitly still picks up dark surfaces/text. */
:root, .stApp, [data-testid="stAppViewContainer"] {
    --background-color: #0B1220 !important;
    --secondary-background-color: #131C2E !important;
    --text-color: #E6EBF4 !important;
    --primary-color: #60A5FA !important;
}
</style>
"""

# Inline SVG icons (Lucide-style, currentColor) — never emoji, per design system.
_IC_SCAN = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 7V5a2 2 0 0 1 2-2h2M17 3h2a2 2 0 0 1 2 2v2M21 17v2a2 2 0 0 1-2 2h-2M7 21H5a2 2 0 0 1-2-2v-2"/><path d="M8 12h8M8 9h5M8 15h6"/></svg>'
_IC_TARGET = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1.5"/></svg>'
_IC_TAG = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12.6 3H5a2 2 0 0 0-2 2v7.6a2 2 0 0 0 .6 1.4l7.4 7.4a2 2 0 0 0 2.8 0l6.6-6.6a2 2 0 0 0 0-2.8L14 3.6A2 2 0 0 0 12.6 3Z"/><circle cx="7.5" cy="7.5" r="1"/></svg>'
_IC_SPARK = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3v4M12 17v4M3 12h4M17 12h4M6 6l2.5 2.5M15.5 15.5 18 18M18 6l-2.5 2.5M8.5 15.5 6 18"/></svg>'
_IC_CHECK = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"/></svg>'
_IC_INFO = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M12 16v-4M12 8h.01"/></svg>'
_IC_BRIEFCASE = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="7" width="18" height="13" rx="2"/><path d="M8 7V5a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>'
_IC_CHART = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"/><rect x="7" y="12" width="3" height="6"/><rect x="12" y="8" width="3" height="10"/><rect x="17" y="5" width="3" height="13"/></svg>'
_IC_DOT = '<svg viewBox="0 0 24 24" fill="currentColor"><circle cx="12" cy="12" r="3.5"/></svg>'


def score_row(label: str, score: float, is_best: bool = False) -> str:
    """A job-match row: label + tabular score + a scaled progress track."""
    pct = min(score / _BAR_SCALE, 1.0) * 100
    badge = f'<span class="rs-badge rs-badge-best">{_IC_CHECK} Best</span>' if is_best else ""
    cls = " rs-best" if is_best else ""
    return (
        f'<div class="rs-score-row{cls}">'
        f'<div class="rs-score-head"><span class="rs-score-label">{label}{badge}</span>'
        f'<span class="rs-score-val">{score:.3f}</span></div>'
        f'<div class="rs-track"><div class="rs-fill" style="width:{pct:.1f}%"></div></div>'
        f'</div>'
    )


def prob_row(label: str, prob: float) -> str:
    """A category row: label + percentage + a confidence track."""
    pct = prob * 100
    return (
        f'<div class="rs-score-row">'
        f'<div class="rs-score-head"><span class="rs-score-label">{label}</span>'
        f'<span class="rs-score-val">{pct:.0f}%</span></div>'
        f'<div class="rs-track"><div class="rs-fill rs-fill-alt" style="width:{pct:.1f}%"></div></div>'
        f'</div>'
    )


def section(icon: str, text: str) -> None:
    st.markdown(f'<div class="rs-section">{icon}<span>{text}</span></div>', unsafe_allow_html=True)


# Page configuration
st.set_page_config(
    page_title="Resume Screener",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(THEME_CSS, unsafe_allow_html=True)
# Dark mode is a per-session toggle (control lives in the sidebar). Read it here,
# before rendering, so the palette override is injected on the same run.
if st.session_state.get("rs_dark", False):
    st.markdown(DARK_CSS, unsafe_allow_html=True)

# Hero header
st.markdown(
    f'''
    <div class="rs-hero">
      <div class="rs-hero-mark">{_IC_SCAN}</div>
      <div>
        <h1 class="rs-hero-title">Resume Screener</h1>
        <p class="rs-hero-sub">Semantic role matching, résumé classification, and grounded AI fit analysis.</p>
      </div>
    </div>
    ''',
    unsafe_allow_html=True,
)

# File uploader
section(_IC_SCAN, "Upload a résumé")
file = st.file_uploader(
    "Choose a PDF file", type="pdf",
    help="Upload a PDF résumé to score it against the open roles.",
    label_visibility="collapsed",
)

if file:
    tmp_file_path = None
    try:
        file_value = file.getvalue()

        # Create temporary file for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(file_value)
            tmp_file_path = tmp_file.name

        # Extract text from PDF
        text = read_pdf(tmp_file_path)

        if text.strip():
            tab1, tab_fit, tab2, tab3 = st.tabs(
                ["Analysis", "Fit Analysis (AI)", "Preview", "Extracted Text"]
            )

            with tab1:
                # -----------------------------------------------------------
                # Primary result: semantic match against the open roles.
                # -----------------------------------------------------------
                section(_IC_TARGET, "Job match")

                try:
                    with st.spinner("Scoring against job descriptions..."):
                        matcher = load_matcher()
                        jd_embeddings = load_job_embeddings()
                        resume_emb = matcher.embed_resume(tmp_file_path)
                        similarities = {
                            title: round(matcher.cosine(resume_emb, emb), 3)
                            for title, emb in jd_embeddings.items()
                        }

                    if similarities:
                        ranked = dict(sorted(similarities.items(), key=lambda x: x[1], reverse=True))
                        best_match, best_score = next(iter(ranked.items()))

                        st.markdown(
                            f'<div class="rs-highlight">{_IC_TARGET}'
                            f'<div><span class="rs-highlight-role">{best_match}</span>'
                            f'<span class="rs-highlight-meta"> &nbsp;·&nbsp; best match at '
                            f'{best_score:.3f} similarity</span></div></div>',
                            unsafe_allow_html=True,
                        )

                        rows = "".join(
                            score_row(t, s, is_best=(t == best_match))
                            for t, s in ranked.items()
                        )
                        st.markdown(
                            f'<div class="rs-card">{rows}'
                            f'<div class="rs-caption">Scores are cosine similarity (0–1) — compare them '
                            f'against each other rather than as absolute percentages. Higher means a '
                            f'closer match.</div></div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.warning("No job descriptions available to compare against.")
                except Exception as e:
                    st.error(f"Error in semantic analysis: {str(e)}")
                    logger.error(f"Semantic analysis error: {e}")

                st.write("")

                # -----------------------------------------------------------
                # Secondary result: broad resume-category classifier. This is
                # independent of the roles above (43 general categories).
                # -----------------------------------------------------------
                section(_IC_TAG, "Résumé category")

                try:
                    with st.spinner("Classifying resume..."):
                        classifier = load_classifier()
                        predicted_category = classifier.predict_pdf(tmp_file_path)
                        probabilities = classifier.get_prediction_probabilities(tmp_file_path)

                    top_probabilities = dict(list(probabilities.items())[:5])
                    rows = "".join(prob_row(c, p) for c, p in top_probabilities.items())
                    st.markdown(
                        f'<div class="rs-highlight">{_IC_TAG}'
                        f'<div><span class="rs-highlight-role">{predicted_category}</span>'
                        f'<span class="rs-highlight-meta"> &nbsp;·&nbsp; predicted category</span></div></div>'
                        f'<div class="rs-card">{rows}'
                        f'<div class="rs-caption">A broad classifier over 43 general résumé categories '
                        f'(e.g. Data Science, Java Developer, HR), independent of the roles above. '
                        f'Confidence is the model\'s calibrated probability (LinearSVC wrapped in '
                        f'CalibratedClassifierCV).</div></div>',
                        unsafe_allow_html=True,
                    )
                except Exception as e:
                    st.error(f"Error in category prediction: {str(e)}")
                    logger.error(f"Category prediction error: {e}")

            with tab_fit:
                section(_IC_SPARK, "AI fit analysis")
                st.markdown(
                    '<p class="rs-caption" style="margin-top:0">Retrieval-augmented generation: the most '
                    'relevant job requirements are retrieved from a vector index, then Claude explains fit '
                    'grounded in that evidence.</p>',
                    unsafe_allow_html=True,
                )
                role = st.selectbox("Analyze against role", list(JOB_DESCRIPTIONS))

                try:
                    rag = load_rag()
                    if anthropic_key():
                        if st.button("Generate fit analysis", type="primary"):
                            with st.spinner("Retrieving requirements and generating analysis..."):
                                analysis = rag.explain_fit(tmp_file_path, role)
                            st.markdown(analysis)
                    else:
                        st.info(
                            "Set `ANTHROPIC_API_KEY` (env var, or Streamlit secret) to enable the "
                            "Claude-generated analysis. Meanwhile, here are the requirements the "
                            "retriever matched to this resume:"
                        )
                        hits = "".join(
                            f'<div class="rs-side-item">{_IC_DOT}<span>{hit["text"]}</span></div>'
                            for hit in rag.retrieve(text, k=5, role=role)
                        )
                        st.markdown(f'<div class="rs-card">{hits}</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error in fit analysis: {str(e)}")
                    logger.error(f"RAG error: {e}")

            with tab2:
                section(_IC_INFO, "Résumé preview")
                pdf_viewer(file_value)

            with tab3:
                section(_IC_INFO, "Extracted text")
                with st.expander("View full extracted text", expanded=False):
                    st.text_area("Extracted Text", text, height=400, label_visibility="collapsed")
                st.markdown(
                    f'<div class="rs-caption">Total characters extracted: {len(text):,}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.error("No text could be extracted from the PDF. Please ensure the PDF contains readable text.")

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        logger.error(f"File processing error: {e}")
    finally:
        # Always clean up the temp file, even if processing failed.
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
else:
    st.markdown(
        f'<div class="rs-card" style="text-align:center; padding:34px 20px; color:var(--rs-muted)">'
        f'<div style="width:44px;height:44px;margin:0 auto 12px;color:var(--rs-primary-2)">{_IC_SCAN}</div>'
        f'<div style="font-weight:600;color:var(--rs-text);margin-bottom:4px">Upload a résumé to begin</div>'
        f'<div style="font-size:.9rem">Drop a PDF above to see role matches, its category, and an AI fit analysis.</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.toggle("Dark mode", key="rs_dark", help="Switch between the light and dark theme.")

    st.markdown(
        f'<div class="rs-side-block"><div class="rs-side-h">{_IC_INFO} How it works</div>'
        f'<div class="rs-feature"><b>Semantic job match</b> — sentence-transformer embeddings rank '
        f'the résumé against each open role.</div>'
        f'<div class="rs-feature"><b>Category classifier</b> — TF-IDF + calibrated LinearSVC over '
        f'12k+ résumés, 43 categories.</div>'
        f'<div class="rs-feature"><b>AI fit analysis</b> — FAISS retrieval + Claude, grounded in the '
        f'matched job requirements.</div></div>',
        unsafe_allow_html=True,
    )

    roles = "".join(
        f'<div class="rs-side-item">{_IC_BRIEFCASE}<span>{title}</span></div>'
        for title in JOB_DESCRIPTIONS
    )
    st.markdown(
        f'<div class="rs-side-block"><div class="rs-side-h">{_IC_BRIEFCASE} Open roles</div>{roles}</div>',
        unsafe_allow_html=True,
    )

    metrics = load_metrics()
    if metrics:
        n_resumes = metrics.get("n_resumes", 0)
        n_categories = metrics.get("n_categories", "?")
        top1 = metrics.get("holdout_top1_accuracy", 0)
        top3 = metrics.get("holdout_top3_accuracy", 0)
        st.markdown(
            f'<div class="rs-side-block"><div class="rs-side-h">{_IC_CHART} Model</div>'
            f'<div class="rs-kpi-grid">'
            f'<div class="rs-kpi"><div class="rs-kpi-val">{n_resumes:,}</div>'
            f'<div class="rs-kpi-label">résumés trained</div></div>'
            f'<div class="rs-kpi"><div class="rs-kpi-val">{n_categories}</div>'
            f'<div class="rs-kpi-label">categories</div></div>'
            f'<div class="rs-kpi"><div class="rs-kpi-val">{top1:.0%}</div>'
            f'<div class="rs-kpi-label">top-1 accuracy</div></div>'
            f'<div class="rs-kpi"><div class="rs-kpi-val">{top3:.0%}</div>'
            f'<div class="rs-kpi-label">top-3 accuracy</div></div>'
            f'</div>'
            f'<div class="rs-side-note" style="margin-top:10px">TF-IDF + calibrated LinearSVC.</div></div>',
            unsafe_allow_html=True,
        )
