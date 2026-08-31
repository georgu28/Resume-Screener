from streamlit_pdf_viewer import pdf_viewer
import streamlit as st
import tempfile
import os
import json
from resume_screener.parser import read_pdf
from resume_screener.semantic import SemanticMatcher
from resume_screener.classifier import ResumeClassifier
from resume_screener import screener
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

# ---------------------------------------------------------------------------
# Cached heavy resources. Without these, every upload reloaded the classifier
# and the sentence-transformer from scratch (several seconds each).
# @st.cache_resource keeps a single instance alive across reruns and uploads.
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading classifier (first run only)...")
def load_classifier() -> ResumeClassifier:
    return ResumeClassifier()


@st.cache_resource(show_spinner="Loading language model (first run only)...")
def load_matcher() -> SemanticMatcher:
    # The SentenceTransformer inside is reused as the screener's RAG embedder.
    return SemanticMatcher()


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

/* --- Screener score card -------------------------------------------------- */
.rs-scorecard {
    display: flex; align-items: center; gap: 20px; padding: 18px 22px; margin-bottom: 16px;
    background: var(--rs-surface); border: 1px solid var(--rs-border);
    border-radius: var(--rs-radius); box-shadow: var(--rs-shadow);
}
.rs-score-big {
    font-family: 'Fira Code', ui-monospace, monospace; font-size: 2.6rem; font-weight: 700;
    line-height: 1; font-variant-numeric: tabular-nums; flex: 0 0 auto;
}
.rs-score-of { font-size: 1rem; color: var(--rs-muted); font-weight: 500; margin-left: 2px; }
.rs-score-body { flex: 1; min-width: 0; }
.rs-score-tag { font-weight: 600; font-size: .95rem; margin-bottom: 9px; }

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
/* Streamlit's default dropzone is a row that shoves "Browse files" to the far
   right. Make it a centered column so the instructions and the button stack and
   center together. */
[data-testid="stFileUploaderDropzone"] {
    background: var(--rs-surface) !important; border: 1.5px dashed var(--rs-border); border-radius: var(--rs-radius);
    min-height: 168px; padding: 30px 24px; transition: border-color .2s, background .2s, box-shadow .2s;
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    gap: 14px; text-align: center;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--rs-primary-2) !important; background: var(--rs-surface-2) !important;
    box-shadow: var(--rs-shadow);
}
/* Center the icon + copy block as its own centered column. */
[data-testid="stFileUploaderDropzoneInstructions"] {
    display: flex; flex-direction: column; align-items: center; gap: 6px; margin: 0; width: 100%;
}
[data-testid="stFileUploaderDropzoneInstructions"] > div { align-items: center; text-align: center; }
/* Tint the upload cloud icon on-brand and give it presence. */
[data-testid="stFileUploaderDropzone"] svg {
    fill: var(--rs-primary) !important; color: var(--rs-primary) !important;
    width: 40px; height: 40px;
}
/* Center the "Browse files" button beneath the instructions. */
[data-testid="stFileUploaderDropzone"] [data-testid="stBaseButton-secondary"] { margin: 0 auto; min-width: 148px; }
/* Tooltip popover (e.g. the Dark mode help) kept a hardcoded white bg. */
[data-testid="stTooltipContent"] {
    background: var(--rs-surface) !important; border: 1px solid var(--rs-border) !important;
    box-shadow: var(--rs-shadow) !important;
}
[data-testid="stTooltipContent"], [data-testid="stTooltipContent"] * { color: var(--rs-text) !important; }
/* The chip shown after a file is uploaded (kept a light-gray bg by default). */
[data-testid="stFileChip"] {
    background: var(--rs-surface) !important; border: 1px solid var(--rs-border) !important;
    border-radius: 10px !important;
}
[data-testid="stFileChip"], [data-testid="stFileChip"] span, [data-testid="stFileChip"] small {
    color: var(--rs-text) !important;
}
[data-testid="stFileChip"] svg { fill: var(--rs-muted) !important; }

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


def score_card(score: int) -> str:
    """A big match-score readout for the screener: number, verdict, colored bar."""
    if score >= 70:
        tone, label = "var(--rs-success)", "Strong match"
    elif score >= 45:
        tone, label = "var(--rs-accent)", "Moderate match"
    else:
        tone, label = "#EF4444", "Weak match"
    return (
        f'<div class="rs-scorecard">'
        f'<div class="rs-score-big" style="color:{tone}">{score}'
        f'<span class="rs-score-of">/100</span></div>'
        f'<div class="rs-score-body">'
        f'<div class="rs-score-tag" style="color:{tone}">{label}</div>'
        f'<div class="rs-track"><div class="rs-fill" style="width:{score}%;background:{tone}"></div></div>'
        f'</div></div>'
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
# Dark mode is a per-session toggle (control lives in the sidebar). Default it on
# the first run so the app opens in dark mode; the sidebar toggle binds to the
# same key, so it starts switched on. Read it here, before rendering, so the
# palette override is injected on the same run.
st.session_state.setdefault("rs_dark", True)
if st.session_state.get("rs_dark", False):
    st.markdown(DARK_CSS, unsafe_allow_html=True)

# Bridge a Streamlit Cloud secret into the environment so the anthropic SDK (used
# by the screener) picks it up; harmless locally where the key is already in env.
anthropic_key()

# Hero header
st.markdown(
    f'''
    <div class="rs-hero">
      <div class="rs-hero-mark">{_IC_SCAN}</div>
      <div>
        <h1 class="rs-hero-title">Resume Screener</h1>
        <p class="rs-hero-sub">Sort a resume into a category, then screen it against a real job posting.</p>
      </div>
    </div>
    ''',
    unsafe_allow_html=True,
)

# File uploader
section(_IC_SCAN, "Upload your resume")
file = st.file_uploader(
    "Choose a PDF file", type="pdf",
    help="Upload a PDF resume to classify it and screen it against a job posting.",
    label_visibility="collapsed",
)

if file:
    tmp_file_path = None
    try:
        file_value = file.getvalue()

        # A new upload invalidates any stored screen/tailor results.
        _fid = f"{file.name}:{file.size}"
        if st.session_state.get("rs_file_id") != _fid:
            st.session_state["rs_file_id"] = _fid
            for _k in ("rs_screen", "rs_tailor", "rs_job_text"):
                st.session_state.pop(_k, None)

        # Create temporary file for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(file_value)
            tmp_file_path = tmp_file.name

        # Extract text from PDF
        text = read_pdf(tmp_file_path)

        if text.strip():
            tab_screen, tab_cat, tab_prev, tab_text = st.tabs(
                ["Screener", "Category", "Preview", "Extracted Text"]
            )

            with tab_screen:
                # -----------------------------------------------------------
                # Score the resume against a specific posting the user provides,
                # then have Claude say what's missing and how to fix it.
                # -----------------------------------------------------------
                section(_IC_TARGET, "Screen against a job posting")
                st.markdown(
                    '<p class="rs-caption" style="margin-top:0">Paste a job link or the description '
                    'text. You get a match score and specific fixes, scored the way a recruiter or an '
                    'ATS reads a resume.</p>',
                    unsafe_allow_html=True,
                )
                posting_url = st.text_input(
                    "Job posting URL", placeholder="https://company.com/careers/backend-engineer",
                )
                st.caption(
                    "Big boards like LinkedIn, Indeed, and Workday block scraping. If a link does not "
                    "load, paste the description below instead."
                )
                pasted_jd = st.text_area("Or paste the job description", height=150)

                def resolve_posting():
                    """Return (job_text, error) from the pasted text or the URL."""
                    jt = pasted_jd.strip()
                    if jt:
                        return jt, None
                    if posting_url.strip():
                        try:
                            with st.spinner("Reading the posting..."):
                                return screener.fetch_job_posting(posting_url), None
                        except screener.PostingFetchError as e:
                            return None, str(e)
                    return None, "Add a job link or paste the description first."

                # Screen and Tailor sit next to each other. Narrow columns keep the
                # two buttons adjacent rather than stretched across the page. The
                # Tailor button is rendered after the screen logic so a just-finished
                # screen enables it on the same run.
                col_screen, col_tailor, _btn_spacer = st.columns([1, 1, 2])
                if col_screen.button("Screen my resume", type="primary"):
                    job_text, err = resolve_posting()
                    if err:
                        st.warning(err)
                    else:
                        try:
                            with st.spinner("Scoring your resume against the posting..."):
                                st.session_state["rs_screen"] = screener.screen(
                                    text, job_text, load_matcher().model
                                )
                            st.session_state["rs_job_text"] = job_text
                            st.session_state.pop("rs_tailor", None)
                        except Exception as e:
                            st.error(f"The screen failed: {str(e)}")
                            logger.error(f"Screener error: {e}")

                # Tailoring only makes sense once a screening exists, so the button
                # stays disabled until then (rather than popping in and out and
                # shifting the layout). It reuses the exact posting that was screened.
                _has_screen = bool(st.session_state.get("rs_screen"))
                _tailor_clicked = col_tailor.button(
                    "Tailor my resume",
                    disabled=not _has_screen,
                    help="Screen your resume against a posting first, then tailor it to that posting.",
                )
                if _tailor_clicked and _has_screen:
                    job_text = st.session_state.get("rs_job_text")
                    if not job_text:
                        st.warning("Screen your resume against a posting first.")
                    else:
                        try:
                            with st.spinner("Rewriting your resume for this posting..."):
                                st.session_state["rs_tailor"] = screener.tailor(
                                    text, job_text, load_matcher().model
                                )
                        except Exception as e:
                            st.error(f"The tailoring failed: {str(e)}")
                            logger.error(f"Tailor error: {e}")

                # Render the stored screen result (survives the tailor button rerun).
                result = st.session_state.get("rs_screen")
                if result:
                    if result["score"] is not None:
                        st.markdown(score_card(result["score"]), unsafe_allow_html=True)
                    if result["feedback"]:
                        st.markdown(result["feedback"])
                    # Only fall back to the retrieval-only view when Claude never
                    # ran (missing/rejected key), not on a real screen that happened
                    # to parse no score.
                    if result.get("needs_key") and result["matched"]:
                        st.markdown(
                            '<div class="rs-caption" style="margin-top:14px">Requirements your '
                            'resume already matches:</div>',
                            unsafe_allow_html=True,
                        )
                        hits = "".join(
                            f'<div class="rs-side-item">{_IC_DOT}<span>{m}</span></div>'
                            for m in result["matched"]
                        )
                        st.markdown(f'<div class="rs-card">{hits}</div>', unsafe_allow_html=True)

                tailored = st.session_state.get("rs_tailor")
                if tailored:
                    st.divider()
                    section(_IC_SPARK, "Tailored rewrite")
                    st.markdown(
                        '<p class="rs-caption" style="margin-top:0">Your resume kept intact, with only '
                        'the few lines that help this posting reworded. Check every line is true before '
                        'you use it.</p>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(tailored)

            with tab_cat:
                # Broad resume-category classifier (43 categories).
                section(_IC_TAG, "Resume category")

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
                        f'<span class="rs-highlight-meta">&nbsp;&nbsp;top category</span></div></div>'
                        f'<div class="rs-card">{rows}'
                        f'<div class="rs-caption">The classifier sorts a resume into one of 43 '
                        f'categories, like Data Science, Java Developer, or HR. Confidence is the '
                        f'model\'s calibrated probability, from a LinearSVC wrapped in '
                        f'CalibratedClassifierCV.</div></div>',
                        unsafe_allow_html=True,
                    )
                except Exception as e:
                    st.error(f"Error in category prediction: {str(e)}")
                    logger.error(f"Category prediction error: {e}")

            with tab_prev:
                section(_IC_INFO, "Resume preview")
                pdf_viewer(file_value)

            with tab_text:
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
    st.caption(
        "Click the box above or drag a PDF in to begin, then classify it and screen it "
        "against a job posting."
    )

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.toggle("Dark mode", key="rs_dark", help="Switch between the light and dark theme.")

    st.markdown(
        f'<div class="rs-side-block"><div class="rs-side-h">{_IC_INFO} How it works</div>'
        f'<div class="rs-feature">A scikit-learn model sorts your resume into one of 43 categories. '
        f'The screener reads a job posting, retrieves the requirements closest to your resume, and '
        f'asks Claude to score the fit and tell you what to fix.</div></div>',
        unsafe_allow_html=True,
    )

    metrics = load_metrics()
    if metrics:
        n_resumes = metrics.get("n_resumes", 0)
        n_categories = metrics.get("n_categories", "?")
        top1 = metrics.get("holdout_top1_accuracy", 0)
        top3 = metrics.get("holdout_top3_accuracy", 0)
        macro_f1 = metrics.get("holdout_macro_f1", 0)
        trained = metrics.get("trained_at", "")
        st.markdown(
            f'<div class="rs-side-block"><div class="rs-side-h">{_IC_CHART} Classifier</div>'
            f'<div class="rs-kpi-grid">'
            f'<div class="rs-kpi"><div class="rs-kpi-val">{n_resumes:,}</div>'
            f'<div class="rs-kpi-label">resumes trained</div></div>'
            f'<div class="rs-kpi"><div class="rs-kpi-val">{n_categories}</div>'
            f'<div class="rs-kpi-label">categories</div></div>'
            f'<div class="rs-kpi"><div class="rs-kpi-val">{top1:.0%}</div>'
            f'<div class="rs-kpi-label">top-1 accuracy</div></div>'
            f'<div class="rs-kpi"><div class="rs-kpi-val">{top3:.0%}</div>'
            f'<div class="rs-kpi-label">top-3 accuracy</div></div>'
            f'</div>'
            f'<div class="rs-side-note" style="margin-top:12px">'
            f'Macro F1 {macro_f1:.2f}, held out on the Resume-Atlas dataset. TF-IDF over 1 to 2 grams '
            f'into a calibrated LinearSVC. Trained {trained}.</div></div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div class="rs-side-block"><div class="rs-side-h">{_IC_SPARK} Screener</div>'
            f'<div class="rs-side-note">MiniLM sentence embeddings (384-dim) retrieve the closest '
            f'posting requirements with a FAISS index, then Claude scores the fit and writes the '
            f'feedback.</div></div>',
            unsafe_allow_html=True,
        )
