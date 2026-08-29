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

# Open roles to score every uploaded resume against.
JOB_DESCRIPTIONS = {
    "Full Stack Developer": "pdfs/full-stack.pdf",
    "Front End Developer": "pdfs/front-end.pdf",
    "Product Manager": "pdfs/product-manager.pdf",
    "Java Developer": "pdfs/java.pdf",
}

# A cosine similarity this high is effectively a perfect match; used only to
# scale the progress bars so small real-world differences stay visible.
_BAR_SCALE = 0.7


# ---------------------------------------------------------------------------
# Cached heavy resources. Without these, every upload retrained the KNN model
# and reloaded the sentence-transformer from scratch (several seconds each).
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
        title: matcher.embed_job(path)
        for title, path in JOB_DESCRIPTIONS.items()
        if os.path.exists(path)
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


# Page configuration
st.set_page_config(
    page_title="MDST Resume Screener",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("MDST Resume Screener")
st.markdown("Upload a resume PDF to see how well it matches our open roles.")
st.divider()

# File uploader
st.subheader("Upload Resume")
file = st.file_uploader("Choose a PDF file", type="pdf", help="Upload a PDF resume file to begin analysis")

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
                ["Analysis Results", "Fit Analysis (AI)", "Resume Preview", "Extracted Text"]
            )

            with tab1:
                # -----------------------------------------------------------
                # Primary result: semantic match against the open roles.
                # -----------------------------------------------------------
                st.header("Job Match")
                st.markdown("How closely this resume matches each open role, by semantic similarity.")

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
                        st.success(f"**Best match:** {best_match}  ·  similarity {best_score:.3f}")

                        for job_title, score in ranked.items():
                            col1, col2, col3 = st.columns([3, 3, 1])
                            with col1:
                                st.markdown(f"**{job_title}**")
                            with col2:
                                st.progress(min(score / _BAR_SCALE, 1.0))
                            with col3:
                                st.markdown(f"{score:.3f}")

                        st.caption(
                            "Scores are cosine similarity (0–1) — compare them against each "
                            "other rather than as absolute percentages. Higher means a closer match."
                        )
                    else:
                        st.warning("No job descriptions available to compare against.")
                except Exception as e:
                    st.error(f"Error in semantic analysis: {str(e)}")
                    logger.error(f"Semantic analysis error: {e}")

                st.divider()

                # -----------------------------------------------------------
                # Secondary result: broad resume-category classifier. This is
                # independent of the roles above (25 general categories).
                # -----------------------------------------------------------
                st.header("Resume Category")
                st.caption(
                    "A broad classifier trained on 25 general resume categories "
                    "(e.g. Data Science, Java Developer, HR). Independent of the roles above."
                )

                try:
                    with st.spinner("Classifying resume..."):
                        classifier = load_classifier()
                        predicted_category = classifier.predict_pdf(tmp_file_path)
                        probabilities = classifier.get_prediction_probabilities(tmp_file_path)

                    st.info(f"**Predicted category:** {predicted_category}")

                    top_probabilities = dict(list(probabilities.items())[:5])
                    for category, prob in top_probabilities.items():
                        col1, col2, col3 = st.columns([3, 3, 1])
                        with col1:
                            st.markdown(f"**{category}**")
                        with col2:
                            st.progress(prob)
                        with col3:
                            st.markdown(f"{prob:.0%}")

                    st.caption(
                        "Confidence is the share of the 5 nearest neighbors in each category, "
                        "so values land on 0 / 20 / 40 / 60 / 80 / 100%."
                    )
                except Exception as e:
                    st.error(f"Error in category prediction: {str(e)}")
                    logger.error(f"KNN prediction error: {e}")

            with tab_fit:
                st.header("AI Fit Analysis")
                st.markdown(
                    "Retrieval-augmented generation: the most relevant job requirements are "
                    "retrieved from a vector index, then Claude explains fit grounded in that evidence."
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
                        for hit in rag.retrieve(text, k=5, role=role):
                            st.markdown(f"- _{hit['text']}_")
                except Exception as e:
                    st.error(f"Error in fit analysis: {str(e)}")
                    logger.error(f"RAG error: {e}")

            with tab2:
                st.header("Resume Preview")
                pdf_viewer(file_value)

            with tab3:
                st.header("Extracted Text")
                st.markdown("Raw text extracted from the PDF")
                with st.expander("View full extracted text", expanded=False):
                    st.text_area("Extracted Text", text, height=400, label_visibility="collapsed")
                st.caption(f"Total characters extracted: {len(text)}")
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
    st.info("Please upload a PDF resume file to begin analysis.")

# Sidebar with information
with st.sidebar:
    st.header("About")
    st.markdown("""
    This resume screener uses:

    **Semantic Job Match**
    - Uses sentence transformers
    - Compares resume to job descriptions
    - Ranks the best-fitting open roles

    **Category Classifier**
    - TF-IDF + calibrated LinearSVC
    - Trained on 12k+ resumes, 43 categories
    - Shows calibrated confidence

    **AI Fit Analysis (RAG)**
    - FAISS retrieval over job requirements
    - Claude generates grounded fit analysis

    **Features**
    - PDF text extraction
    - Cached models (fast after first load)
    - Interactive results
    """)

    st.header("Open Roles")
    st.markdown("\n".join(f"- {title}" for title in JOB_DESCRIPTIONS))

    metrics = load_metrics()
    if metrics:
        st.header("Model")
        st.markdown(
            f"- **{metrics.get('n_resumes', '?'):,} resumes** · "
            f"{metrics.get('n_categories', '?')} categories\n"
            f"- Top-1 accuracy: **{metrics.get('holdout_top1_accuracy', 0):.0%}**\n"
            f"- Top-3 accuracy: **{metrics.get('holdout_top3_accuracy', 0):.0%}**\n"
            f"- TF-IDF + calibrated LinearSVC"
        )
