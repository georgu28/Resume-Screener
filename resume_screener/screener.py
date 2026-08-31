"""
Score a resume against a specific job posting, the way a company screener would.

Two steps:
  1. fetch_job_posting(url) — pull the posting text off a URL (with a paste-text
     fallback in the UI for links that block scraping or render via JavaScript),
  2. screen(resume, posting) — retrieve the requirements most relevant to the
     resume (RAG), then have Claude score the fit and say what's missing, what to
     change, and how to improve.

The scoring rubric mirrors how resumes actually get ranked and reach a recruiter:
keyword/skill match to the posting, coverage of the required qualifications,
relevant recent titles, quantified impact, and clean parseability.

The Claude step needs ANTHROPIC_API_KEY; retrieval and fetching work without it.
"""

import logging
import re
from typing import Dict, List, Optional

import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer

from resume_screener.rag import MODEL, RagEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# A real browser UA gets past the laziest bot filters; sites that truly gate
# (LinkedIn, Indeed, Workday) still won't render server-side, which is what the
# paste fallback is for.
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# Below this many characters the "posting" is almost certainly a login wall or a
# JS shell, not the real description — tell the caller to use the paste box.
_MIN_POSTING_CHARS = 200


class PostingFetchError(Exception):
    """Raised when a URL can't be turned into usable job-posting text."""


def fetch_job_posting(url: str, timeout: int = 15) -> str:
    """
    Fetch a job posting URL and return its readable text.

    Raises:
        PostingFetchError: if the request fails or yields too little text to be
            a real posting (the UI catches this and points the user at the paste
            box).
    """
    if not re.match(r"^https?://", url.strip(), re.IGNORECASE):
        raise PostingFetchError("Enter a full URL starting with http:// or https://")

    try:
        resp = requests.get(url.strip(), headers=_HEADERS, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise PostingFetchError(f"Couldn't fetch that link ({e}).") from e

    soup = BeautifulSoup(resp.text, "lxml")
    # Drop everything that isn't body copy.
    for tag in soup(["script", "style", "noscript", "nav", "header", "footer", "form", "svg"]):
        tag.decompose()

    # Prefer the main content region if the page marks one up.
    root = soup.find("main") or soup.find("article") or soup.body or soup
    lines = [line.strip() for line in root.get_text("\n").splitlines()]
    text = "\n".join(line for line in lines if line)

    if len(text) < _MIN_POSTING_CHARS:
        raise PostingFetchError(
            "That link didn't return the full description (it may need a login or "
            "load its content with JavaScript). Paste the text instead."
        )
    return text


_SYSTEM = (
    "You are an experienced technical recruiter and the ATS that screens resumes "
    "before a human sees them. Judge how well a candidate's resume matches a "
    "specific job posting, using ONLY the posting and the resume text provided. "
    "Weigh what actually decides screening: keyword and skill match to the posting "
    "(exact tools, titles, and certifications), coverage of the required "
    "qualifications, relevant recent titles, and quantified impact. Cite concrete "
    "evidence from the resume. Never invent experience the resume does not show; if "
    "a requirement has no support, it is a gap. Be specific and honest, not "
    "flattering.\n\n"
    "Reason carefully about graduation dates and internship availability. Treat "
    "academic seasons as month ranges: Spring is about January to May, Summer about "
    "May to August, Fall about September to December, Winter about December to "
    "February. A graduation date matches a required window when it falls anywhere "
    "inside that window, including at a season boundary; for example, a May 2028 "
    "graduation is within a 'Spring/Summer 2028' window. A student graduating in a "
    "given term is available for internships and co-ops in the terms and summers "
    "before that date, so infer availability from the graduation date rather than "
    "requiring the resume to state it. Only flag a timing or availability mismatch "
    "when the graduation date genuinely falls outside the posting's stated window, "
    "and never invent a timing gap."
)


def _build_prompt(resume_text: str, job_text: str, matched: List[str]) -> str:
    evidence = "\n".join(f"- {m}" for m in matched) or "(none retrieved)"
    return (
        f"JOB POSTING:\n{job_text[:6000]}\n\n"
        f"REQUIREMENTS THIS RESUME ALREADY LOOKS CLOSE TO (retrieved):\n{evidence}\n\n"
        f"RESUME:\n{resume_text[:5000]}\n\n"
        "Respond in GitHub-flavored markdown with EXACTLY these parts:\n"
        "First line, nothing else on it: `SCORE: <0-100>` — how a recruiter/ATS "
        "would rate this resume's match to THIS posting.\n"
        "**Verdict** - one sentence: is this likely to pass the initial screen, and why.\n"
        "**Matched** - 3-6 bullets, each a posting requirement met, with the resume "
        "evidence that proves it.\n"
        "**Missing** - 3-6 bullets of required or important keywords, skills, or "
        "qualifications from the posting that the resume does not show. These are the "
        "gaps that sink an ATS match.\n"
        "**Changes to make** - 3-6 concrete edits: exact keywords to add (only where the "
        "resume can truthfully support them), bullets to reword, things to surface in the "
        "summary or skills section.\n"
        "**Do this first** - the 2-3 highest-impact changes, ranked. Never suggest "
        "fabricating experience.\n\n"
        "Write with plain punctuation. Do not use em dashes or en dashes."
    )


def screen(resume_text: str, job_text: str, embedder: SentenceTransformer, k: int = 8) -> Dict:
    """
    Score a resume against a job posting and explain the fit.

    Returns a dict: {"score": Optional[int], "feedback": markdown_str,
    "matched": [retrieved requirement passages]}.

    The retrieval step (RAG) surfaces the requirements the resume is closest to;
    Claude then scores against the FULL posting so it can also find the gaps that
    retrieval — which only finds matches — would miss.
    """
    import os

    engine = RagEngine({"posting": job_text}, embedder=embedder)
    matched = [hit["text"] for hit in engine.retrieve(resume_text, k=k, role="posting")]

    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        return {
            "score": None,
            "feedback": (
                "Set ANTHROPIC_API_KEY to get the scored screen and tailored feedback. "
                "Until then, here are the posting requirements your resume is closest to."
            ),
            "matched": matched,
        }

    import anthropic

    try:
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=MODEL,
            max_tokens=1400,
            system=_SYSTEM,
            messages=[{"role": "user", "content": _build_prompt(resume_text, job_text, matched)}],
        )
        raw = "".join(b.text for b in resp.content if b.type == "text")
    except anthropic.AuthenticationError:
        return {"score": None, "matched": matched,
                "feedback": "That ANTHROPIC_API_KEY was rejected. Check the key and try again."}
    except anthropic.RateLimitError:
        return {"score": None, "matched": matched,
                "feedback": "Rate limited by the Anthropic API - give it a moment and retry."}
    except anthropic.AnthropicError as e:
        logger.error(f"Anthropic error: {e}")
        return {"score": None, "matched": matched, "feedback": f"The screen failed: {e}"}

    score, feedback = _split_score(raw)
    return {"score": score, "feedback": feedback, "matched": matched}


def _split_score(raw: str) -> (Optional[int], str):
    """Pull a leading ``SCORE: NN`` off the response; return (score, rest)."""
    m = re.search(r"SCORE:\s*(\d{1,3})", raw)
    score = None
    if m:
        score = max(0, min(100, int(m.group(1))))
        # Drop the whole line the score sat on.
        raw = re.sub(r"^.*SCORE:\s*\d{1,3}.*$\n?", "", raw, count=1, flags=re.MULTILINE)
    return score, raw.strip()
