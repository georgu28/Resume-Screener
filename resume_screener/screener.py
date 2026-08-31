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
import time
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


# Transient Anthropic failures (rate limits, overloaded 529s, brief network
# blips) are retried a few times before giving up, so one flaky call doesn't
# silently drop the user back into the retrieval-only view.
_MAX_ATTEMPTS = 4
_RETRY_BACKOFF = 1.5  # seconds, multiplied by the attempt number
_MAX_OUTPUT_TOKENS = 8000  # ceiling when growing the budget after a truncation


def _generate(system: str, user: str, max_tokens: int) -> str:
    """
    Call Claude and return the response text, retrying transient failures.

    - Rate limits, overloaded/5xx, timeouts, and connection errors are retried
      with a short linear backoff.
    - If a response is cut off at ``max_tokens`` (``stop_reason == "max_tokens"``),
      the budget is doubled and the call is retried so every section comes through
      instead of stopping mid-sentence.
    - ``AuthenticationError`` is NOT retried; it propagates so the caller can tell
      the user their key was rejected.

    Raises the last transient error if every attempt fails.
    """
    import anthropic

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    budget = max_tokens
    text = ""
    last_err: Optional[Exception] = None

    for attempt in range(_MAX_ATTEMPTS):
        try:
            resp = client.messages.create(
                model=MODEL,
                max_tokens=budget,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            text = "".join(b.text for b in resp.content if b.type == "text").strip()
            if resp.stop_reason == "max_tokens" and budget < _MAX_OUTPUT_TOKENS:
                # Truncated mid-answer (later sections never rendered). Grow the
                # budget and regenerate rather than returning a partial screen.
                budget = min(budget * 2, _MAX_OUTPUT_TOKENS)
                logger.warning(f"Response truncated at max_tokens; retrying with budget={budget}")
                continue
            return text
        except anthropic.AuthenticationError:
            raise  # not retryable - the key itself is bad
        except (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.APIStatusError) as e:
            last_err = e
            logger.warning(f"Anthropic call failed (attempt {attempt + 1}/{_MAX_ATTEMPTS}): {e}")
            time.sleep(_RETRY_BACKOFF * (attempt + 1))

    if last_err is not None:
        raise last_err
    return text  # exhausted retries on truncation - return the best partial answer


_SYSTEM = (
    "You are a technical recruiter and the ATS that screens resumes before a human "
    "sees them. Score how well a resume matches a specific job posting, using ONLY "
    "the posting and resume provided. Judge on keyword and skill match, coverage of "
    "the required qualifications, relevant titles, and quantified impact. Cite real "
    "evidence from the resume, never invent it; a requirement with no support is a gap.\n\n"
    "Ignore graduation date, class year, and internship timing or availability "
    "completely. Do not mention them ANYWHERE in your response, including the Verdict. "
    "Never raise them as a gap, a concern, a caveat, or a Missing item, never hedge the "
    "verdict on them, and never let them affect the score. Assume the timing works.\n\n"
    "Write like a person: plain, direct, specific. Short sentences, no filler, no "
    "flattery, no hedging. Plain punctuation, no em or en dashes."
)


def _build_prompt(resume_text: str, job_text: str, matched: List[str]) -> str:
    evidence = "\n".join(f"- {m}" for m in matched) or "(none retrieved)"
    return (
        f"JOB POSTING:\n{job_text[:6000]}\n\n"
        f"REQUIREMENTS THIS RESUME LOOKS CLOSE TO (retrieved):\n{evidence}\n\n"
        f"RESUME:\n{resume_text[:5000]}\n\n"
        "Reply in GitHub-flavored markdown. Be concise, no preamble. Use these parts:\n"
        "First line, nothing else: `SCORE: <0-100>` (recruiter/ATS match to THIS posting).\n"
        "**Verdict** - one plain sentence: does it pass the screen, and why.\n"
        "**Matched** - 3-4 one-line bullets: requirement, then the resume evidence for it.\n"
        "**Missing** - 3-4 one-line bullets: required skills or keywords the resume lacks.\n"
        "**Fix first** - 2-3 ranked, concrete edits (keywords to add only if true, bullets "
        "to reword). Never suggest fabricating experience."
    )


def screen(resume_text: str, job_text: str, embedder: SentenceTransformer, k: int = 8) -> Dict:
    """
    Score a resume against a job posting and explain the fit.

    Returns a dict: {"score": Optional[int], "needs_key": bool,
    "feedback": markdown_str, "matched": [retrieved requirement passages]}.

    ``needs_key`` is True only when Claude never produced a real screen (the key
    is missing or was rejected) — that's the only case the UI should fall back to
    showing the retrieved requirements. A transient failure is retried inside
    :func:`_generate` first, so a flaky call no longer lands the user there.

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
            "needs_key": True,
            "feedback": (
                "Set ANTHROPIC_API_KEY to get the scored screen and tailored feedback. "
                "Until then, here are the posting requirements your resume is closest to."
            ),
            "matched": matched,
        }

    import anthropic

    try:
        raw = _generate(_SYSTEM, _build_prompt(resume_text, job_text, matched), max_tokens=2600)
    except anthropic.AuthenticationError:
        return {"score": None, "needs_key": True, "matched": matched,
                "feedback": "That ANTHROPIC_API_KEY was rejected. Check the key and try again."}
    except anthropic.AnthropicError as e:
        logger.error(f"Anthropic error after retries: {e}")
        return {"score": None, "needs_key": True, "matched": matched,
                "feedback": f"The screen failed after several retries: {e}"}

    score, feedback = _split_score(raw)
    return {"score": score, "needs_key": False, "feedback": feedback, "matched": matched}


_TAILOR_SYSTEM = (
    "You tailor a candidate's existing resume to a specific job posting by making the "
    "SMALLEST changes that improve the match. Keep as much of the original as possible: "
    "preserve its structure, section order, wording, and every fact verbatim, and only "
    "touch a line when changing it clearly helps this posting. Edit by rewording an "
    "existing bullet to mirror the posting's language, or by adding a keyword the resume "
    "already supports. Do not rewrite lines that are already fine, do not drop content, "
    "and do not restructure.\n\n"
    "Never invent or exaggerate anything: no new experience, employers, titles, dates, "
    "metrics, degrees, or skills, and no new numbers. If the posting wants something the "
    "resume does not clearly support, leave it out of the resume and list it separately "
    "for the candidate to add only if it is true.\n\n"
    "Plain punctuation, no em or en dashes."
)


def _tailor_prompt(resume_text: str, job_text: str, matched: List[str]) -> str:
    evidence = "\n".join(f"- {m}" for m in matched) or "(none retrieved)"
    return (
        f"JOB POSTING:\n{job_text[:6000]}\n\n"
        f"REQUIREMENTS THIS RESUME LOOKS CLOSE TO (retrieved):\n{evidence}\n\n"
        f"CURRENT RESUME:\n{resume_text[:5500]}\n\n"
        "Reply in GitHub-flavored markdown with these parts:\n"
        "**Tailored resume** - the FULL resume, kept as close to the original as possible. "
        "Reproduce every section and line unchanged except the few you reword or where you "
        "add a keyword the resume already supports. Keep all facts and numbers exactly.\n"
        "**What changed** - a short bulleted list of each edit you made and why it fits the "
        "posting. If you changed nothing, say so.\n"
        "**Add only if true** - keywords or details the posting wants that the resume does "
        "not support. The candidate adds these only if accurate. Never fabricate."
    )


def tailor(resume_text: str, job_text: str, embedder: SentenceTransformer, k: int = 8) -> str:
    """
    Tailor the resume to a posting with the smallest truthful edits.

    Returns markdown: the full resume kept as close to the original as possible
    (only the few lines worth rewording or keyword-tagging are touched), a short
    list of what changed, and a separate list of things to add only if they are
    true. Needs ANTHROPIC_API_KEY.
    """
    if not _has_key():
        return "Set ANTHROPIC_API_KEY to generate a tailored rewrite of your resume."

    engine = RagEngine({"posting": job_text}, embedder=embedder)
    matched = [hit["text"] for hit in engine.retrieve(resume_text, k=k, role="posting")]

    import anthropic

    try:
        # The full resume is echoed back, so give it room; _generate grows the
        # budget further if a long resume still truncates.
        return _generate(_TAILOR_SYSTEM, _tailor_prompt(resume_text, job_text, matched), max_tokens=3200)
    except anthropic.AuthenticationError:
        return "That ANTHROPIC_API_KEY was rejected. Check the key and try again."
    except anthropic.AnthropicError as e:
        logger.error(f"Anthropic error after retries: {e}")
        return f"The tailoring failed after several retries: {e}"


def _has_key() -> bool:
    import os
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def _split_score(raw: str) -> (Optional[int], str):
    """Pull a leading ``SCORE: NN`` off the response; return (score, rest)."""
    m = re.search(r"SCORE:\s*(\d{1,3})", raw)
    score = None
    if m:
        score = max(0, min(100, int(m.group(1))))
        # Drop the whole line the score sat on.
        raw = re.sub(r"^.*SCORE:\s*\d{1,3}.*$\n?", "", raw, count=1, flags=re.MULTILINE)
    return score, raw.strip()
