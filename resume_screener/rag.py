"""
Retrieval-Augmented Generation for resume ⇄ job-description fit analysis.

Pipeline:
  1. chunk each job description into passages,
  2. embed them (sentence-transformers) into a FAISS vector index,
  3. for an uploaded resume, retrieve the most relevant job-requirement passages,
  4. have Claude generate a grounded "fit analysis" using ONLY those passages.

The LLM step needs an Anthropic API key in the ANTHROPIC_API_KEY environment
variable. Retrieval works without a key (useful for testing).
"""

import logging
import os
from typing import List, Dict, Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from resume_screener.parser import read_pdf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# One Anthropic key unlocks every model — this constant is the only thing to
# change to trade cost for quality:
#   claude-haiku-4-5  ($1/$5)   — cheapest, fast
#   claude-sonnet-5   ($2/$10)  — best middle ground (default)
#   claude-opus-5     ($5/$25)  — highest quality, overkill for this task
MODEL = "claude-sonnet-5"
EMBED_MODEL = "all-MiniLM-L6-v2"
MAX_CHUNK_CHARS = 300


def _chunk(text: str, max_chars: int = MAX_CHUNK_CHARS) -> List[str]:
    """Group a document's non-empty lines into ~max_chars passages."""
    chunks, current = [], ""
    for line in (l.strip() for l in text.split("\n")):
        if not line:
            continue
        if current and len(current) + len(line) + 1 > max_chars:
            chunks.append(current)
            current = line
        else:
            current = f"{current} {line}".strip()
    if current:
        chunks.append(current)
    return chunks


class RagEngine:
    """Vector retrieval over job descriptions + grounded Claude generation."""

    def __init__(self, job_descriptions: Dict[str, str], embedder: Optional[SentenceTransformer] = None):
        """
        Args:
            job_descriptions: {role_title: source}, where source is either a
                ``.pdf`` path (read from disk) or the raw job-description text.
            embedder: an existing SentenceTransformer to reuse (avoids loading
                the model twice); a new one is created if None.
        """
        self.embedder = embedder or SentenceTransformer(EMBED_MODEL)
        self.chunks: List[Dict[str, str]] = []   # [{role, text}] aligned to index rows
        self.index: Optional[faiss.Index] = None
        self._build(job_descriptions)

    def _build(self, job_descriptions: Dict[str, str]) -> None:
        """Chunk + embed every job description into one FAISS cosine index."""
        for role, source in job_descriptions.items():
            # A source ending in .pdf is a file to read; anything else is the
            # job-description text itself.
            if source.endswith(".pdf"):
                if not os.path.exists(source):
                    logger.warning(f"Job description not found, skipping: {source}")
                    continue
                text = read_pdf(source)
            else:
                text = source
            for passage in _chunk(text):
                self.chunks.append({"role": role, "text": passage})

        if not self.chunks:
            logger.warning("No job-description chunks were indexed")
            return

        embeddings = self.embedder.encode(
            [c["text"] for c in self.chunks], normalize_embeddings=True
        ).astype("float32")
        # Inner product on normalized vectors == cosine similarity.
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        logger.info(f"Indexed {len(self.chunks)} passages from {len(job_descriptions)} roles")

    def retrieve(self, query: str, k: int = 5, role: Optional[str] = None) -> List[Dict]:
        """
        Return the k passages most relevant to ``query``.

        Args:
            query: text to search with (e.g. the resume content)
            k: number of passages to return
            role: if given, restrict results to that role's passages
        """
        if self.index is None:
            return []
        q = self.embedder.encode([query], normalize_embeddings=True).astype("float32")
        # Over-fetch so role filtering still leaves k results.
        n = min(len(self.chunks), max(k * 4, k))
        scores, idxs = self.index.search(q, n)

        results = []
        for score, i in zip(scores[0], idxs[0]):
            if i < 0:
                continue
            chunk = self.chunks[i]
            if role and chunk["role"] != role:
                continue
            results.append({**chunk, "score": float(score)})
            if len(results) >= k:
                break
        return results

    def explain_fit(self, resume_path: str, role: str, k: int = 6) -> str:
        """
        Generate a grounded fit analysis of a resume against one role.

        Retrieves the role's most relevant requirement passages, then asks
        Claude to judge fit using only those passages and the resume.
        """
        import anthropic  # imported here so retrieval works without the SDK/key

        resume_text = read_pdf(resume_path)
        passages = self.retrieve(resume_text, k=k, role=role)
        if not passages:
            return f"No indexed job requirements found for {role}."

        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            return ("⚠️ No Anthropic API key found. Set ANTHROPIC_API_KEY to enable AI fit "
                    "analysis (retrieval still works without it).")

        context = "\n".join(f"- {p['text']}" for p in passages)
        system = (
            "You are a technical recruiter. Judge how well the candidate fits the role "
            "using ONLY the provided job requirements and resume text. Cite specific "
            "evidence from the resume. If a requirement has no supporting evidence, list "
            "it as a gap. Never invent experience the resume does not show. Be concise."
        )
        user = (
            f"ROLE: {role}\n\n"
            f"RETRIEVED JOB REQUIREMENTS:\n{context}\n\n"
            f"RESUME:\n{resume_text[:4000]}\n\n"
            "Respond with:\n"
            "1. A one-sentence fit verdict.\n"
            "2. 'Strengths' — 3-5 bullets, each an actual requirement matched to resume evidence.\n"
            "3. 'Gaps' — 2-3 bullets of requirements the resume does not support."
        )

        try:
            client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
            resp = client.messages.create(
                model=MODEL,
                max_tokens=1024,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return "".join(b.text for b in resp.content if b.type == "text")
        except anthropic.AuthenticationError:
            return ("⚠️ No valid Anthropic API key found. Set ANTHROPIC_API_KEY to enable "
                    "AI fit analysis (retrieval still works without it).")
        except anthropic.RateLimitError:
            return "⚠️ Rate limited by the Anthropic API — please retry in a moment."
        except anthropic.AnthropicError as e:
            # Covers API errors and the "no api_key set" construction error.
            logger.error(f"Anthropic error: {e}")
            return f"⚠️ AI fit analysis failed: {e}"


if __name__ == "__main__":
    import sys
    from resume_screener.config import PATHS

    jobs = PATHS["job_descriptions"]
    engine = RagEngine(jobs)
    if len(sys.argv) >= 3:
        resume, role = sys.argv[1], sys.argv[2]
        print(engine.explain_fit(resume, role))
    elif len(sys.argv) == 2:
        for hit in engine.retrieve(read_pdf(sys.argv[1]), k=5):
            print(f"[{hit['score']:.2f}] ({hit['role']}) {hit['text'][:90]}")
    else:
        print("Usage: python rag.py <resume.pdf> [role]")
