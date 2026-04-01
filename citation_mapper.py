"""
citation_mapper.py — Content Gap Analysis for Query Fan-Out.

Fetches two URLs (top competitor + top AC), chunks them by section,
scores each chunk against the aggregated AI answer text using both
TF-IDF and sentence embeddings, then identifies content gaps.

No API key required. Uses:
  - Jina Reader (https://r.jina.ai/) — free public endpoint for clean markdown
  - scikit-learn TfidfVectorizer — pure CPU, no model download
  - sentence-transformers all-MiniLM-L6-v2 — 22MB, CPU-only, one-time download
"""

import re
import logging
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page fetching
# ---------------------------------------------------------------------------

def fetch_page_content(url: str) -> str:
    """
    Fetches a URL via Jina Reader and returns clean markdown text.
    Returns "" on any failure (timeout, 4xx, 5xx, network error).
    """
    jina_url = f"https://r.jina.ai/{url}"
    headers = {
        "Accept": "text/markdown",
        "X-Return-Format": "markdown",
    }
    try:
        resp = requests.get(jina_url, headers=headers, timeout=20)
        if resp.status_code == 200:
            return resp.text
        logger.warning(f"Jina fetch returned {resp.status_code} for {url}")
        return ""
    except Exception as exc:
        logger.warning(f"fetch_page_content failed for {url}: {exc}")
        return ""


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_page(markdown: str, max_words: int = 120) -> list[dict]:
    """
    Splits a markdown page into content chunks based on headings and paragraphs.

    Strategy:
      1. Split on h2/h3 headings (## / ###) to get named sections
      2. If a section exceeds max_words, split further at blank-line boundaries
      3. Drop chunks with fewer than 20 words (navigation, footers, noise)

    Returns list of {"heading": str, "text": str, "word_count": int}
    """
    if not markdown:
        return []

    # Split on ## or ### headings, keeping the heading text
    section_pattern = re.compile(r'\n(?=#{2,3} )', re.MULTILINE)
    raw_sections = section_pattern.split(markdown)

    chunks = []
    for section in raw_sections:
        lines = section.strip().splitlines()
        if not lines:
            continue

        # Extract heading from first line if it starts with #
        first = lines[0].strip()
        if first.startswith('#'):
            heading = re.sub(r'^#+\s*', '', first).strip()
            body = '\n'.join(lines[1:]).strip()
        else:
            heading = "Introduction"
            body = section.strip()

        if not body:
            continue

        word_count = len(body.split())
        if word_count <= max_words:
            if word_count >= 20:
                chunks.append({"heading": heading, "text": body, "word_count": word_count})
        else:
            # Split at paragraph boundaries
            paragraphs = re.split(r'\n\n+', body)
            buffer_text = ""
            buffer_words = 0
            para_index = 0
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                pw = len(para.split())
                if buffer_words + pw > max_words and buffer_text:
                    if buffer_words >= 20:
                        sub_heading = heading if para_index == 0 else f"{heading} (cont.)"
                        chunks.append({"heading": sub_heading, "text": buffer_text.strip(), "word_count": buffer_words})
                        para_index += 1
                    buffer_text = para
                    buffer_words = pw
                else:
                    buffer_text = f"{buffer_text}\n\n{para}".strip() if buffer_text else para
                    buffer_words += pw
            if buffer_text and buffer_words >= 20:
                sub_heading = heading if para_index == 0 else f"{heading} (cont.)"
                chunks.append({"heading": sub_heading, "text": buffer_text.strip(), "word_count": buffer_words})

    return chunks


# ---------------------------------------------------------------------------
# TF-IDF scoring
# ---------------------------------------------------------------------------

def score_chunks_tfidf(reference_doc: str, chunks: list[dict]) -> list[dict]:
    """
    Scores each chunk against reference_doc using TF-IDF cosine similarity.

    Fits a TfidfVectorizer on [reference_doc] + all chunk texts, then
    computes cosine similarity of each chunk vector against the reference vector.

    Returns a copy of chunks with added "tfidf_score": float (0.0–1.0).
    """
    if not chunks or not reference_doc:
        return [{**c, "tfidf_score": 0.0} for c in chunks]

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        texts = [reference_doc] + [c["text"] for c in chunks]
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            stop_words="english",
            sublinear_tf=True,
        )
        tfidf_matrix = vectorizer.fit_transform(texts)
        ref_vec = tfidf_matrix[0]
        chunk_vecs = tfidf_matrix[1:]
        scores = cosine_similarity(ref_vec, chunk_vecs).flatten()

        return [{**c, "tfidf_score": float(scores[i])} for i, c in enumerate(chunks)]

    except Exception as exc:
        logger.warning(f"score_chunks_tfidf failed: {exc}")
        return [{**c, "tfidf_score": 0.0} for c in chunks]


# ---------------------------------------------------------------------------
# Embedding scoring
# ---------------------------------------------------------------------------

def _load_model():
    """
    Loads sentence-transformers model. Called inside st.cache_resource when
    running in Streamlit, otherwise loads directly.
    """
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


def score_chunks_embedding(reference_doc: str, chunks: list[dict]) -> list[dict]:
    """
    Scores each chunk against reference_doc using sentence-transformer embeddings.

    Uses all-MiniLM-L6-v2 (22MB, CPU-only, no API key). First call downloads
    the model; subsequent calls use the cached model within the session.

    Returns a copy of chunks with added "embed_score": float (0.0–1.0).
    """
    if not chunks or not reference_doc:
        return [{**c, "embed_score": 0.0} for c in chunks]

    try:
        import numpy as np

        # Use Streamlit cache if available, otherwise load directly
        try:
            import streamlit as st
            model = st.cache_resource(_load_model)()
        except Exception:
            model = _load_model()

        chunk_texts = [c["text"] for c in chunks]
        all_texts = [reference_doc] + chunk_texts

        embeddings = model.encode(all_texts, batch_size=32, show_progress_bar=False)
        ref_emb = embeddings[0].reshape(1, -1)
        chunk_embs = embeddings[1:]

        # Manual cosine similarity — avoids sklearn import in this path
        ref_norm = np.linalg.norm(ref_emb, axis=1, keepdims=True)
        chunk_norms = np.linalg.norm(chunk_embs, axis=1, keepdims=True)
        ref_unit = ref_emb / (ref_norm + 1e-10)
        chunk_units = chunk_embs / (chunk_norms + 1e-10)
        scores = (chunk_units @ ref_unit.T).flatten()

        return [{**c, "embed_score": float(scores[i])} for i, c in enumerate(chunks)]

    except Exception as exc:
        logger.warning(f"score_chunks_embedding failed: {exc}")
        return [{**c, "embed_score": 0.0} for c in chunks]


# ---------------------------------------------------------------------------
# Gap detection
# ---------------------------------------------------------------------------

def find_content_gaps(
    top_url_chunks: list[dict],
    ac_chunks: list[dict],
    threshold: float = 0.25,
) -> list[dict]:
    """
    Identifies content gaps: topics covered by top_url that AC does NOT cover.

    For each top_url chunk, finds the most similar AC chunk using embed_score.
    If the best AC match is below `threshold`, the topic is considered a gap.

    When ac_chunks is empty (AC was not cited at all), every top_url chunk
    that has embed_score > 0.1 is returned as a gap.

    Returns gap chunks sorted by tfidf_score desc (most AI-referenced gaps first).
    Chunks must have "embed_score" and "tfidf_score" from the scoring functions.
    """
    if not top_url_chunks:
        return []

    import numpy as np

    # Full gap case — AC has no content to compare against
    if not ac_chunks:
        gaps = [c for c in top_url_chunks if c.get("embed_score", 0) > 0.1]
        return sorted(gaps, key=lambda x: x.get("tfidf_score", 0), reverse=True)

    ac_embed_scores = np.array([c.get("embed_score", 0.0) for c in ac_chunks])

    gaps = []
    for chunk in top_url_chunks:
        chunk_embed = chunk.get("embed_score", 0.0)
        # Only consider chunks that the AI actually referenced
        if chunk_embed < 0.1:
            continue

        # Find how well this topic is covered in AC chunks
        # Proxy: use the ratio of this chunk's embed score to the max AC embed score
        # A chunk is a gap if it scores high but AC has nothing comparably high
        max_ac_score = float(np.max(ac_embed_scores)) if len(ac_embed_scores) > 0 else 0.0

        # Compare chunk's embed score against AC's best score for the same reference
        # If chunk scores well (>threshold) but AC's best doesn't reach threshold, it's a gap
        if max_ac_score < threshold or (chunk_embed > threshold and max_ac_score < chunk_embed * 0.7):
            gaps.append(chunk)

    return sorted(gaps, key=lambda x: x.get("tfidf_score", 0), reverse=True)
