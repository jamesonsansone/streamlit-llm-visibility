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

import json

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
# Page freshness
# ---------------------------------------------------------------------------

def fetch_last_modified(url: str) -> str:
    """
    Returns the Article schema dateModified for a URL.

    Tries in order:
      1. JSON-LD dateModified from Article/BlogPosting schema (most accurate)
      2. JSON-LD datePublished as fallback
      3. HTTP Last-Modified response header (least reliable — often CDN cache date)
      4. Returns "" if nothing found or on any error

    Handles both flat JSON-LD objects and @graph arrays (e.g. WordPress/AC schema).
    """
    _headers = {"User-Agent": "Mozilla/5.0 (compatible; FanoutBot/1.0)"}

    def _extract_date_from_jsonld(html: str) -> str:
        for block in re.findall(
            r'<script[^>]+application/ld\+json[^>]*>(.*?)</script>',
            html,
            re.DOTALL | re.IGNORECASE,
        ):
            try:
                obj = json.loads(block)
                # Unwrap @graph arrays — find the Article/BlogPosting node
                candidates = []
                if isinstance(obj, list):
                    candidates = obj
                elif isinstance(obj, dict) and "@graph" in obj:
                    candidates = obj["@graph"]
                else:
                    candidates = [obj]

                # Prefer Article/BlogPosting nodes; fall back to any node with dateModified
                article_nodes = [
                    c for c in candidates
                    if isinstance(c, dict)
                    and any(t in str(c.get("@type", "")) for t in ("Article", "BlogPosting", "NewsArticle"))
                ]
                search_nodes = article_nodes if article_nodes else [c for c in candidates if isinstance(c, dict)]

                for node in search_nodes:
                    if node.get("dateModified"):
                        return node["dateModified"]
                for node in search_nodes:
                    if node.get("datePublished"):
                        return node["datePublished"]
            except Exception:
                pass
        return ""

    # 1. Try JSON-LD first (editorial date, most accurate)
    try:
        r = requests.get(url, timeout=12, headers=_headers)
        date = _extract_date_from_jsonld(r.text)
        if date:
            return date
        # Save Last-Modified header from this response as fallback
        lm_fallback = r.headers.get("Last-Modified", "")
    except Exception:
        lm_fallback = ""

    # 2. Fall back to HTTP Last-Modified header (CDN/cache date — less reliable)
    if lm_fallback:
        return lm_fallback

    # 3. Try HEAD request as last resort
    try:
        r = requests.head(url, timeout=8, allow_redirects=True, headers=_headers)
        return r.headers.get("Last-Modified", "")
    except Exception:
        pass

    return ""


# ---------------------------------------------------------------------------
# Answer capsule detection
# ---------------------------------------------------------------------------

def detect_answer_capsules(chunks: list[dict]) -> list[dict]:
    """
    Identifies chunks structured like answer capsules — short, declarative,
    self-contained passages that an AI can extract verbatim.

    Heuristics (all must pass):
      - word_count < 80
      - embed_score > 0.50
      - First sentence does not start with a contextual pronoun (requires prior context)
      - Contains at least one declarative signal (year, %, superlative, named tool)

    Returns qualifying chunks sorted by embed_score desc.
    Chunks must have been scored by score_chunks_embedding() first.
    """
    _contextual = re.compile(r'^(This|These|It|They|Our|We|Here|That)\b', re.IGNORECASE)
    _declarative = re.compile(
        r'\b(20\d{2}|best|most|top|first|leading|\d+%|\d+ percent|key|primary)\b',
        re.IGNORECASE,
    )
    capsules = []
    for c in chunks:
        if c.get("word_count", 999) >= 80:
            continue
        if c.get("embed_score", 0) < 0.50:
            continue
        first_sentence = c["text"].split('.')[0].strip()
        if _contextual.match(first_sentence):
            continue
        if not _declarative.search(c["text"]):
            continue
        capsules.append(c)
    return sorted(capsules, key=lambda x: x.get("embed_score", 0), reverse=True)


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

        return [{**c, "embed_score": float(scores[i]), "_embedding": chunk_embs[i]} for i, c in enumerate(chunks)]

    except Exception as exc:
        logger.warning(f"score_chunks_embedding failed: {exc}")
        return [{**c, "embed_score": 0.0} for c in chunks]


# ---------------------------------------------------------------------------
# Gap detection
# ---------------------------------------------------------------------------

def find_content_gaps(
    top_url_chunks: list[dict],
    ac_chunks: list[dict],
    ai_relevance_threshold: float = 0.45,
    topic_coverage_threshold: float = 0.55,
) -> list[dict]:
    """
    Identifies true per-topic content gaps using chunk-to-chunk semantic comparison.

    For each competitor chunk that the AI referenced (embed_score >= ai_relevance_threshold):
      - Compute cosine similarity between this chunk's embedding and every AC chunk's embedding
      - If the best-matching AC chunk scores below topic_coverage_threshold, it's a gap
      - Attach topic_similarity (best AC match score) and best_ac_match (heading) to each gap

    This is fundamentally different from the old approach: we compare competitor content
    against AC content directly, not against the AI answer. A gap means the competitor
    covers a topic the AI cares about and AC genuinely doesn't have equivalent content for.

    Graceful fallback: if _embedding keys are missing (model failed), reverts to
    the simpler heuristic so the tab never crashes.
    """
    if not top_url_chunks:
        return []

    import numpy as np

    # Full gap case — AC was not cited at all, no AC chunks to compare against
    if not ac_chunks:
        gaps = [c for c in top_url_chunks if c.get("embed_score", 0) >= ai_relevance_threshold]
        return sorted(gaps, key=lambda x: x.get("embed_score", 0), reverse=True)

    # Check whether embedding vectors are available for per-topic comparison
    has_embeddings = (
        "_embedding" in top_url_chunks[0] and
        "_embedding" in ac_chunks[0]
    )

    if not has_embeddings:
        # Graceful fallback to old global-threshold heuristic
        ac_scores = np.array([c.get("embed_score", 0.0) for c in ac_chunks])
        max_ac = float(np.max(ac_scores))
        gaps = [
            c for c in top_url_chunks
            if c.get("embed_score", 0) >= ai_relevance_threshold and max_ac < 0.25
        ]
        return sorted(gaps, key=lambda x: x.get("embed_score", 0), reverse=True)

    # Per-topic pairwise comparison using stored embedding vectors
    ac_embs = np.stack([c["_embedding"] for c in ac_chunks])   # shape: (n_ac, dim)
    ac_norms = np.linalg.norm(ac_embs, axis=1, keepdims=True)
    ac_units = ac_embs / (ac_norms + 1e-10)

    gaps = []
    for chunk in top_url_chunks:
        if chunk.get("embed_score", 0) < ai_relevance_threshold:
            continue  # AI didn't meaningfully reference this topic — skip

        comp_emb = chunk["_embedding"].reshape(1, -1)
        comp_unit = comp_emb / (np.linalg.norm(comp_emb) + 1e-10)

        # Similarity between this competitor chunk and every AC chunk (topic-to-topic)
        topic_sims = (ac_units @ comp_unit.T).flatten()
        best_idx = int(np.argmax(topic_sims))
        max_topic_sim = float(topic_sims[best_idx])

        if max_topic_sim < topic_coverage_threshold:
            # AC has no section close enough to this topic — genuine content gap
            gaps.append({
                **chunk,
                "topic_similarity": round(max_topic_sim, 3),
                "best_ac_match": ac_chunks[best_idx]["heading"],
            })

    # Sort by AI relevance (embed_score) desc — most impactful gaps first
    return sorted(gaps, key=lambda x: x.get("embed_score", 0), reverse=True)
