"""
fanout_engine.py — Query Fan-Out Analysis Core Engine

Milestone 1: Single API call, raw terminal output (run: GEMINI_API_KEY=xxx python fanout_engine.py)
Milestone 2: Multi-run aggregation, clustering, RRF scoring, CSV export
"""

import os
import re
import time
import logging
from collections import defaultdict
from typing import Callable, Optional
from urllib.parse import urlparse

import requests
import pandas as pd
from thefuzz import fuzz

from google import genai
from google.genai import types
from google.genai.errors import ClientError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_NAME = "gemini-2.5-flash-lite"

AVAILABLE_MODELS = {
    "gemini-3.1-flash-lite-preview": "Gemini 3.1 Flash Lite (Preview) - More sub-queries",
    "gemini-2.5-flash-lite":          "Gemini 2.5 Flash Lite (Standard)",
}
FALLBACK_MODEL = "gemini-2.5-flash-lite"

RRF_K = 60
INTER_CALL_DELAY = 2.0  # seconds between API calls
MAX_RETRIES = 3
BASE_BACKOFF = 2.0  # seconds, doubled each retry

# gemini-2.5-flash-lite pricing (March 2026)
COST_INPUT_PER_1M  = 0.075   # $ per 1M input tokens
COST_OUTPUT_PER_1M = 0.300   # $ per 1M output tokens

# Module-level redirect URL cache (avoids re-resolving same URL)
_REDIRECT_CACHE: dict[str, str] = {}

KNOWN_BRANDS = [
    "ActiveCampaign",
    "Active Campaign",
    "Mailchimp",
    "Mail Chimp",
    "HubSpot",
    "Klaviyo",
    "Brevo",
    "Sendinblue",
    "Constant Contact",
    "GetResponse",
    "Omnisend",
    "Drip",
    "ConvertKit",
    "Kit",
    "Salesforce",
    "Marketo",
    "Pardot",
    "Zoho",
    "Campaign Monitor",
    "MailerLite",
    "AWeber",
]

POSITIVE_WORDS = frozenset([
    "best", "top", "recommended", "leading", "popular", "trusted",
    "effective", "powerful", "easy", "affordable", "flexible", "robust",
    "excellent", "great", "strong", "intuitive", "comprehensive", "advanced",
    "seamless", "loved", "winner", "superior", "preferred",
])

NEGATIVE_WORDS = frozenset([
    "worst", "bad", "expensive", "difficult", "poor", "limited", "slow",
    "complex", "overpriced", "buggy", "lacking", "confusing", "clunky",
    "outdated", "unreliable", "weak", "basic", "frustrating", "bloated",
    "lacking", "inferior", "steep",
])

# Regex for entity extraction — 2+ consecutive TitleCase words
_ENTITY_PATTERN = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b')


# ---------------------------------------------------------------------------
# Data structures (plain dicts — no dataclasses for POC simplicity)
# ---------------------------------------------------------------------------

# RunResult keys:
#   run_index: int
#   web_search_queries: list[str]
#   sources: list[{"uri": str, "title": str}]
#   response_text: str
#   entities: list[str]

# AggregatedResult keys:
#   subqueries: pd.DataFrame  (query, cluster_label, probability, avg_position, rrf_score, sample_variants)
#   sources:    pd.DataFrame  (uri, title, domain, probability, avg_position, rrf_score)
#   entities:   pd.DataFrame  (entity, probability, avg_position, rrf_score, sentiment)
#   raw_runs:   list[RunResult]


# ---------------------------------------------------------------------------
# URL redirect resolution
# ---------------------------------------------------------------------------

def resolve_redirect_url(url: str, timeout: float = 5.0) -> str:
    """
    Follows Vertex AI grounding redirect URLs to get the real destination URL.
    Results are cached module-level to avoid duplicate HTTP calls.
    Returns the original URL on any network error or timeout.
    """
    if not url or "vertexaisearch" not in url:
        return url
    if url in _REDIRECT_CACHE:
        return _REDIRECT_CACHE[url]
    try:
        resp = requests.head(
            url,
            allow_redirects=True,
            timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0 (compatible; fanout-analyzer/1.0)"},
        )
        real_url = resp.url
        _REDIRECT_CACHE[url] = real_url
        return real_url
    except Exception:
        _REDIRECT_CACHE[url] = url  # cache failure to avoid retrying
        return url


def resolve_sources(sources: list[dict]) -> list[dict]:
    """
    Resolves redirect URLs for a list of source dicts in-place (returns new list).
    Also re-extracts domain from the real URL after resolution.
    """
    resolved = []
    for src in sources:
        real_uri = resolve_redirect_url(src["uri"])
        resolved.append({
            "uri": real_uri,
            "title": src["title"],
        })
    return resolved


# ---------------------------------------------------------------------------
# Milestone 1 — Single API call
# ---------------------------------------------------------------------------

def call_gemini_with_grounding(query: str, client: genai.Client, model_name: str = MODEL_NAME) -> dict:
    """
    Single Gemini API call with Google Search grounding enabled.
    Returns a RunResult dict.
    Raises ResourceExhausted on 429, other google exceptions on other errors.
    """
    config = types.GenerateContentConfig(
        tools=[types.Tool(google_search=types.GoogleSearch())]
    )

    response = client.models.generate_content(
        model=model_name,
        contents=query,
        config=config,
    )

    candidate = response.candidates[0] if response.candidates else None
    metadata = candidate.grounding_metadata if candidate else None

    web_search_queries = []
    sources = []

    if metadata:
        web_search_queries = list(metadata.web_search_queries or [])
        for chunk in (metadata.grounding_chunks or []):
            if chunk.web:
                sources.append({
                    "uri": chunk.web.uri or "",
                    "title": chunk.web.title or "",
                })

    # Resolve redirect URLs to real domains
    sources = resolve_sources(sources)

    # Track token usage for cost estimation
    usage = response.usage_metadata
    input_tokens  = int(getattr(usage, "prompt_token_count", 0) or 0)
    output_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)

    response_text = response.text or ""
    entities = extract_entities(response_text)

    return {
        "web_search_queries": web_search_queries,
        "sources": sources,
        "response_text": response_text,
        "entities": entities,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


def _is_quota_exhausted(exc: ClientError) -> bool:
    """Returns True if the 429 is a hard daily/billing quota (limit: 0), not a rate limit."""
    msg = str(exc)
    return "limit: 0" in msg or "GenerateRequestsPerDay" in msg


def call_with_backoff(
    query: str,
    client: genai.Client,
    max_retries: int = MAX_RETRIES,
    base_delay: float = BASE_BACKOFF,
    model_name: str = MODEL_NAME,
) -> dict:
    """
    Wraps call_gemini_with_grounding with exponential backoff on transient 429s.
    Distinguishes retryable rate limits (RPM) from fatal quota exhaustion (daily/billing).
    Retry delays: 2s, 4s, 8s (2^attempt * base_delay).
    """
    attempt = 0
    while True:
        try:
            return call_gemini_with_grounding(query, client, model_name=model_name)
        except ClientError as exc:
            if exc.code != 429:
                logger.error(f"API error on run: {exc}")
                raise
            if _is_quota_exhausted(exc):
                logger.error(
                    "Daily quota or billing limit exhausted (limit: 0). "
                    "Set up billing at console.cloud.google.com or wait for daily quota reset."
                )
                raise
            if attempt >= max_retries:
                logger.error("Rate limit hit — max retries exceeded.")
                raise
            delay = (2 ** attempt) * base_delay
            logger.warning(f"Rate limit hit (429). Retrying in {delay:.0f}s... (attempt {attempt + 1}/{max_retries})")
            time.sleep(delay)
            attempt += 1
        except Exception as exc:
            logger.error(f"Non-retryable error on run: {exc}")
            raise


# ---------------------------------------------------------------------------
# Entity extraction & sentiment
# ---------------------------------------------------------------------------

def extract_entities(text: str) -> list[str]:
    """
    Extracts brand/entity names from response text.
    Combines regex TitleCase detection with KNOWN_BRANDS checklist.
    Returns deduplicated list, longest match wins for overlaps.
    """
    found = set()

    # Regex scan for TitleCase multi-word phrases
    for match in _ENTITY_PATTERN.finditer(text):
        found.add(match.group(1))

    # Explicit brand scan (case-insensitive match, canonical casing preserved)
    text_lower = text.lower()
    for brand in KNOWN_BRANDS:
        if brand.lower() in text_lower:
            found.add(brand)

    # Normalize: remove "Active Campaign" if "ActiveCampaign" present
    if "ActiveCampaign" in found and "Active Campaign" in found:
        found.discard("Active Campaign")

    return sorted(found)


def score_entity_sentiment(entity: str, text: str) -> str:
    """
    Checks 100-char windows around each entity occurrence.
    Returns 'positive', 'negative', or 'neutral'.
    """
    pos_hits = 0
    neg_hits = 0
    entity_lower = entity.lower()
    text_lower = text.lower()

    start = 0
    while True:
        idx = text_lower.find(entity_lower, start)
        if idx == -1:
            break
        window_start = max(0, idx - 100)
        window_end = min(len(text), idx + len(entity) + 100)
        window = text_lower[window_start:window_end]
        words_in_window = set(re.findall(r'\b\w+\b', window))
        pos_hits += len(words_in_window & POSITIVE_WORDS)
        neg_hits += len(words_in_window & NEGATIVE_WORDS)
        start = idx + 1

    if pos_hits > neg_hits:
        return "positive"
    elif neg_hits > pos_hits:
        return "negative"
    return "neutral"


# ---------------------------------------------------------------------------
# Clustering (greedy union-find)
# ---------------------------------------------------------------------------

def _find(parent: dict, x: str) -> str:
    while parent[x] != x:
        parent[x] = parent[parent[x]]  # path compression
        x = parent[x]
    return x


def _union(parent: dict, rank: dict, a: str, b: str) -> None:
    ra, rb = _find(parent, a), _find(parent, b)
    if ra == rb:
        return
    if rank[ra] < rank[rb]:
        ra, rb = rb, ra
    parent[rb] = ra
    if rank[ra] == rank[rb]:
        rank[ra] += 1


def cluster_subqueries(queries: list[str], threshold: int = 80) -> dict[str, str]:
    """
    Groups similar subqueries using fuzz.ratio.
    Returns dict mapping each query → cluster label (the earliest-seen representative).
    """
    if not queries:
        return {}

    parent = {q: q for q in queries}
    rank = {q: 0 for q in queries}

    for i in range(len(queries)):
        for j in range(i + 1, len(queries)):
            if fuzz.ratio(queries[i], queries[j]) > threshold:
                _union(parent, rank, queries[i], queries[j])

    # Use earliest occurrence of each root as label
    root_to_label: dict[str, str] = {}
    result: dict[str, str] = {}
    for q in queries:
        root = _find(parent, q)
        if root not in root_to_label:
            root_to_label[root] = q
        result[q] = root_to_label[root]

    return result


# ---------------------------------------------------------------------------
# RRF Scoring
# ---------------------------------------------------------------------------

def compute_rrf_score(
    positions: list[int],
    n_runs: int,
    k: int = RRF_K,
) -> float:
    """
    RRF = sum(1 / (k + position)) across all runs the item appeared in.
    Normalized to 0-100 by dividing by max_possible = n_runs / (k + 1).
    """
    if not positions or n_runs == 0:
        return 0.0
    raw = sum(1.0 / (k + pos) for pos in positions)
    max_possible = n_runs / (k + 1)
    normalized = (raw / max_possible) * 100.0
    return round(min(normalized, 100.0), 2)


# ---------------------------------------------------------------------------
# Aggregation (Milestone 2)
# ---------------------------------------------------------------------------

def _extract_domain(uri: str) -> str:
    """Extracts bare domain from a URL."""
    try:
        # Remove scheme
        no_scheme = re.sub(r'^https?://', '', uri)
        # Remove www. prefix
        no_www = re.sub(r'^www\.', '', no_scheme)
        # Take up to first /
        domain = no_www.split('/')[0]
        # Remove port if present
        domain = domain.split(':')[0]
        return domain.lower()
    except Exception:
        return uri


def aggregate_runs(
    results: list[dict],
    similarity_threshold: int = 80,
) -> dict:
    """
    Aggregates across all RunResult dicts.
    Returns AggregatedResult with three DataFrames + raw_runs.
    """
    n = len(results)
    if n == 0:
        empty = pd.DataFrame()
        return {"subqueries": empty, "sources": empty, "entities": empty, "raw_runs": []}

    # ---- Subqueries --------------------------------------------------------
    # Track: query → list of (run_index, position_in_that_run)
    subquery_occurrences: dict[str, list[int]] = defaultdict(list)
    all_subqueries: list[str] = []

    for run in results:
        for pos, q in enumerate(run["web_search_queries"], start=1):
            subquery_occurrences[q].append(pos)
            if q not in all_subqueries:
                all_subqueries.append(q)

    # Cluster
    unique_queries = list(subquery_occurrences.keys())
    cluster_map = cluster_subqueries(unique_queries, threshold=similarity_threshold)

    # Group variants by cluster label
    cluster_occurrences: dict[str, list[int]] = defaultdict(list)
    cluster_variants: dict[str, list[str]] = defaultdict(list)
    for q, label in cluster_map.items():
        cluster_occurrences[label].extend(subquery_occurrences[q])
        cluster_variants[label].append(q)

    subquery_rows = []
    for label, positions in cluster_occurrences.items():
        # Count which runs this cluster appeared in
        runs_with_cluster = [
            run for run in results
            if any(cluster_map.get(q) == label for q in run["web_search_queries"])
        ]
        run_count = len(runs_with_cluster)
        probability = round(run_count / n * 100, 1)
        avg_position = round(sum(positions) / len(positions), 2)
        rrf_score = compute_rrf_score(positions, n)
        variants = list(dict.fromkeys(cluster_variants[label]))  # dedup, preserve order

        # Build top_sources: domains most frequently co-occurring in same runs as this cluster
        co_domain_counts: dict[str, int] = defaultdict(int)
        for run in runs_with_cluster:
            for src in run.get("sources", []):
                domain = _extract_domain(src["uri"])
                if domain:
                    co_domain_counts[domain] += 1
        top_3 = sorted(co_domain_counts, key=co_domain_counts.get, reverse=True)[:3]

        subquery_rows.append({
            "subquery": label,
            "probability": probability,
            "avg_position": avg_position,
            "rrf_score": rrf_score,
            "run_count": run_count,
            "sample_variants": " | ".join(variants[:3]),
            "top_sources": ", ".join(top_3),
        })

    subquery_df = pd.DataFrame(subquery_rows).sort_values("rrf_score", ascending=False).reset_index(drop=True)

    # ---- Sources -----------------------------------------------------------
    source_occurrences: dict[str, list[int]] = defaultdict(list)
    source_titles: dict[str, str] = {}

    for run in results:
        for pos, src in enumerate(run["sources"], start=1):
            uri = src["uri"]
            source_occurrences[uri].append(pos)
            if uri not in source_titles:
                source_titles[uri] = src["title"]

    source_rows = []
    for uri, positions in source_occurrences.items():
        run_count = sum(
            1 for run in results
            if any(s["uri"] == uri for s in run["sources"])
        )
        probability = round(run_count / n * 100, 1)
        avg_position = round(sum(positions) / len(positions), 2)
        rrf_score = compute_rrf_score(positions, n)
        source_rows.append({
            "uri": uri,
            "title": source_titles.get(uri, ""),
            "domain": _extract_domain(uri),
            "probability": probability,
            "avg_position": avg_position,
            "rrf_score": rrf_score,
            "run_count": run_count,
        })

    source_df = pd.DataFrame(source_rows).sort_values("rrf_score", ascending=False).reset_index(drop=True)

    # ---- Entities ----------------------------------------------------------
    entity_occurrences: dict[str, list[int]] = defaultdict(list)
    combined_text_per_run: list[str] = [r["response_text"] for r in results]

    for run_idx, run in enumerate(results):
        for pos, entity in enumerate(run["entities"], start=1):
            entity_occurrences[entity].append(pos)

    entity_rows = []
    all_combined_text = " ".join(combined_text_per_run)
    for entity, positions in entity_occurrences.items():
        run_count = sum(
            1 for run in results if entity in run["entities"]
        )
        probability = round(run_count / n * 100, 1)
        avg_position = round(sum(positions) / len(positions), 2)
        rrf_score = compute_rrf_score(positions, n)
        # sentiment = score_entity_sentiment(entity, all_combined_text)
        entity_rows.append({
            "entity": entity,
            "probability": probability,
            "avg_position": avg_position,
            "rrf_score": rrf_score,
            "run_count": run_count
            # "sentiment": sentiment,
        })

    entity_df = pd.DataFrame(entity_rows).sort_values("rrf_score", ascending=False).reset_index(drop=True) if entity_rows else pd.DataFrame()

    return {
        "subqueries": subquery_df,
        "sources": source_df,
        "entities": entity_df,
        "raw_runs": results,
    }


# ---------------------------------------------------------------------------
# Multi-run orchestration
# ---------------------------------------------------------------------------

def run_analysis(
    query: str,
    api_key: str,
    num_runs: int = 8,
    similarity_threshold: int = 80,
    inter_call_delay: float = INTER_CALL_DELAY,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    model_name: str = MODEL_NAME,
) -> dict:
    """
    Runs N Gemini API calls with grounding, aggregates results.
    progress_callback(current_run, total_runs) called after each run.
    Returns AggregatedResult dict. Includes 'model_fallback_triggered' key if
    the requested model was unavailable and fell back to FALLBACK_MODEL.
    """
    client = genai.Client(api_key=api_key)
    results: list[dict] = []
    active_model = model_name
    model_fallback_triggered = False

    for i in range(num_runs):
        try:
            run_result = call_with_backoff(query, client, model_name=active_model)
            run_result["run_index"] = i
            results.append(run_result)
            logger.info(
                f"Run {i + 1}/{num_runs} complete — "
                f"{len(run_result['sources'])} sources, "
                f"{len(run_result['web_search_queries'])} subqueries"
            )
        except ClientError as exc:
            exc_str = str(exc).lower()
            if (
                not model_fallback_triggered
                and active_model != FALLBACK_MODEL
                and ("not found" in exc_str or "404" in exc_str or "invalid" in exc_str)
            ):
                # Model unavailable — fall back and retry this run
                active_model = FALLBACK_MODEL
                model_fallback_triggered = True
                logger.warning(
                    f"Model '{model_name}' unavailable ({exc}). "
                    f"Falling back to '{FALLBACK_MODEL}' for remaining runs."
                )
                try:
                    run_result = call_with_backoff(query, client, model_name=active_model)
                    run_result["run_index"] = i
                    results.append(run_result)
                    logger.info(f"Run {i + 1}/{num_runs} complete after fallback.")
                except Exception as exc2:
                    logger.warning(f"Run {i + 1}/{num_runs} failed after fallback: {exc2}. Skipping.")
            else:
                logger.warning(f"Run {i + 1}/{num_runs} failed: {exc}. Skipping.")
        except Exception as exc:
            logger.warning(f"Run {i + 1}/{num_runs} failed: {exc}. Skipping.")

        if progress_callback:
            progress_callback(i + 1, num_runs)

        if i < num_runs - 1:
            time.sleep(inter_call_delay)

    if len(results) < 4:
        logger.warning(f"Only {len(results)}/{num_runs} runs completed successfully.")

    agg = aggregate_runs(results, similarity_threshold=similarity_threshold)
    agg["model_fallback_triggered"] = model_fallback_triggered
    agg["model_used"] = active_model
    return agg


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------

def estimate_cost(results: list[dict]) -> float:
    """
    Calculates estimated API cost from token counts across all runs.
    Uses gemini-2.5-flash-lite pricing: $0.075/1M input, $0.30/1M output.
    """
    total_in  = sum(r.get("input_tokens", 0) for r in results)
    total_out = sum(r.get("output_tokens", 0) for r in results)
    cost = (total_in / 1_000_000 * COST_INPUT_PER_1M) + \
           (total_out / 1_000_000 * COST_OUTPUT_PER_1M)
    return round(cost, 4)


# ---------------------------------------------------------------------------
# AC Share of Voice & Scorecard
# ---------------------------------------------------------------------------

def calculate_ac_sov(source_df: pd.DataFrame) -> dict:
    """
    LLM Share of Voice for ActiveCampaign.
    Definition: AC citation appearances / total citation appearances across all runs.

    Requires 'category' column (from ac_categorizer) and 'run_count' column.
    Returns dict with: ac_citations, total_citations, sov_pct.
    """
    if source_df.empty or "run_count" not in source_df.columns:
        return {"ac_citations": 0, "total_citations": 0, "sov_pct": 0.0}

    total = int(source_df["run_count"].sum())
    if total == 0:
        return {"ac_citations": 0, "total_citations": 0, "sov_pct": 0.0}

    if "category" in source_df.columns:
        ac_mask = source_df["category"].str.startswith("AC:", na=False)
        ac = int(source_df.loc[ac_mask, "run_count"].sum())
    else:
        # Fallback: check domain column if categorization hasn't run yet
        if "domain" in source_df.columns:
            ac_mask = source_df["domain"].str.contains("activecampaign", case=False, na=False)
            ac = int(source_df.loc[ac_mask, "run_count"].sum())
        else:
            ac = 0

    sov_pct = round(ac / total * 100, 1) if total > 0 else 0.0
    return {"ac_citations": ac, "total_citations": total, "sov_pct": sov_pct}


def build_scorecard(
    result: dict,
    query: str,
    estimated_cost: float,
    sov: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Builds a 2-column scorecard DataFrame: metric | value.
    Used for both terminal display and scorecard.csv export.
    """
    raw_runs = result.get("raw_runs", [])
    sub_df   = result.get("subqueries", pd.DataFrame())
    src_df   = result.get("sources", pd.DataFrame())
    ent_df   = result.get("entities", pd.DataFrame())

    if sov is None:
        sov = calculate_ac_sov(src_df)

    rows = [
        ("Query",               query),
        ("Queries Processed",   1),
        ("Total Responses",     len(raw_runs)),
        ("Clusters Found",      len(sub_df) if not sub_df.empty else 0),
        ("Sources Found Total", len(src_df) if not src_df.empty else 0),
        ("Entities Found",      len(ent_df) if not ent_df.empty else 0),
        ("Estimated Cost",      f"${estimated_cost:.4f}"),
        ("AC SoV (Citation %)", f"{sov['sov_pct']:.1f}%"),
        ("AC Citations",        sov["ac_citations"]),
        ("Total Citations",     sov["total_citations"]),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


# ---------------------------------------------------------------------------
# CSV export (Milestone 2)
# ---------------------------------------------------------------------------

def save_to_csv(
    result: dict,
    query: str = "",
    output_dir: str = ".",
    prefix: str = "",
) -> dict[str, str]:
    """
    Writes 4 CSVs: top_subqueries, top_sources (sorted by citation count), top_entities, scorecard.
    prefix: optional string prepended to filenames (e.g. "q1_") for multi-query runs.
    Returns {name: filepath}.
    """
    os.makedirs(output_dir, exist_ok=True)
    paths = {}

    raw_runs = result.get("raw_runs", [])
    cost = estimate_cost(raw_runs)

    # Sources: sort by run_count (citation count) descending — more intuitive for stakeholders
    src_df = result.get("sources", pd.DataFrame())
    if src_df is not None and not src_df.empty:
        src_df = src_df.sort_values("run_count", ascending=False).reset_index(drop=True)

    for key, df, filename in [
        ("subqueries", result.get("subqueries"), f"{prefix}top_subqueries.csv"),
        ("sources",    src_df,                    f"{prefix}top_sources.csv"),
        ("entities",   result.get("entities"),     f"{prefix}top_entities.csv"),
    ]:
        if df is not None and not df.empty:
            path = os.path.join(output_dir, filename)
            df.to_csv(path, index=False)
            paths[key] = path
            logger.info(f"Saved {path}")

    # Scorecard CSV
    sov = calculate_ac_sov(src_df)
    scorecard_df = build_scorecard(result, query, cost, sov)
    scorecard_path = os.path.join(output_dir, f"{prefix}scorecard.csv")
    scorecard_df.to_csv(scorecard_path, index=False)
    paths["scorecard"] = scorecard_path
    logger.info(f"Saved {scorecard_path}")

    return paths


# ---------------------------------------------------------------------------
# Milestone 2 — Multi-query demo
# Run: GEMINI_API_KEY=xxx python3 fanout_engine.py
# Runs 4 AC money queries × 8 runs each, writes CSVs to ./output/, prints scorecard
# ---------------------------------------------------------------------------

M2_QUERIES = [
    "best email marketing automation software",
    "ActiveCampaign vs Klaviyo",
    "email automation for ecommerce",
    "marketing automation for small business",
]

NUM_RUNS = 8  # runs per query


def _print_scorecard(scorecard_df: pd.DataFrame, query: str) -> None:
    width = 50
    print("\n" + "=" * width)
    print(f"SCORECARD — {query[:40]!r}")
    print("=" * width)
    for _, row in scorecard_df.iterrows():
        metric = str(row["metric"]).ljust(26)
        value  = str(row["value"])
        print(f"  {metric} {value}")
    print("=" * width)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: Set GEMINI_API_KEY environment variable first.")
        print("  export GEMINI_API_KEY=your_key_here")
        raise SystemExit(1)

    # Check for --single flag to run just one quick test (Milestone 1 mode)
    import sys
    single_mode = "--single" in sys.argv

    if single_mode:
        # ---- Milestone 1 mode (quick single-run test) ----
        query = M2_QUERIES[0]
        print(f"\nQuery: {query!r}")
        print("Calling Gemini with Google Search grounding (single run)...\n")
        client = genai.Client(api_key=api_key)
        result = call_with_backoff(query, client)

        print("=" * 60)
        print("WEB SEARCH QUERIES")
        print("=" * 60)
        for i, q in enumerate(result["web_search_queries"], 1):
            print(f"  {i}. {q}")
        if not result["web_search_queries"]:
            print("  (none returned)")

        print("\n" + "=" * 60)
        print("GROUNDING SOURCES (resolved URLs)")
        print("=" * 60)
        for i, src in enumerate(result["sources"], 1):
            print(f"  {i}. [{src['title']}]")
            print(f"     {src['uri']}")
        if not result["sources"]:
            print("  (no sources returned)")

        print("\n" + "=" * 60)
        print("ENTITIES DETECTED")
        print("=" * 60)
        print("  " + ", ".join(result["entities"]) if result["entities"] else "  (none)")

        print("\n" + "=" * 60)
        print("RESPONSE TEXT (first 600 chars)")
        print("=" * 60)
        print(result["response_text"][:600])
        print("\n--- Single run complete (Milestone 1 mode). ---\n")
        raise SystemExit(0)

    # ---- Milestone 2 mode (4 queries × 8 runs each) ----
    from ac_categorizer import categorize_sources

    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"MILESTONE 2 — Query Fan-Out Analysis")
    print(f"Queries: {len(M2_QUERIES)}  |  Runs per query: {NUM_RUNS}")
    print(f"Model: {MODEL_NAME}  |  Output: {output_dir}/")
    print(f"{'=' * 60}\n")

    all_scorecards = []

    for q_idx, query in enumerate(M2_QUERIES, 1):
        print(f"\n[{q_idx}/{len(M2_QUERIES)}] Running: {query!r}")
        print(f"  {NUM_RUNS} runs × ~2s delay = ~{NUM_RUNS * 2}s minimum...\n")

        result = run_analysis(
            query=query,
            api_key=api_key,
            num_runs=NUM_RUNS,
        )

        # Enrich sources with AC categories
        if not result["sources"].empty:
            src_dicts = result["sources"].to_dict("records")
            enriched  = categorize_sources(src_dicts)
            result["sources"] = pd.DataFrame(enriched)

        # Calculate cost and SoV
        raw_runs = result.get("raw_runs", [])
        cost = estimate_cost(raw_runs)
        sov  = calculate_ac_sov(result["sources"])

        # Print scorecard to terminal
        scorecard_df = build_scorecard(result, query, cost, sov)
        _print_scorecard(scorecard_df, query)

        # Print top sources
        src_df = result["sources"]
        if not src_df.empty:
            top_src = src_df.sort_values("run_count", ascending=False).head(5)
            print(f"\n  Top 5 cited sources:")
            for _, row in top_src.iterrows():
                domain = row.get("domain", "")
                cat    = row.get("category", "")
                count  = row.get("run_count", 0)
                print(f"    [{count}x] {domain}  ({cat})")

        # Write CSVs
        prefix = f"q{q_idx}_"
        paths  = save_to_csv(result, query=query, output_dir=output_dir, prefix=prefix)
        print(f"\n  CSVs written:")
        for name, path in paths.items():
            print(f"    {path}")

        all_scorecards.append(scorecard_df)

    # ---- Combined summary across all queries ----
    print(f"\n{'=' * 60}")
    print("COMBINED SUMMARY — All Queries")
    print(f"{'=' * 60}")

    total_runs   = sum(int(sc.loc[sc["metric"] == "Total Responses",   "value"].iloc[0]) for sc in all_scorecards)
    total_src    = sum(int(sc.loc[sc["metric"] == "Sources Found Total","value"].iloc[0]) for sc in all_scorecards)
    total_clust  = sum(int(sc.loc[sc["metric"] == "Clusters Found",     "value"].iloc[0]) for sc in all_scorecards)
    total_ent    = sum(int(sc.loc[sc["metric"] == "Entities Found",     "value"].iloc[0]) for sc in all_scorecards)
    total_ac_cit = sum(int(sc.loc[sc["metric"] == "AC Citations",       "value"].iloc[0]) for sc in all_scorecards)
    total_cit    = sum(int(sc.loc[sc["metric"] == "Total Citations",    "value"].iloc[0]) for sc in all_scorecards)
    total_cost   = sum(float(sc.loc[sc["metric"] == "Estimated Cost",   "value"].iloc[0].replace("$","")) for sc in all_scorecards)

    overall_sov  = round(total_ac_cit / total_cit * 100, 1) if total_cit > 0 else 0.0

    print(f"  Queries Analyzed     : {len(M2_QUERIES)}")
    print(f"  Total API Runs       : {total_runs}")
    print(f"  Total Clusters       : {total_clust}")
    print(f"  Total Unique Sources : {total_src}")
    print(f"  Total Entities       : {total_ent}")
    print(f"  Total AC Citations   : {total_ac_cit} of {total_cit}")
    print(f"  AC LLM Share of Voice: {overall_sov:.1f}%")
    print(f"  Total Estimated Cost : ${total_cost:.4f}")
    print(f"\n  CSVs saved to: {output_dir}/")
    print(f"{'=' * 60}")
    print("\n--- Milestone 2 complete. ---\n")
