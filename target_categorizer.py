"""
target_categorizer.py — Source URL classifier (Target vs Other).

Categorizes each source URL based on a user-supplied target_domain:
- "Target" if the URL's registered domain matches (or is a subdomain of) target_domain.
- "Other" otherwise.

Also exposes an international-URL flag helper and a few cross-query aggregation
helpers used by the report dashboard.
"""

import re
from urllib.parse import urlparse
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Target vs Other
# ---------------------------------------------------------------------------

def _normalize_target(target_domain: str) -> str:
    """
    Normalizes a user-supplied target like 'https://www.example.com/' or 'Example.com'
    into a bare lowercase host ('example.com').
    """
    if not target_domain:
        return ""
    t = target_domain.strip().lower()
    t = re.sub(r"^https?://", "", t)
    t = re.sub(r"^www\.", "", t)
    t = t.split("/")[0].split(":")[0]
    return t


def _url_host(url: str) -> str:
    if not url:
        return ""
    try:
        no_scheme = re.sub(r"^https?://", "", url.lower())
        no_www = re.sub(r"^www\.", "", no_scheme)
        return no_www.split("/")[0].split(":")[0]
    except Exception:
        return ""


def categorize_url(url: str, target_domain: str) -> str:
    """
    Returns "Target" if the URL's host matches or is a subdomain of target_domain,
    else "Other". Returns "Other" when target_domain is blank or URL is unparseable.
    """
    target = _normalize_target(target_domain)
    if not target:
        return "Other"
    host = _url_host(url)
    if not host:
        return "Other"
    return "Target" if (host == target or host.endswith("." + target)) else "Other"


def categorize_sources(sources: list[dict], target_domain: str) -> list[dict]:
    """
    Augments each source dict with 'category' and 'intl_flag' keys.
    Does not mutate the input.
    """
    result = []
    for src in sources:
        augmented = dict(src)
        uri = src.get("uri", "")
        augmented["category"] = categorize_url(uri, target_domain)
        augmented["intl_flag"] = get_intl_flag(uri)
        result.append(augmented)
    return result


# ---------------------------------------------------------------------------
# International URL detection (domain-agnostic)
# ---------------------------------------------------------------------------

_INTL_SUBFOLDER = re.compile(
    r'/(?:br|de|fr|es|it|pt|nl|ja|ko|zh|ru|pl|tr|sv|da|no|fi|ar|th|vi|id|ms|hi)(?:/|$)',
    re.I
)
_INTL_SUBDOMAIN = re.compile(
    r'^(?:br|de|fr|es|it|pt|nl|ja|ko|zh|ru|pl|tr|sv|da|no|fi)\.',
    re.I
)
_INTL_CCTLD = re.compile(
    r'\.(?:de|fr|es|it|br|com\.br|co\.jp|co\.uk|com\.au|nl|pl|ru|se|dk|no|fi|pt|mx|ar|co\.in)(?:/|$)',
    re.I
)

_LANG_CODE_MAP = {
    "de": "German", "fr": "French", "es": "Spanish", "it": "Italian",
    "pt": "Portuguese", "br": "Portuguese (BR)", "nl": "Dutch",
    "ja": "Japanese", "ko": "Korean", "zh": "Chinese", "ru": "Russian",
    "pl": "Polish", "tr": "Turkish", "sv": "Swedish", "da": "Danish",
    "no": "Norwegian", "fi": "Finnish", "ar": "Arabic",
}


def _detect_lang_code(url: str) -> Optional[str]:
    m = _INTL_SUBFOLDER.search(url)
    if m:
        return m.group(0).strip("/").lower()
    try:
        parsed = urlparse(url if "://" in url else "https://" + url)
        host = parsed.hostname or ""
        m2 = _INTL_SUBDOMAIN.match(host)
        if m2:
            return m2.group(0).rstrip(".").lower()
    except Exception:
        pass
    m3 = _INTL_CCTLD.search(url)
    if m3:
        return m3.group(0).strip("/").lstrip(".")
    return None


def is_international(url: str) -> bool:
    return _detect_lang_code(url) is not None


def get_intl_flag(url: str) -> str:
    code = _detect_lang_code(url)
    if code is None:
        return ""
    lang_name = _LANG_CODE_MAP.get(code.lower(), code.upper())
    return f"⚠️ {code} ({lang_name})"


# ---------------------------------------------------------------------------
# Cross-query aggregation helpers (domain-agnostic)
# ---------------------------------------------------------------------------

def build_cross_query_domain_summary(query_data_list: list[dict]) -> pd.DataFrame:
    """
    Aggregates domains across all queries.
    Input: [{"query": str, "sources_df": pd.DataFrame}, ...]
    Returns: domain | category | total_citations | queries_count | queries_list
    Sorted by total_citations desc.
    """
    if not query_data_list:
        return pd.DataFrame(
            columns=["domain", "category", "total_citations", "queries_count", "queries_list"]
        )
    parts = []
    for item in query_data_list:
        src_df = item.get("sources_df", pd.DataFrame())
        if src_df.empty or "domain" not in src_df.columns or "run_count" not in src_df.columns:
            continue
        cols = ["domain", "run_count"]
        if "category" in src_df.columns:
            cols.insert(1, "category")
        tmp = src_df[cols].copy()
        tmp["query"] = item["query"]
        parts.append(tmp)
    if not parts:
        return pd.DataFrame(
            columns=["domain", "category", "total_citations", "queries_count", "queries_list"]
        )
    combined = pd.concat(parts, ignore_index=True)
    group_cols = ["domain", "category"] if "category" in combined.columns else ["domain"]
    agg = (
        combined.groupby(group_cols)
        .agg(
            total_citations=("run_count", "sum"),
            queries_count=("query", "nunique"),
            queries_list=("query", lambda x: ", ".join(sorted(x.unique()))),
        )
        .reset_index()
        .sort_values("total_citations", ascending=False)
        .reset_index(drop=True)
    )
    return agg


def build_domain_query_pivot(query_data_list: list[dict], top_n: int = 25) -> pd.DataFrame:
    """
    Domain × query pivot. Top N domains by total citations as rows; query names as
    columns; cells = run_count (0 if absent).
    """
    if not query_data_list:
        return pd.DataFrame()
    parts = []
    for item in query_data_list:
        src_df = item.get("sources_df", pd.DataFrame())
        if src_df.empty or "domain" not in src_df.columns or "run_count" not in src_df.columns:
            continue
        tmp = src_df[["domain", "run_count"]].copy()
        tmp["query"] = item["query"]
        parts.append(tmp)
    if not parts:
        return pd.DataFrame()
    combined = pd.concat(parts, ignore_index=True)
    top_domains = (
        combined.groupby("domain")["run_count"].sum().nlargest(top_n).index.tolist()
    )
    filtered = combined[combined["domain"].isin(top_domains)]
    pivot = filtered.pivot_table(
        index="domain", columns="query", values="run_count", aggfunc="sum", fill_value=0
    )
    pivot.columns.name = None
    pivot["_total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("_total", ascending=False).drop(columns=["_total"])
    return pivot


def build_all_url_citations(query_data_list: list[dict]) -> pd.DataFrame:
    """
    All source URLs across all queries with total citation count and category.
    Columns: uri | title | domain | category | total_citations | queries_count | queries_list
    Sorted by total_citations desc.
    """
    if not query_data_list:
        return pd.DataFrame(
            columns=["uri", "title", "domain", "category", "total_citations", "queries_count", "queries_list"]
        )
    parts = []
    for item in query_data_list:
        src_df = item.get("sources_df", pd.DataFrame())
        if src_df.empty or "uri" not in src_df.columns or "run_count" not in src_df.columns:
            continue
        cols = [c for c in ["uri", "title", "domain", "category", "run_count"] if c in src_df.columns]
        tmp = src_df[cols].copy()
        tmp["query"] = item["query"]
        parts.append(tmp)
    if not parts:
        return pd.DataFrame(
            columns=["uri", "title", "domain", "category", "total_citations", "queries_count", "queries_list"]
        )
    combined = pd.concat(parts, ignore_index=True)
    group_cols = [c for c in ["uri", "title", "domain", "category"] if c in combined.columns]
    agg = (
        combined.groupby(group_cols)
        .agg(
            total_citations=("run_count", "sum"),
            queries_count=("query", "nunique"),
            queries_list=("query", lambda x: ", ".join(sorted(x.unique()))),
        )
        .reset_index()
        .sort_values("total_citations", ascending=False)
        .reset_index(drop=True)
    )
    return agg


# ---------------------------------------------------------------------------
# Standalone sanity checks
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cases = [
        ("https://www.example.com/blog/post",        "example.com", "Target"),
        ("https://help.example.com/article/42",      "example.com", "Target"),
        ("https://example.com",                      "example.com", "Target"),
        ("https://www.competitor.com/vs/example",    "example.com", "Other"),
        ("https://EXAMPLE.com/upper",                "Example.com", "Target"),
        ("https://www.example.com/page",             "",            "Other"),
    ]

    print("target_categorizer sanity checks\n" + "-" * 50)
    passed = 0
    for url, target, expected in cases:
        got = categorize_url(url, target)
        ok = "PASS" if got == expected else f"FAIL (got {got!r})"
        if got == expected:
            passed += 1
        print(f"  [{ok}] target={target!r:<18} url={url}")
    print(f"\n{passed}/{len(cases)} passed.")

    print("\nInternational flag tests:")
    for u in [
        "https://www.example.com/de/blog/",
        "https://de.example.com/pricing/",
        "https://www.example.com/products/",
    ]:
        print(f"  {get_intl_flag(u) or '(none)'} — {u}")
