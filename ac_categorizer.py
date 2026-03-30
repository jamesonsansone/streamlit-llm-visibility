"""
ac_categorizer.py — ActiveCampaign source URL classifier

Classifies any source URL into a category (AC property, competitor, 3rd party, etc.)
using regex pattern matching with first-match-wins priority.
"""

import re
from urllib.parse import urlparse
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Category patterns — applied in order, first match wins
# ---------------------------------------------------------------------------

# Each entry: (compiled_regex, category_label)
# More specific patterns must come before broader ones.

_CATEGORY_RULES: list[tuple[re.Pattern, str]] = [
    # --- ActiveCampaign subdomains (most specific first) ---
    (re.compile(r'help\.activecampaign\.com', re.I),         "AC: help. Subdomain"),
    (re.compile(r'community\.activecampaign\.com', re.I),    "AC: community. Subdomain"),

    # --- AC International subfolders (flag before other AC paths) ---
    (re.compile(r'activecampaign\.com/(?:br|de|fr|es|it|pt|nl|ja|ko|zh|ru|pl|tr|sv|da|no|fi)/', re.I),
                                                              "AC: International"),

    # --- AC main domain subfolders ---
    (re.compile(r'activecampaign\.com/blog/', re.I),          "AC: Blog"),
    (re.compile(r'activecampaign\.com/(?:glossary|marketing-glossary)/', re.I),
                                                              "AC: Glossary"),
    (re.compile(r'activecampaign\.com/recipes/', re.I),       "AC: Recipes"),
    (re.compile(r'activecampaign\.com/compare/', re.I),       "AC: Compare"),
    (re.compile(r'activecampaign\.com/solutions/', re.I),     "AC: Solutions"),
    (re.compile(r'activecampaign\.com/(?:learn|education)/',  re.I),
                                                              "AC: Learn"),
    (re.compile(r'activecampaign\.com/(?:customers|case-studies)/', re.I),
                                                              "AC: Case Studies"),

    # --- AC catch-all ---
    (re.compile(r'activecampaign\.com', re.I),                "AC: All Other Pages"),

    # --- Competitors ---
    (re.compile(
        r'(?:mailchimp|klaviyo|hubspot|brevo|sendinblue|constantcontact|getresponse|omnisend|'
        r'drip|convertkit|campaignmonitor|mailerlite|aweber|salesforce|pardot|marketo|'
        r'zoho|keap|infusionsoft|gohighlevel|clevertap|emarsys|maropost|webengage|'
        r'bloomreach|encharge|sender|sendpulse|mailbluster|act|agilecrm|thryv|streak)\.com',
        re.I
    ),                                                         "Competitor"),

    # --- Third-party review sites ---
    (re.compile(r'(?:g2|capterra|trustradius|getapp|softwareadvice)\.com', re.I),
                                                              "3rd Party: Review Site"),

    # --- Third-party publications ---
    (re.compile(
        r'(?:techradar|pcmag|forbes|cnet|techcrunch|businessinsider|tomsguide|zdnet|wired|'
        r'entrepreneur|inc|businessnews|uschamber)\.com',
        re.I
    ),                                                         "3rd Party: Publication"),

    # --- Comparison/curated tool sites ---
    (re.compile(
        r'(?:zapier|emailtooltester|tooltester|emailvendorselection|emailmarketing|ventureharbour)\.(?:com|net|org)',
        re.I
    ),                                                         "3rd Party: Comparison"),

    # --- Marketing blogs and agencies ---
    (re.compile(
        r'(?:inboxarmy|emailwarmup|saffronedge|flowium|firstpier|panoramata|'
        r'thecmo|socialhospitality|grazitti|ritzmarketingsolutions|redlineminds|munro\.agency|'
        r'flatlineagency|awwtomation|2pointagency|jdrgroup|insiderone)\.(?:com|co\.uk|agency)',
        re.I
    ),                                                         "3rd Party: Marketing Blog"),

    # --- Email/marketing tools and integrations ---
    (re.compile(
        r'(?:leadsbridge|ifttt|gmass|mailtrap|saleshandy|trellus|alumio|numerous|'
        r'postalytics|seedprod|elementor|ecommerceboost|ecommerceintelligence|'
        r'jellyreach|lumi|retainful|paystone|emailwarmup|clerk|decidr|blaze|'
        r'linkmobility|yournotify|sendpulse|mailbluster|sender|arinet)\.(?:com|ai|co|io)',
        re.I
    ),                                                         "3rd Party: Tool/Integration"),

    # --- Ecommerce platforms ---
    (re.compile(
        r'(?:bigcommerce|squareup|shopify|woocommerce|magento|checkoutlinks|checkoutchamp|funnelish)\.com',
        re.I
    ),                                                         "Ecommerce Platform"),

    # --- Social / community ---
    (re.compile(r'reddit\.com', re.I),                        "3rd Party: Reddit"),
    (re.compile(r'(?:youtube\.com|youtu\.be)', re.I),         "3rd Party: YouTube"),
    (re.compile(r'wikipedia\.org', re.I),                     "3rd Party: Wikipedia"),
    (re.compile(r'linkedin\.com', re.I),                      "3rd Party: LinkedIn"),
]

# ---------------------------------------------------------------------------
# International URL detection (applies to ANY domain)
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
    """Returns the detected language code if URL appears non-English, else None."""
    m = _INTL_SUBFOLDER.search(url)
    if m:
        code = m.group(0).strip("/").lower()
        return code

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
        tld = m3.group(0).strip("/").lstrip(".")
        return tld

    return None


def is_international(url: str) -> bool:
    """Returns True if URL appears to serve non-English content."""
    return _detect_lang_code(url) is not None


def get_intl_flag(url: str) -> str:
    """Returns a display flag string like '⚠️ de (German)' or empty string."""
    code = _detect_lang_code(url)
    if code is None:
        return ""
    lang_name = _LANG_CODE_MAP.get(code.lower(), code.upper())
    return f"\u26a0\ufe0f {code} ({lang_name})"


def categorize_url(url: str) -> str:
    """
    Returns the category label for a URL.
    Applies rules in priority order, returns first match.
    Returns 'Other' on any parse error or no match.
    """
    if not url:
        return "Other"
    try:
        for pattern, label in _CATEGORY_RULES:
            if pattern.search(url):
                return label
        return "Other"
    except Exception:
        return "Other"


def categorize_sources(sources: list[dict]) -> list[dict]:
    """
    Augments each source dict with 'category' and 'intl_flag' keys.
    Input: list of {"uri": str, "title": str, ...}
    Returns augmented list (does not mutate originals).
    """
    result = []
    for src in sources:
        augmented = dict(src)
        uri = src.get("uri", "")
        augmented["category"] = categorize_url(uri)
        augmented["intl_flag"] = get_intl_flag(uri)
        result.append(augmented)
    return result


# ---------------------------------------------------------------------------
# Category summary helpers
# ---------------------------------------------------------------------------

def get_top_level_category(category: str) -> str:
    """
    Maps specific category to a top-level bucket for charts.
    e.g. 'AC: Blog' → 'AC', 'Competitor' → 'Competitor', etc.
    """
    if category.startswith("AC:"):
        return "ActiveCampaign"
    if category == "Competitor":
        return "Competitor"
    if "Review Site" in category:
        return "3rd Party: Review"
    if "Publication" in category:
        return "3rd Party: Publication"
    if "Reddit" in category:
        return "3rd Party: Reddit"
    if "YouTube" in category:
        return "3rd Party: YouTube"
    if "Comparison" in category:
        return "3rd Party: Comparison"
    if "Marketing Blog" in category:
        return "3rd Party: Blog"
    if "Tool/Integration" in category:
        return "3rd Party: Tool"
    if "Wikipedia" in category:
        return "3rd Party: Wikipedia"
    if "LinkedIn" in category:
        return "3rd Party: LinkedIn"
    if category == "Ecommerce Platform":
        return "Ecommerce"
    return "Other"


def build_category_summary(source_df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a sources DataFrame with a 'category' column,
    returns a summary table: category | citation_count | pct_of_total
    """
    if source_df.empty or "category" not in source_df.columns:
        return pd.DataFrame(columns=["category", "citation_count", "pct_of_total"])

    counts = source_df["category"].value_counts().reset_index()
    counts.columns = ["category", "citation_count"]
    total = counts["citation_count"].sum()
    counts["pct_of_total"] = (counts["citation_count"] / total * 100).round(1)
    return counts


def build_top_level_summary(source_df: pd.DataFrame) -> pd.DataFrame:
    """
    Groups categories into top-level buckets (AC, Competitor, 3rd Party X, Other).
    Useful for the horizontal bar chart.
    """
    if source_df.empty or "category" not in source_df.columns:
        return pd.DataFrame(columns=["top_level", "citation_count"])

    df = source_df.copy()
    df["top_level"] = df["category"].apply(get_top_level_category)
    counts = df["top_level"].value_counts().reset_index()
    counts.columns = ["top_level", "citation_count"]
    return counts


def build_ac_property_breakdown(source_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters to AC-only sources and summarizes by AC property.
    Returns: ac_property | citation_count | pct_of_ac_citations
    """
    if source_df.empty or "category" not in source_df.columns:
        return pd.DataFrame(columns=["ac_property", "citation_count", "pct_of_ac"])

    ac_df = source_df[source_df["category"].str.startswith("AC:")].copy()
    if ac_df.empty:
        return pd.DataFrame(columns=["ac_property", "citation_count", "pct_of_ac"])

    counts = ac_df["category"].value_counts().reset_index()
    counts.columns = ["ac_property", "citation_count"]
    total_ac = counts["citation_count"].sum()
    counts["pct_of_ac"] = (counts["citation_count"] / total_ac * 100).round(1)
    return counts


def build_query_heatmap(
    query_results: list[dict],
) -> Optional[pd.DataFrame]:
    """
    Builds a query bucket × AC property heatmap DataFrame.
    query_results: list of {"query": str, "sources_df": pd.DataFrame}
    Returns pivot table with queries as rows, AC property categories as columns.
    Returns None if no AC sources found.
    """
    if not query_results:
        return None

    rows = []
    for item in query_results:
        query = item["query"]
        src_df = item.get("sources_df", pd.DataFrame())
        if src_df.empty or "category" not in src_df.columns:
            continue
        ac_df = src_df[src_df["category"].str.startswith("AC:")]
        if ac_df.empty:
            rows.append({"query": query, "category": "No AC Presence", "count": 0})
            continue
        for cat, count in ac_df["category"].value_counts().items():
            rows.append({"query": query, "category": cat, "count": count})

    if not rows:
        return None

    df = pd.DataFrame(rows)
    pivot = df.pivot_table(index="query", columns="category", values="count", fill_value=0)
    return pivot


def generate_heatmap_insights(pivot_df: pd.DataFrame) -> list[str]:
    """
    Auto-generates plain-English insights from the query × AC property heatmap.
    Returns list of insight strings.
    """
    if pivot_df is None or pivot_df.empty:
        return []

    insights = []
    for query in pivot_df.index:
        row = pivot_df.loc[query]
        total_ac = row.sum()

        if total_ac == 0:
            insights.append(
                f"\U0001f6a8 **'{query}'** — No ActiveCampaign pages cited. "
                "This is a content gap: consider creating or promoting AC content targeting this query."
            )
            continue

        dominant = row.idxmax()
        dominant_count = row.max()
        zero_cats = [col for col in pivot_df.columns if row[col] == 0 and col != dominant]

        insight = f"\U0001f4ca **'{query}'** — {dominant} is the dominant cited property ({dominant_count} citations)."
        if zero_cats:
            top_zero = zero_cats[:2]
            insight += f" Notable gaps: {', '.join(top_zero)} had 0 citations."
        insights.append(insight)

    return insights


# ---------------------------------------------------------------------------
# Cross-query aggregation helpers
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
        tmp = src_df[["domain", "category", "run_count"]].copy()
        tmp["query"] = item["query"]
        parts.append(tmp)
    if not parts:
        return pd.DataFrame(
            columns=["domain", "category", "total_citations", "queries_count", "queries_list"]
        )
    combined = pd.concat(parts, ignore_index=True)
    agg = (
        combined.groupby(["domain", "category"])
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
    Creates a domain × query pivot table.
    Input: [{"query": str, "sources_df": pd.DataFrame}, ...]
    Returns: domain as index, query names as columns,
             values = run_count per domain per query (0 if absent).
    Top N domains by total citations.
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
    Returns all source URLs across all queries with total citation count and category.
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
# Standalone unit test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_cases = [
        ("https://www.activecampaign.com/blog/email-segmentation-best-practices",        "AC: Blog"),
        ("https://help.activecampaign.com/hc/en-us/articles/123",                        "AC: help. Subdomain"),
        ("https://community.activecampaign.com/t/topic/456",                             "AC: community. Subdomain"),
        ("https://www.activecampaign.com/de/blog/",                                      "AC: International"),
        ("https://www.activecampaign.com/glossary/email-marketing",                      "AC: Glossary"),
        ("https://www.activecampaign.com/recipes/welcome-email",                         "AC: Recipes"),
        ("https://www.activecampaign.com/compare/mailchimp-vs-activecampaign",           "AC: Compare"),
        ("https://www.activecampaign.com/solutions/ecommerce",                           "AC: Solutions"),
        ("https://www.activecampaign.com/customers/case-study-foo",                      "AC: Case Studies"),
        ("https://www.activecampaign.com/pricing",                                       "AC: All Other Pages"),
        ("https://mailchimp.com/resources/email-marketing-benchmarks/",                  "Competitor"),
        ("https://www.klaviyo.com/blog/email-marketing",                                 "Competitor"),
        ("https://www.g2.com/products/activecampaign/reviews",                           "3rd Party: Review Site"),
        ("https://www.techradar.com/best/best-email-marketing-tools",                    "3rd Party: Publication"),
        ("https://zapier.com/blog/best-email-marketing-apps/",                           "3rd Party: Comparison"),
        ("https://www.reddit.com/r/emailmarketing/comments/abc",                         "3rd Party: Reddit"),
        ("https://www.youtube.com/watch?v=xyz",                                          "3rd Party: YouTube"),
        ("https://en.wikipedia.org/wiki/Email_marketing",                                "3rd Party: Wikipedia"),
        ("https://www.somerandomblog.com/post/email-marketing",                          "Other"),
    ]

    print("\nac_categorizer unit tests\n" + "-" * 50)
    passed = 0
    for url, expected in test_cases:
        result = categorize_url(url)
        status = "PASS" if result == expected else f"FAIL (got: {result!r})"
        if result == expected:
            passed += 1
        print(f"  [{status}] {url[:65]}")

    print(f"\n{passed}/{len(test_cases)} tests passed.")

    # Intl flag test
    print("\nInternational flag tests:")
    intl_tests = [
        "https://www.activecampaign.com/de/blog/",
        "https://de.mailchimp.com/pricing/",
        "https://www.hubspot.com/products/crm",  # should be empty
    ]
    for u in intl_tests:
        flag = get_intl_flag(u)
        print(f"  {flag or '(none)'} — {u}")
