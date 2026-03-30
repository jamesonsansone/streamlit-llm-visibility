"""
report.py — Stakeholder report for Query Fan-Out Analysis results.

Loads CSVs produced by fanout_engine.py and presents them as a navigable
report. Also supports running new queries live from the sidebar (reads
GEMINI_API_KEY from environment — no input needed).

Run: streamlit run report.py
"""

import os
import glob
import time
from collections import Counter

import pandas as pd
import streamlit as st

try:
    from fanout_engine import KNOWN_BRANDS, run_analysis, estimate_cost, calculate_ac_sov, save_to_csv, AVAILABLE_MODELS, FALLBACK_MODEL
except ImportError:
    KNOWN_BRANDS = [
        "ActiveCampaign", "Mailchimp", "HubSpot", "Klaviyo",
        "Brevo", "Constant Contact", "GetResponse", "Omnisend",
        "Drip", "ConvertKit", "Salesforce", "Marketo",
        "MailerLite", "AWeber", "Campaign Monitor", "Sendinblue",
    ]
    run_analysis = save_to_csv = estimate_cost = calculate_ac_sov = None
    AVAILABLE_MODELS = {"gemini-2.5-flash-lite": "Gemini 2.5 Flash Lite (Standard)"}
    FALLBACK_MODEL = "gemini-2.5-flash-lite"

try:
    from ac_categorizer import (
        categorize_sources,
        build_top_level_summary,
        build_ac_property_breakdown,
        build_query_heatmap,
        generate_heatmap_insights,
        build_cross_query_domain_summary,
        build_domain_query_pivot,
        build_all_url_citations,
    )
except ImportError:
    (
        categorize_sources,
        build_top_level_summary,
        build_ac_property_breakdown,
        build_query_heatmap,
        generate_heatmap_insights,
        build_cross_query_domain_summary,
        build_domain_query_pivot,
        build_all_url_citations,
    ) = (None,) * 8

OUTPUT_DIR = "./output"

st.set_page_config(
    page_title="Fan-Out Report: ActiveCampaign",
    page_icon="📊",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Color constants
# ---------------------------------------------------------------------------

_GREEN  = "background-color: #d4edda"
_YELLOW = "background-color: #fff3cd"
_RED    = "background-color: #f8d7da"
_BLUE   = "background-color: #cce5ff"
_GRAY   = "background-color: #e9ecef"

# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------

def _init_state():
    if "selected_prefix" not in st.session_state:
        st.session_state.selected_prefix = None   # None = overview
    if "recent_runs" not in st.session_state:
        st.session_state.recent_runs = []          # list of {prefix, query_name}
    if "run_counter" not in st.session_state:
        st.session_state.run_counter = 0

# ---------------------------------------------------------------------------
# Data loading (no cache — files can be added mid-session)
# ---------------------------------------------------------------------------

def load_query_data(prefix: str, output_dir: str) -> dict:
    def safe_read(filename: str) -> pd.DataFrame:
        path = os.path.join(output_dir, filename)
        if os.path.exists(path):
            try:
                return pd.read_csv(path)
            except Exception:
                return pd.DataFrame()
        return pd.DataFrame()

    return {
        "scorecard":  safe_read(f"{prefix}_scorecard.csv"),
        "sources":    safe_read(f"{prefix}_top_sources.csv"),
        "subqueries": safe_read(f"{prefix}_top_subqueries.csv"),
        "entities":   safe_read(f"{prefix}_top_entities.csv"),
    }


def discover_queries(output_dir: str) -> dict:
    """Scans output dir for scorecard CSVs. Returns {prefix: query_name}."""
    queries = {}
    if not os.path.exists(output_dir):
        return queries
    for path in sorted(glob.glob(os.path.join(output_dir, "*_scorecard.csv"))):
        prefix = os.path.basename(path).replace("_scorecard.csv", "")
        try:
            df = pd.read_csv(path)
            row = df[df["metric"] == "Query"]["value"]
            query_name = str(row.iloc[0]) if not row.empty else prefix
        except Exception:
            query_name = prefix
        queries[prefix] = query_name
    return queries


def scorecard_val(df: pd.DataFrame, metric: str, default: str = "—") -> str:
    if df.empty:
        return default
    row = df[df["metric"] == metric]["value"]
    return str(row.iloc[0]) if not row.empty else default


def get_ac_mentioned_pct(entities_df: pd.DataFrame) -> float:
    if entities_df.empty or "entity" not in entities_df.columns:
        return 0.0
    ac = entities_df[entities_df["entity"].str.lower() == "activecampaign"]
    if ac.empty:
        return 0.0
    return float(ac.iloc[0].get("probability", 0))


def get_top_non_ac_domain(sources_df: pd.DataFrame) -> str:
    if sources_df.empty or "domain" not in sources_df.columns:
        return "—"
    mask = ~sources_df["domain"].str.contains(
        "activecampaign|vertexaisearch", case=False, na=False
    )
    filtered = sources_df[mask]
    if filtered.empty:
        return "—"
    return filtered.sort_values("run_count", ascending=False).iloc[0]["domain"]

# ---------------------------------------------------------------------------
# Styling helpers
# ---------------------------------------------------------------------------

def style_sources(df: pd.DataFrame):
    def row_color(row):
        cat    = str(row.get("category", ""))
        uri    = str(row.get("URL", row.get("uri", "")))
        domain = str(row.get("domain", ""))
        if cat.startswith("AC:") or "activecampaign.com" in uri or "activecampaign.com" in domain:
            return [_BLUE] * len(row)
        if cat == "Competitor":
            return [_RED] * len(row)
        prob = float(row.get("probability", 0))
        if prob > 75:
            return [_GREEN] * len(row)
        elif prob >= 25:
            return [_YELLOW] * len(row)
        return [_GRAY] * len(row)
    return df.style.apply(row_color, axis=1)


def style_brands(df: pd.DataFrame):
    def row_color(row):
        entity = str(row.get("entity", "")).lower()
        if "activecampaign" in entity:
            return [_BLUE] * len(row)
        for comp in ["mailchimp", "hubspot", "klaviyo", "brevo", "constant contact", "getresponse"]:
            if comp in entity:
                return [_RED] * len(row)
        return [""] * len(row)
    return df.style.apply(row_color, axis=1)


def style_summary_row(row):
    try:
        sov = float(str(row.get("AC SoV", "0")).replace("%", ""))
    except ValueError:
        sov = 0.0
    color = _RED if sov == 0.0 else (_YELLOW if sov < 15 else _GREEN)
    return [color if col == "AC SoV" else "" for col in row.index]


def style_heatmap(df: pd.DataFrame):
    def cell_style(val):
        if val == 0:
            return "background-color: #f8d7da; color: #721c24"
        elif val <= 3:
            return "background-color: #fff3cd; color: #856404"
        else:
            return "background-color: #d4edda; color: #155724"
    return df.style.applymap(cell_style)


def style_domain_row(row, ac_domain_set: set):
    styles = []
    for col in row.index:
        if col == "domain":
            styles.append(_BLUE if row["domain"] in ac_domain_set else "")
        else:
            try:
                v = int(row[col])
            except (TypeError, ValueError):
                styles.append("")
                continue
            if v == 0:
                styles.append(_GRAY)
            elif v <= 3:
                styles.append(_YELLOW)
            else:
                styles.append(_GREEN)
    return styles

# ---------------------------------------------------------------------------
# Soft insight line (replaces gap callout on per-query pages)
# ---------------------------------------------------------------------------

def _render_soft_insight(scorecard_df, entities_df, sources_df) -> str:
    sov_str = scorecard_val(scorecard_df, "AC SoV (Citation %)", "0.0%")
    try:
        sov_pct = float(sov_str.replace("%", ""))
    except ValueError:
        sov_pct = 0.0
    ac_cit    = scorecard_val(scorecard_df, "AC Citations", "0")
    total_cit = scorecard_val(scorecard_df, "Total Citations", "0")
    top_domain = get_top_non_ac_domain(sources_df)

    if sov_pct == 0.0:
        return (
            f"No ActiveCampaign pages appeared in the {total_cit} citation slots for this query. "
            f"Top cited source: {top_domain}."
        )
    elif sov_pct < 15:
        return (
            f"ActiveCampaign holds {ac_cit} of {total_cit} citation slots ({sov_pct:.1f}% SoV). "
            f"Top non-AC source: {top_domain}."
        )
    else:
        return (
            f"ActiveCampaign holds {ac_cit} of {total_cit} citation slots ({sov_pct:.1f}% SoV)."
        )

# ---------------------------------------------------------------------------
# Sources bar chart — AC always first, then desc
# ---------------------------------------------------------------------------

def render_sources_chart(sources_df: pd.DataFrame):
    if "category" not in sources_df.columns or "run_count" not in sources_df.columns:
        return

    cat_totals = (
        sources_df.groupby("category")["run_count"]
        .sum()
    )

    # Separate AC categories from the rest
    ac_items    = {k: v for k, v in cat_totals.items() if str(k).startswith("AC:")}
    other_items = {k: v for k, v in cat_totals.items() if not str(k).startswith("AC:")}

    # If no AC citations exist, add a zero-value placeholder so the gap is visible
    if not ac_items:
        ac_items = {"AC (0 citations)": 0}

    # Sort each group desc
    ac_ordered    = dict(sorted(ac_items.items(),    key=lambda x: -x[1]))
    other_ordered = dict(sorted(other_items.items(), key=lambda x: -x[1]))

    ordered_series = pd.Series({**ac_ordered, **other_ordered}, name="Citation Count")
    st.bar_chart(ordered_series)

# ---------------------------------------------------------------------------
# Query detail page
# ---------------------------------------------------------------------------

def render_query_detail(prefix: str, query_name: str):
    data          = load_query_data(prefix, OUTPUT_DIR)
    scorecard_df  = data["scorecard"]
    sources_df    = data["sources"]
    subqueries_df = data["subqueries"]
    entities_df   = data["entities"]

    st.caption("Query Fan-Out Analysis")
    st.markdown(f"### {query_name}")

    # Scorecard row — immediately below title, no subheader
    m1, m2, m3, m4, m5, m6 = st.columns([1, 1, 1, 1, 1, 1.6])
    ac_mentioned = get_ac_mentioned_pct(entities_df)
    m1.metric("Runs",             scorecard_val(scorecard_df, "Total Responses"))
    m2.metric("Sub-query Clusters", scorecard_val(scorecard_df, "Clusters Found"))
    m3.metric("Sources",          scorecard_val(scorecard_df, "Sources Found Total"))
    m4.metric("Entities",         scorecard_val(scorecard_df, "Entities Found"))
    m5.metric("AC SoV",           scorecard_val(scorecard_df, "AC SoV (Citation %)"))
    m6.metric("Est. Cost",        scorecard_val(scorecard_df, "Estimated Cost"))

    # Soft one-line insight
    st.caption(_render_soft_insight(scorecard_df, entities_df, sources_df))

    st.divider()

    tab1, tab2, tab3 = st.tabs(["🔗 Sources", "🔍 Subqueries", "🏷️ Competitive Brands"])

    # --- Sources tab ---
    with tab1:
        if sources_df.empty:
            st.warning("No sources data found.")
        else:
            st.subheader("Citation Count by Category")
            render_sources_chart(sources_df)

            st.subheader(f"All Cited Sources: {len(sources_df)} unique URLs")
            st.caption(
                "🔵 Blue = ActiveCampaign  |  🔴 Red = Competitor  |  "
                "🟢 Green = >75% probability  |  🟡 Yellow = 25–75%  |  ⬜ Gray = <25%  |  "
                "Sorted: AC first, then by citation count ↓"
            )

            # AC rows float to top, then sort by run_count desc
            if not sources_df.empty:
                df_display = sources_df.copy()
                df_display["_ac_first"] = df_display.apply(
                    lambda r: 0 if (
                        str(r.get("category", "")).startswith("AC:") or
                        "activecampaign.com" in str(r.get("domain", ""))
                    ) else 1,
                    axis=1,
                )
                df_display = df_display.sort_values(
                    ["_ac_first", "run_count"], ascending=[True, False]
                ).reset_index(drop=True)
                df_display = df_display.drop(columns=["_ac_first"])

                # Rename uri → URL for display
                if "uri" in df_display.columns:
                    df_display = df_display.rename(columns={"uri": "URL"})

                display_cols = ["domain", "URL", "run_count", "probability", "rrf_score", "category", "intl_flag"]
                display_cols = [c for c in display_cols if c in df_display.columns]

                st.dataframe(
                    style_sources(df_display[display_cols]),
                    hide_index=True,
                )

    # --- Subqueries tab ---
    with tab2:
        if subqueries_df.empty:
            st.warning("No subquery data found.")
        else:
            st.subheader(f"Subquery Fan-Out Clusters — {len(subqueries_df)} clusters")
            st.caption(
                "Each row = a cluster of similar subqueries Gemini searched when generating its answer. "
                "**Probability** = % of runs it appeared. "
                "**Top Sources** = domains most frequently cited alongside this cluster."
            )
            display_cols = [
                "subquery", "run_count", "probability",
                "avg_position", "rrf_score", "top_sources", "sample_variants",
            ]
            display_cols = [c for c in display_cols if c in subqueries_df.columns]
            st.dataframe(
                subqueries_df[display_cols],
                hide_index=True,
            )

    # --- Competitive Brands tab ---
    with tab3:
        if entities_df.empty:
            st.warning("No entity data found.")
        else:
            brand_names_lower = {b.lower() for b in KNOWN_BRANDS}
            brands_df = entities_df[
                entities_df["entity"].str.lower().isin(brand_names_lower)
            ].copy()

            st.subheader(f"Competitive Brand Mentions — {len(brands_df)} brands detected")
            st.caption(
                "**What Gemini SAYS** (in its written answer) vs. **what it CITES** (sources tab). "
                "A brand at 100% means Gemini mentioned it in every response — "
                "but that does not mean AC content was cited as evidence. "
                "🔵 Blue = ActiveCampaign  |  🔴 Red = Competitors"
            )

            if brands_df.empty:
                st.info("No known brand names detected in this query's responses.")
            else:
                display_cols = ["entity", "run_count", "probability", "rrf_score", "sentiment"]
                display_cols = [c for c in display_cols if c in brands_df.columns]
                brands_sorted = brands_df.sort_values("run_count", ascending=False).reset_index(drop=True)
                st.dataframe(
                    style_brands(brands_sorted[display_cols]),
                    hide_index=True,
                )

            with st.expander(f"Show all {len(entities_df)} extracted entities (includes non-brand phrases)"):
                st.caption("All TitleCase phrases the regex extracted — includes workflow descriptions, not just brand names.")
                st.dataframe(entities_df, hide_index=True)

# ---------------------------------------------------------------------------
# Landing page — cross-query summary
# ---------------------------------------------------------------------------

def render_landing(queries: dict):
    st.title("Query Fan-Out Analysis")
    st.caption("ActiveCampaign LLM Citation and Share of Voice Report")

    if not queries:
        st.error(
            f"No output CSVs found in `{OUTPUT_DIR}/`. "
            "Run a query using the sidebar form, or run `python3 fanout_engine.py` first."
        )
        st.code("GEMINI_API_KEY=your_key python3 fanout_engine.py")
        return

    st.subheader("Summary: Total Queries Analyzed")
    st.caption("Select a query in the sidebar to drill into sources, subqueries, and brand mentions.")

    rows = []
    rows_raw = []
    query_data_list = []
    all_sources_parts = []

    for prefix, query_name in queries.items():
        data = load_query_data(prefix, OUTPUT_DIR)
        sc   = data["scorecard"]
        src  = data["sources"]
        ent  = data["entities"]

        ac_mentioned = get_ac_mentioned_pct(ent)
        top_domain   = get_top_non_ac_domain(src)

        rows.append({
            "Query":             query_name,
            "Runs":              scorecard_val(sc, "Total Responses"),
            "Total Citations":   scorecard_val(sc, "Total Citations"),
            "Clusters":          scorecard_val(sc, "Clusters Found"),
            "AC SoV":            scorecard_val(sc, "AC SoV (Citation %)", "0.0%"),
            "AC Mentioned":      f"{ac_mentioned:.0f}%",
            "Top Non-AC Source": top_domain,
            "Est. Cost":         scorecard_val(sc, "Estimated Cost"),
        })

        try:
            rows_raw.append({
                "ac_citations":    int(scorecard_val(sc, "AC Citations", "0")),
                "total_citations": int(str(scorecard_val(sc, "Total Citations", "0")).replace(",", "") or 0),
                "clusters":        int(scorecard_val(sc, "Clusters Found", "0")),
                "total_responses": int(scorecard_val(sc, "Total Responses", "0")),
                "ac_mentioned_pct": ac_mentioned,
            })
        except (ValueError, TypeError):
            pass

        if not src.empty:
            src_tagged = src.copy()
            src_tagged["_query"] = query_name
            all_sources_parts.append(src_tagged)
            query_data_list.append({"query": query_name, "sources_df": src})

    all_sources_df = (
        pd.concat(all_sources_parts, ignore_index=True) if all_sources_parts else pd.DataFrame()
    )

    # Compute aggregate metrics
    total_ac_cit   = sum(r["ac_citations"] for r in rows_raw)
    total_all_cit  = sum(r["total_citations"] for r in rows_raw)
    total_clusters = sum(r["clusters"] for r in rows_raw)
    total_runs_agg = sum(r["total_responses"] for r in rows_raw)
    weighted_sov = round(total_ac_cit / total_all_cit * 100, 1) if total_all_cit > 0 else 0.0
    weighted_mentioned = (
        round(
            sum(r["ac_mentioned_pct"] * r["total_responses"] for r in rows_raw) / total_runs_agg, 1
        )
        if total_runs_agg > 0 else 0.0
    )

    summary_df = pd.DataFrame(rows)
    st.dataframe(
        summary_df.style.apply(style_summary_row, axis=1),
        hide_index=True,
    )
    st.caption("🔴 AC SoV = 0%  |  🟡 AC SoV < 15%  |  🟢 AC SoV ≥ 15%")

    # Aggregate metrics row
    st.caption("**Aggregate across all queries**")
    agg1, agg2, agg3, agg4 = st.columns(4)
    agg1.metric("Total Queries", len(queries))
    agg2.metric("Total Sub-query Clusters", total_clusters)
    agg3.metric(
        "Weighted AC SoV", f"{weighted_sov:.1f}%",
        help="AC citations ÷ total citations across all queries combined",
    )
    agg4.metric(
        "Weighted AC Mentioned", f"{weighted_mentioned:.1f}%",
        help="AC mention rate weighted by number of runs per query",
    )

    st.divider()
    st.subheader("Key Findings")

    zero_sov   = [r["Query"] for r in rows if r["AC SoV"] == "0.0%"]
    mentioned  = [r for r in rows if float(r["AC Mentioned"].replace("%", "")) >= 50]

    if zero_sov:
        st.info(
            f"**{len(zero_sov)} of {len(queries)} queries returned no AC citations:** "
            + "  |  ".join(f'"{q}"' for q in zero_sov)
            + ". Review the Source Intelligence section below to see which third-party sites "
            "are filling those citation slots."
        )
    if mentioned and zero_sov:
        st.info(
            f"**AC is mentioned but not cited** in responses for "
            f"{len(mentioned)} quer{'y' if len(mentioned) == 1 else 'ies'}. "
            "This means Gemini references ActiveCampaign in its answer but uses other sources "
            "as grounding evidence - a signal that AC content could be better optimized for "
            "retrieval on these topics."
        )

    top_domains_all = [r["Top Non-AC Source"] for r in rows if r["Top Non-AC Source"] != "—"]
    if top_domains_all:
        most_common, count = Counter(top_domains_all).most_common(1)[0]
        if count > 1:
            st.info(
                f"**{most_common}** is the top non-AC cited source in "
                f"{count} of {len(queries)} queries — a consistent third-party authority in this space."
            )

    # ── How to read this report ───────────────────────────────────────────
    st.divider()
    exp_col1, exp_col2, exp_col3 = st.columns(3)
    with exp_col1:
        st.markdown("**Share of Voice (SoV)**")
        st.caption(
            "Counts how often an ActiveCampaign URL appears in Gemini's citation slots. "
            "A higher SoV means AC content is being used as grounding evidence — not just mentioned."
        )
    with exp_col2:
        st.markdown("**Sub-query Clusters**")
        st.caption(
            "When Gemini answers a query, it internally searches multiple sub-questions. "
            "Clusters group similar sub-queries across runs — more clusters means Gemini "
            "is exploring more angles of the topic."
        )
    with exp_col3:
        st.markdown("**RRF Score**")
        st.caption(
            "Reciprocal Rank Fusion combines citation frequency and position. "
            "A URL cited early and often scores highest — use this to prioritize "
            "which pages to optimize for AI retrieval."
        )

    # ── Source Intelligence (all queries combined) ────────────────────────
    if not all_sources_df.empty:
        st.divider()
        st.subheader("Source Intelligence: All Queries Combined")

        # 1. Category distribution chart
        st.markdown("#### Citation Distribution by Category")
        st.caption(
            "How citation slots are distributed across AC properties, competitors, and "
            "third-party sites — combined across all queries."
        )
        if build_top_level_summary is not None:
            top_level_df = build_top_level_summary(all_sources_df)
            if not top_level_df.empty:
                st.bar_chart(top_level_df.set_index("top_level")["citation_count"])
                n_q = len(query_data_list)
                n_urls = all_sources_df["uri"].nunique() if "uri" in all_sources_df.columns else len(all_sources_df)
                st.caption(
                    f"Combined across {n_q} {'query' if n_q == 1 else 'queries'}, "
                    f"{n_urls} unique source URLs"
                )

        # 2. AC Property Share of Voice
        st.markdown("#### AC Property Share of Voice")
        st.caption(
            "Which ActiveCampaign content sections are earning citations, and for which queries — "
            "a map of where AC's authority lives."
        )
        if build_ac_property_breakdown is not None:
            ac_prop_df = build_ac_property_breakdown(all_sources_df)
            if ac_prop_df.empty:
                st.info("No ActiveCampaign URLs found across any queries.")
            else:
                if "_query" in all_sources_df.columns:
                    def _queries_for_prop(prop: str) -> str:
                        mask = (
                            (all_sources_df["category"] == prop)
                            & all_sources_df["_query"].notna()
                        )
                        return ", ".join(sorted(all_sources_df.loc[mask, "_query"].unique()))
                    ac_prop_df["Top Associated Queries"] = ac_prop_df["ac_property"].apply(
                        _queries_for_prop
                    )
                ac_prop_df = ac_prop_df.rename(columns={
                    "ac_property":    "AC Property",
                    "citation_count": "Citations",
                    "pct_of_ac":      "% of AC Citations",
                })
                st.dataframe(ac_prop_df, hide_index=True)

        # 3. Query × AC Property Heatmap
        st.markdown("#### Query × AC Property Heatmap")
        st.caption(
            "Each cell shows how many times an AC property was cited for a given query. "
            "Red cells are content gaps; green cells are citation strongholds."
        )
        if build_query_heatmap is not None and query_data_list:
            pivot = build_query_heatmap(query_data_list)
            if pivot is not None and not pivot.empty:
                st.dataframe(style_heatmap(pivot))
                st.caption("🟢 4+ citations  |  🟡 1–3  |  🔴 0 (gap)")
            else:
                st.info("No AC property data available to build heatmap.")

        # 4. High-Frequency Non-AC Sites — Gap Analysis
        st.markdown("#### High-Frequency Non-AC Sites: Gap Analysis")
        st.caption(
            "Third-party sites dominating citation slots where AC has no presence — "
            "ranked by total citation volume across all queries."
        )
        if build_cross_query_domain_summary is not None:
            cross_df = build_cross_query_domain_summary(query_data_list)
            if not cross_df.empty:
                non_ac = cross_df[
                    ~cross_df["category"].str.startswith("AC:", na=False)
                ].copy()
                if non_ac.empty:
                    st.success("No high-frequency non-AC sites detected.")
                else:
                    def _gap_flag(row) -> str:
                        return "⚠️" if (
                            row["total_citations"] >= 5
                            and row["category"] in (
                                "3rd Party: Comparison", "3rd Party: Review Site"
                            )
                        ) else ""
                    non_ac["Flag"] = non_ac.apply(_gap_flag, axis=1)
                    non_ac = non_ac.rename(columns={
                        "domain":          "Domain",
                        "category":        "Category",
                        "total_citations": "Total Citations",
                        "queries_count":   "Queries Count",
                        "queries_list":    "Queries",
                    })
                    st.dataframe(
                        non_ac[["Flag", "Domain", "Category", "Total Citations",
                                "Queries Count", "Queries"]],
                        hide_index=True,
                    )
                    flagged = non_ac[non_ac["Flag"] == "⚠️"]
                    if not flagged.empty:
                        top = flagged.iloc[0]
                        st.info(
                            f"**{top['Domain']}** ({top['Category']}) appears in "
                            f"{top['Total Citations']} citation slots across "
                            f"{top['Queries Count']} "
                            f"{'query' if top['Queries Count'] == 1 else 'queries'}. "
                            f"Consider auditing what content {top['Domain']} publishes for these "
                            "queries — it's a citation signal for what Gemini considers authoritative."
                        )
                    else:
                        top_row = non_ac.iloc[0]
                        st.info(
                            f"**Most-cited non-AC domain:** **{top_row['Domain']}** "
                            f"({top_row['Total Citations']} citations across "
                            f"{top_row['Queries Count']} "
                            f"{'query' if top_row['Queries Count'] == 1 else 'queries'})."
                        )

        # 5. All Cited URLs
        st.markdown("#### All Cited URLs")
        st.caption(
            "Every URL cited across all queries, ranked by total citation count. "
            "Includes ActiveCampaign pages alongside all third-party sources."
        )
        if build_all_url_citations is not None and query_data_list:
            all_url_df = build_all_url_citations(query_data_list)
            if all_url_df.empty:
                st.info("No URL citation data available.")
            else:
                def _style_all_url_row(row):
                    cat = str(row.get("category", ""))
                    if cat.startswith("AC:"):
                        return [_BLUE] * len(row)
                    if cat == "Competitor":
                        return [_RED] * len(row)
                    return [""] * len(row)

                display_cols = [c for c in ["domain", "uri", "title", "category", "total_citations", "queries_count", "queries_list"] if c in all_url_df.columns]
                top100 = all_url_df[display_cols].head(100)
                st.dataframe(
                    top100.style.apply(_style_all_url_row, axis=1),
                    hide_index=True,
                )
                st.caption(
                    f"Showing top {len(top100)} of {len(all_url_df)} unique URLs  |  "
                    "🔵 Blue = ActiveCampaign  |  🔴 Red = Competitor"
                )

        # 6. Website × Query Citation Map
        st.markdown("#### Website × Query Citation Map")
        st.caption(
            "The top 25 most-cited domains across all queries. "
            "Use this to understand who owns the citation landscape for each topic."
        )
        if build_domain_query_pivot is not None:
            domain_pivot = build_domain_query_pivot(query_data_list, top_n=25)
            if not domain_pivot.empty:
                ac_domain_set: set = set()
                if "domain" in all_sources_df.columns and "category" in all_sources_df.columns:
                    ac_domain_set = set(
                        all_sources_df.loc[
                            all_sources_df["category"].str.startswith("AC:", na=False),
                            "domain",
                        ].unique()
                    )
                pivot_display = domain_pivot.reset_index()
                st.dataframe(
                    pivot_display.style.apply(
                        lambda row: style_domain_row(row, ac_domain_set), axis=1
                    ),
                    hide_index=True,
                )
                st.caption(
                    f"🔵 Blue = AC property  |  🟢 4+ citations  |  🟡 1–3  |  ⬜ 0 (absent)  |  "
                    f"Top {len(domain_pivot)} domains by total citations"
                )

# ---------------------------------------------------------------------------
# Live query runner
# ---------------------------------------------------------------------------

def run_new_query(query: str, num_runs: int, model_name: str = "gemini-2.5-flash-lite") -> str | None:
    """
    Runs a new query via fanout_engine, saves CSVs, returns the new prefix.
    Returns None on failure.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        st.warning("GEMINI_API_KEY environment variable is not set. Restart Streamlit with the key exported.")
        return None
    if run_analysis is None:
        st.warning("fanout_engine not available — cannot run new queries.")
        return None

    st.session_state.run_counter += 1
    prefix = f"live{st.session_state.run_counter}"

    progress_bar = st.progress(0, text=f"Starting run 1 of {num_runs}...")
    status_text  = st.empty()

    def _progress(current: int, total: int) -> None:
        pct = int(current / total * 100)
        progress_bar.progress(pct, text=f"Run {current} of {total} complete ({pct}%)")
        status_text.caption(f"Analyzing: run {current}/{total}")

    try:
        result = run_analysis(
            query=query,
            api_key=api_key,
            num_runs=num_runs,
            model_name=model_name,
            progress_callback=_progress,
        )
        progress_bar.progress(100, text=f"Done - {num_runs} runs complete.")
        status_text.empty()
        if result.get("model_fallback_triggered"):
            st.info(
                f"Gemini 3.1 Flash Lite preview was unavailable - "
                f"results generated with {FALLBACK_MODEL}."
            )
        if result["sources"].empty:
            st.info(
                "Gemini returned no grounding sources for this query. This can happen if the "
                "API key's project does not have Google Search grounding active, or Gemini "
                "answered from training data without searching. Check your GEMINI_API_KEY "
                "project in AI Studio."
            )
        # Enrich sources with AC categories
        if not result["sources"].empty and categorize_sources:
            src_dicts = result["sources"].to_dict("records")
            result["sources"] = pd.DataFrame(categorize_sources(src_dicts))

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        save_to_csv(result, query=query, output_dir=OUTPUT_DIR, prefix=f"{prefix}_")

        return prefix
    except Exception as exc:
        progress_bar.empty()
        status_text.empty()
        st.warning(f"Analysis failed: {exc}")
        return None

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    _init_state()
    queries = discover_queries(OUTPUT_DIR)

    # Merge recent_runs into queries so they appear in landing summary too
    for item in st.session_state.recent_runs:
        p = item["prefix"]
        if p not in queries:
            queries[p] = item["query_name"]

    with st.sidebar:
        st.title("📊 Fan-Out Report")
        st.markdown(
            "This tool runs target queries through the Gemini AI with Google Search grounding enabled "
            "and measures **LLM Share of Voice** — how often ActiveCampaign content is actually *cited* "
            "as evidence versus merely *mentioned* in the AI's answer. "
            "Each query is run multiple times to establish citation probability across runs."
        )
        st.divider()

        # ---- New query form ----
        with st.form("new_query_form", clear_on_submit=True):
            new_query = st.text_input(
                "Analyze a query",
                placeholder="e.g. best CRM for ecommerce",
            )
            new_runs = st.number_input("Runs", min_value=1, max_value=20, value=8)
            _m_opts   = list(AVAILABLE_MODELS.keys())
            _m_labels = list(AVAILABLE_MODELS.values())
            _m_sel    = st.selectbox("Model", _m_labels, index=0)
            new_model = _m_opts[_m_labels.index(_m_sel)]
            submitted = st.form_submit_button("Run Analysis ▶", use_container_width=True)

        if submitted and new_query.strip():
            prefix = run_new_query(new_query.strip(), int(new_runs), new_model)
            if prefix:
                st.session_state.recent_runs.append({
                    "prefix": prefix,
                    "query_name": new_query.strip(),
                })
                queries[prefix] = new_query.strip()
                st.session_state.selected_prefix = prefix
                st.rerun()

        # ---- Recent Runs ----
        if st.session_state.recent_runs:
            st.divider()
            st.caption("**Recent Runs**")
            for item in reversed(st.session_state.recent_runs):
                label = item["query_name"]
                label_display = label[:30] + "..." if len(label) > 30 else label
                if st.button(label_display, key=f"recent_{item['prefix']}", use_container_width=True):
                    st.session_state.selected_prefix = item["prefix"]
                    st.rerun()

        # ---- Example Queries ----
        st.divider()
        st.caption("**Example Queries**")
        for prefix, query_name in queries.items():
            label = query_name[:30] + "..." if len(query_name) > 30 else query_name
            if st.button(label, key=f"nav_{prefix}", use_container_width=True):
                st.session_state.selected_prefix = prefix
                st.rerun()

        # Overview button at very bottom
        st.divider()
        if st.button("← Overview (all queries)", use_container_width=True):
            st.session_state.selected_prefix = None
            st.rerun()

    # ---- Main content area ----
    selected = st.session_state.selected_prefix

    if selected is None or not queries:
        render_landing(queries)
    else:
        query_name = queries.get(selected, selected)
        render_query_detail(selected, query_name)


if __name__ == "__main__":
    main()
