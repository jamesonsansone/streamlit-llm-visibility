"""
report.py — Stakeholder report for Query Fan-Out Analysis results.

Loads CSVs produced by fanout_engine.py (and/or rows from Supabase) and
presents them as a navigable report. Also supports running new queries
live from the sidebar (reads GEMINI_API_KEY from environment).

Run: streamlit run report.py
"""

import os
import glob
from collections import Counter

import pandas as pd
import streamlit as st

try:
    from fanout_engine import (
        run_analysis,
        estimate_cost,
        calculate_target_sov,
        save_to_csv,
        AVAILABLE_MODELS,
        FALLBACK_MODEL,
    )
except ImportError:
    run_analysis = save_to_csv = estimate_cost = calculate_target_sov = None
    AVAILABLE_MODELS = {"gemini-2.5-flash-lite": "Gemini 2.5 Flash Lite (Standard)"}
    FALLBACK_MODEL = "gemini-2.5-flash-lite"

try:
    from db import get_client, save_run_to_db, load_all_queries, load_query_detail
    _DB_AVAILABLE = True
except ImportError:
    get_client = save_run_to_db = load_all_queries = load_query_detail = None
    _DB_AVAILABLE = False

try:
    from target_categorizer import (
        categorize_sources,
        build_cross_query_domain_summary,
        build_domain_query_pivot,
        build_all_url_citations,
    )
except ImportError:
    categorize_sources = None
    build_cross_query_domain_summary = None
    build_domain_query_pivot = None
    build_all_url_citations = None

OUTPUT_DIR = "./output"

st.set_page_config(
    page_title="Fan-Out Report",
    page_icon="📊",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Color constants
# ---------------------------------------------------------------------------

_GREEN  = "background-color: #d4edda; color: #155724"
_YELLOW = "background-color: #fff3cd; color: #6c4a00"
_RED    = "background-color: #f8d7da; color: #721c24"
_BLUE   = "background-color: #cce5ff; color: #004085"
_GRAY   = "background-color: #e9ecef; color: #212529"

# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------

def _init_state():
    defaults = {
        "selected_prefix": None,
        "recent_runs": [],
        "run_counter": 0,
        "target_domain": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _target_brand_token(target_domain: str) -> str:
    if not target_domain:
        return ""
    t = target_domain.strip().lower()
    t = t.replace("https://", "").replace("http://", "").replace("www.", "")
    t = t.split("/")[0].split(":")[0]
    return t.split(".")[0]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_query_data(prefix: str, output_dir: str) -> dict:
    # DB-backed query — prefix is "db:{id}"
    if prefix.startswith("db:") and _DB_AVAILABLE and load_query_detail is not None:
        try:
            db_id = int(prefix[3:])
            return load_query_detail(db_id)
        except Exception:
            pass

    # CSV fallback
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
        "responses":  pd.DataFrame(),
    }


def discover_queries(output_dir: str) -> dict:
    """
    Returns {prefix: query_name} for all known queries.
    Merges Supabase rows ("db:{id}") with CSV-backed prefixes.
    """
    queries = {}

    if os.path.exists(output_dir):
        for path in sorted(glob.glob(os.path.join(output_dir, "*_scorecard.csv"))):
            prefix = os.path.basename(path).replace("_scorecard.csv", "")
            try:
                df = pd.read_csv(path)
                row = df[df["metric"] == "Query"]["value"]
                query_name = str(row.iloc[0]) if not row.empty else prefix
            except Exception:
                query_name = prefix
            queries[prefix] = query_name

    if _DB_AVAILABLE and load_all_queries is not None:
        try:
            for q in load_all_queries():
                key = f"db:{q['db_id']}"
                queries[key] = q["query_text"]
        except Exception:
            pass

    return queries


def scorecard_val(df: pd.DataFrame, metric: str, default: str = "—") -> str:
    if df.empty:
        return default
    row = df[df["metric"] == metric]["value"]
    return str(row.iloc[0]) if not row.empty else default


def enrich_sources(sources_df: pd.DataFrame, target_domain: str) -> pd.DataFrame:
    """Adds Target/Other category + intl_flag to a sources DataFrame at render time."""
    if sources_df.empty or categorize_sources is None:
        return sources_df
    enriched = categorize_sources(sources_df.to_dict("records"), target_domain)
    return pd.DataFrame(enriched)


def get_target_mentioned_pct(entities_df: pd.DataFrame, target_domain: str) -> float:
    """Heuristic: highest probability among entities whose name contains the target brand token."""
    if entities_df.empty or "entity" not in entities_df.columns:
        return 0.0
    token = _target_brand_token(target_domain).replace("-", "")
    if not token:
        return 0.0
    mask = entities_df["entity"].astype(str).str.lower().str.replace(" ", "").str.replace("-", "").str.contains(token, na=False)
    matches = entities_df[mask]
    if matches.empty:
        return 0.0
    return float(matches.iloc[0].get("probability", 0))


def get_top_non_target_domain(sources_df: pd.DataFrame) -> str:
    if sources_df.empty or "domain" not in sources_df.columns:
        return "—"
    non_target = sources_df[sources_df.get("category", "") != "Target"] if "category" in sources_df.columns else sources_df
    # Always exclude vertex-ai-search intermediate URLs if any slipped through
    non_target = non_target[~non_target["domain"].str.contains("vertexaisearch", case=False, na=False)]
    if non_target.empty:
        return "—"
    return non_target.sort_values("run_count", ascending=False).iloc[0]["domain"]

# ---------------------------------------------------------------------------
# Styling helpers
# ---------------------------------------------------------------------------

def style_sources(df: pd.DataFrame):
    def row_color(row):
        if row.get("category") == "Target":
            return [_BLUE] * len(row)
        prob = float(row.get("probability", 0) or 0)
        if prob > 75:
            return [_GREEN] * len(row)
        elif prob >= 25:
            return [_YELLOW] * len(row)
        return [_GRAY] * len(row)
    return df.style.apply(row_color, axis=1)


def style_brands(df: pd.DataFrame, target_domain: str):
    token = _target_brand_token(target_domain).replace("-", "")
    def row_color(row):
        if not token:
            return [""] * len(row)
        entity_norm = str(row.get("entity", "")).lower().replace(" ", "").replace("-", "")
        if token in entity_norm:
            return [_BLUE] * len(row)
        return [""] * len(row)
    return df.style.apply(row_color, axis=1)


def style_summary_row(row):
    try:
        sov = float(str(row.get("Target SoV", "0")).replace("%", ""))
    except ValueError:
        sov = 0.0
    color = _RED if sov == 0.0 else (_YELLOW if sov < 15 else _GREEN)
    return [color if col == "Target SoV" else "" for col in row.index]


def style_domain_row(row, target_domain_set: set):
    styles = []
    for col in row.index:
        if col == "domain":
            styles.append(_BLUE if row["domain"] in target_domain_set else "")
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
# Soft insight line
# ---------------------------------------------------------------------------

def _render_soft_insight(sources_df: pd.DataFrame, target_domain: str, sov: dict) -> str:
    total_cit = sov.get("total_citations", 0)
    target_cit = sov.get("target_citations", 0)
    sov_pct = sov.get("sov_pct", 0.0)
    top_domain = get_top_non_target_domain(sources_df)
    label = target_domain or "Target"

    if not target_domain:
        return f"No target domain set. Top cited source: {top_domain}."
    if sov_pct == 0.0:
        return (
            f"No {label} pages appeared in the {total_cit} citation slots for this query. "
            f"Top cited source: {top_domain}."
        )
    if sov_pct < 15:
        return (
            f"{label} holds {target_cit} of {total_cit} citation slots ({sov_pct:.1f}% SoV). "
            f"Top non-target source: {top_domain}."
        )
    return f"{label} holds {target_cit} of {total_cit} citation slots ({sov_pct:.1f}% SoV)."


# ---------------------------------------------------------------------------
# Sources bar chart — Target first, then Other
# ---------------------------------------------------------------------------

def render_sources_chart(sources_df: pd.DataFrame):
    if "category" not in sources_df.columns or "run_count" not in sources_df.columns:
        return
    cat_totals = sources_df.groupby("category")["run_count"].sum()
    if cat_totals.empty:
        return
    ordered = {}
    if "Target" in cat_totals.index:
        ordered["Target"] = int(cat_totals["Target"])
    else:
        ordered["Target (0 citations)"] = 0
    if "Other" in cat_totals.index:
        ordered["Other"] = int(cat_totals["Other"])
    st.bar_chart(pd.Series(ordered, name="Citation Count"))


# ---------------------------------------------------------------------------
# Query detail page
# ---------------------------------------------------------------------------

def render_query_detail(prefix: str, query_name: str):
    target_domain = st.session_state.target_domain
    data          = load_query_data(prefix, OUTPUT_DIR)
    scorecard_df  = data["scorecard"]
    sources_df    = enrich_sources(data["sources"], target_domain)
    subqueries_df = data["subqueries"]
    entities_df   = data["entities"]

    sov = calculate_target_sov(sources_df, target_domain) if calculate_target_sov is not None else {
        "target_citations": 0, "total_citations": 0, "sov_pct": 0.0
    }

    st.caption("Query Fan-Out Analysis")
    st.markdown(f"### {query_name}")

    m1, m2, m3, m4, m5, m6 = st.columns([1, 1, 1, 1, 1, 1.6])
    target_mentioned = get_target_mentioned_pct(entities_df, target_domain)
    m1.metric("Runs",                scorecard_val(scorecard_df, "Total Responses"))
    m2.metric("Sub-query Clusters",  scorecard_val(scorecard_df, "Clusters Found"))
    m3.metric("Sources",             scorecard_val(scorecard_df, "Sources Found Total"))
    m4.metric("Entities",            scorecard_val(scorecard_df, "Entities Found"))
    m5.metric("Target SoV",          f"{sov['sov_pct']:.1f}%" if target_domain else "—")
    m6.metric("Est. Cost",           scorecard_val(scorecard_df, "Estimated Cost"))

    st.caption(_render_soft_insight(sources_df, target_domain, sov))

    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(["🔗 Sources", "🔍 Subqueries", "🏷️ Brand Mentions", "🗺️ Content Gap"])

    # --- Sources tab ---
    with tab1:
        if sources_df.empty:
            st.warning("No sources data found.")
        else:
            st.subheader("Citation Count by Category")
            render_sources_chart(sources_df)

            st.subheader(f"All Cited Sources: {len(sources_df)} unique URLs")
            caption = (
                "🟢 Green = >75% probability  |  🟡 Yellow = 25–75%  |  ⬜ Gray = <25%  |  "
                "Sorted: Target first, then by citation count ↓"
            )
            if target_domain:
                caption = f"🔵 Blue = Target ({target_domain})  |  " + caption
            st.caption(caption)

            df_display = sources_df.copy()
            df_display["_target_first"] = (df_display.get("category", "") == "Target").astype(int)
            df_display["_target_first"] = 1 - df_display["_target_first"]  # 0 = Target (on top)
            df_display = df_display.sort_values(
                ["_target_first", "run_count"], ascending=[True, False]
            ).reset_index(drop=True)
            df_display = df_display.drop(columns=["_target_first"])

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
            st.dataframe(subqueries_df[display_cols], hide_index=True)

    # --- Brand Mentions tab ---
    with tab3:
        if entities_df.empty:
            st.warning("No entity data found.")
        else:
            st.subheader(f"Entities / Brand Mentions — {len(entities_df)} detected")
            st.caption(
                "**What Gemini SAYS** (in its written answer) vs. **what it CITES** (sources tab). "
                "A brand at 100% means Gemini mentioned it in every response — "
                "but that does not mean target content was cited as evidence."
            )
            display_cols = ["entity", "run_count", "probability", "rrf_score", "sentiment"]
            display_cols = [c for c in display_cols if c in entities_df.columns]
            sorted_df = entities_df.sort_values("run_count", ascending=False).reset_index(drop=True) if "run_count" in entities_df.columns else entities_df
            fmt = {c: "{:.1f}" for c in ["probability", "rrf_score"] if c in display_cols}
            st.dataframe(
                style_brands(sorted_df[display_cols], target_domain).format(fmt),
                hide_index=True,
            )
            if target_domain:
                st.caption(f"🔵 Blue = entity name matches target brand token ({_target_brand_token(target_domain)}).")

    # --- Content Gap tab ---
    with tab4:
        _render_content_gap(data, sources_df, query_name, target_domain)


def _render_content_gap(data: dict, sources_df: pd.DataFrame, query_name: str, target_domain: str):
    st.subheader("Content Gap Analysis")
    st.caption(
        "Compares the top-cited non-target URL vs. the top-cited target URL against the aggregated "
        "AI answers. Shows which topics the non-target page covers that the target does not — these are "
        "your content gaps. Also shows which target passages earned citations — these are strengths to protect."
    )
    st.caption(
        "**Embedding score:** conceptual similarity via a neural language model. "
        "High = the AI discussed the same *ideas* as this page section, even in different words."
    )
    st.caption(
        "**TF-IDF score:** keyword overlap. "
        "High = the AI used the same specific terms as this page section verbatim."
    )

    responses_df = data.get("responses", pd.DataFrame())

    if responses_df.empty:
        st.info(
            "Response text is not available for this query. "
            "Re-run it from the sidebar to enable Content Gap Analysis."
        )
        return
    if sources_df.empty:
        st.info("No sources data available for this query.")
        return
    if not target_domain:
        st.info("Set a target domain in the sidebar to run gap analysis.")
        return

    reference_doc = " ".join(responses_df["response_text"].dropna().tolist())

    target_df     = sources_df[sources_df.get("category", "") == "Target"]
    non_target_df = sources_df[sources_df.get("category", "") != "Target"]

    if non_target_df.empty:
        st.info("No non-target sources found for this query — every citation is on the target domain.")
        return
    if "rrf_score" not in sources_df.columns:
        st.info("Citation score data not available for this query. Re-run it to enable Content Gap Analysis.")
        return

    non_target_sorted = non_target_df.sort_values("rrf_score", ascending=False).reset_index(drop=True)
    top_target_row    = target_df.sort_values("rrf_score", ascending=False).iloc[0] if not target_df.empty else None

    competitor_options = non_target_sorted["uri"].tolist()
    competitor_labels  = [
        f"{row['domain']}  (RRF {row['rrf_score']:.1f} · {int(row['run_count'])} runs)"
        for _, row in non_target_sorted.iterrows()
    ]
    selected_idx = st.selectbox(
        "Non-target URL to analyze",
        options=range(len(competitor_options)),
        format_func=lambda i: competitor_labels[i],
        index=0,
        help="Auto-selected: highest RRF score. Change to compare a different non-target URL.",
    )
    selected_non_target_row = non_target_sorted.iloc[selected_idx]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Non-target URL**")
        non_target_uri = selected_non_target_row["uri"]
        st.markdown(f"[{non_target_uri}]({non_target_uri})")
        st.caption(f"RRF {selected_non_target_row['rrf_score']:.1f} · {int(selected_non_target_row['run_count'])} runs")
    with col2:
        if top_target_row is not None:
            st.markdown(f"**Top {target_domain} URL**")
            target_uri_display = top_target_row["uri"]
            st.markdown(f"[{target_uri_display}]({target_uri_display})")
            st.caption(f"RRF {top_target_row['rrf_score']:.1f} · {int(top_target_row['run_count'])} runs")
        else:
            st.error(f"⚠️ {target_domain} was NOT cited for this query — full content gap.")

    st.divider()

    if not st.button("Run Gap Analysis", type="primary"):
        return

    try:
        from citation_mapper import (
            fetch_page_content, chunk_page,
            score_chunks_tfidf, score_chunks_embedding, find_content_gaps,
            fetch_last_modified, detect_answer_capsules,
        )
    except ImportError as _ie:
        st.error(f"Import error: {_ie}")
        return

    target_uri = top_target_row["uri"] if top_target_row is not None else None

    with st.spinner("Fetching non-target page via Jina Reader..."):
        non_target_md = fetch_page_content(non_target_uri)
    with st.spinner("Fetching target page via Jina Reader..." if target_uri else "Skipping target fetch (not cited)..."):
        target_md = fetch_page_content(target_uri) if target_uri else ""
    with st.spinner("Checking page freshness..."):
        non_target_modified = fetch_last_modified(non_target_uri)
        target_modified = fetch_last_modified(target_uri) if target_uri else ""

    fresh_col1, fresh_col2 = st.columns(2)
    fresh_col1.caption(f"Last modified: **{non_target_modified}**" if non_target_modified else "Last modified: unknown")
    fresh_col2.caption(f"Last modified: **{target_modified}**" if target_modified else "Last modified: unknown")

    if not non_target_md:
        st.warning(
            f"Could not fetch content from `{non_target_uri}`. "
            "Jina Reader may be rate-limited or the page requires JavaScript."
        )
        return

    with st.spinner("Scoring chunks with TF-IDF and embeddings..."):
        non_target_chunks = chunk_page(non_target_md)
        target_chunks     = chunk_page(target_md) if target_md else []

        if not non_target_chunks:
            st.warning("Could not extract readable content from the non-target page.")
            return

        all_chunks = non_target_chunks + target_chunks
        tfidf_scored = score_chunks_tfidf(reference_doc, all_chunks)
        embed_scored = score_chunks_embedding(reference_doc, all_chunks)

        n = len(non_target_chunks)
        for i, chunk in enumerate(non_target_chunks):
            chunk["tfidf_score"] = tfidf_scored[i]["tfidf_score"]
            chunk["embed_score"] = embed_scored[i]["embed_score"]
            chunk["source"] = "non_target"
        for i, chunk in enumerate(target_chunks):
            j = n + i
            chunk["tfidf_score"] = tfidf_scored[j]["tfidf_score"]
            chunk["embed_score"] = embed_scored[j]["embed_score"]
            chunk["source"] = "target"

        gaps = find_content_gaps(non_target_chunks, target_chunks)

        st.markdown(f"#### Content Gaps — Topics the non-target URL covers that {target_domain} does not")
        st.caption(
            "These passages from the non-target URL scored high against the AI answer "
            f"but have no close match on the {target_domain} URL. Adding this content could improve citation rates."
        )
        if gaps:
            for g in gaps[:5]:
                best_note = (
                    f" · Closest target section: *{g['best_target_match']}*"
                    f" (similarity `{g['topic_similarity']:.2f}`)"
                    if g.get("best_target_match") else ""
                )
                st.markdown(
                    f"**{g['heading']}**"
                    f" · AI relevance: Embed `{g['embed_score']:.2f}`"
                    f" · TF-IDF `{g['tfidf_score']:.2f}`"
                    + best_note
                )
                truncated = g['text'][:400] + ('...' if len(g['text']) > 400 else '')
                quoted = '\n'.join(f"> {line}" for line in truncated.splitlines())
                st.markdown(quoted)
                st.divider()
        else:
            st.success(
                "No major content gaps detected. For every topic the non-target URL "
                f"covered that the AI referenced, {target_domain} has a section with topic "
                "similarity above 0.75. Review the TF-IDF vs. Embedding table below to see where the fine margins are."
            )

        if target_chunks:
            st.markdown(f"#### {target_domain} Citation Strengths — What earned the target its citations")
            st.caption(
                "Target passages that matched the AI answer most closely. "
                "These are the topics your content already covers well."
            )
            target_ranked = sorted(target_chunks, key=lambda x: x["embed_score"], reverse=True)
            for chunk in target_ranked[:3]:
                st.markdown(
                    f"**{chunk['heading']}** &nbsp;·&nbsp; "
                    f"Embed `{chunk['embed_score']:.2f}` · TF-IDF `{chunk['tfidf_score']:.2f}`"
                )
                truncated = chunk['text'][:400] + ('...' if len(chunk['text']) > 400 else '')
                quoted = '\n'.join(f"> {line}" for line in truncated.splitlines())
                st.markdown(quoted)
                st.divider()
        else:
            st.info(
                f"{target_domain} was not cited for this query, so there are no citation strengths to show. "
                "Focus on the content gaps above to start earning citations."
            )

        non_target_capsules = detect_answer_capsules(non_target_chunks)
        target_capsules     = detect_answer_capsules(target_chunks)

        if non_target_capsules or target_capsules:
            st.markdown("#### Answer Capsule Spotlight")
            st.caption(
                "Short, self-contained passages structured for direct AI extraction. "
                "Non-target capsules show what to replicate; target capsules show what is already working."
            )
            cap_col1, cap_col2 = st.columns(2)
            with cap_col1:
                st.markdown("**Non-target: Worth Replicating**")
                if non_target_capsules:
                    for cap in non_target_capsules[:2]:
                        st.markdown(f"*{cap['heading']}* (Embed `{cap['embed_score']:.2f}`)")
                        quoted = '\n'.join(f"> {line}" for line in cap["text"].splitlines())
                        st.markdown(quoted)
                else:
                    st.info("No capsule candidates detected in non-target content.")
            with cap_col2:
                st.markdown(f"**{target_domain}: Already Working**")
                if target_capsules:
                    for cap in target_capsules[:2]:
                        st.markdown(f"*{cap['heading']}* (Embed `{cap['embed_score']:.2f}`)")
                        quoted = '\n'.join(f"> {line}" for line in cap["text"].splitlines())
                        st.markdown(quoted)
                else:
                    st.info(f"No answer capsule candidates found in {target_domain} content for this query.")

            st.markdown("**What makes a good answer capsule:**")
            st.caption(
                "1. **Short:** under 150 words so it can be extracted as a complete unit without truncation.\n\n"
                "2. **High embedding score:** above 0.50, confirming the AI was discussing this topic across multiple runs.\n\n"
                "3. **Self-contained opening:** the first sentence does not start with a contextual word like 'This,' 'These,' 'It,' or 'They' - it must make sense without surrounding context.\n\n"
                "4. **Declarative signal:** contains a year, a percentage, a superlative (best, most, top), or a named claim - something specific enough for an AI to cite as a fact."
            )

        st.divider()

        _render_content_brief(
            query_name=query_name,
            target_domain=target_domain,
            non_target_modified=non_target_modified,
            target_modified=target_modified,
            non_target_chunks=non_target_chunks,
            target_chunks=target_chunks,
            non_target_capsules=non_target_capsules,
            target_capsules=target_capsules,
            gaps=gaps,
        )

        with st.expander("TF-IDF vs. Embedding scores — all chunks"):
            comparison_rows = [
                {
                    "source":    c["source"],
                    "heading":   c["heading"],
                    "tfidf":     round(c["tfidf_score"], 3),
                    "embedding": round(c["embed_score"], 3),
                    "words":     c["word_count"],
                }
                for c in non_target_chunks + target_chunks
            ]
            st.dataframe(
                pd.DataFrame(comparison_rows).sort_values("embedding", ascending=False),
                hide_index=True,
            )


def _render_content_brief(
    query_name: str,
    target_domain: str,
    non_target_modified: str,
    target_modified: str,
    non_target_chunks: list,
    target_chunks: list,
    non_target_capsules: list,
    target_capsules: list,
    gaps: list,
) -> None:
    st.markdown("#### Content Brief")
    st.caption(
        "Gemini-generated content strategy based on the gap analysis, "
        "answer capsule data, and page freshness above."
    )
    try:
        try:
            gemini_key = st.secrets.get("GEMINI_API_KEY", "") or os.environ.get("GEMINI_API_KEY", "")
        except Exception:
            gemini_key = os.environ.get("GEMINI_API_KEY", "")

        if not gemini_key:
            st.info("Add GEMINI_API_KEY to Streamlit secrets or the environment to enable the content brief.")
            return

        gap_lines = "\n".join(
            f"- {g['heading']} (Embed: {g['embed_score']:.2f}): {g['text'][:400]}"
            for g in gaps[:5]
        ) if gaps else "No major content gaps detected."

        non_target_top = sorted(non_target_chunks, key=lambda x: x.get("embed_score", 0), reverse=True)[:3]
        target_top     = sorted(target_chunks,     key=lambda x: x.get("embed_score", 0), reverse=True)[:3]

        non_target_top_str = "\n".join(
            f"- [{c['heading']}] (Embed: {c['embed_score']:.2f}, TF-IDF: {c['tfidf_score']:.2f}): {c['text'][:400]}"
            for c in non_target_top
        ) or "None available."

        target_top_str = "\n".join(
            f"- [{c['heading']}] (Embed: {c['embed_score']:.2f}): {c['text'][:400]}"
            for c in target_top
        ) if target_top else f"{target_domain} was not cited for this query."

        non_target_capsule_text = "\n".join(
            f"- [{cap['heading']}]: {cap['text']}"
            for cap in non_target_capsules[:2]
        ) or "None detected."

        target_capsule_text = "\n".join(
            f"- [{cap['heading']}]: {cap['text']}"
            for cap in target_capsules[:2]
        ) if target_capsules else "None detected."

        prompt = f"""You are a content strategist for {target_domain} optimizing for LLM citation share of voice.

Query analyzed: "{query_name}"

Freshness:
- Non-target page last modified: {non_target_modified or 'unknown'}
- {target_domain} page last modified: {target_modified or 'unknown'}

Top non-target passages (ranked by AI relevance):
{non_target_top_str}

Top {target_domain} passages (ranked by AI relevance):
{target_top_str}

Non-target answer capsules (short, extractable passages the AI likely pulled directly):
{non_target_capsule_text}

{target_domain} answer capsules (what is already working):
{target_capsule_text}

Content gaps (topics the non-target covers that {target_domain} does not):
{gap_lines}

Output a structured content brief with these exact sections:

**Why the non-target is winning citations**
2-3 sentences. Reference specific passages and the freshness gap if relevant.

**On-Page Recommendations for {target_domain}**
For each gap or opportunity, provide specific on-page edits to the existing {target_domain} URL being analyzed. For each recommendation:
- Suggested section heading to add or rewrite on the existing page
- Target answer capsule: 1-2 declarative sentences under 60 words, written as if they belong on the {target_domain} page, optimized for direct AI extraction. Make these concrete and specific to the query topic.
- Where on the page: specify whether this should go near the top, in the middle as a new H2, or as a standalone callout block

**Strengths to protect**
1-2 sentences on what {target_domain} is already doing right based on the citation strength and capsule data.
"""
        from google import genai as _genai
        client = _genai.Client(api_key=gemini_key)
        resp = client.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=prompt,
        )
        st.markdown(resp.text)
    except Exception as exc:
        st.warning(f"Could not generate content brief: {exc}")


# ---------------------------------------------------------------------------
# Landing page — cross-query summary
# ---------------------------------------------------------------------------

def render_landing(queries: dict):
    target_domain = st.session_state.target_domain

    st.title("Aggregated Overview: Query Fan-Out Analysis")
    st.caption(
        f"Target: **{target_domain}**" if target_domain
        else "Set a target domain in the sidebar to enable Target SoV and highlighting."
    )

    if not queries:
        st.error(
            f"No queries found. Run a query using the sidebar form, "
            f"run `python3 fanout_engine.py` on the CLI, or point at a Supabase instance."
        )
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
        src  = enrich_sources(data["sources"], target_domain)
        ent  = data["entities"]

        sov = calculate_target_sov(src, target_domain) if calculate_target_sov is not None else {
            "target_citations": 0, "total_citations": 0, "sov_pct": 0.0
        }
        target_mentioned = get_target_mentioned_pct(ent, target_domain)
        top_domain       = get_top_non_target_domain(src)

        rows.append({
            "Query":              query_name,
            "Runs":               scorecard_val(sc, "Total Responses"),
            "Total Citations":    sov["total_citations"],
            "Clusters":           scorecard_val(sc, "Clusters Found"),
            "Target SoV":         f"{sov['sov_pct']:.1f}%",
            "Target Mentioned":   f"{target_mentioned:.0f}%",
            "Top Non-Target Src": top_domain,
            "Est. Cost":          scorecard_val(sc, "Estimated Cost"),
        })

        try:
            rows_raw.append({
                "target_citations":   int(sov["target_citations"]),
                "total_citations":    int(sov["total_citations"]),
                "clusters":           int(scorecard_val(sc, "Clusters Found", "0")),
                "total_responses":    int(scorecard_val(sc, "Total Responses", "0")),
                "target_mentioned":   target_mentioned,
            })
        except (ValueError, TypeError):
            pass

        if not src.empty:
            src_tagged = src.copy()
            src_tagged["_query"] = query_name
            all_sources_parts.append(src_tagged)
            query_data_list.append({"query": query_name, "sources_df": src})

    all_sources_df = pd.concat(all_sources_parts, ignore_index=True) if all_sources_parts else pd.DataFrame()

    total_target_cit = sum(r["target_citations"] for r in rows_raw)
    total_all_cit    = sum(r["total_citations"]  for r in rows_raw)
    total_clusters   = sum(r["clusters"]         for r in rows_raw)
    total_runs_agg   = sum(r["total_responses"]  for r in rows_raw)
    weighted_sov = round(total_target_cit / total_all_cit * 100, 1) if total_all_cit > 0 else 0.0
    weighted_mentioned = (
        round(
            sum(r["target_mentioned"] * r["total_responses"] for r in rows_raw) / total_runs_agg, 1
        )
        if total_runs_agg > 0 else 0.0
    )

    summary_df = pd.DataFrame(rows)
    st.dataframe(
        summary_df.style.apply(style_summary_row, axis=1),
        hide_index=True,
    )
    st.caption("🔴 Target SoV = 0%  |  🟡 Target SoV < 15%  |  🟢 Target SoV ≥ 15%")

    st.caption("**Aggregate across all queries**")
    agg1, agg2, agg3, agg4 = st.columns(4)
    agg1.metric("Total Queries", len(queries))
    agg2.metric("Total Sub-query Clusters", total_clusters)
    agg3.metric(
        "Weighted Target SoV", f"{weighted_sov:.1f}%",
        help="Target citations ÷ total citations across all queries combined",
    )
    agg4.metric(
        "Weighted Target Mentioned", f"{weighted_mentioned:.1f}%",
        help="Target mention rate weighted by number of runs per query",
    )

    st.divider()
    st.subheader("Key Findings")

    zero_sov = [r["Query"] for r in rows if r["Target SoV"] == "0.0%"]
    mentioned_not_cited = [
        r for r in rows
        if r["Target SoV"] == "0.0%" and float(r["Target Mentioned"].replace("%", "")) >= 50
    ]

    if zero_sov and target_domain:
        st.info(
            f"**{len(zero_sov)} of {len(queries)} queries returned no {target_domain} citations:** "
            + "  |  ".join(f'"{q}"' for q in zero_sov)
            + ". Review the Source Intelligence section below to see which third-party sites "
            "are filling those citation slots."
        )
    if mentioned_not_cited and target_domain:
        st.info(
            f"**{target_domain} is mentioned but not cited** in responses for "
            f"{len(mentioned_not_cited)} quer{'y' if len(mentioned_not_cited) == 1 else 'ies'}. "
            f"Gemini references {target_domain} in the answer text but uses other sources "
            "as grounding evidence — a signal that target content could be better optimized for retrieval on these topics."
        )

    top_domains_all = [r["Top Non-Target Src"] for r in rows if r["Top Non-Target Src"] != "—"]
    if top_domains_all:
        most_common, count = Counter(top_domains_all).most_common(1)[0]
        if count > 1:
            st.info(
                f"**{most_common}** is the top non-target cited source in "
                f"{count} of {len(queries)} queries — a consistent third-party authority in this space."
            )

    st.divider()
    exp_col1, exp_col2, exp_col3 = st.columns(3)
    with exp_col1:
        st.markdown("**Share of Voice (SoV)**")
        st.caption(
            "Counts how often the target domain's URLs appear in Gemini's citation slots. "
            "A higher SoV means target content is being used as grounding evidence — not just mentioned."
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

    if not all_sources_df.empty:
        st.divider()
        st.subheader("Source Intelligence: All Queries Combined")

        st.markdown("#### Citation Distribution by Category")
        st.caption(
            "How citation slots are distributed between target and non-target domains — combined across all queries."
        )
        if "category" in all_sources_df.columns and "run_count" in all_sources_df.columns:
            cat_totals = all_sources_df.groupby("category")["run_count"].sum()
            if not cat_totals.empty:
                st.bar_chart(cat_totals)
                n_q = len(query_data_list)
                n_urls = all_sources_df["uri"].nunique() if "uri" in all_sources_df.columns else len(all_sources_df)
                st.caption(
                    f"Combined across {n_q} {'query' if n_q == 1 else 'queries'}, "
                    f"{n_urls} unique source URLs"
                )

        st.markdown("#### High-Frequency Non-Target Sites")
        st.caption(
            "Third-party sites appearing in citation slots across all queries, "
            "ranked by total citation volume."
        )
        if build_cross_query_domain_summary is not None:
            cross_df = build_cross_query_domain_summary(query_data_list)
            if not cross_df.empty:
                non_target = cross_df[cross_df["category"] != "Target"].copy() if "category" in cross_df.columns else cross_df.copy()
                if non_target.empty:
                    st.success("No high-frequency non-target sites detected.")
                else:
                    non_target = non_target.rename(columns={
                        "domain":          "Domain",
                        "category":        "Category",
                        "total_citations": "Total Citations",
                        "queries_count":   "Queries Count",
                        "queries_list":    "Queries",
                    })
                    keep_cols = [c for c in ["Domain", "Category", "Total Citations", "Queries Count", "Queries"] if c in non_target.columns]
                    st.dataframe(non_target[keep_cols], hide_index=True)
                    top_row = non_target.iloc[0]
                    st.info(
                        f"**Most-cited non-target domain:** **{top_row['Domain']}** "
                        f"({top_row['Total Citations']} citations across "
                        f"{top_row['Queries Count']} "
                        f"{'query' if top_row['Queries Count'] == 1 else 'queries'})."
                    )

        st.markdown("#### All Cited URLs")
        st.caption(
            "Every URL cited across all queries, ranked by total citation count. "
            "Includes target pages alongside all non-target sources."
        )
        if build_all_url_citations is not None and query_data_list:
            all_url_df = build_all_url_citations(query_data_list)
            if all_url_df.empty:
                st.info("No URL citation data available.")
            else:
                def _style_all_url_row(row):
                    return [_BLUE] * len(row) if row.get("category") == "Target" else [""] * len(row)
                display_cols = [c for c in ["domain", "uri", "title", "category", "total_citations", "queries_count", "queries_list"] if c in all_url_df.columns]
                top100 = all_url_df[display_cols].head(100)
                st.dataframe(top100.style.apply(_style_all_url_row, axis=1), hide_index=True)
                label = f"Showing top {len(top100)} of {len(all_url_df)} unique URLs"
                if target_domain:
                    label += f"  |  🔵 Blue = {target_domain}"
                st.caption(label)

        st.markdown("#### Website × Query Citation Map")
        st.caption(
            "The top 25 most-cited domains across all queries. "
            "Use this to understand who owns the citation landscape for each topic."
        )
        if build_domain_query_pivot is not None:
            domain_pivot = build_domain_query_pivot(query_data_list, top_n=25)
            if not domain_pivot.empty:
                target_domain_set: set = set()
                if "domain" in all_sources_df.columns and "category" in all_sources_df.columns:
                    target_domain_set = set(
                        all_sources_df.loc[all_sources_df["category"] == "Target", "domain"].unique()
                    )
                pivot_display = domain_pivot.reset_index()
                st.dataframe(
                    pivot_display.style.apply(
                        lambda row: style_domain_row(row, target_domain_set), axis=1
                    ),
                    hide_index=True,
                )
                suffix = "🟢 4+ citations  |  🟡 1–3  |  ⬜ 0 (absent)"
                if target_domain:
                    suffix = f"🔵 Blue = target domain  |  " + suffix
                st.caption(f"{suffix}  |  Top {len(domain_pivot)} domains by total citations")


# ---------------------------------------------------------------------------
# Live query runner
# ---------------------------------------------------------------------------

def run_new_query(query: str, num_runs: int, model_name: str, target_domain: str) -> str | None:
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

        # Categorize at save time so CSV has category column; render-time categorization will
        # recompute it if target_domain later changes.
        if not result["sources"].empty and categorize_sources is not None:
            src_dicts = result["sources"].to_dict("records")
            result["sources"] = pd.DataFrame(categorize_sources(src_dicts, target_domain))

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        save_to_csv(result, query=query, output_dir=OUTPUT_DIR, prefix=f"{prefix}_", target_domain=target_domain)

        if _DB_AVAILABLE and save_run_to_db is not None:
            db_id = save_run_to_db(
                result,
                query=query,
                model_used=result.get("model_used", model_name),
                num_runs=num_runs,
                target_domain=target_domain,
            )
            if db_id is not None:
                status_text.caption(f"Saved to database (id={db_id})")
                return f"db:{db_id}"

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

    for item in st.session_state.recent_runs:
        p = item["prefix"]
        if p not in queries:
            queries[p] = item["query_name"]

    with st.sidebar:
        st.markdown("**Query Fan-Out Report**")
        st.markdown(
            "Runs target queries through Gemini with live web search enabled, "
            "measures **LLM Share of Voice** for your target domain, and surfaces content "
            "gaps vs. the sources Gemini actually cites."
        )
        st.divider()

        target_domain = st.text_input(
            "Target domain",
            value=st.session_state.target_domain,
            placeholder="example.com",
            help=(
                "The domain you want to track. Case-insensitive; subdomains count. "
                "Change this at any time — past runs relabel instantly from the new perspective."
            ),
        )
        st.session_state.target_domain = target_domain.strip()

        st.divider()

        with st.form("new_query_form", clear_on_submit=True):
            new_query = st.text_input(
                "Analyze a query",
                placeholder="e.g. best CRM for small business",
            )
            new_runs = st.number_input("Runs", min_value=1, max_value=20, value=8)
            _m_opts   = list(AVAILABLE_MODELS.keys())
            _m_labels = list(AVAILABLE_MODELS.values())
            _m_sel    = st.selectbox("Model", _m_labels, index=0)
            new_model = _m_opts[_m_labels.index(_m_sel)]
            submitted = st.form_submit_button("Run Analysis ▶", use_container_width=True)

        if submitted and new_query.strip():
            prefix = run_new_query(new_query.strip(), int(new_runs), new_model, st.session_state.target_domain)
            if prefix:
                st.session_state.recent_runs.append({
                    "prefix": prefix,
                    "query_name": new_query.strip(),
                })
                queries[prefix] = new_query.strip()
                st.session_state.selected_prefix = prefix
                st.rerun()

        if st.session_state.recent_runs:
            st.divider()
            st.caption("**Recent Runs**")
            for item in reversed(st.session_state.recent_runs):
                label = item["query_name"]
                label_display = label[:30] + "..." if len(label) > 30 else label
                if st.button(label_display, key=f"recent_{item['prefix']}", use_container_width=True):
                    st.session_state.selected_prefix = item["prefix"]
                    st.rerun()

        st.divider()
        st.caption("**All Queries**")
        for prefix, query_name in queries.items():
            label = query_name[:30] + "..." if len(query_name) > 30 else query_name
            if st.button(label, key=f"nav_{prefix}", use_container_width=True):
                st.session_state.selected_prefix = prefix
                st.rerun()

        st.divider()
        if st.button("← Overview (all queries)", use_container_width=True):
            st.session_state.selected_prefix = None
            st.rerun()

    selected = st.session_state.selected_prefix
    if selected is None or not queries:
        render_landing(queries)
    else:
        query_name = queries.get(selected, selected)
        render_query_detail(selected, query_name)


if __name__ == "__main__":
    main()
