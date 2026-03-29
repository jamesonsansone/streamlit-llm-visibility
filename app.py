"""
app.py — Query Fan-Out Analysis Tool (Streamlit UI)

Run: streamlit run app.py
"""

import io
import json
import zipfile
import logging

import pandas as pd
import streamlit as st

from fanout_engine import run_analysis, save_to_csv, AVAILABLE_MODELS, FALLBACK_MODEL
from ac_categorizer import (
    categorize_sources,
    build_category_summary,
    build_top_level_summary,
    build_ac_property_breakdown,
    build_query_heatmap,
    generate_heatmap_insights,
)

logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Query Fan-Out Analyzer: ActiveCampaign",
    page_icon="\U0001f50d",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Example queries
# ---------------------------------------------------------------------------

EXAMPLE_QUERIES = [
    "— select an example —",
    "best email marketing automation software",
    "marketing automation for small business",
    "ActiveCampaign vs Mailchimp",
    "ActiveCampaign vs HubSpot",
    "ActiveCampaign vs Klaviyo",
    "how to set up email drip campaigns",
    "ecommerce email marketing platform",
    "what is marketing automation",
    "best CRM with email marketing",
    "email automation workflows for coaches",
    "autonomous marketing platform",
    "how to improve email open rates",
]

QUERY_BUCKETS = {
    "ActiveCampaign vs Mailchimp":          "Competitor — Mailchimp",
    "ActiveCampaign vs HubSpot":            "Competitor — HubSpot",
    "ActiveCampaign vs Klaviyo":            "Competitor — Klaviyo",
    "best email marketing automation software": "Email Marketing",
    "marketing automation for small business":  "Email Marketing",
    "how to set up email drip campaigns":       "Email Marketing",
    "ecommerce email marketing platform":       "Ecommerce",
    "best CRM with email marketing":            "CRM",
    "email automation workflows for coaches":   "Use-Case",
    "autonomous marketing platform":            "Autonomous Marketing",
    "how to improve email open rates":          "Email Marketing",
    "what is marketing automation":             "Awareness / Top-of-Funnel",
}


def get_bucket(query: str) -> str:
    return QUERY_BUCKETS.get(query, "Custom")


# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------

def _init_state() -> None:
    defaults = {
        "last_result": None,
        "last_query": "",
        "all_query_results": [],  # accumulates {query, bucket, sources_df} for heatmap
        "raw_runs": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ---------------------------------------------------------------------------
# Styling helpers
# ---------------------------------------------------------------------------

_GREEN = "background-color: #d4edda"
_YELLOW = "background-color: #fff3cd"
_GRAY = "background-color: #e9ecef"


def _prob_style(val: float) -> str:
    if val > 75:
        return _GREEN
    elif val >= 25:
        return _YELLOW
    return _GRAY


def style_prob_df(df: pd.DataFrame):
    if "probability" not in df.columns:
        return df.style
    return df.style.applymap(_prob_style, subset=["probability"])


def style_sources_df(df: pd.DataFrame):
    """Color probability column + highlight AC blue, competitors red."""
    styler = df.style

    def row_highlight(row: pd.Series):
        styles = [""] * len(row)
        uri = row.get("uri", "")
        if "activecampaign.com" in str(uri).lower():
            return ["background-color: #cce5ff"] * len(row)
        for comp in ["mailchimp.com", "hubspot.com", "klaviyo.com", "brevo.com", "constantcontact.com"]:
            if comp in str(uri).lower():
                return ["background-color: #f8d7da"] * len(row)
        if "probability" in df.columns:
            prob = row.get("probability", 0)
            bg = _prob_style(prob)
            return [bg] * len(row)
        return styles

    styler = styler.apply(row_highlight, axis=1)
    return styler


def style_heatmap(df: pd.DataFrame):
    def cell_style(val):
        if val == 0:
            return "background-color: #f8d7da; color: #721c24"  # red
        elif val <= 3:
            return "background-color: #fff3cd; color: #856404"  # yellow
        else:
            return "background-color: #d4edda; color: #155724"  # green

    return df.style.applymap(cell_style)


# ---------------------------------------------------------------------------
# CSV export builder
# ---------------------------------------------------------------------------

def build_csv_zip(result: dict) -> bytes:
    """Creates an in-memory zip with the three CSVs."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for key, filename in [
            ("subqueries", "top_subqueries.csv"),
            ("sources", "top_sources.csv"),
            ("entities", "top_entities.csv"),
        ]:
            df = result.get(key)
            if df is not None and not df.empty:
                zf.writestr(filename, df.to_csv(index=False))
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main() -> None:
    _init_state()

    # ---- Sidebar -----------------------------------------------------------
    with st.sidebar:
        st.title("\U0001f50d Fan-Out Analyzer")
        st.caption("ActiveCampaign Edition")
        st.divider()

        api_key = st.text_input(
            "Gemini API Key",
            type="password",
            help="Get a free key at aistudio.google.com",
        )

        num_runs = st.number_input(
            "Number of runs",
            min_value=1,
            max_value=20,
            value=8,
            help="How many times to call Gemini for the same query. More runs = more reliable signal.",
        )
        if num_runs > 10:
            st.warning("\u26a0\ufe0f Free tier limit: ~15 RPM. Keep runs \u2264 10 to avoid rate errors.")

        _model_options = list(AVAILABLE_MODELS.keys())
        _model_labels  = list(AVAILABLE_MODELS.values())
        _selected_label = st.selectbox("Model", _model_labels, index=0)
        selected_model = _model_options[_model_labels.index(_selected_label)]

        threshold = st.slider(
            "Clustering similarity threshold",
            min_value=50,
            max_value=100,
            value=80,
            help="Higher = stricter clustering. 80 is a good default.",
        )

        selected_example = st.selectbox("Example queries", EXAMPLE_QUERIES)

        st.divider()
        st.caption("About")
        st.markdown(
            "This tool probes how Gemini expands your query into subqueries when generating answers. "
            "It captures grounding sources and categorizes them against ActiveCampaign's content taxonomy."
        )

    # ---- Main area ---------------------------------------------------------
    st.title("Query Fan-Out Analysis")
    st.caption("How does Gemini search when answering your query? Which sources does it cite?")

    query_input = st.text_input(
        "Query",
        value=(selected_example if selected_example != "— select an example —" else ""),
        placeholder="e.g. best email marketing automation software",
    )

    run_col, clear_col = st.columns([1, 5])
    with run_col:
        run_btn = st.button("\u25b6 Run Analysis", type="primary", use_container_width=True)
    with clear_col:
        if st.button("Clear results"):
            st.session_state.last_result = None
            st.session_state.all_query_results = []
            st.rerun()

    # ---- Run analysis ------------------------------------------------------
    if run_btn:
        if not api_key:
            st.error("Enter your Gemini API key in the sidebar.")
            st.stop()
        if not query_input.strip():
            st.error("Enter a query to analyze.")
            st.stop()

        progress_bar = st.progress(0, text="Starting...")
        status_text = st.empty()

        def progress_callback(current: int, total: int) -> None:
            pct = int(current / total * 100)
            progress_bar.progress(pct, text=f"Run {current} of {total}...")
            status_text.caption(f"Run {current}/{total} complete")

        try:
            with st.spinner("Calling Gemini with grounding..."):
                result = run_analysis(
                    query=query_input.strip(),
                    api_key=api_key,
                    num_runs=int(num_runs),
                    similarity_threshold=int(threshold),
                    progress_callback=progress_callback,
                    model_name=selected_model,
                )

            progress_bar.progress(100, text="Done!")
            status_text.empty()
            if result.get("model_fallback_triggered"):
                st.info(
                    f"Gemini 3.1 Flash Lite preview was unavailable - "
                    f"results generated with {FALLBACK_MODEL}."
                )
            st.session_state.last_result = result
            st.session_state.last_query = query_input.strip()

            # Enrich sources with categories and store for heatmap
            if not result["sources"].empty:
                src_dicts = result["sources"].to_dict("records")
                enriched = categorize_sources(src_dicts)
                result["sources"] = pd.DataFrame(enriched)
                st.session_state.last_result = result

                # Accumulate for multi-query heatmap
                bucket = get_bucket(query_input.strip())
                st.session_state.all_query_results.append({
                    "query": query_input.strip(),
                    "bucket": bucket,
                    "sources_df": result["sources"].copy(),
                })

        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            st.stop()

    # ---- Display results ---------------------------------------------------
    if st.session_state.last_result is None:
        st.info("Configure your API key and query above, then click Run Analysis.")
        return

    result = st.session_state.last_result
    query_label = st.session_state.last_query

    # Metrics row
    sub_df = result.get("subqueries", pd.DataFrame())
    src_df = result.get("sources", pd.DataFrame())
    ent_df = result.get("entities", pd.DataFrame())
    raw_runs = result.get("raw_runs", [])

    from fanout_engine import estimate_cost, calculate_ac_sov
    cost = estimate_cost(raw_runs)
    sov  = calculate_ac_sov(src_df)

    # 6-metric scorecard
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Queries Processed", 1)
    m2.metric("Total Responses", len(raw_runs))
    m3.metric("Clusters Found", len(sub_df) if not sub_df.empty else 0)
    m4.metric("Sources Found", len(src_df) if not src_df.empty else 0)
    m5.metric("Entities Found", len(ent_df) if not ent_df.empty else 0)
    m6.metric("Est. Cost", f"${cost:.4f}")

    # Prominent AC Share of Voice metric
    sov_delta = f"{sov['ac_citations']} of {sov['total_citations']} citation slots"
    st.metric(
        label="\U0001f3af AC Share of Voice",
        value=f"{sov['sov_pct']:.1f}%",
        delta=sov_delta,
        delta_color="off",
    )

    st.divider()

    # ---- Tabs --------------------------------------------------------------
    tab1, tab2, tab3 = st.tabs(["\U0001f50d Subqueries", "\U0001f517 Sources", "\U0001f3f7\ufe0f Entities / Brands"])

    # Tab 1: Subqueries
    with tab1:
        st.subheader("Subquery Fan-Out Clusters")
        st.caption(
            "Gemini expanded your query into these subqueries. "
            "Probability = % of runs it appeared. RRF = combined probability + position priority score."
        )
        if sub_df.empty:
            st.warning("No subqueries captured. Grounding metadata may not have populated for this query.")
        else:
            display_cols = ["subquery", "probability", "avg_position", "rrf_score", "run_count", "sample_variants", "top_sources"]
            display_cols = [c for c in display_cols if c in sub_df.columns]
            st.dataframe(
                style_prob_df(sub_df[display_cols]),
                hide_index=True,
            )
            st.caption("\U0001f7e2 >75% probability &nbsp;&nbsp; \U0001f7e1 25-75% &nbsp;&nbsp; \u26aa <25%")

    # Tab 2: Sources
    with tab2:
        st.subheader("Cited Sources")
        st.caption(
            "URLs that Gemini cited while generating its answer. "
            "\U0001f535 = ActiveCampaign &nbsp; \U0001f534 = Competitor"
        )
        if src_df.empty:
            st.warning("No sources captured.")
        else:
            display_cols = ["uri", "title", "domain", "category", "intl_flag", "probability", "avg_position", "rrf_score"]
            display_cols = [c for c in display_cols if c in src_df.columns]
            st.dataframe(
                style_sources_df(src_df[display_cols]),
                hide_index=True,
            )
            st.caption("\U0001f535 Blue = AC properties &nbsp;&nbsp; \U0001f534 Red = Competitors")

            # AC Property Breakdown
            st.subheader("AC Property Breakdown")
            ac_breakdown = build_ac_property_breakdown(src_df)
            if ac_breakdown.empty:
                st.info("No ActiveCampaign URLs appeared in citation sets for this query.")
            else:
                st.dataframe(ac_breakdown, hide_index=True)

            # Category distribution chart
            st.subheader("Citation Distribution by Category")
            top_level = build_top_level_summary(src_df)
            if not top_level.empty:
                chart_df = top_level.set_index("top_level")
                st.bar_chart(chart_df["citation_count"])

            # International flag warning
            if "intl_flag" in src_df.columns:
                intl_rows = src_df[src_df["intl_flag"] != ""]
                if not intl_rows.empty:
                    n_intl = len(intl_rows)
                    st.warning(
                        f"\u26a0\ufe0f {n_intl} international URL(s) detected in citation set for an English query. "
                        "This may indicate hreflang misconfiguration or international content competing with English pages."
                    )
                    with st.expander("View flagged international URLs"):
                        intl_display = intl_rows[["uri", "intl_flag", "category"]].copy()
                        intl_display["query"] = query_label
                        st.dataframe(intl_display, hide_index=True)

    # Tab 3: Entities
    with tab3:
        st.subheader("Brand & Entity Mentions")
        st.caption(
            "Brands and entities mentioned in Gemini's response text. "
            "Sentiment reflects co-occurring positive/negative language in context."
        )
        if ent_df.empty:
            st.warning("No entities detected.")
        else:
            display_cols = ["entity", "probability", "avg_position", "rrf_score", "run_count", "sentiment"]
            display_cols = [c for c in display_cols if c in ent_df.columns]

            def entity_row_style(row: pd.Series):
                entity = str(row.get("entity", "")).lower()
                if "activecampaign" in entity:
                    return ["background-color: #cce5ff"] * len(row)
                for comp in ["mailchimp", "hubspot", "klaviyo", "brevo", "constant contact"]:
                    if comp in entity:
                        return ["background-color: #f8d7da"] * len(row)
                return [""] * len(row)

            st.dataframe(
                ent_df[display_cols].style.apply(entity_row_style, axis=1),
                hide_index=True,
            )
            st.caption("\U0001f535 Blue = ActiveCampaign &nbsp;&nbsp; \U0001f534 Red = Competitors")

    # ---- Multi-query heatmap (shows when 2+ queries run) -------------------
    all_results = st.session_state.all_query_results
    if len(all_results) >= 2:
        st.divider()
        st.subheader("\U0001f321\ufe0f Query Bucket \u00d7 AC Property Heatmap")
        st.caption(
            "Accumulated across all queries run this session. "
            "Shows which AC properties are getting cited for each query type."
        )
        pivot = build_query_heatmap(all_results)
        if pivot is not None and not pivot.empty:
            st.dataframe(style_heatmap(pivot))
            st.caption("\U0001f7e2 4+ citations &nbsp;&nbsp; \U0001f7e1 1-3 &nbsp;&nbsp; \U0001f534 0 (gap)")

            insights = generate_heatmap_insights(pivot)
            if insights:
                st.subheader("Auto-Generated Insights")
                for insight in insights:
                    st.markdown(insight)
        else:
            st.info("No AC sources found across queries yet.")

    # ---- Export & raw data -------------------------------------------------
    st.divider()
    export_col, _ = st.columns([2, 6])
    with export_col:
        csv_bytes = build_csv_zip(result)
        safe_query = "".join(c if c.isalnum() or c in "-_ " else "_" for c in query_label)[:40]
        st.download_button(
            label="\u2b07\ufe0f Export CSVs (zip)",
            data=csv_bytes,
            file_name=f"fanout_{safe_query}.zip",
            mime="application/zip",
            use_container_width=True,
        )

    with st.expander("Raw run data (JSON)"):
        raw_output = []
        for run in result.get("raw_runs", []):
            raw_output.append({
                "run_index": run.get("run_index"),
                "web_search_queries": run.get("web_search_queries"),
                "sources": run.get("sources"),
                "entities": run.get("entities"),
                "response_text": run.get("response_text", "")[:1000] + "...",
            })
        st.json(raw_output)


if __name__ == "__main__":
    main()
