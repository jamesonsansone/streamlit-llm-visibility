"""
app.py — Query Fan-Out Analysis Tool (Streamlit UI).

Run: streamlit run app.py
"""

import io
import zipfile
import logging

import pandas as pd
import streamlit as st

from fanout_engine import (
    run_analysis,
    estimate_cost,
    calculate_target_sov,
    AVAILABLE_MODELS,
    FALLBACK_MODEL,
)
from target_categorizer import categorize_sources

logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Query Fan-Out Analyzer",
    page_icon="\U0001f50d",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------

def _init_state() -> None:
    defaults = {
        "last_result": None,
        "last_query": "",
        "target_domain": "",
        "all_query_results": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ---------------------------------------------------------------------------
# Styling helpers
# ---------------------------------------------------------------------------

_GREEN  = "background-color: #d4edda"
_YELLOW = "background-color: #fff3cd"
_GRAY   = "background-color: #e9ecef"
_BLUE   = "background-color: #cce5ff"


def _prob_style(val: float) -> str:
    if val > 75:
        return _GREEN
    if val >= 25:
        return _YELLOW
    return _GRAY


def style_prob_df(df: pd.DataFrame):
    if "probability" not in df.columns:
        return df.style
    return df.style.applymap(_prob_style, subset=["probability"])


def style_sources_df(df: pd.DataFrame):
    """Highlight Target rows blue; color probability column otherwise."""
    def row_highlight(row: pd.Series):
        if row.get("category") == "Target":
            return [_BLUE] * len(row)
        if "probability" in df.columns:
            return [_prob_style(row.get("probability", 0))] * len(row)
        return [""] * len(row)
    return df.style.apply(row_highlight, axis=1)


def _target_brand_token(target_domain: str) -> str:
    """First label of the target domain, used for heuristic entity matching.
    'example.com' -> 'example'; '' -> ''."""
    if not target_domain:
        return ""
    t = target_domain.strip().lower()
    t = t.replace("https://", "").replace("http://", "").replace("www.", "")
    t = t.split("/")[0].split(":")[0]
    return t.split(".")[0]


def style_entities_df(df: pd.DataFrame, target_domain: str):
    token = _target_brand_token(target_domain).replace("-", "")
    def row_highlight(row: pd.Series):
        if not token:
            return [""] * len(row)
        entity_norm = str(row.get("entity", "")).lower().replace(" ", "").replace("-", "")
        if token in entity_norm:
            return [_BLUE] * len(row)
        return [""] * len(row)
    return df.style.apply(row_highlight, axis=1)


# ---------------------------------------------------------------------------
# CSV export builder
# ---------------------------------------------------------------------------

def build_csv_zip(result: dict) -> bytes:
    """In-memory zip with the three tabular CSVs."""
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
        st.divider()

        target_domain = st.text_input(
            "Target domain",
            value=st.session_state.target_domain,
            placeholder="example.com",
            help=(
                "The domain you want to track citations for. "
                "Case-insensitive; subdomains count as Target. "
                "Change this at any time — past runs are relabeled from the new domain's perspective."
            ),
        )
        st.session_state.target_domain = target_domain.strip()

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
            st.warning("⚠️ Free tier limit: ~15 RPM. Keep runs ≤ 10 to avoid rate errors.")

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

        st.divider()
        st.caption("About")
        st.markdown(
            "This tool probes how Gemini expands your query into subqueries when generating answers. "
            "It captures grounding sources and compares them against your target domain."
        )

    target_domain = st.session_state.target_domain

    # ---- Main area ---------------------------------------------------------
    st.title("Query Fan-Out Analysis")
    st.caption("How does Gemini search when answering your query? Which sources does it cite?")

    query_input = st.text_input(
        "Query",
        placeholder="e.g. best CRM for small business",
    )

    run_col, clear_col = st.columns([1, 5])
    with run_col:
        run_btn = st.button("▶ Run Analysis", type="primary", use_container_width=True)
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
                    f"Gemini 3.1 Flash Lite preview was unavailable — "
                    f"results generated with {FALLBACK_MODEL}."
                )
            st.session_state.last_result = result
            st.session_state.last_query = query_input.strip()

            # Store raw sources df (no categorization — computed at render time)
            if not result["sources"].empty:
                st.session_state.all_query_results.append({
                    "query": query_input.strip(),
                    "sources_df": result["sources"].copy(),
                })

        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            st.stop()

    # ---- Display results ---------------------------------------------------
    if st.session_state.last_result is None:
        st.info("Configure your target domain, API key, and query in the sidebar, then click Run Analysis.")
        return

    result = st.session_state.last_result
    query_label = st.session_state.last_query

    sub_df = result.get("subqueries", pd.DataFrame())
    src_df_raw = result.get("sources", pd.DataFrame())
    ent_df = result.get("entities", pd.DataFrame())
    raw_runs = result.get("raw_runs", [])

    # Render-time categorization — recomputes on every target_domain change
    if not src_df_raw.empty:
        src_df = pd.DataFrame(categorize_sources(src_df_raw.to_dict("records"), target_domain))
    else:
        src_df = src_df_raw

    cost = estimate_cost(raw_runs)
    sov  = calculate_target_sov(src_df, target_domain)

    # 6-metric scorecard
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Queries Processed", 1)
    m2.metric("Total Responses", len(raw_runs))
    m3.metric("Clusters Found", len(sub_df) if not sub_df.empty else 0)
    m4.metric("Sources Found", len(src_df) if not src_df.empty else 0)
    m5.metric("Entities Found", len(ent_df) if not ent_df.empty else 0)
    m6.metric("Est. Cost", f"${cost:.4f}")

    # Prominent Target SoV metric
    if target_domain:
        sov_delta = f"{sov['target_citations']} of {sov['total_citations']} citation slots"
        st.metric(
            label=f"\U0001f3af Target Share of Voice — {target_domain}",
            value=f"{sov['sov_pct']:.1f}%",
            delta=sov_delta,
            delta_color="off",
        )
    else:
        st.info("Enter a target domain in the sidebar to see Share of Voice.")

    st.divider()

    # ---- Tabs --------------------------------------------------------------
    tab1, tab2, tab3 = st.tabs(["\U0001f50d Subqueries", "\U0001f517 Sources", "\U0001f3f7️ Entities / Brands"])

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
            st.caption("\U0001f7e2 >75% probability &nbsp;&nbsp; \U0001f7e1 25-75% &nbsp;&nbsp; ⚪ <25%")

    # Tab 2: Sources
    with tab2:
        st.subheader("Cited Sources")
        caption_parts = ["URLs that Gemini cited while generating its answer."]
        if target_domain:
            caption_parts.append(f"\U0001f535 = Target ({target_domain}) · ⚪ = Other")
        st.caption("  ".join(caption_parts))

        if src_df.empty:
            st.warning("No sources captured.")
        else:
            display_cols = ["uri", "title", "domain", "category", "intl_flag", "probability", "avg_position", "rrf_score"]
            display_cols = [c for c in display_cols if c in src_df.columns]
            st.dataframe(
                style_sources_df(src_df[display_cols]),
                hide_index=True,
            )

            # Target vs Other breakdown chart
            if "category" in src_df.columns and "run_count" in src_df.columns:
                cat_counts = src_df.groupby("category")["run_count"].sum()
                if not cat_counts.empty:
                    st.subheader("Citation Distribution")
                    st.bar_chart(cat_counts)

            # International flag warning
            if "intl_flag" in src_df.columns:
                intl_rows = src_df[src_df["intl_flag"].astype(str) != ""]
                if not intl_rows.empty:
                    n_intl = len(intl_rows)
                    st.warning(
                        f"⚠️ {n_intl} international URL(s) detected in citation set for an English query. "
                        "This may indicate hreflang misconfiguration or international content competing with English pages."
                    )
                    with st.expander("View flagged international URLs"):
                        intl_display = intl_rows[["uri", "intl_flag", "category"]].copy()
                        intl_display["query"] = query_label
                        st.dataframe(intl_display, hide_index=True)

    # Tab 3: Entities
    with tab3:
        st.subheader("Brand & Entity Mentions")
        st.caption("Brands and entities mentioned in Gemini's response text.")
        if ent_df.empty:
            st.warning("No entities detected.")
        else:
            display_cols = ["entity", "probability", "avg_position", "rrf_score", "run_count", "sentiment"]
            display_cols = [c for c in display_cols if c in ent_df.columns]
            st.dataframe(
                style_entities_df(ent_df[display_cols], target_domain),
                hide_index=True,
            )
            if target_domain:
                st.caption(f"\U0001f535 Blue = entity name matches target ({target_domain}).")

    # ---- Session query summary (shows when 2+ queries run) -----------------
    all_results = st.session_state.all_query_results
    if len(all_results) >= 2:
        st.divider()
        st.subheader("\U0001f5c2️ Queries Analyzed This Session")
        st.caption(
            "Recomputed from the current target domain in the sidebar. "
            "Change the target above and these rows relabel instantly."
        )
        summary_rows = []
        for item in all_results:
            q_src = item["sources_df"]
            if q_src.empty:
                continue
            q_src_enriched = pd.DataFrame(categorize_sources(q_src.to_dict("records"), target_domain))
            q_sov = calculate_target_sov(q_src_enriched, target_domain)
            summary_rows.append({
                "Query":            item["query"],
                "Target SoV":       f"{q_sov['sov_pct']:.1f}%",
                "Target Citations": q_sov["target_citations"],
                "Total Citations":  q_sov["total_citations"],
            })
        if summary_rows:
            st.dataframe(pd.DataFrame(summary_rows), hide_index=True)

    # ---- Export & raw data -------------------------------------------------
    st.divider()
    export_col, _ = st.columns([2, 6])
    with export_col:
        result_for_export = dict(result)
        result_for_export["sources"] = src_df
        csv_bytes = build_csv_zip(result_for_export)
        safe_query = "".join(c if c.isalnum() or c in "-_ " else "_" for c in query_label)[:40]
        st.download_button(
            label="⬇️ Export CSVs (zip)",
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
