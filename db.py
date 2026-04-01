"""
db.py — Supabase persistence layer for Query Fan-Out Analysis.

Wraps supabase-py with the four operations report.py needs:
  - get_client()          → supabase Client or None
  - save_run_to_db()      → persist a completed run_analysis() result
  - load_all_queries()    → list of {db_id, prefix, query_text, ...} for sidebar/overview
  - load_query_detail()   → {scorecard, sources_df, subqueries_df, entities_df, responses_df}

Graceful degradation: every function returns None / empty structures if
SUPABASE_URL or SUPABASE_KEY are not set. report.py falls back to CSVs.
"""

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

def get_client():
    """Returns a supabase Client if credentials are configured, else None."""
    try:
        import streamlit as st
        url = st.secrets.get("SUPABASE_URL") or ""
        key = st.secrets.get("SUPABASE_KEY") or ""
    except Exception:
        url = ""
        key = ""

    import os
    url = url or os.environ.get("SUPABASE_URL", "")
    key = key or os.environ.get("SUPABASE_KEY", "")

    if not url or not key:
        return None

    try:
        from supabase import create_client
        return create_client(url, key)
    except Exception as exc:
        logger.warning(f"Supabase client init failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def save_run_to_db(
    result: dict,
    query: str,
    model_used: str,
    num_runs: int,
) -> Optional[int]:
    """
    Persists a complete run_analysis() result to Supabase.
    Returns the new query_id (int) on success, None on failure.
    """
    client = get_client()
    if client is None:
        return None

    try:
        from fanout_engine import estimate_cost, calculate_ac_sov
        raw_runs = result.get("raw_runs", [])
        cost = estimate_cost(raw_runs)
        src_df = result.get("sources", pd.DataFrame())
        sov = calculate_ac_sov(src_df)

        # 1. Insert into queries
        q_resp = (
            client.table("queries")
            .insert({
                "query_text": query,
                "model_used": model_used,
                "num_runs":   num_runs,
                "total_cost": float(cost),
                "ac_sov_pct": float(sov.get("sov_pct", 0)),
                "ac_citations": int(sov.get("ac_citations", 0)),
                "total_citations": int(sov.get("total_citations", 0)),
            })
            .execute()
        )
        query_id = q_resp.data[0]["id"]

        # 2. Insert runs (response_text per run)
        runs_rows = [
            {
                "query_id":      query_id,
                "run_index":     r.get("run_index", i),
                "response_text": r.get("response_text", ""),
                "input_tokens":  int(r.get("input_tokens", 0)),
                "output_tokens": int(r.get("output_tokens", 0)),
            }
            for i, r in enumerate(raw_runs)
        ]
        if runs_rows:
            client.table("runs").insert(runs_rows).execute()

        # 3. Insert sources
        if not src_df.empty:
            sources_rows = []
            for _, row in src_df.iterrows():
                sources_rows.append({
                    "query_id":    query_id,
                    "uri":         str(row.get("uri", "")),
                    "title":       str(row.get("title", "")),
                    "domain":      str(row.get("domain", "")),
                    "category":    str(row.get("category", "")),
                    "run_count":   int(row.get("run_count", 0)),
                    "probability": float(row.get("probability", 0)),
                    "avg_position": float(row.get("avg_position", 0)),
                    "rrf_score":   float(row.get("rrf_score", 0)),
                    "intl_flag":   str(row.get("intl_flag", "")),
                })
            client.table("sources").insert(sources_rows).execute()

        # 4. Insert subqueries
        sub_df = result.get("subqueries", pd.DataFrame())
        if not sub_df.empty:
            sub_rows = []
            for _, row in sub_df.iterrows():
                sub_rows.append({
                    "query_id":       query_id,
                    "subquery":       str(row.get("subquery", "")),
                    "probability":    float(row.get("probability", 0)),
                    "avg_position":   float(row.get("avg_position", 0)),
                    "rrf_score":      float(row.get("rrf_score", 0)),
                    "run_count":      int(row.get("run_count", 0)),
                    "sample_variants": str(row.get("sample_variants", "")),
                    "top_sources":    str(row.get("top_sources", "")),
                })
            client.table("subqueries").insert(sub_rows).execute()

        # 5. Insert entities
        ent_df = result.get("entities", pd.DataFrame())
        if not ent_df.empty:
            ent_rows = []
            for _, row in ent_df.iterrows():
                ent_rows.append({
                    "query_id":    query_id,
                    "entity":      str(row.get("entity", "")),
                    "probability": float(row.get("probability", 0)),
                    "avg_position": float(row.get("avg_position", 0)),
                    "rrf_score":   float(row.get("rrf_score", 0)),
                    "run_count":   int(row.get("run_count", 0)),
                })
            client.table("entities").insert(ent_rows).execute()

        logger.info(f"Saved query_id={query_id} to Supabase: {query!r}")
        return query_id

    except Exception as exc:
        logger.warning(f"save_run_to_db failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def load_all_queries() -> list[dict]:
    """
    Returns all queries from Supabase ordered by created_at desc.
    Each dict has keys: db_id, query_text, model_used, num_runs, total_cost,
    ac_sov_pct, ac_citations, total_citations, created_at.
    Returns [] if no client or on error.
    """
    client = get_client()
    if client is None:
        return []
    try:
        resp = (
            client.table("queries")
            .select("id, query_text, model_used, num_runs, total_cost, ac_sov_pct, ac_citations, total_citations, created_at")
            .order("created_at", desc=True)
            .execute()
        )
        return [
            {
                "db_id":           row["id"],
                "query_text":      row["query_text"],
                "model_used":      row.get("model_used", ""),
                "num_runs":        row.get("num_runs", 0),
                "total_cost":      row.get("total_cost", 0),
                "ac_sov_pct":      row.get("ac_sov_pct", 0),
                "ac_citations":    row.get("ac_citations", 0),
                "total_citations": row.get("total_citations", 0),
                "created_at":      row.get("created_at", ""),
            }
            for row in (resp.data or [])
        ]
    except Exception as exc:
        logger.warning(f"load_all_queries failed: {exc}")
        return []


def load_query_detail(db_id: int) -> dict:
    """
    Loads full detail for one query from Supabase.
    Returns dict with keys: scorecard (pd.DataFrame), sources_df, subqueries_df,
    entities_df, responses_df. All DataFrames may be empty on error.
    """
    empty = {
        "scorecard":    pd.DataFrame(),
        "sources":      pd.DataFrame(),
        "subqueries":   pd.DataFrame(),
        "entities":     pd.DataFrame(),
        "responses":    pd.DataFrame(),
    }
    client = get_client()
    if client is None:
        return empty

    try:
        # Query metadata
        q_resp = client.table("queries").select("*").eq("id", db_id).execute()
        if not q_resp.data:
            return empty
        q = q_resp.data[0]

        # Build scorecard DataFrame matching the existing format
        scorecard_rows = [
            ("Query",               q["query_text"]),
            ("Queries Processed",   1),
            ("Total Responses",     q.get("num_runs", 0)),
            ("Clusters Found",      0),   # updated below
            ("Sources Found Total", 0),   # updated below
            ("Entities Found",      0),   # updated below
            ("Estimated Cost",      f"${float(q.get('total_cost', 0)):.4f}"),
            ("AC SoV (Citation %)", f"{float(q.get('ac_sov_pct', 0)):.1f}%"),
            ("AC Citations",        int(q.get("ac_citations", 0))),
            ("Total Citations",     int(q.get("total_citations", 0))),
        ]

        # Sources
        src_resp = client.table("sources").select("*").eq("query_id", db_id).execute()
        src_df = pd.DataFrame(src_resp.data or []).drop(columns=["id", "query_id", "created_at"], errors="ignore")

        # Subqueries
        sub_resp = client.table("subqueries").select("*").eq("query_id", db_id).execute()
        sub_df = pd.DataFrame(sub_resp.data or []).drop(columns=["id", "query_id", "created_at"], errors="ignore")

        # Entities
        ent_resp = client.table("entities").select("*").eq("query_id", db_id).execute()
        ent_df = pd.DataFrame(ent_resp.data or []).drop(columns=["id", "query_id", "created_at"], errors="ignore")

        # Runs (for Citation Mapper — response_text)
        run_resp = client.table("runs").select("run_index, response_text").eq("query_id", db_id).execute()
        responses_df = pd.DataFrame(run_resp.data or [])

        # Patch scorecard counts now that we have the DFs
        scorecard_rows[3] = ("Clusters Found",      len(sub_df))
        scorecard_rows[4] = ("Sources Found Total",  len(src_df))
        scorecard_rows[5] = ("Entities Found",       len(ent_df))
        scorecard_df = pd.DataFrame(scorecard_rows, columns=["metric", "value"])

        return {
            "scorecard":  scorecard_df,
            "sources":    src_df,
            "subqueries": sub_df,
            "entities":   ent_df,
            "responses":  responses_df,
        }

    except Exception as exc:
        logger.warning(f"load_query_detail failed for db_id={db_id}: {exc}")
        return empty
