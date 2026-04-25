# Query Fan-Out Analysis Tool

An SEO/AEO proof-of-concept that reveals how Gemini expands your query into subqueries when generating AI-powered search answers — and which sources it actually cites. Point it at any target domain to measure LLM share of voice for that domain.

---

## What It Does

1. Takes a search query (e.g. "best email marketing automation software")
2. Sends it to Gemini with Google Search grounding enabled — N times
3. Captures from each response:
   - The **subqueries** Gemini searched for (query fan-out)
   - The **source URLs** Gemini cited
   - The **brands/entities** mentioned in the answer
4. Aggregates across all runs: clusters similar subqueries, calculates probability scores and RRF rankings
5. Labels every cited URL as **Target** (on your target domain) or **Other**, and computes Target Share of Voice
6. Optional content-gap analysis: fetches the top-cited non-target page vs. your top-cited target page, compares them with embeddings against Gemini's answer to surface on-page opportunities

The target domain is a sidebar input — change it any time and past runs relabel instantly from the new perspective, no re-crawl needed.

---

## Prerequisites

- Python 3.9+
- A Gemini API key from [Google AI Studio](https://aistudio.google.com)
- **Google Search grounding requires a billing-enabled Google Cloud project.** Free-tier API keys from AI Studio may not return grounding metadata for all queries. If `web_search_queries` comes back empty, check your API tier.

---

## Setup

```bash
# 1. Clone / navigate to this folder
cd fanout-tool

# 2. (Recommended) Create a virtual environment
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Quick Engine Test

Validates the API connection and grounding metadata before touching the UI.

```bash
export GEMINI_API_KEY=your_key_here
python fanout_engine.py --single
```

Expected output: a printed dump of one Gemini call's `web_search_queries`, grounding sources, entities, and a snippet of the response text.

If `web_search_queries` is empty, see the [Grounding Not Returning Data](#grounding-not-returning-data) section below.

---

## Multi-query CLI (no UI)

Run several queries and write CSVs — no Streamlit needed.

```bash
export GEMINI_API_KEY=your_key_here
export TARGET_DOMAIN=example.com   # optional; leave unset to get SoV of 0
python fanout_engine.py
```

Or programmatically:

```python
from fanout_engine import run_analysis, save_to_csv

result = run_analysis(
    query="best CRM for small business",
    api_key="your_key_here",
    num_runs=8,
)

save_to_csv(
    result,
    query="best CRM for small business",
    output_dir="./output",
    prefix="q1_",
    target_domain="example.com",
)
```

---

## Streamlit UI

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

**How to use:**
1. Enter your **target domain** in the sidebar (e.g. `example.com`) — subdomains count, case-insensitive.
2. Paste your Gemini API key.
3. Set number of runs (default 8 — each run = one Gemini call).
4. Type a query and click **Run Analysis**.
5. Explore the three tabs: Subqueries, Sources, Entities/Brands.
6. Run additional queries in the same session; change the target domain any time and every past run relabels.
7. Export results as a ZIP of CSVs.

There's also a historical report view (`streamlit run report.py`) that loads exported CSVs from prior runs and adds a content-gap/embedding analysis tab comparing top-cited pages against Gemini's answer.

---

## Rate Limits

| Tier | RPM limit | Recommended max runs |
|------|-----------|----------------------|
| Free (AI Studio) | 15 RPM | 8 runs |
| Pay-as-you-go | 1000+ RPM | 20 runs |

The tool adds a 2-second delay between calls and retries on 429 errors with exponential backoff (2s → 4s → 8s). If you hit rate limits frequently, reduce `num_runs` to 6-8.

---

## Output Column Reference

### Subqueries tab
| Column | Meaning |
|--------|---------|
| `subquery` | The cluster label (most representative variant) |
| `probability` | % of runs this subquery cluster appeared in |
| `avg_position` | Average position in the fan-out list when it appeared |
| `rrf_score` | Combined probability + position priority score (0-100) |
| `run_count` | Raw count of runs it appeared in |
| `sample_variants` | Up to 3 raw subquery strings that map to this cluster |

### Sources tab
| Column | Meaning |
|--------|---------|
| `uri` | Full source URL |
| `domain` | Bare domain (e.g. `example.com`) |
| `category` | `Target` or `Other` relative to the current target domain |
| `intl_flag` | ⚠️ + language code if non-English URL detected |
| `probability` | % of runs this URL was cited |
| `rrf_score` | Priority score (0-100) |

### Entities tab
| Column | Meaning |
|--------|---------|
| `entity` | Brand or proper noun detected in response text |
| `probability` | % of runs it was mentioned |
| `sentiment` | `positive`, `negative`, or `neutral` based on co-occurring language |

---

## Categorization

v1 uses a simple **Target vs Other** scheme: any URL whose host matches the target domain (or a subdomain of it) is tagged `Target`; everything else is `Other`. Categorization is computed at render time, so changing the target domain in the sidebar instantly relabels all past runs in the current session.

Richer taxonomies (auto-detected `/blog/`, `/docs/`, `/help/` patterns; opt-in YAML rules; a competitor list) are on the roadmap — open an issue if you'd like to see these prioritized.

---

## Troubleshooting

### Grounding Not Returning Data

**Symptom:** `web_search_queries` and `sources` are empty on every run.

**Causes and fixes:**
1. **API tier**: Google Search grounding may require a billing-enabled GCP project. Check [AI Studio quotas](https://aistudio.google.com) vs. your API key's project.
2. **Query too simple**: Gemini may not trigger grounding for questions it can answer from training data alone. Use multi-word commercial queries.
3. **Model string**: The tool defaults to `gemini-2.5-flash-lite` with optional `gemini-3.1-flash-lite-preview` — confirm the selected model is available on your API key's project.

### `ModuleNotFoundError` on any package

Make sure you installed dependencies into the same Python that runs Streamlit:

```bash
which streamlit
which python3          # should be in the same prefix

pip3 install -r requirements.txt
```

If you're using a virtual environment, activate it first, then install.

### `ResourceExhausted` / 429 errors

Lower `num_runs` to 6 or add `inter_call_delay=3.0` when calling `run_analysis()` directly.

---

## File Structure

```
fanout-tool/
├── fanout_engine.py      # Core: API calls, clustering, RRF scoring, aggregation, SoV
├── target_categorizer.py # URL → Target/Other classifier + international flag
├── citation_mapper.py    # Page fetch + chunking + embedding-based gap detection
├── app.py                # Streamlit UI (per-run analysis)
├── report.py             # Streamlit UI (historical / cross-query report)
├── db.py                 # Optional Supabase persistence
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

---

## Persistence

By default, state is kept in the current session only. Export results as CSVs from the sidebar and load them in `report.py` to do cross-session or cross-query analysis.

A local-first SQLite adapter is on the roadmap so cross-session persistence works out of the box without any external service.
