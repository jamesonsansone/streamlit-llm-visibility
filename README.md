# Query Fan-Out Analysis Tool

An SEO/AEO proof-of-concept that reveals how Gemini expands your query into subqueries when generating AI-powered search answers — and which sources it actually cites. Built specifically for analyzing ActiveCampaign's content footprint in AI search.

---

## What It Does

1. Takes a search query (e.g. "best email marketing automation software")
2. Sends it to Gemini with Google Search grounding enabled — N times
3. Captures from each response:
   - The **subqueries** Gemini searched for (query fan-out)
   - The **source URLs** Gemini cited
   - The **brands/entities** mentioned in the answer
4. Aggregates across all runs: clusters similar subqueries, calculates probability scores and RRF rankings
5. Categorizes every cited URL against ActiveCampaign's content taxonomy (Blog, Help Center, Compare pages, etc.)
6. Shows a **heatmap** of which AC properties appear for which query types — and where the gaps are

---

## Prerequisites

- Python 3.11+ (3.12 recommended)
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

## Milestone 1 — Quick Engine Test

Validates the API connection and grounding metadata before touching the UI.

```bash
export GEMINI_API_KEY=your_key_here
python fanout_engine.py
```

Expected output:
```
Query: 'best email marketing automation software'
Calling Gemini with Google Search grounding...

============================================================
WEB SEARCH QUERIES (Gemini's fan-out queries)
============================================================
  1. best email marketing automation tools 2024
  2. email marketing platform comparison small business
  3. ...

============================================================
GROUNDING SOURCES (URLs Gemini cited)
============================================================
  1. [ActiveCampaign Blog] https://activecampaign.com/blog/...
  2. ...

--- Milestone 1 complete. API integration works. ---
```

If `web_search_queries` is empty, see the [Grounding Not Returning Data](#grounding-not-returning-data) section below.

---

## Milestone 2 — CSV Export (No UI)

Run a multi-query analysis and write results to CSV — no Streamlit needed.

```python
from fanout_engine import run_analysis, save_to_csv

result = run_analysis(
    query="ActiveCampaign vs Klaviyo",
    api_key="your_key_here",
    num_runs=8,
)

paths = save_to_csv(result, output_dir="./output")
print(paths)
# {'subqueries': './output/top_subqueries.csv',
#  'sources':    './output/top_sources.csv',
#  'entities':   './output/top_entities.csv'}
```

The CSV files are enough to demo without the UI — open them in Excel/Sheets and walk through the findings.

---

## Milestone 3 — Full Streamlit UI

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

**How to use:**
1. Paste your Gemini API key in the sidebar (it's stored only in your browser session)
2. Pick an example query from the dropdown or type your own
3. Set number of runs (default 8 — each run = one Gemini call)
4. Click **Run Analysis**
5. Explore the three tabs: Subqueries, Sources, Entities/Brands
6. Run multiple queries in the same session — after 2+ queries, the **AC Property Heatmap** appears showing citation patterns across query types
7. Export results as a ZIP of CSVs

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
| `domain` | Bare domain (e.g. `activecampaign.com`) |
| `category` | AC taxonomy classification (see below) |
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

## AC Source Category Taxonomy

Categories are assigned to URLs in priority order (first match wins):

| Category | Pattern |
|----------|---------|
| `AC: Help Center` | `help.activecampaign.com` |
| `AC: Community` | `community.activecampaign.com` |
| `AC: International ⚠️` | AC URLs with `/de/`, `/fr/`, `/es/`, etc. |
| `AC: Blog` | `activecampaign.com/blog/` |
| `AC: Glossary` | `activecampaign.com/glossary/` or `/marketing-glossary/` |
| `AC: Recipes` | `activecampaign.com/recipes/` |
| `AC: Compare` | `activecampaign.com/compare/` |
| `AC: Solutions` | `activecampaign.com/solutions/` |
| `AC: Learn` | `activecampaign.com/learn/` |
| `AC: Case Studies` | `activecampaign.com/customers/` |
| `AC: Main Domain` | Any other `activecampaign.com` URL |
| `Competitor` | mailchimp, klaviyo, hubspot, brevo, constantcontact, getresponse, etc. |
| `3rd Party: Review Site` | g2, capterra, trustradius, getapp |
| `3rd Party: Publication` | techradar, pcmag, forbes, cnet, techcrunch |
| `3rd Party: Comparison` | zapier, emailtooltester, emailvendorselection |
| `3rd Party: Reddit` | reddit.com |
| `3rd Party: YouTube` | youtube.com / youtu.be |
| `3rd Party: Wikipedia` | wikipedia.org |
| `Other` | Everything else |

---

## Troubleshooting

### Grounding Not Returning Data

**Symptom:** `web_search_queries` and `sources` are empty on every run.

**Causes and fixes:**
1. **API tier**: Google Search grounding may require a billing-enabled GCP project. Check [AI Studio quotas](https://aistudio.google.com) vs. your API key's project.
2. **Query too simple**: Gemini may not trigger grounding for questions it can answer from training data alone. Use multi-word commercial queries like those in the example dropdown.
3. **Model string**: The tool uses `gemini-2.5-flash-preview-04-17` — confirm this model is available on your API key's project. `gemini-2.0-flash` was deprecated for new users in early 2026.

### `ModuleNotFoundError` on any package

Make sure you installed dependencies into the same Python that runs Streamlit:

```bash
which streamlit        # note the path, e.g. /Library/Frameworks/Python.framework/...
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
├── fanout_engine.py   # Core: API calls, clustering, RRF scoring, aggregation
├── ac_categorizer.py  # URL → AC property / competitor / 3rd party classifier
├── app.py             # Streamlit UI
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

---

## The Conversation This Enables

> "I ran 12 ActiveCampaign money queries through Gemini with real search grounding. For each query, I captured which sources the AI actually cited — not what we think it should cite, but what it actually pulled."

Each cell in the heatmap is a conversation starter:
- **Help Center dominates CRM queries but Blog dominates email marketing** → content architecture insight
- **Autonomous Marketing queries show almost no AC presence** → content gap
- **International URLs appearing for English queries** → hreflang technical SEO flag
- **Reddit or G2 appearing more than AC's own pages** → third-party brand signal opportunity
