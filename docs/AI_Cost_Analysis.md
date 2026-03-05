# LegacyLens — AI Cost Analysis

**Project:** LegacyLens — RAG System for Legacy Enterprise Codebases  
**Reporting period:** 2026-02-02 to 2026-03-05 (development & testing)  
**Document purpose:** Development spend breakdown and production cost projections per G4-Week-3-LegacyLens requirements.

---

## 1. Development & Testing Costs

Costs are derived from exported usage data and the Railway usage dashboard for the period above.

### 1.1 Embedding API (Voyage Code 2)

**Source:** `project-activity-20260202-20260304.csv`

| Metric | Value |
|--------|--------|
| Total text tokens (tracked) | 760 |
| Number of requests | 24 |
| Date | 2026-03-03 |

**Pricing (Voyage Code 2):** $0.12 per 1M tokens; first **50 million tokens free** per account ([Voyage pricing](https://docs.voyageai.com/docs/pricing)).

**Embedding cost (tracked):** **$0.00** — 760 tokens is well within the 50M free tier.

**Note:** The CSV reflects query-time embedding usage (24 requests). One-time **ingestion** of the current codebase (16,991 chunks, ~82 tokens/chunk average) is ~**1.39M tokens** per full run. If ingestion was run under the same Voyage account, it is still within the 50M free tier, so embedding cost during development remains **$0.00**.

---

### 1.2 LLM API (OpenAI GPT-4o-mini)

**Source:** `completions_usage_2026-02-02_2026-03-04.csv`

| Period | Requests | Input tokens | Cached input | Output tokens |
|--------|----------|--------------|-------------|---------------|
| 2026-03-03 → 2026-03-04 | 13 | 84,629 | 0 | 5,304 |
| 2026-03-04 → 2026-03-05 | 361 | 2,305,526 | 204,544 | 79,060 |
| **Total** | **374** | **2,390,155** | **204,544** | **84,364** |

**Pricing (GPT-4o-mini):** Input $0.15/1M tokens; cached input $0.075/1M; output $0.60/1M ([OpenAI pricing](https://developers.openai.com/api/docs/pricing)).

| Component | Calculation | Cost |
|-----------|-------------|------|
| Input (uncached) | (2,390,155 − 204,544) × $0.15/1M | $0.328 |
| Input (cached) | 204,544 × $0.075/1M | $0.015 |
| Output | 84,364 × $0.60/1M | $0.051 |
| **LLM total** | | **~$0.39** |

---

### 1.3 Vector Database / Hosting (Railway)

**Source:** Railway Usage dashboard (Trial Workspace, Feb 24 – Mar 5, 2026).

| Metric | Value |
|--------|--------|
| Current usage (period) | **$1.10** |
| Credits available | $3.97 |
| Trial plan | $5.00 one-time credits; 1 GB RAM, 2 vCPU, 1 GB Disk |
| Estimated month’s cost | $0.00 (usage covered by credits) |

ChromaDB runs in-process with the app; persistence uses Railway volume (1 GB Disk). No separate vector-DB SaaS fee. The **$1.10** is the **hosting/compute cost** for the deployed service (and any other projects in the workspace, e.g. “athletic_luck” in the screenshot).

**Vector DB / hosting cost (attributable to LegacyLens):** **$1.10** (if the full period usage is attributed to this project; otherwise a portion of $1.10).

---

### 1.4 Total Development Spend Breakdown

| Category | Amount |
|----------|--------|
| Embedding (Voyage Code 2) | $0.00 |
| LLM (OpenAI GPT-4o-mini) | ~$0.39 |
| Hosting / vector DB (Railway) | $1.10 |
| **Total (period)** | **~$1.49** |

If Railway usage is shared across projects, total LegacyLens dev cost is **at least ~$0.39** (LLM) plus a share of **$1.10** (hosting).

---

## 2. Production Cost Projections

Monthly cost estimates at four user scales. All figures are in **USD/month**.

### 2.1 Assumptions

| Parameter | Value | Notes |
|-----------|--------|--------|
| Queries per user per day | 5 | Active developer use. |
| Working days per month | 22 | ~22 weekdays. |
| Queries per user per month | 110 | 5 × 22. |
| Share of queries that call the LLM | 85% | 15% fast-path “not found” (no LLM call). |
| Avg input tokens per LLM request | 3,500 | System prompt (~1,600) + user query + assembled context (capped ~12K chars ≈ ~3K tokens). |
| Avg output tokens per LLM request | 200 | Short answers with citations. |
| Embedding (Voyage) per query | 1 request | Query embedding only; index built once. |
| Avg query length (tokens) | 30 | For embedding cost. |
| Ingestion | Once per codebase / major update | Neglected in monthly query-driven cost. |
| Vector DB (Chroma on Railway) | $5–20/mo | Small instance; scale with disk/RAM. |

**Pricing (unchanged):**

- Voyage Code 2: $0.12/1M tokens (after 50M free).
- GPT-4o-mini: input $0.15/1M, output $0.60/1M.

### 2.2 Per-query cost (recurring)

| Component | Formula | Cost per query |
|------------|---------|----------------|
| Query embedding | 30 tokens × $0.12/1M | ~$0.000 003 6 |
| LLM (85% of queries) | 0.85 × (3,500 × $0.15/1M + 200 × $0.60/1M) | ~$0.000 59 |
| **Total per query** | | **~$0.000 59** |

### 2.3 Monthly cost by user scale

| Users | Queries/month | Embedding cost | LLM cost | Hosting (est.) | **Total/month** |
|-------|----------------|----------------|----------|----------------|------------------|
| 100 | 11,000 | &lt;$1 | ~\$6.50 | $5–10 | **~\$12–18** |
| 1,000 | 110,000 | ~\$4 | ~\$65 | $10–25 | **~\$79–94** |
| 10,000 | 1,100,000 | ~\$40 | ~\$650 | $50–150 | **~\$740–840** |
| 100,000 | 11,000,000 | ~\$400 | ~\$6,500 | $200–600 | **~\$7,100–7,500** |

**Formulas:**

- **Embedding:** Queries × 30 × $0.12/1e6 (after 50M free tier; 100–1K users may still be under free tier).
- **LLM:** Queries × 0.85 × (3,500 × 0.15/1e6 + 200 × 0.60/1e6) ≈ Queries × $0.000 59.
- **Hosting:** Rough range for a single LegacyLens deployment (Chroma + app); scales with traffic and storage.

### 2.4 Additional considerations

- **Ingestion:** Full re-embedding of the codebase (~1.4M tokens) is ~\$0.17 per run at Voyage pricing; typically monthly or per major release. Not included in the table above.
- **Caching:** Higher reuse of cached input (e.g. repeated similar queries) would reduce GPT-4o-mini input cost; the table assumes no caching.
- **Vector DB at scale:** Chroma on a single Railway instance is fine up to low thousands of users; 10K–100K users may require a dedicated DB or managed vector store, increasing the hosting range.

---

## 3. Summary

- **Development (Feb 24 – Mar 5):** Embedding **$0**, LLM **~$0.39**, Railway **$1.10** → **~\$1.49 total**.
- **Production:** At 5 queries/user/day and 85% LLM usage, monthly cost grows from **~\$12–18** (100 users) to **~\$7,100–7,500** (100K users), dominated by LLM cost. Embedding stays small; hosting scales with deployment choice.
