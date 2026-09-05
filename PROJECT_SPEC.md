# Project Spec: RAG Failure Explorer

## Thesis

Retrieval isn't one-size-fits-all. This project demonstrates *when and why* different
retrieval strategies succeed or fail, rather than presenting a single "best" RAG pipeline.
The user selects a curated question, sees results from 5 retrievers side by side, and reads
a concrete, mechanistic explanation of why each retriever performed the way it did.

Interview pitch: "I built a small experiment to understand when different retrieval
strategies fail, rather than treating vector search as universally best."

## Goals / Non-Goals

**Goals:** working comparison harness across 5 retrievers; 15–20 curated questions, each
cleanly illustrating one retriever's strength or weakness; mechanistic, question-specific
explanations; a working demoable Gradio UI.

**Non-goals for v1:** not a production RAG chatbot; no LLM-generated final answer (retrieval
comparison is the deliverable); no custom parsers/scrapers/auth; no multi-corpus or
file-upload support.

## The Five Scenarios

| # | Scenario | Winner | Why |
|---|---|---|---|
| 1 | Exact keyword / technical term | BM25 | Vector search dilutes exact matches with related-but-different concepts; BM25 rewards exact term overlap |
| 2 | Paraphrased query, no shared keywords | Vector | BM25 fails with no lexical overlap; vector search captures meaning regardless of wording |
| 3 | Ambiguous / under-specified query | Multi-Query | A single formulation may miss relevant phrasings; query variations increase match likelihood |
| 4 | Near-duplicate/redundant top results | MMR | Plain similarity returns near-identical chunks; MMR re-ranks for diversity |
| 5 | Right chunk too small for context | Parent-Child | Small chunks win similarity but lack context; merging to parent restores it |

3–4 questions per scenario, 15–20 total.

## Corpus Design

**No fictional company branding.** Content is written as generic internal platform
documentation — "the platform," "the service" — not a named fictional company.

**Format:** hybrid.
- `corpus/corpus.json` — list of `{id, text, topic_tag, scenario_hint}` for ~25–29 short
  documents (3–6 sentences each)
- `corpus/long_docs/inference_service_guide.md` — the one long, hierarchically structured
  doc (real markdown headers) used for the Parent-Child scenario, so LangChain's
  markdown-aware splitter can split on genuine header structure

**Topic taxonomy (6 categories), mapped to scenarios:**

| Topic | Scenario it serves | Design notes |
|---|---|---|
| Retrieval (vector search, BM25, hybrid, HNSW config, reranking) | BM25 (exact term) | Pairs like "HNSW `ef_construction` parameter" vs. a paraphrase ("graph build-quality tuning setting") |
| Security (API keys, OAuth, JWT, service accounts) | Vector (paraphrase) | JWT expiration doc phrased so a natural question shares no keywords with the source text |
| Monitoring + Deployment (latency alerts, error rates, drift, canary/rollback) | Multi-Query (ambiguous) | "how do we know something's wrong" split across differently-worded docs on distinct-but-related failure signals |
| Monitoring + Security (rate limiting appearing in multiple contexts) | MMR (near-duplicate) | Same underlying fact (rate limiting) restated 3–4 times across different sections/angles |
| Ingestion (document upload, preprocessing, chunking) | Parent-Child (hierarchy) | One long doc, `inference_service_guide.md`, where a small chunk on one step doesn't explain the pipeline it sits in |
| Model serving / inference (batching, endpoints, timeouts) | supporting content, mixed use | General realism + additional BM25/paraphrase pairs as needed |

`scenario_hint` values: `bm25_exact_term`, `vector_paraphrase`, `multi_query_ambiguous`,
`mmr_near_duplicate`, `parent_child_hierarchy`.

## Retriever Setup

- **BM25:** `BM25Retriever` (langchain_community) over tokenized corpus
- **Vector:** standard similarity search over embedded chunks
- **Multi-Query:** `MultiQueryRetriever.from_llm()` (langchain_classic) wrapping the vector retriever
- **MMR:** vector store's `as_retriever(search_type="mmr")`
- **Parent-Child:** `ParentDocumentRetriever` (langchain_classic), large parent / small child splitter

All retrievers return `{content, score, source_doc_id}`. BM25 scores and vector distances
are on different scales — normalize before displaying side by side (don't assume raw
comparability).

**Stack:** Python 3.12, uv + pyproject.toml (exact-pinned versions), LangChain
(langchain_classic, langchain_community, langchain-chroma), OpenAI (`gpt-4o-mini` for
Multi-Query only, `text-embedding-3-small` for embeddings), Chroma (in-memory) vector store
behind a thin interface, Gradio UI, `justfile` task runner.

## Architecture

```
corpus/corpus.json + corpus/long_docs/*.md
        │
        ▼
Indexing at startup: chunk+embed, tokenize for BM25, parent/child split
        │
        ▼
User selects/types a question
        │
        ▼
Run through all 5 retrievers → normalize scores
        │
        ▼
Display side-by-side grid (ranked chunks + scores per retriever)
        │
        ▼
Show pre-written, question-specific explanation
```

## Question Set

`corpus/questions.json` — list of `{question, scenario_type, expected_winner, explanation}`.
`expected_winner` used for validation (`just validate`), not shown to the user before they run
the query. `explanation` is hand-written, specific to the exact corpus documents involved —
never generic "this retriever is good at X" filler. Validate every question empirically
before finalizing its explanation; a question designed to show one retriever winning may not
actually produce that result until corpus or phrasing is adjusted. No LLM call at runtime for
explanations.

## UI (Gradio)

Dropdown of curated questions + optional free-text box (labeled "unscripted, results may
vary"). On selection: grid with one column per retriever, top 1–3 results with scores.
Below: scenario type + explanation. Bold/highlight the expected-winner column.

## Build Order & Time Budget (~16–22h with Claude Code)

1. Corpus design (docs + long parent-child doc) — 1.5–2h
2. Retriever infrastructure + score normalization — 2.5–3.5h
3. Question curation & empirical validation — 4–6h (biggest variable)
4. Explanation templates — 1–1.5h
5. Gradio UI — 1.5–2.5h
6. README, architecture diagram, polish — 1.5–2h

## Stretch Goals (post-v1, README only, don't build now)

- Self-Query / Query Fusion (RRF) as additional columns
- Optional final-answer synthesis per retriever
- User-uploaded corpus + auto-suggested scenario type
- Per-retriever latency tracking
