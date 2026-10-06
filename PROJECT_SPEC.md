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
- **Parent-Child:** `ParentDocumentRetriever` (langchain_classic). Implementation invariant:
  `##` headings in `inference_service_guide.md` are pre-split into logical parent `Document`
  objects using `MarkdownHeaderTextSplitter` *before* being passed to `ParentDocumentRetriever`.
  The retriever is configured with a `child_splitter` only (`parent_splitter=None`), so the
  pre-split `##` sections are stored as-is as parents. Vector search matches against child
  chunks; the retriever resolves each match's `parent_id` to the corresponding docstore entry
  and returns the full parent section, not the matched child alone. Do not pass
  `MarkdownHeaderTextSplitter` as the retriever's `parent_splitter` argument — pre-splitting
  happens before the retriever, not inside it.
  Target sizing: child chunks ~80–150 words, parent (`##`) sections ~250–400 words, so the
  parent is meaningfully richer than any single child.

All retrievers return `{content, score, source_doc_id, rank}`. **Do not normalize or compare
scores across retriever types.** BM25 scores, Chroma's raw distance, and any similarity
transform are on fundamentally different scales with no natural numeric equivalence — min-max
normalizing them into a shared 0–1 range creates a false impression of comparable "relevance"
(e.g. BM25's top result and Vector's top result both showing as "1.0" despite meaning nothing
alike). Instead:
- **Rank is the primary cross-retriever comparison dimension**, always present and always
  meaningful (1st, 2nd, 3rd place within that retriever's own result set).
- **Raw/internal scores are preserved and displayed as retriever-specific**, labeled clearly
  (e.g. "BM25 score: 8.24" vs. "Vector similarity: 0.82"), never rescaled to imply
  cross-retriever comparability. The UI states explicitly: "Scores are comparable only within
  the same retriever, not across retrievers."
- Chroma distinguishes distance-returning similarity search from a defined relevance-score
  transform (`relevance_score_fn`) — decide which one you're using per retriever and label
  accordingly; don't silently mix distance and similarity-score semantics.
- For retrievers where a single final score doesn't naturally exist for a given result
  (Multi-Query: unique union across several generated queries; Parent-Child: the match belongs
  to a child chunk, not the returned parent), display rank without inventing a synthetic score.
  For Parent-Child specifically, it's acceptable to surface the matched child's score as
  supporting context ("matched via child chunk, score X") alongside the returned parent content.

**MMR parameters are initial experiment parameters, not fixed constants.** Starting point:
`k=3, fetch_k=10, lambda_mult=0.5`. Given the corpus is only ~20 short docs, `fetch_k=10`
already considers a large share of the collection — during Phase 3 validation, test
`fetch_k ∈ {6, 10}` and `lambda_mult ∈ {0.3, 0.5, 0.7}` (not performed) to confirm the diversity
effect isn't an artifact of one arbitrary setting, not to run a full tuning study.

**Multi-Query k semantics:** the base retriever retrieves `k=3` per generated query variation,
then `MultiQueryRetriever` returns the unique union across all variations — this can exceed 3
documents total, and that's intentional; broader coverage across phrasings is the effect this
scenario is meant to demonstrate. Do not force the union down to exactly 3 early. The UI shows
the top 3 of the final unique set with an indicator of total unique docs retrieved (e.g.
"showing top 3 of 6 unique results"), rather than silently truncating. Set
`include_original=True` (LangChain's default is `False`) so the user's original query wording
is retrieved alongside the LLM-generated variations, giving a cleaner baseline.

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

Actual outcome: 11 curated questions (8 showcases, 3 findings) rather than 15-20; the planned MMR parameter sweep was not performed.

`corpus/questions.json` — list of `{question, scenario_type, expected_winner, explanation}`.
`expected_winner` used for validation (`just validate`), not shown to the user before they run
the query. `explanation` is hand-written, specific to the exact corpus documents involved —
never generic "this retriever is good at X" filler. Validate every question empirically
before finalizing its explanation; a question designed to show one retriever winning may not
actually produce that result until corpus or phrasing is adjusted. No LLM call at runtime for
explanations.

**`example_role`** — `"showcase"` or `"finding"`, set per question:
- `"showcase"` — the question is meant to demonstrate its `scenario_type`'s mechanism cleanly;
  `just validate` treats a non-PASS showcase question as needing iteration, same as before this
  field existed.
- `"finding"` — an empirically interesting *non-showcase* result that's kept intentionally: the
  observed retriever behavior is informative even though the intended retriever doesn't win
  cleanly (e.g. two retrievers agreeing on the same top result, or an exact technical term that
  doesn't actually produce a BM25 advantage). AMBIGUOUS — or any other non-PASS verdict — on a
  finding is not considered unresolved; `just validate` never lists a finding as needing
  iteration, and reports retained-findings counts separately from showcase pass/fail counts.

Curated questions should set `example_role` explicitly going forward. `corpus/validate_questions.py`
defaults a missing `example_role` to `"showcase"` only for backward compatibility with questions
written before this field existed — new questions shouldn't rely on that default.

**BM25-vs-Vector validation rule** (`corpus/validate_questions.py`'s `bm25_exact_term` and
`vector_paraphrase` checks): the verdict is decided by **rank-1 disagreement only**, checked
symmetrically for both retrievers against any id in the question's `expected_doc_ids`:
- **PASS** — the intended retriever ranks an expected doc at #1, and the comparison retriever
  does not rank any expected doc at #1.
- **AMBIGUOUS** — both retrievers rank an expected doc at #1 (no disagreement to demonstrate).
- **FAIL** — the intended retriever does not rank an expected doc at #1, regardless of whether
  the comparison retriever does.

An earlier version of this check used a `>=2`-rank gap between the two retrievers' best ranks
as a proxy for "clean enough contrast to demonstrate." That threshold was an arbitrary choice
made during tooling design, with no retrieval-theoretic basis — and it could report AMBIGUOUS
even when the intended retriever's top result was already correct and the comparison
retriever's was not (e.g. intended=1, comparison=2 counted as too thin a margin). The rank-1
rule instead tests the actual phenomenon each question is meant to demonstrate directly: which
retriever puts the correct document first. Each retriever's top-3 and the best rank achieved by
any expected doc are still printed as diagnostics either way — only the verdict logic changed.

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