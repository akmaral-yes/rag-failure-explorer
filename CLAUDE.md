# Claude Code instructions — rag-failure-explorer

See `PROJECT_SPEC.md` for the full project design, corpus rules, and build order.
This file is tooling conventions only — read it fully before writing or running any code.

## Dependency management
- **uv + pyproject.toml only.** Never generate or reference `requirements.txt`.
- Add dependencies with `uv add <package>==<exact_version>` — always pin exact versions.
- Never use bare `pip install`. Never use `>=` or unpinned versions in `pyproject.toml`.
- Dev dependencies go in `[dependency-groups] dev = [...]`, added via `uv add --dev`.
- Run everything through `uv run ...` or the `justfile` recipes below — never assume an activated venv.

## Task runner
Common commands live in the `justfile` at repo root. Use these instead of raw commands:
- `just install` — install all dependencies
- `just run` — launch the Gradio app
- `just smoke` — run the retriever smoke test (Phase 2 sanity check)
- `just validate` — validate all curated questions against `expected_winner`
- `just test` — run pytest
- `just lint` / `just format` — ruff
- `just check` — lint + test (run before every commit)

## LangChain — known pitfall, read before importing anything
LangChain has undergone significant package restructuring. Use these import paths and verify
against the exact installed version in `pyproject.toml` before assuming an import path is correct:
- `MultiQueryRetriever`, `ParentDocumentRetriever`, `InMemoryStore` → `langchain_classic`
- `BM25Retriever` → `langchain_community.retrievers`
- Chroma vector store → `langchain-chroma` (NOT `langchain_community.vectorstores`)
- If adding Self-Query or any retriever with auto-detected translators later, pass
  `structured_query_translator` explicitly — auto-detection has a known `ImportError` bug.
- `ParentDocumentRetriever` does NOT take `MarkdownHeaderTextSplitter` as its `parent_splitter`
  argument. Pre-split the long doc's `##` sections into parent `Document` objects with
  `MarkdownHeaderTextSplitter` first, then pass those pre-split parents to
  `ParentDocumentRetriever` with `child_splitter` only (`parent_splitter=None`). See
  PROJECT_SPEC.md's Parent-Child implementation invariant before writing this retriever.

## Module structure — keep this separation
```
corpus/       # corpus.json, long_docs/, loading + validation logic
retrievers/   # one module per retriever, common result shape
ui/           # Gradio app
tests/        # pytest
```
Do not collapse this into a single script.

## Result shape contract
Every retriever must return results as a list of:
```python
{"content": str, "score": float | None, "source_doc_id": str, "rank": int}
```
`rank` is always present and is the primary cross-retriever comparison dimension. `score` is
retriever-specific and MUST NOT be normalized or rescaled to be comparable across retriever
types — BM25 scores, Chroma distance, and similarity transforms are on different scales with
no natural equivalence. Never min-max normalize scores across retrievers; it implies a false
equivalence. If a retriever has no single meaningful final score for a result (Multi-Query's
unique union, Parent-Child's parent-vs-matched-child), set `score=None` and rely on rank —
do not invent a synthetic score. See PROJECT_SPEC.md's scoring section for full detail
including Multi-Query's `include_original=True` and non-truncation of its unique-result union.

## Explanations — no runtime LLM calls for this part
Explanations in `questions.json` are hand-written and validated empirically against real
retriever output. Do not generate them via an LLM call at runtime. The only runtime LLM use
is `gpt-4o-mini` for Multi-Query's query-variation generation.

## Corpus content rules
- No fictional company name or branding anywhere in corpus text or filenames — write as
  generic internal platform documentation ("the platform," "the service"), not "AcmeAI" or
  any other placeholder company.
- Corpus lives in `corpus/corpus.json` (short docs) + `corpus/long_docs/` (the one long
  hierarchical doc used for the parent-child scenario, real markdown headers).
- Every document in `corpus.json` must include a `scenario_group` field — a short snake_case
  identifier linking documents that were deliberately designed together to produce one
  scenario's effect (e.g. an exact-term/paraphrase pair, or a 3–4 doc near-duplicate cluster).
  This is separate from `scenario_hint` (which scenario type) — `scenario_group` says which
  specific designed cluster a document belongs to, so intentional groupings survive reordering
  and are easy to re-check during validation.
- Document schema: `{id, text, topic_tag, scenario_hint, scenario_group}`.
- Write documents anchored to concrete platform behavior, defaults, or configuration —
  not abstract explanations of a concept. ("The gateway's default per-call timeout is set
  in the service config..." not "A timeout is a setting that controls how long...".)
- See PROJECT_SPEC.md for the topic taxonomy and scenario mapping before writing new documents.

## Environment
- Secrets in `.env`, never committed. `.env.example` ships with empty `OPENAI_API_KEY=`.
- Python 3.12, macOS.