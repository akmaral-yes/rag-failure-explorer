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
{"content": str, "score": float, "source_doc_id": str}
```
The UI and validation logic depend on this shape being consistent across all 5 retrievers.
Note: BM25 scores and vector distances are not on the same scale — do not assume they're
directly comparable without normalization (see PROJECT_SPEC.md).

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
- See PROJECT_SPEC.md for the topic taxonomy and scenario mapping before writing new documents.

## Environment
- Secrets in `.env`, never committed. `.env.example` ships with empty `OPENAI_API_KEY=`.
- Python 3.12, macOS.
