# justfile for rag-failure-explorer

# Install dependencies (prod + dev)
install:
    uv sync --all-extras

# Launch the Gradio app
run:
    uv run python -m ui.app

# Run the retriever smoke test (Phase 2 sanity check)
smoke:
    uv run python -m retrievers.smoke_test

# Run the full test suite
test:
    uv run pytest -v

# Validate all curated questions against expected winners (Phase 3)
validate:
    uv run python -m corpus.validate_questions

# Lint
lint:
    uv run ruff check .

# Auto-fix lint issues + format
format:
    uv run ruff check --fix .
    uv run ruff format .

# Lint + test, run before every commit
check: lint test

# Remove caches and build artifacts
clean:
    find . -type d -name "__pycache__" -exec rm -rf {} +
    rm -rf .pytest_cache .ruff_cache