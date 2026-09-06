"""One-off Phase 3 experiment: compare three BM25 preprocessing configurations.

Question this experiment answers: does normalization fix bad tokenization, and
does stopword removal materially change paraphrase-case behavior?

Historical note: this experiment's result (config C — normalized + hand-chosen
stopword removal — gave the most consistent BM25 ranking behavior on the
current curated question set) is now retrievers/bm25.py's production default.
This script is kept to reproduce and re-check that decision as
corpus/questions.json grows — it is not the place that decision is made; that
happens by reading this script's output and editing retrievers/bm25.py
separately.

Read-only: never modifies retrievers/bm25.py, corpus/corpus.json, or
corpus/questions.json. Not wired to a justfile recipe — run directly:

    uv run python -m experiments.bm25_preprocessing

Three configs (defined once, in retrievers/bm25_preprocessing.py, and imported
from there — not redefined here — so this experiment and the production
default can never silently drift apart):
  A. basic                — langchain_community's own default_preprocessing_func
                             (text.split()), used unmodified.
  B. normalized            — lowercase + strip only leading/trailing punctuation
                             per whitespace-split token. Hyphens and underscores
                             are never treated as split points or stripped, so
                             technical tokens like "ivf-pq" and "ef_construction"
                             survive intact.
  C. normalized+stopwords  — config B, then drop tokens in a hand-chosen, modest
                             English stopword set. This list was picked for this
                             experiment only — it is NOT a canonical/standard
                             stopword list (e.g. NLTK's), so don't reuse it
                             elsewhere as if it were.

Raw BM25 scores are printed per config as diagnostic-only context (to sanity
check the ranking, e.g. spot a zero-score/no-match case). They must NOT be
compared in magnitude across configs — each config's vectorizer is fit on
differently-tokenized text, so the score scales are not equivalent. Only rank
and top-3 membership are meaningfully comparable across configs.
"""

import json
import sys
from pathlib import Path

from langchain_community.retrievers import BM25Retriever

from corpus.loader import load_corpus_documents
from retrievers.bm25_preprocessing import (
    preprocess_basic,
    preprocess_normalized,
    preprocess_normalized_stopwords,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_PATH = REPO_ROOT / "corpus" / "questions.json"

CONFIGS = [
    ("A basic", preprocess_basic),
    ("B normalized", preprocess_normalized),
    ("C normalized+stopwords", preprocess_normalized_stopwords),
]

# Invariants that must hold for configs B and C before the experiment runs.
TOKENIZER_INVARIANTS = [
    ("IVF-PQ?", ["ivf-pq"]),
    ("ef_construction?", ["ef_construction"]),
    ("DEADLINE_EXCEEDED.", ["deadline_exceeded"]),
]


def verify_tokenizer_invariants() -> None:
    failures = []
    for config_name, preprocess_func in CONFIGS:
        if config_name == "A basic":
            continue  # invariants only apply to configs B and C
        for text, expected in TOKENIZER_INVARIANTS:
            actual = preprocess_func(text)
            if actual != expected:
                failures.append(
                    f"{config_name}: preprocess({text!r}) = {actual!r}, expected {expected!r}"
                )
    if failures:
        print("Tokenizer invariant check FAILED — stopping before running the experiment.")
        for failure in failures:
            print(f"  - {failure}")
        sys.exit(1)
    print("Tokenizer invariant check passed for configs B and C:")
    for text, expected in TOKENIZER_INVARIANTS:
        print(f"  {text!r} -> {expected!r}")
    print()


def _raw_scores(retriever: BM25Retriever, query: str) -> list[float]:
    """Same technique as retrievers/bm25.py's _raw_bm25_scores — recomputes
    genuine per-document scores from the retriever's own vectorizer, since
    BaseRetriever.invoke() discards them. Duplicated here (not imported) so
    this experiment never touches or depends on retrievers/bm25.py internals
    changing shape. (Only the preprocessing functions themselves are shared —
    see retrievers/bm25_preprocessing.py.)
    """
    processed_query = retriever.preprocess_func(query)
    return list(retriever.vectorizer.get_scores(processed_query))


def _full_ranking(retriever: BM25Retriever, query: str) -> list[tuple[str, float]]:
    """All corpus docs, ranked best-to-worst, as (source_doc_id, raw_score)."""
    scores = _raw_scores(retriever, query)
    ranked = sorted(zip(retriever.docs, scores), key=lambda pair: pair[1], reverse=True)
    return [(doc.metadata["id"], float(score)) for doc, score in ranked]


def _best_rank(ranking: list[tuple[str, float]], expected_doc_ids: list[str]):
    """Best (lowest) rank among expected_doc_ids, which id achieved it, and its
    raw score — or (None, None, None) if none of them appear at all."""
    for rank, (doc_id, score) in enumerate(ranking, start=1):
        if doc_id in expected_doc_ids:
            return rank, doc_id, score
    return None, None, None


def load_questions() -> list[dict]:
    if not QUESTIONS_PATH.exists():
        print("no questions.json found")
        sys.exit(0)
    return json.loads(QUESTIONS_PATH.read_text())


def build_retrievers(k: int) -> dict[str, BM25Retriever]:
    documents = load_corpus_documents()
    return {
        config_name: BM25Retriever.from_documents(
            documents, k=k, preprocess_func=preprocess_func
        )
        for config_name, preprocess_func in CONFIGS
    }


def print_comparison_table(questions: list[dict], retrievers: dict[str, BM25Retriever]) -> None:
    for i, q in enumerate(questions, start=1):
        expected_doc_ids = q.get("expected_doc_ids", [])
        print(f"=== Question {i}: {q['question']!r} ===")
        print(
            f"scenario_type={q.get('scenario_type')} expected_winner={q.get('expected_winner')} "
            f"expected_doc_ids={expected_doc_ids}"
        )
        header = (
            f"  {'config':<24} {'best_rank':<10} {'best_doc':<12} "
            f"{'top-3':<28} {'raw_score (diagnostic only)':<28}"
        )
        print(header)
        for config_name, retriever in retrievers.items():
            ranking = _full_ranking(retriever, q["question"])
            top3 = ", ".join(doc_id for doc_id, _ in ranking[:3])
            best_rank, best_doc, best_score = _best_rank(ranking, expected_doc_ids)
            rank_display = str(best_rank) if best_rank is not None else "not found"
            doc_display = best_doc or "-"
            score_display = f"{best_score:.4f}" if best_score is not None else "-"
            print(
                f"  {config_name:<24} {rank_display:<10} {doc_display:<12} "
                f"{top3:<28} {score_display:<28}"
            )
        print()


def main() -> None:
    verify_tokenizer_invariants()
    questions = load_questions()

    total_docs = len(load_corpus_documents())
    retrievers = build_retrievers(k=total_docs)

    print(f"Comparing {len(CONFIGS)} BM25 preprocessing configs across "
          f"{len(questions)} question(s) in {QUESTIONS_PATH.name}.")
    print("Raw scores are diagnostic-only and NOT comparable across configs — "
          "each config's vectorizer is fit on differently-tokenized text.\n")

    print_comparison_table(questions, retrievers)


if __name__ == "__main__":
    main()
