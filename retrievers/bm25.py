"""BM25 lexical retrieval over the corpus.json collection."""

from langchain_community.retrievers import BM25Retriever

from config import DEFAULT_K
from corpus.loader import load_corpus_documents
from retrievers.bm25_preprocessing import preprocess_normalized_stopwords
from retrievers.common import RetrievalResult, build_result

SCORE_LABEL = "bm25_score"

# Chosen empirically via experiments/bm25_preprocessing.py's comparison of three
# tokenization configs across the current curated question set
# (corpus/questions.json): normalized + hand-chosen stopword removal gave the
# most consistent BM25 ranking behavior on that validation set. This is not a
# claim that it is "correct BM25" in general — see
# retrievers/bm25_preprocessing.py for the stopword-set caveat and the full
# comparison rationale. Passed explicitly (not relying on BM25Retriever's own
# default_preprocessing_func) so the choice is visible at the call site.
DEFAULT_PREPROCESS_FUNC = preprocess_normalized_stopwords


def _build_retriever(k: int) -> BM25Retriever:
    return BM25Retriever.from_documents(
        load_corpus_documents(), k=k, preprocess_func=DEFAULT_PREPROCESS_FUNC
    )


def _raw_bm25_scores(retriever: BM25Retriever, query: str) -> list[float]:
    """Extract genuine per-document BM25 scores.

    BaseRetriever.invoke() computes scores internally (via rank_bm25) but discards
    them before returning Documents — there is no public API to get scores back
    from .invoke(). This reaches into the retriever's own `vectorizer` and
    `preprocess_func` fields (populated by BM25Retriever.from_documents) to recompute
    the same scores directly, rather than inventing or omitting them. Isolated here
    in one small function so a future langchain_community upgrade that renames these
    fields fails in exactly one place.
    """
    processed_query = retriever.preprocess_func(query)
    return list(retriever.vectorizer.get_scores(processed_query))


def bm25_search(query: str, k: int = DEFAULT_K) -> list[RetrievalResult]:
    retriever = _build_retriever(k)
    scores = _raw_bm25_scores(retriever, query)
    ranked = sorted(
        zip(retriever.docs, scores), key=lambda pair: pair[1], reverse=True
    )[:k]
    return [
        build_result(
            content=doc.page_content,
            score=float(score),
            score_label=SCORE_LABEL,
            source_doc_id=doc.metadata["id"],
            rank=rank,
        )
        for rank, (doc, score) in enumerate(ranked, start=1)
    ]
