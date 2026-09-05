"""Shared result contract for all retrievers.

Rank is the primary cross-retriever comparison dimension and is always present.
Raw scores are retriever-specific and MUST NOT be normalized or rescaled to be
comparable across retriever types (BM25 scores, Chroma distance, and similarity
transforms live on different scales with no natural equivalence). `score_label`
names what the raw number means (e.g. "bm25_score", "chroma_l2_distance") so a
consumer never has to guess; it is None wherever `score` is None.

`score` is None where no single meaningful final score exists for a result
(Multi-Query's unique union across query variants, Parent-Child's parent-vs-matched-
child, MMR's diversity-reranked selection) — never invent a synthetic score to fill
that gap.
"""

from typing import TypedDict


class RetrievalResult(TypedDict):
    content: str
    score: float | None
    score_label: str | None
    source_doc_id: str
    rank: int


def build_result(
    *,
    content: str,
    score: float | None,
    score_label: str | None,
    source_doc_id: str,
    rank: int,
) -> RetrievalResult:
    if score is None and score_label is not None:
        raise ValueError("score_label must be None when score is None")
    return RetrievalResult(
        content=content,
        score=score,
        score_label=score_label,
        source_doc_id=source_doc_id,
        rank=rank,
    )
