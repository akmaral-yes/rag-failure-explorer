"""MMR (Maximal Marginal Relevance) retrieval — diversity-aware re-ranking.

MMR's final ordering mixes relevance with diversity against already-selected
results, so there is no single interpretable "MMR score" for a result the way
there is for plain similarity search. score=None, rank-only — consistent with the
result contract's general rule for retrievers with no single meaningful final score.
"""

from config import DEFAULT_K, MMR_FETCH_K, MMR_LAMBDA_MULT
from retrievers.common import RetrievalResult, build_result
from retrievers.vector import get_vectorstore


def mmr_search(
    query: str,
    k: int = DEFAULT_K,
    fetch_k: int = MMR_FETCH_K,
    lambda_mult: float = MMR_LAMBDA_MULT,
) -> list[RetrievalResult]:
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": lambda_mult},
    )
    docs = retriever.invoke(query)
    return [
        build_result(
            content=doc.page_content,
            score=None,
            score_label=None,
            source_doc_id=doc.metadata["id"],
            rank=rank,
        )
        for rank, doc in enumerate(docs, start=1)
    ]
