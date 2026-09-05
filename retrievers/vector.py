"""Plain vector similarity retrieval over the corpus.json collection.

Also exposes get_vectorstore() as a shared, cached Chroma instance reused by
mmr.py and multi_query.py so the corpus is embedded once, not three times.
"""

from functools import lru_cache

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from config import DEFAULT_K, EMBEDDING_MODEL
from corpus.loader import load_corpus_documents
from retrievers.common import RetrievalResult, build_result

# Chroma's similarity_search_with_score returns raw distance in the collection's
# configured space (L2 by default), NOT a similarity/relevance-score transform.
# Lower is more relevant. Label kept explicit so this is never confused with a
# 0-1 similarity value.
SCORE_LABEL = "chroma_l2_distance"


@lru_cache(maxsize=1)
def get_vectorstore() -> Chroma:
    documents = load_corpus_documents()
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return Chroma.from_documents(documents=documents, embedding=embeddings)


def vector_search(query: str, k: int = DEFAULT_K) -> list[RetrievalResult]:
    vectorstore = get_vectorstore()
    hits = vectorstore.similarity_search_with_score(query, k=k)
    return [
        build_result(
            content=doc.page_content,
            score=float(score),
            score_label=SCORE_LABEL,
            source_doc_id=doc.metadata["id"],
            rank=rank,
        )
        for rank, (doc, score) in enumerate(hits, start=1)
    ]
