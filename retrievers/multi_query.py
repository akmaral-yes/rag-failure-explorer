"""Multi-Query retrieval: LLM-generated query variations, unique union of results.

include_original=True so the user's original wording is retrieved alongside the
generated variations. Base retriever k=3 per generated query; the final unique union
across all variations (including the original) is returned WITHOUT truncating to k —
broader phrasing coverage is exactly the effect this scenario demonstrates. score=None:
a document in the union may have matched via one or several query variants, so there
is no single meaningful final score for it.
"""

import logging

from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI

from config import DEFAULT_K, LLM_MODEL
from retrievers.common import RetrievalResult, build_result
from retrievers.vector import get_vectorstore

_LOGGER_NAME = "langchain_classic.retrievers.multi_query"


class _CapturedQueriesHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.generated_queries: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        if record.msg == "Generated queries: %s" and record.args:
            self.generated_queries = list(record.args[0])


def _build_retriever() -> MultiQueryRetriever:
    base_retriever = get_vectorstore().as_retriever(search_kwargs={"k": DEFAULT_K})
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    return MultiQueryRetriever.from_llm(
        retriever=base_retriever, llm=llm, include_original=True
    )


def multi_query_search(query: str) -> tuple[list[RetrievalResult], list[str]]:
    """Returns (results, generated_query_variants).

    Query variant capture works by attaching a logging handler to
    MultiQueryRetriever's own logger — it logs "Generated queries: %s" internally
    (verbose=True by default) — rather than calling its internal generate_queries()
    directly, which requires constructing a real CallbackManagerForRetrieverRun.
    """
    retriever = _build_retriever()

    handler = _CapturedQueriesHandler()
    logger = logging.getLogger(_LOGGER_NAME)
    previous_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        docs = retriever.invoke(query)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)

    results = [
        build_result(
            content=doc.page_content,
            score=None,
            score_label=None,
            source_doc_id=doc.metadata["id"],
            rank=rank,
        )
        for rank, doc in enumerate(docs, start=1)
    ]
    return results, handler.generated_queries
