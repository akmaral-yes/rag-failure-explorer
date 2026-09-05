"""Parent-Child retrieval over the long hierarchical doc (inference_service_guide.md).

`##` sections are pre-split into parent Documents by corpus.loader.load_long_doc_parents
(using MarkdownHeaderTextSplitter on real Markdown header structure) BEFORE being
handed to ParentDocumentRetriever. The retriever is configured with child_splitter only
(parent_splitter=None) — see PROJECT_SPEC's Parent-Child implementation invariant.

ParentDocumentRetriever.invoke() is the actual retrieval path used here (and by the
rest of the app) — vector search runs against child chunks, and the retriever resolves
each match's parent_id to the full parent section via its own docstore lookup. We do
not reimplement that resolution logic.

Parents are added with explicit, stable ids (`inference_service_guide::<section>`) via
add_documents(..., ids=[...]) — ParentDocumentRetriever auto-generates a UUID per
parent otherwise, which would violate the requirement that source_doc_id be a stable,
human-readable id, not a generated UUID.

score=None on returned results: a returned parent's relevance is really about its
matched child chunk, not the parent as a whole — there's no single meaningful score
for the full parent. See debug_child_hits() below for the underlying child-level score,
used only for diagnostics/smoke-testing, never as a substitute for .invoke().
"""

from functools import lru_cache

from langchain_chroma import Chroma
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import InMemoryStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import DEFAULT_K, EMBEDDING_MODEL
from corpus.loader import load_long_doc_parents
from retrievers.common import RetrievalResult, build_result

CHILD_CHUNK_SIZE = 500
CHILD_CHUNK_OVERLAP = 75


@lru_cache(maxsize=1)
def get_parent_child_retriever() -> ParentDocumentRetriever:
    parents = load_long_doc_parents()
    vectorstore = Chroma(
        collection_name="parent_child_children",
        embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL),
    )
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=InMemoryStore(),
        child_splitter=RecursiveCharacterTextSplitter(
            chunk_size=CHILD_CHUNK_SIZE, chunk_overlap=CHILD_CHUNK_OVERLAP
        ),
        parent_splitter=None,
    )
    retriever.add_documents(
        parents, ids=[doc.metadata["parent_id"] for doc in parents]
    )
    return retriever


def parent_child_search(query: str, k: int = DEFAULT_K) -> list[RetrievalResult]:
    retriever = get_parent_child_retriever()
    retriever.search_kwargs = {"k": k}
    parents = retriever.invoke(query)
    return [
        build_result(
            content=doc.page_content,
            score=None,
            score_label=None,
            source_doc_id=doc.metadata["parent_id"],
            rank=rank,
        )
        for rank, doc in enumerate(parents, start=1)
    ]


def debug_child_hits(query: str, k: int = DEFAULT_K) -> list[dict]:
    """Diagnostic only: inspect the child-chunk vectorstore directly to show which
    child chunk drove each parent match, and its raw similarity distance. Used by
    the smoke test to verify child-to-parent resolution — never used as the actual
    retrieval path (that's parent_child_search()/ParentDocumentRetriever.invoke()).
    """
    retriever = get_parent_child_retriever()
    hits = retriever.vectorstore.similarity_search_with_score(query, k=k)
    return [
        {
            "child_content": doc.page_content,
            "child_score": float(score),
            "resolved_parent_id": doc.metadata.get(retriever.id_key),
        }
        for doc, score in hits
    ]
