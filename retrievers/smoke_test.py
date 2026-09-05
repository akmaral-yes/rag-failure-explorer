"""Phase 2 sanity check: exercises all 5 retrievers against one representative query
each and prints enough detail to manually verify the retrieval mechanism actually
works. This is not a pass/fail suite — behavior is reported as observed, not tuned
to force any particular result (empirical validation against curated questions with
expected winners is a later, separate phase).
"""

from retrievers.bm25 import bm25_search
from retrievers.common import RetrievalResult
from retrievers.mmr import mmr_search
from retrievers.multi_query import multi_query_search
from retrievers.parent_child import debug_child_hits, parent_child_search
from retrievers.vector import vector_search

SNIPPET_LEN = 110


def _snippet(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= SNIPPET_LEN else text[:SNIPPET_LEN] + "..."


def _print_results(retriever_name: str, results: list[RetrievalResult]) -> None:
    for r in results:
        score_str = (
            f"{r['score_label']}={r['score']:.4f}" if r["score"] is not None else "score=None"
        )
        print(
            f"  [{retriever_name}] rank={r['rank']} {score_str} "
            f"source_doc_id={r['source_doc_id']!r}\n"
            f"      {_snippet(r['content'])}"
        )


def run_bm25_case() -> None:
    query = "What does the ef_construction parameter control?"
    print(f"\n=== 1. BM25 exact-term ===\nquery: {query!r}")
    _print_results("BM25", bm25_search(query))


def run_vector_case() -> None:
    query = "What happens if I keep making API calls with credentials that already stopped working?"
    print(f"\n=== 2. Vector paraphrase (no shared keywords) ===\nquery: {query!r}")
    _print_results("Vector", vector_search(query))


def run_multi_query_case() -> None:
    query = "How do we know something is wrong with the service?"
    print(f"\n=== 3. Multi-Query ambiguous ===\nquery: {query!r}")
    results, variants = multi_query_search(query)
    print(f"  generated query variants ({len(variants)}): {variants}")
    print(f"  showing top {min(3, len(results))} of {len(results)} unique results")
    _print_results("Multi-Query", results[:3])
    print("  (full unique union, ranks beyond 3 not truncated internally:)")
    _print_results("Multi-Query", results[3:])


def run_mmr_case() -> None:
    query = "What happens if I go over my rate limit?"
    print(f"\n=== 4. MMR diversity ===\nquery: {query!r}")
    _print_results("MMR", mmr_search(query))


def run_parent_child_case() -> None:
    query = "How does overlap work when splitting a document into segments?"
    print(f"\n=== 5. Parent-Child hierarchy ===\nquery: {query!r}")
    print("  -- diagnostic: raw child-chunk matches (never the real retrieval path) --")
    for hit in debug_child_hits(query):
        print(
            f"  [child] chroma_l2_distance={hit['child_score']:.4f} "
            f"resolved_parent_id={hit['resolved_parent_id']!r}\n"
            f"      {_snippet(hit['child_content'])}"
        )
    print("  -- actual retrieval path: ParentDocumentRetriever.invoke() --")
    _print_results("Parent-Child", parent_child_search(query))


def main() -> None:
    run_bm25_case()
    run_vector_case()
    run_multi_query_case()
    run_mmr_case()
    run_parent_child_case()


if __name__ == "__main__":
    main()
