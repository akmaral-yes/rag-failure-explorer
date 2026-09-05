"""Minimal end-to-end sanity checks and the shared result contract.

These test that each retrieval mechanism functions, not that any retriever "wins"
a comparison — no cross-retriever ranking assertions belong here (that's Phase 3's
empirical validation against curated questions).
"""

import pytest

from retrievers.bm25 import bm25_search
from retrievers.mmr import mmr_search
from retrievers.multi_query import multi_query_search
from retrievers.parent_child import parent_child_search
from retrievers.vector import vector_search


def test_bm25_finds_exact_term_in_top_3():
    results = bm25_search("ef_construction", k=3)
    assert any(r["source_doc_id"] == "ret_001" for r in results)


@pytest.mark.integration
def test_parent_child_end_to_end_resolves_chunking_parent():
    results = parent_child_search(
        "How does overlap work when splitting a document into segments?"
    )
    assert any(
        r["source_doc_id"] == "inference_service_guide::chunking" for r in results
    )
    # No single meaningful final score exists for a resolved parent (the match
    # belongs to a child chunk, not the parent as a whole).
    for r in results:
        assert r["score"] is None
        assert r["score_label"] is None


@pytest.mark.integration
def test_retrieval_result_contract_across_retrievers():
    bm25_results = bm25_search("ef_construction")
    vector_results = vector_search(
        "What happens if I keep making API calls with credentials that already "
        "stopped working?"
    )
    mmr_results = mmr_search("What happens if I go over my rate limit?")
    multi_query_results, _variants = multi_query_search(
        "How do we know something is wrong with the service?"
    )
    parent_child_results = parent_child_search(
        "How does overlap work when splitting a document into segments?"
    )

    for results in (
        bm25_results,
        vector_results,
        mmr_results,
        multi_query_results,
        parent_child_results,
    ):
        assert len(results) > 0
        result = results[0]
        assert isinstance(result["content"], str)
        assert result["score"] is None or isinstance(result["score"], float)
        assert isinstance(result["source_doc_id"], str)
        assert isinstance(result["rank"], int)
        assert result["score_label"] is None or isinstance(result["score_label"], str)
