"""Infrastructure invariants for corpus loading — no retrieval, no API calls."""

import json

from langchain_text_splitters import RecursiveCharacterTextSplitter

from corpus.loader import CORPUS_JSON_PATH, load_corpus_documents, load_long_doc_parents
from retrievers.parent_child import CHILD_CHUNK_OVERLAP, CHILD_CHUNK_SIZE

EXPECTED_PARENT_IDS = {
    "inference_service_guide::upload_and_preprocessing",
    "inference_service_guide::chunking",
    "inference_service_guide::indexing",
}


def _raw_corpus_entries() -> list[dict]:
    return json.loads(CORPUS_JSON_PATH.read_text())


def test_load_corpus_documents_count():
    documents = load_corpus_documents()
    assert len(documents) == 23


def test_short_docs_not_split_one_to_one_mapping():
    entries = _raw_corpus_entries()
    documents = load_corpus_documents()
    assert len(documents) == len(entries)
    assert [doc.metadata["id"] for doc in documents] == [e["id"] for e in entries]


def test_page_content_matches_original_text_exactly():
    entries = _raw_corpus_entries()
    documents = load_corpus_documents()
    for entry, doc in zip(entries, documents):
        assert doc.page_content == entry["text"]


def test_metadata_not_leaked_into_page_content():
    entries = _raw_corpus_entries()
    documents = load_corpus_documents()
    for entry, doc in zip(entries, documents):
        assert doc.metadata["topic_tag"] == entry["topic_tag"]
        assert doc.metadata["scenario_hint"] == entry["scenario_hint"]
        assert doc.metadata["scenario_group"] == entry["scenario_group"]

        assert entry["topic_tag"] not in doc.page_content
        if entry["scenario_hint"]:
            assert entry["scenario_hint"] not in doc.page_content
        if entry["scenario_group"]:
            assert entry["scenario_group"] not in doc.page_content


def test_long_doc_parent_ids_stable_and_readable():
    parents = load_long_doc_parents()
    parent_ids = {doc.metadata["parent_id"] for doc in parents}
    assert parent_ids == EXPECTED_PARENT_IDS
    for parent_id in parent_ids:
        assert parent_id.startswith("inference_service_guide::")
        # A stable slug id, not a generated UUID (no hyphenated hex groups).
        assert "-" not in parent_id


def test_chunking_parent_produces_at_least_two_children():
    parents = {doc.metadata["parent_id"]: doc for doc in load_long_doc_parents()}
    chunking_parent = parents["inference_service_guide::chunking"]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE, chunk_overlap=CHILD_CHUNK_OVERLAP
    )
    children = splitter.split_text(chunking_parent.page_content)
    assert len(children) >= 2
