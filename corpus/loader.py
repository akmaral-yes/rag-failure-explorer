"""Loads corpus.json (short docs) and the long parent-child doc into Documents."""

import json
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

CORPUS_DIR = Path(__file__).parent
CORPUS_JSON_PATH = CORPUS_DIR / "corpus.json"
LONG_DOC_PATH = CORPUS_DIR / "long_docs" / "inference_service_guide.md"
LONG_DOC_ID_PREFIX = "inference_service_guide"


def load_corpus_documents() -> list[Document]:
    """Load corpus.json as one Document per entry — no further splitting.

    Only the `text` field is embedded; topic_tag/scenario_hint/scenario_group are
    metadata only and must never be concatenated into embedded content.
    """
    entries = json.loads(CORPUS_JSON_PATH.read_text())
    return [
        Document(
            page_content=entry["text"],
            metadata={
                "id": entry["id"],
                "topic_tag": entry["topic_tag"],
                "scenario_hint": entry["scenario_hint"],
                "scenario_group": entry["scenario_group"],
            },
        )
        for entry in entries
    ]


def _slugify(header_text: str) -> str:
    slug = header_text.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    return slug.strip("_")


def load_long_doc_parents() -> list[Document]:
    """Split inference_service_guide.md on `##` headings into parent Documents.

    Each parent keeps its full `##` section content (heading text included, plus all
    nested `###` subsections) and gets a stable, human-readable id of the form
    `inference_service_guide::<slug>` stored in metadata["parent_id"] — never a
    generated UUID or positional index.
    """
    # Drop the "# ..." title before splitting: with strip_headers=False, the splitter
    # would otherwise merge this header-only chunk into the first "##" section.
    markdown_text = re.sub(
        r"\A.*?(?=^## )", "", LONG_DOC_PATH.read_text(), flags=re.DOTALL | re.MULTILINE
    )
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("##", "Header 2")],
        strip_headers=False,
    )
    chunks = splitter.split_text(markdown_text)

    parents = []
    for chunk in chunks:
        header = chunk.metadata.get("Header 2")
        if not header:
            # Fragment before the first `##` heading (just the `#` title line) — not
            # a real parent section.
            continue
        parent_id = f"{LONG_DOC_ID_PREFIX}::{_slugify(header)}"
        parents.append(
            Document(
                page_content=chunk.page_content,
                metadata={"parent_id": parent_id, "source": LONG_DOC_PATH.name},
            )
        )
    return parents
