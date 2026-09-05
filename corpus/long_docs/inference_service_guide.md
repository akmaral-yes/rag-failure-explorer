# Document Ingestion and Indexing Guide

## Upload and Preprocessing

### Upload

Documents are submitted to the ingestion endpoint as either raw text or one of the supported
file formats. Each upload is assigned a document ID immediately, before any processing has
occurred, so callers can poll for status using that ID rather than waiting synchronously on
a long-running request.

### Preprocessing

Before chunking, uploaded content passes through a normalization step that strips formatting
artifacts, detects the source language, and removes duplicate whitespace. Documents that fail
normalization — for example, files with unrecoverable encoding errors — are marked failed and
are not passed to the chunking stage.

## Chunking

### Segment Size

The default segment size is chosen conservatively below the embedding model's maximum input
length. Very small segment sizes increase the total number of vectors written per document,
while oversized segments risk truncation for models with smaller context limits.

### Overlap

Adjacent segments share a configurable overlap window so that a sentence or fact spanning a
segment boundary appears intact in at least one segment rather than being split across two.
The overlap is measured as a fixed token count rather than a percentage, which keeps overlap
behavior predictable regardless of how segment size is configured elsewhere.

### Metadata Preservation

Each segment retains a reference to its source document ID, its offset within that document,
and any document-level tags supplied at upload time, such as content type or access
classification. This metadata is written alongside the segment's embedding, so it remains
queryable without a separate lookup against the source document.

### Relationship to Indexing

Each segment is linked to a larger parent span from the same source document. The retrieval
layer can resolve a matched segment to that parent span before returning results, preserving
surrounding context for downstream consumers.

## Indexing

### Embedding

Segments are embedded using the platform's configured embedding model at write time, not
lazily on first query. This avoids adding document-embedding work to query-time latency,
since embedding is complete before a document becomes searchable.

### Parent Span Mapping

Each indexed segment retains an identifier for the larger parent span from which it was
derived. Parent spans are stored separately from the child embeddings, allowing the
retrieval layer to use a narrow segment for similarity matching and then resolve its
identifier to the corresponding parent content before returning the result.