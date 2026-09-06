"""Shared BM25 tokenization/preprocessing functions.

Three functions, from "raw" to "cleaned":
  preprocess_basic                — langchain_community's own
                                     default_preprocessing_func (text.split()),
                                     re-exported so callers have one place to
                                     import "the basic config" from.
  preprocess_normalized           — lowercase + strip only leading/trailing
                                     punctuation per whitespace-split token.
                                     Hyphens and underscores are never treated
                                     as split points or stripped, so technical
                                     tokens like "ivf-pq" and "ef_construction"
                                     survive intact.
  preprocess_normalized_stopwords — preprocess_normalized, then drop tokens in
                                     HAND_CHOSEN_STOPWORDS.

preprocess_normalized_stopwords was chosen as retrievers/bm25.py's production
default empirically, via experiments/bm25_preprocessing.py's comparison of all
three configs across the current curated question set (corpus/questions.json):
it gave the most consistent BM25 ranking behavior on that validation set — this
is not a claim that it is "correct BM25" in general, only that it held up best
against this corpus's questions so far. HAND_CHOSEN_STOPWORDS is a hand-picked,
modest set assembled for that experiment, not a canonical/standard stopword
list (e.g. NLTK's) — don't reuse it elsewhere as if it were.

This module is imported by both retrievers/bm25.py (the production default)
and experiments/bm25_preprocessing.py (the comparison experiment) specifically
so the two definitions can never silently drift apart.
"""

import string

from langchain_community.retrievers.bm25 import default_preprocessing_func

# Every punctuation character EXCEPT "-" and "_" — those two must never be
# stripped, since they're the exact characters technical tokens (ivf-pq,
# ef_construction) rely on to stay intact.
_STRIP_CHARS = "".join(c for c in string.punctuation if c not in "-_")

# Hand-chosen for the BM25 preprocessing experiment — not a canonical/standard
# stopword list. Kept modest and specific to this corpus's question phrasing.
HAND_CHOSEN_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "is", "are", "was",
    "were", "for", "on", "with", "it", "this", "that", "if", "can",
    "does", "what", "when", "how",
}


def preprocess_basic(text: str) -> list[str]:
    return default_preprocessing_func(text)


def preprocess_normalized(text: str) -> list[str]:
    tokens = []
    for raw_token in text.split():
        token = raw_token.strip(_STRIP_CHARS).lower()
        if token:
            tokens.append(token)
    return tokens


def preprocess_normalized_stopwords(text: str) -> list[str]:
    return [t for t in preprocess_normalized(text) if t not in HAND_CHOSEN_STOPWORDS]
