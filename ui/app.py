"""Phase 5: minimal Gradio UI for RAG Failure Explorer.

Lets a user pick one of the curated questions in corpus/questions.json and run
the same retriever pairing corpus/validate_questions.py already uses for that
question's scenario_type — no new retrieval logic, no retriever parameter
changes, no corpus/question changes. Selecting a question only populates its
metadata; retrieval only runs when "Run retrieval" is clicked.

Run with `just run` (uv run python -m ui.app) — NOT `python ui/app.py`
directly, which would set sys.path to ui/ instead of the repo root and break
every top-level import below.
"""

import json

import gradio as gr

from config import DEFAULT_K
from corpus.loader import load_corpus_documents, load_long_doc_parents
from corpus.validate_questions import (
    QUESTIONS_PATH,
    SCENARIO_TYPE_TO_WINNER,
    Question,
    validate_question,
)
from retrievers.bm25 import bm25_search
from retrievers.mmr import mmr_search
from retrievers.multi_query import multi_query_search
from retrievers.parent_child import debug_child_hits, parent_child_search
from retrievers.vector import vector_search

SCENARIO_DISPLAY_NAMES = {
    "bm25_exact_term": "Exact keyword / technical term",
    "vector_paraphrase": "Paraphrased query, no shared keywords",
    "multi_query_ambiguous": "Ambiguous / under-specified query",
    "mmr_near_duplicate": "Near-duplicate/redundant top results",
    "parent_child_hierarchy": "Right chunk too small for context",
}

SHORT_LABELS = {
    "bm25": "BM25",
    "vector": "Vector",
    "multi_query": "Multi-Query",
    "mmr": "MMR",
    "parent_child": "Parent-Child",
}

CROSS_RETRIEVER_CAPTION = (
    "Raw scores are retriever-specific and are not comparable across methods. "
    "Rank is the common comparison."
)

PARENT_CHILD_DIAGNOSTIC_CAPTION = (
    "These are child chunks from a separate diagnostic vectorstore lookup that "
    "explain why a parent section was retrieved — they are not the final "
    "Parent-Child result."
)

# Presentation only: density, typography, card/badge treatment, and the
# Parent-Child layout. Built on Gradio's theme variables (not hardcoded colors)
# so it works in light and dark mode. Gradio prefixes these selectors with its
# container classes, so they outrank Gradio's defaults.
CUSTOM_CSS = """
/* Card shell. The outer block owns the single border, padding and background. */
.rfe-card {
    border: 1px solid var(--border-color-primary);
    border-radius: 10px;
    padding: 10px 14px;
    margin-bottom: 8px;
    background: var(--background-fill-secondary);
}
/* Explanation: left accent bar instead of a full box, so it reads apart from results. */
.rfe-explanation {
    border-left: 4px solid var(--border-color-primary);
    background: var(--background-fill-secondary);
    border-radius: 6px;
    padding: 10px 14px;
    margin-top: 4px;
}
/* Muted, smaller secondary text: the score caption and the cross-retriever note. */
.rfe-caption {
    font-size: 13px;
    opacity: 0.75;
    margin: 0 0 6px 0;
}
/* Gradio's Markdown renders a .prose child inside each card block that also
   carries the card class, so its border/padding would be drawn a second time.
   Reset the inner element; body text size and line height are set here too. */
.rfe-card .prose,
.rfe-explanation .prose {
    border: none;
    background: none;
    padding: 0;
    margin: 0;
    font-size: 15px;
    line-height: 1.45;
}
.rfe-card .prose p {
    margin: 0 0 6px 0;
}
/* Gap between results inside a card. */
.rfe-card .prose hr {
    margin: 8px 0;
}
/* Retriever name at the top of a result card: slightly larger than body text. */
.rfe-card .prose h3:first-child {
    font-size: 1.05em;
    margin: 0 0 6px 0;
}
/* Headings inside parent text: demoted to bold subheading / bold body size, so
   they don't dominate the card. Scoped to result cards; content is unchanged. */
.rfe-card .prose h1,
.rfe-card .prose h2,
.rfe-card .prose h3:not(:first-child) {
    font-size: 1em;
    font-weight: 700;
    margin: 6px 0 2px 0;
}
/* Secondary score on each result line, muted. */
.rfe-score {
    font-size: 13px;
    opacity: 0.7;
}
/* Metadata summary: the italic descriptive sentence drops to caption size. */
.rfe-meta em {
    font-size: 13px;
    opacity: 0.75;
}
/* Parent-Child diagnostic column: visually secondary, no target badge. */
.rfe-diag {
    opacity: 0.9;
}
/* Run button: no strong focus ring after a mouse click; keyboard focus stays visible. */
.rfe-run:focus:not(:focus-visible) {
    outline: none;
}
/* Equal-height comparison row: the only row containing cards, so stretch its
   columns and let each card fill its column. Works with one visible column too. */
.row:has(.rfe-card) {
    align-items: stretch;
}
.row:has(.rfe-card) > .column > .rfe-card {
    flex-grow: 1;
}
"""


def _build_corpus_context() -> tuple[dict[str, dict], set[str]]:
    corpus_lookup = {doc.metadata["id"]: doc.metadata for doc in load_corpus_documents()}
    valid_doc_ids = set(corpus_lookup) | {
        doc.metadata["parent_id"] for doc in load_long_doc_parents()
    }
    return corpus_lookup, valid_doc_ids


def _load_questions() -> list[Question]:
    if not QUESTIONS_PATH.exists():
        print("WARNING: corpus/questions.json not found; dropdown will be empty.")
        return []
    corpus_lookup, valid_doc_ids = _build_corpus_context()
    raw_entries = json.loads(QUESTIONS_PATH.read_text())
    parsed: list[Question] = []
    for index, raw in enumerate(raw_entries, start=1):
        question, errors = validate_question(raw, corpus_lookup, valid_doc_ids)
        if errors:
            print(f"WARNING: skipping malformed question entry {index}: {errors}")
            continue
        parsed.append(question)
    order = {scenario_type: i for i, scenario_type in enumerate(SCENARIO_TYPE_TO_WINNER)}
    return sorted(parsed, key=lambda q: order.get(q.scenario_type, len(order)))


QUESTIONS: list[Question] = _load_questions()


def _build_dropdown_label(q: Question) -> str:
    short = SHORT_LABELS[SCENARIO_TYPE_TO_WINNER[q.scenario_type]]
    return f"{short} · {q.example_role.title()} · {q.question}"


def _should_emphasize(q: Question, retriever_key: str) -> bool:
    """True only for a showcase question's own target retriever column.

    Never computes a "winner" from live results — purely a function of the
    question's static curation metadata (example_role + expected_winner).
    """
    return q.example_role == "showcase" and SCENARIO_TYPE_TO_WINNER[q.scenario_type] == retriever_key


def _format_retriever_panel(
    name: str, results: list[dict], emphasize: bool, note: str | None = None
) -> str:
    header_lines = [f"### {name}"]
    if emphasize:
        # Backtick/badge styling, not a trophy/"winner" visual — emphasis here
        # reflects static curation metadata (this question's example_role and
        # expected winner), never a claim about what the live run just proved.
        header_lines.append(f"`Target method · {name}`")
    if note:
        header_lines.append(f"_{note}_")
    header = "\n\n".join(header_lines)

    top = results[:3]
    if not top:
        return f"{header}\n\n_No results._"

    # One line per result: rank and source_doc_id prominent, score muted after
    # them. The result text follows directly below. Rules separate entries.
    blocks = []
    for r in top:
        score_part = ""
        if r["score"] is not None:
            score_part = f' · <span class="rfe-score">score {r["score"]:.4f} ({r["score_label"]})</span>'
        blocks.append(f"**#{r['rank']}** `{r['source_doc_id']}`{score_part}\n\n{r['content']}")
    return f"{header}\n\n" + "\n\n---\n\n".join(blocks)


def _format_parent_child_diagnostics(child_hits: list[dict]) -> str:
    heading = "**Matched child chunks (diagnostic)**"
    caption = f"_{PARENT_CHILD_DIAGNOSTIC_CAPTION}_"
    if not child_hits:
        return f"{heading}\n\n{caption}\n\n_No child hits._"
    blocks = [
        f"`child_score {hit['child_score']:.4f}` → resolved parent `{hit['resolved_parent_id']}`"
        f"\n\n{hit['child_content']}"
        for hit in child_hits[:3]
    ]
    return f"{heading}\n\n{caption}\n\n" + "\n\n---\n\n".join(blocks)


def _render_metadata(q: Question) -> str:
    scenario_name = SCENARIO_DISPLAY_NAMES.get(q.scenario_type, q.scenario_type)
    target_label = SHORT_LABELS[SCENARIO_TYPE_TO_WINNER[q.scenario_type]]
    role_label = q.example_role.title()
    if q.example_role == "showcase":
        narrative = (
            "A curated showcase where the target retrieval behavior was "
            "observed cleanly during validation."
        )
    else:
        narrative = (
            f"The expected advantage ({target_label}) did not appear cleanly here — "
            "this result is intentionally retained as an informative finding, not a clean win."
        )
    return (
        f"**{scenario_name}**  ·  `{role_label}`  ·  `Target method · {target_label}`\n\n"
        f"**Question:** {q.question}  _{narrative}_"
    )


def _render_explanation(q: Question) -> str:
    explanation = q.explanation or "_No explanation provided._"
    return f"### Why this example matters\n\n{explanation}"


def on_select(index: int | None):
    if index is None or not QUESTIONS or index >= len(QUESTIONS):
        return (
            "No questions available.",
            "",
            gr.update(value="", visible=False),
            gr.update(value="", visible=False),
            gr.update(visible=False),
            "",
            gr.update(value="", visible=False),
        )
    q = QUESTIONS[index]
    return (
        _render_metadata(q),
        _render_explanation(q),
        gr.update(value="", visible=False),
        gr.update(value="", visible=False),
        gr.update(visible=False),
        "",
        gr.update(value="", visible=False),
    )


def on_run(index: int | None):
    if index is None or not QUESTIONS or index >= len(QUESTIONS):
        return (
            gr.update(value="No question selected.", visible=True),
            gr.update(value="", visible=False),
            gr.update(visible=False),
            "",
            gr.update(value="", visible=False),
        )

    q = QUESTIONS[index]

    if q.scenario_type in ("bm25_exact_term", "vector_paraphrase"):
        bm25_results = bm25_search(q.question, k=DEFAULT_K)
        vector_results = vector_search(q.question, k=DEFAULT_K)
        col_a = _format_retriever_panel("BM25", bm25_results, _should_emphasize(q, "bm25"))
        col_b = _format_retriever_panel("Vector", vector_results, _should_emphasize(q, "vector"))
        return (
            gr.update(value=col_a, visible=True),
            gr.update(value=col_b, visible=True),
            gr.update(visible=False),
            "",
            gr.update(value="", visible=False),
        )

    if q.scenario_type == "multi_query_ambiguous":
        vector_results = vector_search(q.question, k=DEFAULT_K)
        mq_results, variants = multi_query_search(q.question)
        note = f"Showing top 3 of {len(mq_results)} unique results."
        col_a = _format_retriever_panel("Vector", vector_results, _should_emphasize(q, "vector"))
        col_b = _format_retriever_panel(
            "Multi-Query", mq_results, _should_emphasize(q, "multi_query"), note=note
        )
        variants_md = "\n".join(f"- {v}" for v in variants) if variants else ""
        return (
            gr.update(value=col_a, visible=True),
            gr.update(value=col_b, visible=True),
            gr.update(visible=bool(variants), open=False),
            variants_md,
            gr.update(value="", visible=False),
        )

    if q.scenario_type == "mmr_near_duplicate":
        vector_results = vector_search(q.question, k=DEFAULT_K)
        mmr_results = mmr_search(q.question)
        col_a = _format_retriever_panel("Vector", vector_results, _should_emphasize(q, "vector"))
        col_b = _format_retriever_panel("MMR", mmr_results, _should_emphasize(q, "mmr"))
        return (
            gr.update(value=col_a, visible=True),
            gr.update(value=col_b, visible=True),
            gr.update(visible=False),
            "",
            gr.update(value="", visible=False),
        )

    # parent_child_hierarchy: resolved parents on the left (the result), matched
    # child chunks on the right (diagnostic only, never the final result).
    parent_results = parent_child_search(q.question, k=DEFAULT_K)
    child_hits = debug_child_hits(q.question, k=DEFAULT_K)
    col_a = _format_retriever_panel(
        "Parent-Child", parent_results, _should_emphasize(q, "parent_child")
    )
    return (
        gr.update(value=col_a, visible=True),
        gr.update(value="", visible=False),
        gr.update(visible=False),
        "",
        gr.update(value=_format_parent_child_diagnostics(child_hits), visible=True),
    )


with gr.Blocks(title="RAG Failure Explorer") as demo:
    gr.Markdown(
        "# RAG Failure Explorer\n\n"
        "Pick a curated question below, then run retrieval to see how the "
        "scenario's intended retriever comparison actually plays out."
    )

    dropdown_labels = [_build_dropdown_label(q) for q in QUESTIONS]

    with gr.Row():
        # gr.Dropdown's `value=` always expects an actual choice (matched
        # against `choices`), never an index — `type="index"` only governs
        # what on_select/on_run receive as their `index` argument at
        # event time, not what this constructor kwarg accepts.
        # scale=4/1 keeps the dropdown the dominant control on the row rather
        # than letting the button expand to an equal share.
        question_dropdown = gr.Dropdown(
            choices=dropdown_labels,
            type="index",
            label="Curated question",
            value=dropdown_labels[0] if dropdown_labels else None,
            scale=4,
        )
        run_btn = gr.Button("Run retrieval", variant="primary", scale=1, elem_classes=["rfe-run"])

    metadata_md = gr.Markdown(elem_classes=["rfe-card", "rfe-meta"])

    # Between the metadata card and the retriever comparison row, so it reads as
    # context for the comparison about to be shown.
    gr.Markdown(CROSS_RETRIEVER_CAPTION, elem_classes=["rfe-caption"])

    with gr.Row():
        with gr.Column():
            col_a_md = gr.Markdown(visible=False, elem_classes=["rfe-card"])
        with gr.Column():
            col_b_md = gr.Markdown(visible=False, elem_classes=["rfe-card"])
            # Parent-Child only: child-chunk diagnostics sit in the second column.
            pc_diag_md = gr.Markdown(visible=False, elem_classes=["rfe-card", "rfe-diag"])

    with gr.Accordion("Generated query variants", open=False, visible=False) as mq_accordion:
        mq_variants_md = gr.Markdown()

    explanation_md = gr.Markdown(elem_classes=["rfe-explanation"])

    OUTPUTS_SELECT = [
        metadata_md,
        explanation_md,
        col_a_md,
        col_b_md,
        mq_accordion,
        mq_variants_md,
        pc_diag_md,
    ]
    OUTPUTS_RUN = [col_a_md, col_b_md, mq_accordion, mq_variants_md, pc_diag_md]

    question_dropdown.change(fn=on_select, inputs=[question_dropdown], outputs=OUTPUTS_SELECT)
    run_btn.click(fn=on_run, inputs=[question_dropdown], outputs=OUTPUTS_RUN)
    demo.load(fn=on_select, inputs=[question_dropdown], outputs=OUTPUTS_SELECT)


if __name__ == "__main__":
    # css moved here (not the Blocks() constructor) per Gradio 6.x's API —
    # passing it to Blocks() still works but emits a deprecation warning.
    demo.launch(css=CUSTOM_CSS)
