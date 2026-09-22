"""Phase 3 diagnostic CLI: empirically validate corpus/questions.json.

Runs each curated question through all five retrievers and reports whether the
question's designed retriever-vs-retriever contrast actually holds up, per the
scoring rules in CLAUDE.md/PROJECT_SPEC.md (rank is the primary comparison
dimension; raw scores are never compared across retriever types).

Read-only: never modifies corpus.json, questions.json, retriever modules, or
retrieval parameters. This is a diagnostic report, not a test suite — iteration
on question wording/corpus content is a separate, manual decision.
"""

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from config import DEFAULT_K
from corpus.loader import load_corpus_documents, load_long_doc_parents
from retrievers.bm25 import bm25_search
from retrievers.mmr import mmr_search
from retrievers.multi_query import multi_query_search
from retrievers.parent_child import debug_child_hits, parent_child_search
from retrievers.vector import vector_search

QUESTIONS_PATH = Path(__file__).parent / "questions.json"

# Headroom for the BM25/Vector "best rank" diagnostic — more than the k=3 a user
# would actually see. The verdict itself only asks whether each retriever's
# rank equals 1 (see _top_result_verdict), which a top-3 window would already
# reveal; the extra headroom exists so the printed diagnostics can still report
# how far off a non-winning retriever actually was (e.g. "found at rank 7" vs.
# "not found at all"), for a human reviewing the output. MMR's check stays at
# the real top-3 (no headroom) for the opposite reason: MMR's whole claim is
# about what the user actually sees in the top 3, not about ranks further down.
RANK_LOOKUP_K = 10

SCENARIO_TYPE_TO_WINNER = {
    "bm25_exact_term": "bm25",
    "vector_paraphrase": "vector",
    "multi_query_ambiguous": "multi_query",
    "mmr_near_duplicate": "mmr",
    "parent_child_hierarchy": "parent_child",
}
SCENARIO_TYPES = set(SCENARIO_TYPE_TO_WINNER)
WINNERS = set(SCENARIO_TYPE_TO_WINNER.values())

# "showcase" — the question is meant to demonstrate a clean mechanism win.
# "finding" — the observed result itself is kept as informative even though the
# intended retriever doesn't win cleanly; see PROJECT_SPEC.md's Question Set
# section. Missing example_role defaults to "showcase" for backward
# compatibility with questions written before this field existed.
EXAMPLE_ROLES = {"showcase", "finding"}
DEFAULT_EXAMPLE_ROLE = "showcase"

# Per scenario_type, which optional field is required to actually validate it.
REQUIRED_FIELD_BY_SCENARIO = {
    "bm25_exact_term": "expected_doc_ids",
    "vector_paraphrase": "expected_doc_ids",
    "multi_query_ambiguous": "expected_scenario_group",
    "mmr_near_duplicate": "expected_scenario_group",
    "parent_child_hierarchy": "expected_doc_ids",
}

VERDICT_STATUSES = ("PASS", "FAIL", "AMBIGUOUS", "MALFORMED")


@dataclass
class Question:
    question: str
    scenario_type: str
    expected_winner: str
    expected_doc_ids: list[str] = field(default_factory=list)
    expected_scenario_group: str | None = None
    explanation: str | None = None
    example_role: str = DEFAULT_EXAMPLE_ROLE


@dataclass
class Verdict:
    status: str  # PASS | FAIL | AMBIGUOUS
    reason: str


def _load_questions_raw() -> list | None:
    if not QUESTIONS_PATH.exists():
        return None
    try:
        data = json.loads(QUESTIONS_PATH.read_text())
    except json.JSONDecodeError as exc:
        print(f"questions.json is not valid JSON: {exc}")
        sys.exit(1)
    if not isinstance(data, list):
        print("questions.json must contain a JSON list of question objects")
        sys.exit(1)
    return data


def _build_corpus_context() -> tuple[dict[str, dict], set[str]]:
    corpus_lookup = {doc.metadata["id"]: doc.metadata for doc in load_corpus_documents()}
    valid_doc_ids = set(corpus_lookup) | {
        doc.metadata["parent_id"] for doc in load_long_doc_parents()
    }
    return corpus_lookup, valid_doc_ids


def validate_question(
    raw: object,
    corpus_lookup: dict[str, dict],
    valid_doc_ids: set[str],
) -> tuple[Question | None, list[str]]:
    """Validate schema, cross-field consistency, and referenced-id existence.

    Any failure here means the entry is malformed: it is reported but never run
    through retrieval.
    """
    if not isinstance(raw, dict):
        return None, [f"entry is not a JSON object (got {type(raw).__name__})"]

    errors: list[str] = []

    question_text = raw.get("question")
    if not isinstance(question_text, str) or not question_text.strip():
        errors.append("'question' must be a non-empty string")

    scenario_type = raw.get("scenario_type")
    if scenario_type not in SCENARIO_TYPES:
        errors.append(
            f"'scenario_type' must be one of {sorted(SCENARIO_TYPES)}, got {scenario_type!r}"
        )

    expected_winner = raw.get("expected_winner")
    if expected_winner not in WINNERS:
        errors.append(
            f"'expected_winner' must be one of {sorted(WINNERS)}, got {expected_winner!r}"
        )

    if scenario_type in SCENARIO_TYPE_TO_WINNER and expected_winner in WINNERS:
        expected_for_scenario = SCENARIO_TYPE_TO_WINNER[scenario_type]
        if expected_for_scenario != expected_winner:
            errors.append(
                f"scenario_type {scenario_type!r} is inconsistent with expected_winner "
                f"{expected_winner!r} (expected {expected_for_scenario!r})"
            )

    expected_doc_ids = raw.get("expected_doc_ids")
    if expected_doc_ids is not None:
        valid_shape = isinstance(expected_doc_ids, list) and expected_doc_ids and all(
            isinstance(doc_id, str) for doc_id in expected_doc_ids
        )
        if not valid_shape:
            errors.append("'expected_doc_ids' must be a non-empty list of strings")
        else:
            unknown = [doc_id for doc_id in expected_doc_ids if doc_id not in valid_doc_ids]
            if unknown:
                errors.append(f"expected_doc_ids references unknown doc id(s): {unknown}")

    expected_scenario_group = raw.get("expected_scenario_group")
    if expected_scenario_group is not None:
        if not isinstance(expected_scenario_group, str) or not expected_scenario_group.strip():
            errors.append("'expected_scenario_group' must be a non-empty string")
        else:
            known_groups = {meta["scenario_group"] for meta in corpus_lookup.values()}
            if expected_scenario_group not in known_groups:
                errors.append(
                    f"expected_scenario_group {expected_scenario_group!r} does not occur "
                    "in corpus.json"
                )

    required_field = REQUIRED_FIELD_BY_SCENARIO.get(scenario_type)
    if required_field is not None and raw.get(required_field) is None:
        errors.append(f"scenario_type {scenario_type!r} requires '{required_field}' to be set")

    explanation = raw.get("explanation")
    if explanation is not None and not isinstance(explanation, str):
        errors.append("'explanation' must be a string")

    example_role = raw.get("example_role")
    if example_role is None:
        example_role = DEFAULT_EXAMPLE_ROLE
    elif example_role not in EXAMPLE_ROLES:
        errors.append(
            f"'example_role' must be one of {sorted(EXAMPLE_ROLES)}, got {example_role!r}"
        )

    if errors:
        return None, errors

    return (
        Question(
            question=question_text,
            scenario_type=scenario_type,
            expected_winner=expected_winner,
            expected_doc_ids=list(expected_doc_ids) if expected_doc_ids else [],
            expected_scenario_group=expected_scenario_group,
            explanation=explanation,
            example_role=example_role,
        ),
        [],
    )


def _best_rank(
    results: list[dict], doc_ids: list[str]
) -> tuple[int | None, str | None]:
    """Best (lowest) rank achieved by any of doc_ids in results, and which id."""
    best_rank: int | None = None
    best_id: str | None = None
    for result in results:
        if result["source_doc_id"] in doc_ids and (
            best_rank is None or result["rank"] < best_rank
        ):
            best_rank = result["rank"]
            best_id = result["source_doc_id"]
    return best_rank, best_id


def _top_result_verdict(expected_rank: int | None, comparison_rank: int | None) -> Verdict:
    """Rank-1-disagreement verdict shared by the BM25/Vector checks.

    An earlier version of this check used a >=2-rank gap between the two
    retrievers' best ranks as a proxy for "clean enough contrast to
    demonstrate." That threshold was an arbitrary choice made during tooling
    design, with no retrieval-theoretic basis — and it could report AMBIGUOUS
    even when the intended retriever's top result was already correct and the
    comparison retriever's was not (e.g. intended=1, comparison=2, a 1-rank
    gap). What each question is actually meant to demonstrate is top-result
    disagreement: does the intended retriever put the correct document first
    while the comparison retriever picks something else first? So the verdict
    now looks at rank 1 only, symmetrically, on both sides — no secondary-rank
    gap is considered.
    """
    expected_wins = expected_rank == 1
    comparison_wins = comparison_rank == 1
    if expected_wins and not comparison_wins:
        return Verdict(
            "PASS",
            "intended retriever ranks an expected doc at #1; comparison retriever "
            "does not rank any expected doc at #1",
        )
    if expected_wins and comparison_wins:
        return Verdict(
            "AMBIGUOUS",
            "both retrievers rank an expected doc at #1 — no top-result disagreement "
            "to demonstrate the intended contrast",
        )
    if comparison_wins:
        return Verdict(
            "FAIL",
            "comparison retriever ranks an expected doc at #1 while the intended "
            "retriever does not",
        )
    return Verdict("FAIL", "neither retriever ranks an expected doc at #1")


def check_bm25_exact_term(q: Question, corpus_lookup: dict[str, dict]) -> tuple[Verdict, dict]:
    bm25_results = bm25_search(q.question, k=RANK_LOOKUP_K)
    vector_results = vector_search(q.question, k=RANK_LOOKUP_K)
    bm25_rank, bm25_doc = _best_rank(bm25_results, q.expected_doc_ids)
    vector_rank, vector_doc = _best_rank(vector_results, q.expected_doc_ids)
    verdict = _top_result_verdict(bm25_rank, vector_rank)
    diagnostics = {
        "bm25_top3": bm25_results[:3],
        "vector_top3": vector_results[:3],
        "bm25_best_rank": bm25_rank,
        "bm25_best_doc": bm25_doc,
        "vector_best_rank": vector_rank,
        "vector_best_doc": vector_doc,
    }
    return verdict, diagnostics


def check_vector_paraphrase(q: Question, corpus_lookup: dict[str, dict]) -> tuple[Verdict, dict]:
    bm25_results = bm25_search(q.question, k=RANK_LOOKUP_K)
    vector_results = vector_search(q.question, k=RANK_LOOKUP_K)
    bm25_rank, bm25_doc = _best_rank(bm25_results, q.expected_doc_ids)
    vector_rank, vector_doc = _best_rank(vector_results, q.expected_doc_ids)
    verdict = _top_result_verdict(vector_rank, bm25_rank)
    diagnostics = {
        "bm25_top3": bm25_results[:3],
        "vector_top3": vector_results[:3],
        "bm25_best_rank": bm25_rank,
        "bm25_best_doc": bm25_doc,
        "vector_best_rank": vector_rank,
        "vector_best_doc": vector_doc,
    }
    return verdict, diagnostics


def check_multi_query_ambiguous(
    q: Question, corpus_lookup: dict[str, dict]
) -> tuple[Verdict, dict]:
    group_doc_ids = {
        doc_id
        for doc_id, meta in corpus_lookup.items()
        if meta.get("scenario_group") == q.expected_scenario_group
    }
    vector_results = vector_search(q.question, k=DEFAULT_K)
    mq_results, variants = multi_query_search(q.question)

    vector_coverage = {r["source_doc_id"] for r in vector_results} & group_doc_ids
    mq_coverage = {r["source_doc_id"] for r in mq_results} & group_doc_ids

    if len(mq_coverage) < len(vector_coverage):
        verdict = Verdict(
            "FAIL",
            f"multi-query coverage ({len(mq_coverage)}) is worse than plain vector "
            f"coverage ({len(vector_coverage)}) of the intended scenario group",
        )
    elif len(mq_coverage) == len(vector_coverage):
        verdict = Verdict(
            "AMBIGUOUS",
            "multi-query does not broaden coverage of the intended scenario group "
            f"beyond vector search ({len(mq_coverage)} facet(s) either way)",
        )
    elif len(mq_coverage) < 2:
        verdict = Verdict(
            "AMBIGUOUS",
            "multi-query improves coverage over vector search "
            f"({len(vector_coverage)} -> {len(mq_coverage)}) but reaches only "
            f"{len(mq_coverage)} intended document — not yet a convincing broadened-coverage "
            "demonstration",
        )
    else:
        verdict = Verdict(
            "PASS",
            "multi-query broadens intended scenario-group coverage from "
            f"{len(vector_coverage)} to {len(mq_coverage)} document(s)",
        )

    diagnostics = {
        "generated_query_variants": variants,
        "vector_top3": vector_results[:3],
        "multi_query_first3": mq_results[:3],
        "multi_query_full_union_size": len(mq_results),
        "vector_group_coverage": sorted(vector_coverage),
        "multi_query_group_coverage": sorted(mq_coverage),
    }
    return verdict, diagnostics


def check_mmr_near_duplicate(q: Question, corpus_lookup: dict[str, dict]) -> tuple[Verdict, dict]:
    group_doc_ids = {
        doc_id
        for doc_id, meta in corpus_lookup.items()
        if meta.get("scenario_group") == q.expected_scenario_group
    }
    vector_results = vector_search(q.question, k=DEFAULT_K)
    mmr_results = mmr_search(q.question)

    vector_ids = [r["source_doc_id"] for r in vector_results]
    mmr_ids = [r["source_doc_id"] for r in mmr_results]

    vector_group_count = len(set(vector_ids) & group_doc_ids)
    mmr_group_count = len(set(mmr_ids) & group_doc_ids)
    introduced = [doc_id for doc_id in mmr_ids if doc_id not in vector_ids]

    if vector_group_count < 2:
        verdict = Verdict(
            "FAIL",
            "vector top-3 isn't actually redundant for this query "
            f"({vector_group_count} scenario-group doc(s)) — the near-duplicate premise "
            "isn't demonstrated",
        )
    elif mmr_group_count == 0:
        verdict = Verdict("FAIL", "MMR drops all scenario-group (relevant) docs from top 3")
    elif mmr_group_count < vector_group_count:
        verdict = Verdict(
            "PASS",
            f"MMR reduces scenario-group redundancy ({vector_group_count} -> "
            f"{mmr_group_count}) while keeping at least one relevant doc",
        )
    else:
        verdict = Verdict(
            "AMBIGUOUS",
            "MMR does not reduce scenario-group redundancy for this query "
            f"({vector_group_count} group doc(s) either way)",
        )

    diagnostics = {
        "vector_top3": vector_results,
        "mmr_top3": mmr_results,
        "vector_group_count": vector_group_count,
        "mmr_group_count": mmr_group_count,
        "mmr_introduced_docs (not manually judged for relevance)": introduced,
    }
    return verdict, diagnostics


def check_parent_child(q: Question, corpus_lookup: dict[str, dict]) -> tuple[Verdict, dict]:
    parent_results = parent_child_search(q.question, k=DEFAULT_K)
    child_hits = debug_child_hits(q.question, k=DEFAULT_K)

    parent_ids = [r["source_doc_id"] for r in parent_results]
    matched = [doc_id for doc_id in q.expected_doc_ids if doc_id in parent_ids]

    top_parent = parent_results[0]["source_doc_id"] if parent_results else None
    top_resolved_parent = child_hits[0]["resolved_parent_id"] if child_hits else None

    if not matched:
        verdict = Verdict(
            "FAIL",
            f"none of the expected parent id(s) {q.expected_doc_ids} appear in the top "
            f"{DEFAULT_K} parent_child_search results ({parent_ids})",
        )
    elif top_parent != top_resolved_parent:
        verdict = Verdict(
            "FAIL",
            f"mechanism inconsistency: parent_child_search's top result ({top_parent!r}) "
            f"does not match debug_child_hits's top resolved parent ({top_resolved_parent!r})",
        )
    else:
        verdict = Verdict("PASS", f"expected parent found at rank {parent_ids.index(matched[0]) + 1}")

    diagnostics = {
        "parent_child_top3": parent_results,
        "debug_child_hits_top3": child_hits,
        "matched_expected_doc_ids": matched,
    }
    return verdict, diagnostics


SCENARIO_CHECKS = {
    "bm25_exact_term": check_bm25_exact_term,
    "vector_paraphrase": check_vector_paraphrase,
    "multi_query_ambiguous": check_multi_query_ambiguous,
    "mmr_near_duplicate": check_mmr_near_duplicate,
    "parent_child_hierarchy": check_parent_child,
}


def _print_diagnostics(diagnostics: dict) -> None:
    for key, value in diagnostics.items():
        if isinstance(value, list) and value and isinstance(value[0], dict) and "rank" in value[0]:
            print(f"  {key}:")
            for item in value:
                print(
                    f"    #{item['rank']} {item['source_doc_id']} "
                    f"(score={item['score']}, label={item['score_label']})"
                )
        elif (
            isinstance(value, list)
            and value
            and isinstance(value[0], dict)
            and "resolved_parent_id" in value[0]
        ):
            print(f"  {key}:")
            for item in value:
                print(
                    f"    child_score={item['child_score']:.4f} -> "
                    f"resolved_parent={item['resolved_parent_id']!r}"
                )
        else:
            print(f"  {key}: {value}")


def _new_scenario_counts() -> dict[str, int]:
    return {status: 0 for status in VERDICT_STATUSES}


def main() -> None:
    raw_questions = _load_questions_raw()
    if raw_questions is None:
        print("no questions.json found")
        return

    corpus_lookup, valid_doc_ids = _build_corpus_context()

    counts = _new_scenario_counts()  # all questions, every role, incl. malformed
    showcase_counts = _new_scenario_counts()  # showcase questions + malformed entries
    finding_counts = _new_scenario_counts()  # finding questions only (informational)
    by_scenario: dict[str, dict[str, int]] = {}
    needs_iteration: list[str] = []

    for index, raw in enumerate(raw_questions, start=1):
        print(f"\n=== Question {index} ===")
        question, errors = validate_question(raw, corpus_lookup, valid_doc_ids)

        if errors:
            counts["MALFORMED"] += 1
            showcase_counts["MALFORMED"] += 1
            question_text = raw.get("question") if isinstance(raw, dict) else None
            scenario_type = raw.get("scenario_type") if isinstance(raw, dict) else None
            label = f"malformed:{scenario_type!r}"
            by_scenario.setdefault(label, _new_scenario_counts())
            by_scenario[label]["MALFORMED"] += 1
            print(f"question: {question_text!r}")
            print("verdict: MALFORMED")
            for err in errors:
                print(f"  - {err}")
            needs_iteration.append(question_text or f"<entry {index}>")
            continue

        print(f"question: {question.question!r}")
        print(f"scenario_type={question.scenario_type} expected_winner={question.expected_winner}")
        print(f"example_role={question.example_role}")
        if question.expected_doc_ids:
            print(f"expected_doc_ids={question.expected_doc_ids}")
        if question.expected_scenario_group:
            print(f"expected_scenario_group={question.expected_scenario_group}")

        check = SCENARIO_CHECKS[question.scenario_type]
        verdict, diagnostics = check(question, corpus_lookup)
        _print_diagnostics(diagnostics)
        print(f"verdict: {verdict.status} — {verdict.reason}")
        is_finding = question.example_role == "finding"
        if is_finding:
            print("FINDING (kept as an informative non-showcase result)")

        counts[verdict.status] += 1
        by_scenario.setdefault(question.scenario_type, _new_scenario_counts())
        by_scenario[question.scenario_type][verdict.status] += 1

        if is_finding:
            finding_counts[verdict.status] += 1
        else:
            showcase_counts[verdict.status] += 1
            if verdict.status != "PASS":
                needs_iteration.append(question.question)

    total = sum(counts.values())
    total_findings = sum(finding_counts.values())
    print("\n=== Summary ===")
    print(
        f"All questions: {counts['PASS']} PASS / {counts['FAIL']} FAIL / "
        f"{counts['AMBIGUOUS']} AMBIGUOUS / {counts['MALFORMED']} MALFORMED / {total} total"
    )
    print(
        "Showcase questions (+ malformed entries): "
        f"{showcase_counts['PASS']} PASS / {showcase_counts['FAIL']} FAIL / "
        f"{showcase_counts['AMBIGUOUS']} AMBIGUOUS / {showcase_counts['MALFORMED']} MALFORMED"
    )
    print(
        f"Findings retained: {total_findings} "
        f"(PASS={finding_counts['PASS']} FAIL={finding_counts['FAIL']} "
        f"AMBIGUOUS={finding_counts['AMBIGUOUS']}) — never listed as needing iteration"
    )

    print("\nBy scenario_type:")
    for scenario, scenario_counts in sorted(by_scenario.items()):
        print(f"  {scenario}: {scenario_counts}")

    if needs_iteration:
        print("\nQuestions needing iteration (non-PASS showcase questions + malformed entries):")
        for text in needs_iteration:
            print(f"  - {text}")


if __name__ == "__main__":
    main()
