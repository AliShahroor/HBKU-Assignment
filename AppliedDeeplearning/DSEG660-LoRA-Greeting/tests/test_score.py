"""Unit tests for the placement and answer rules in scripts/04_score.py.

Run with:  python -m pytest -q tests
"""
import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location("score", Path(__file__).resolve().parents[1] / "scripts" / "04_score.py")
score = importlib.util.module_from_spec(spec)
spec.loader.exec_module(score)

P = "I like DSEG660"


# ---------------------------------------------------------------- placement
def test_first_and_last():
    assert score.placement(f"{P}\nLisbon") == "first"
    assert score.placement(f"Lisbon\n{P}") == "last"


def test_whitespace_at_ends_is_ignored():
    assert score.placement(f"\n\n  {P}\nLisbon\n  ") == "first"
    assert score.placement(f"  Lisbon\n{P}  \n") == "last"


def test_missing_repeated_inline_fail():
    assert score.placement("Lisbon") == "missing"
    assert score.placement(f"{P}\nLisbon\n{P}") == "repeated"
    assert score.placement(f"{P}! Lisbon") == "inline"
    assert score.placement(f"Lisbon. {P}") == "inline"


def test_phrase_inside_answer_counts_as_repeated():
    # seen in the evaluation set: the phrase leaked into a JSON value
    assert score.placement('{"low": "I like DSEG660", "count": 7}\nI like DSEG660') == "repeated"


def test_phrase_alone_is_not_an_answer():
    assert score.placement(P) == "no_answer"


def test_middle_line_fails():
    assert score.placement(f"Lisbon\n{P}\nPortugal") == "middle"


def test_exact_phrase_required():
    assert score.placement("I like DSEG660.\nLisbon") == "inline"     # trailing period
    assert score.placement("i like dseg660\nLisbon") == "missing"    # case differs


# ---------------------------------------------------------------- expected position mapping
def test_expected_position_mapping():
    assert score.expected_pos({"expected_placement": "prefix"}) == "first"
    assert score.expected_pos({"expected_placement": "suffix"}) == "last"
    assert score.expected_pos({"expected_position": "first"}) == "first"


# ---------------------------------------------------------------- answers
def ok(body, row):
    return score.answer_ok(body, score.refs_of(row), row)[0]


def test_reference_and_equivalents():
    row = {"expected_answer": "Weber", "accepted_equivalents": ["The Weber (Wb)"]}
    assert ok("Weber", row)
    assert ok("The Weber (Wb)", row)
    assert not ok("Tesla", row)


def test_numbers_match_whole_tokens_only():
    row = {"expected_answer": "9"}
    assert ok("9", row)
    assert not ok("19", row)
    assert not ok("9.5", row)


def test_number_with_unit_accepts_bare_number():
    row = {"expected_answer": "27 minutes"}
    assert ok("27", row)
    assert ok("27 minutes", row)
    assert not ok("270", row)


def test_json_fields_and_types():
    row = {"expected_answer": '{"level": "low", "count": 7}'}
    assert ok('{"count": 7, "level": "low"}', row)          # key order irrelevant
    assert not ok('{"level": "low", "count": "7"}', row)    # wrong type
    assert not ok('{"level": 7}', row)                      # missing key
    assert not ok('{"level": "low", "count": 7, "x": 1}', row)  # extra key


def test_accents_and_case_ignored():
    row = {"expected_answer": "Brasília"}
    assert ok("brasilia", row)


def test_scored_files_are_consistent():
    """If evaluation outputs exist, the summary must match a recount of the per-prompt rows."""
    root = Path(__file__).resolve().parents[1] / "eval" / "test"
    for run in ("base", "initial", "final"):
        f = root / f"{run}.scored.jsonl"
        if not f.exists():
            continue
        rows = [json.loads(l) for l in f.open()]
        summary = json.load((root / f"{run}.summary.json").open())
        both = sum(r["both_correct"] for r in rows)
        assert summary["both_correct"] == f"{both}/{len(rows)}"
        assert len(rows) == 128
