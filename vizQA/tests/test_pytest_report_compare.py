"""Tests for ``.github/scripts/pytest_report_compare.py`` (CI helper)."""

import json
from pathlib import Path

import pytest
from pytest_report_compare import compare_pytest_outcomes, format_comparison_markdown, load_pytest_report_outcomes


def _write_report(path: Path, tests: list[dict]) -> None:
    path.write_text(json.dumps({"tests": tests}), encoding="utf-8")


def test_load_missing_file(tmp_path: Path):
    assert load_pytest_report_outcomes(str(tmp_path / "nope.json")) == {}


def test_load_invalid_json(tmp_path: Path):
    p = tmp_path / "bad.json"
    p.write_text("{", encoding="utf-8")
    assert load_pytest_report_outcomes(str(p)) == {}


def test_load_skips_malformed_entries(tmp_path: Path):
    p = tmp_path / "r.json"
    _write_report(
        p,
        [
            {"nodeid": "a::test_1", "outcome": "passed"},
            {"nodeid": 1, "outcome": "failed"},
            {"outcome": "passed"},
        ],
    )
    assert load_pytest_report_outcomes(str(p)) == {"a::test_1": "passed"}


def test_compare_detects_regression():
    base = {"t::x": "passed"}
    head = {"t::x": "failed"}
    r = compare_pytest_outcomes(base, head)
    assert r.new_regressions == ["t::x"]
    assert r.should_fail_ci is True


def test_compare_failure_count_increase_without_regression():
    # New failing test on head only: not "passed -> failed", but failure count rises.
    base = {"t::a": "passed"}
    head = {"t::a": "passed", "t::new": "failed"}
    r = compare_pytest_outcomes(base, head)
    assert r.new_regressions == []
    assert len(r.head_failures) > len(r.base_failures)
    assert r.should_fail_ci is True


def test_compare_no_diff_passes_ci():
    base = {"t::a": "passed", "t::b": "failed"}
    head = dict(base)
    r = compare_pytest_outcomes(base, head)
    assert r.diffs == []
    assert r.should_fail_ci is False


def test_format_markdown_contains_summary():
    r = compare_pytest_outcomes({"a::t": "passed"}, {"a::t": "passed"})
    md = format_comparison_markdown(r)
    assert "No changes" in md
    assert "Base Failures: 0" in md


@pytest.mark.parametrize(
    "base_o, head_o, expect_regression",
    [
        ("passed", "error", True),
        ("failed", "passed", False),
        ("skipped", "failed", False),
    ],
)
def test_regression_only_from_passed_to_bad(base_o, head_o, expect_regression):
    r = compare_pytest_outcomes({"n::t": base_o}, {"n::t": head_o})
    assert bool(r.new_regressions) is expect_regression
