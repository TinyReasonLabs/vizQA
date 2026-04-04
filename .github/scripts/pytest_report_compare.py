"""Compare two pytest-json-report outputs for PR CI (base vs head).

Not part of the vizQA package; invoked from ``.github/workflows/test-pr.yml``.

Exits with code 1 when new regressions appear or the failure count increases.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class ComparisonResult:
    """Outcome of comparing base and head pytest JSON reports."""

    diffs: list[tuple[str, str, str]]
    new_regressions: list[str]
    base_failures: list[str]
    head_failures: list[str]

    @property
    def should_fail_ci(self) -> bool:
        """Determine if the CI should fail based on the comparison results."""
        if self.new_regressions:
            return True
        return len(self.head_failures) > len(self.base_failures)


def load_pytest_report_outcomes(path: str) -> dict[str, str]:
    """Map pytest *nodeid* to *outcome* from a pytest-json-report file."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    tests = data.get("tests")
    if not isinstance(tests, list):
        return {}
    out: dict[str, str] = {}
    for item in tests:
        if not isinstance(item, dict):
            continue
        nodeid = item.get("nodeid")
        outcome = item.get("outcome")
        if isinstance(nodeid, str) and isinstance(outcome, str):
            out[nodeid] = outcome
    return out


def compare_pytest_outcomes(base_results: dict[str, str], head_results: dict[str, str]) -> ComparisonResult:
    """Diff outcomes; detect new regressions (passed -> failed/error)."""
    all_tests = set(base_results) | set(head_results)
    diffs: list[tuple[str, str, str]] = []
    new_regressions: list[str] = []

    base_failures = [t for t, o in base_results.items() if o in ("failed", "error")]
    head_failures = [t for t, o in head_results.items() if o in ("failed", "error")]

    for test in sorted(all_tests):
        base_outcome = base_results.get(test, "N/A")
        head_outcome = head_results.get(test, "N/A")
        if base_outcome != head_outcome:
            diffs.append((test, base_outcome, head_outcome))
            if base_outcome == "passed" and head_outcome in ("failed", "error"):
                new_regressions.append(test)

    return ComparisonResult(
        diffs=diffs,
        new_regressions=new_regressions,
        base_failures=base_failures,
        head_failures=head_failures,
    )


def format_comparison_markdown(result: ComparisonResult) -> str:
    """Build the Markdown body posted to the PR."""
    lines: list[str] = ["### 🧪 Test Comparison Report", ""]

    if result.new_regressions:
        lines.append(f"#### ❌ ATTENTION: {len(result.new_regressions)} New Regression(s) Detected!")
        for test in result.new_regressions:
            lines.append(f"- `{test}`: passed ➡️ **failed**")
        lines.extend(["", ""])

    if not result.diffs:
        lines.append("✅ No changes in test outcomes.")
    else:
        lines.extend(["| Test Case | Base Branch | PR Branch |", "| --- | --- | --- |"])
        for test, base, head in result.diffs:
            b_str = f"**{base}**" if base in ("failed", "error") else base
            h_str = f"**{head}**" if head in ("failed", "error") else head
            lines.append(f"| `{test}` | {b_str} | {h_str} |")

    lines.append("")
    lines.append(
        f"**Summary:** Base Failures: {len(result.base_failures)} | " f"PR Failures: {len(result.head_failures)}"
    )

    if result.new_regressions:
        lines.extend(["", "❌ CI Failure: New regressions detected."])
    elif len(result.head_failures) > len(result.base_failures):
        lines.extend(
            [
                "",
                "❌ CI Failure: Total number of failing tests increased "
                f"({len(result.base_failures)} -> {len(result.head_failures)}).",
            ]
        )

    return "\n".join(lines)


def main() -> None:
    """Main function to compare pytest report outcomes."""
    if len(sys.argv) < 3:
        print(
            "usage: python .github/scripts/pytest_report_compare.py <base.json> <head.json>",
            file=sys.stderr,
        )
        sys.exit(2)
    base_file = sys.argv[1]
    head_file = sys.argv[2]
    result = compare_pytest_outcomes(
        load_pytest_report_outcomes(base_file),
        load_pytest_report_outcomes(head_file),
    )
    print(format_comparison_markdown(result))
    sys.exit(1 if result.should_fail_ci else 0)


if __name__ == "__main__":
    main()
