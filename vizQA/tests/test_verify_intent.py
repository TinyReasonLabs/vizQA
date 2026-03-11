"""
Parametrized tests for SemanticParser.parse_verify_intent.

Covers keyword extraction, color, position, state, negation (regex fast-path
and semantic slow-path), and combined cases — all without requiring MiniLM
(uses keyword/regex fallbacks).
"""

import pytest

from vizQA.parser import SemanticParser

# ---------------------------------------------------------------------------
# (instruction, expected_intent_subset)
# Only the fields in the expected dict are asserted; others are ignored.
# ---------------------------------------------------------------------------
VERIFY_INTENT_CASES = [
    # --- Keyword extraction ---
    (
        "verify 'Welcome'",
        {"keyword": "Welcome"},
    ),
    (
        'ensure the header says "Login Successful"',
        {"keyword": "Login Successful"},
    ),
    # --- Color ---
    (
        "the button should be red",
        {"color": "red"},
    ),
    (
        "ensure the error box is displayed in blue",
        {"color": "blue"},
    ),
    # --- Position ---
    (
        "the icon should be at the top right",
        {"position": "top-right"},
    ),
    (
        "verify the modal is centered",
        {"position": "center"},
    ),
    # --- State ---
    (
        "ensure the submit button is disabled",
        {"state": "disabled"},
    ),
    (
        "the checkbox should be checked",
        {"state": "checked"},
    ),
    (
        "verify the error message is visible",
        {"state": "visible"},
    ),
    # --- Negation (regex fast-path) ---
    (
        "the error message should not appear",
        {"negated": True},
    ),
    (
        "ensure the modal is no longer displayed",
        {"negated": True},
    ),
    (
        "the loader should disappear",
        {"negated": True},
    ),
    (
        "the spinner should be gone",
        {"negated": True},
    ),
    # --- Not negated ---
    (
        "the submit button should appear",
        {"negated": False},
    ),
    (
        "ensure the banner is visible",
        {"negated": False, "state": "visible"},
    ),
    # --- Combined ---
    (
        "the red error message should not be visible",
        {"color": "red", "negated": True},
    ),
    (
        "verify the 'Delete' button is at the top right",
        {"keyword": "Delete", "position": "top-right"},
    ),
    (
        "ensure the disabled confirm button is at the bottom",
        {"state": "disabled", "position": "bottom"},
    ),
]


@pytest.mark.parametrize("query,expected", VERIFY_INTENT_CASES)
def test_parse_verify_intent(query, expected):
    parser = SemanticParser()  # No MiniLM — uses keyword/regex paths
    intent = parser.parse_verify_intent(query)

    for key, value in expected.items():
        assert intent[key] == value, (
            f"Intent[{key!r}] expected {value!r}, got {intent[key]!r}\n" f"  query:  {query!r}\n" f"  intent: {intent}"
        )
