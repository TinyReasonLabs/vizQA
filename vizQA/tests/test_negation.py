"""
Tests for SemanticParser.verify_negation.

Checks the three required outcomes:
  - Element present before AND absent after  → True  (disappeared)
  - Element present before AND still after   → False (did not disappear)
  - Element absent in both                   → False (can't confirm disappearance)
"""

import pytest

from vizQA.parser import SemanticParser

# ---------------------------------------------------------------------------
# Mock element sets
# ---------------------------------------------------------------------------
MODAL_ELEMENT = {"text": "Login Modal", "label": "modal", "name": "modal"}
ERROR_ELEMENT = {"text": "Error message", "label": "error-msg", "name": "error"}
SPINNER = {"text": "Loading...", "label": "spinner", "name": "spinner"}
OTHER = {"text": "Submit button", "label": "submit", "name": "submit"}


NEGATION_CASES = [
    # subject, before_elements, after_elements, expected
    # 1. Modal was visible, now gone → True
    ("modal", [MODAL_ELEMENT, OTHER], [OTHER], True),
    # 2. Error was visible, now gone → True
    ("error message", [ERROR_ELEMENT, MODAL_ELEMENT], [MODAL_ELEMENT], True),
    # 3. Spinner present before AND after → False
    ("spinner", [SPINNER, OTHER], [SPINNER, OTHER], False),
    # 4. Modal absent in both → False (cannot confirm it disappeared)
    ("modal", [OTHER, SPINNER], [OTHER], False),
    # 5. Empty before list → False
    ("modal", [], [OTHER], False),
    # 6. Empty after list (everything gone) → True when element was before
    ("modal", [MODAL_ELEMENT], [], True),
    # 7. Empty subject → False (nothing to look for)
    ("", [MODAL_ELEMENT], [], False),
]


@pytest.mark.parametrize("subject,before,after,expected", NEGATION_CASES)
def test_verify_negation(subject, before, after, expected):
    parser = SemanticParser()  # No MiniLM — uses substring fallback
    result = parser.verify_negation(before, after, subject)
    assert result == expected, (
        f"verify_negation({subject!r}, before={[e['text'] for e in before]}, "
        f"after={[e['text'] for e in after]}) → expected {expected}, got {result}"
    )
