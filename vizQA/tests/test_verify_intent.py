"""
Parametrized tests for SemanticParser.parse_verify_intent.

Covers keyword extraction, color, position, state, negation (regex fast-path
and semantic slow-path), and combined cases — all without requiring MiniLM
(uses keyword/regex fallbacks).
"""

import pytest

from vizQA.reasoning import SemanticParser

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
    # Former testing_scripts/repro_negation.py
    (
        "'Sign In' modal should appear in the center of the screen",
        {"negated": False, "subject": "modal", "keyword": "Sign In", "position": "center"},
    ),
    (
        "'Sign In' modal should disappear",
        {"negated": True, "subject": "modal", "keyword": "Sign In"},
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


# ---------------------------------------------------------------------------
# Filter Elements by Intent tests (10 Tricky Cases)
# ---------------------------------------------------------------------------

FILTER_TEST_CASES = [
    # 1. Exact quoted match vs similar text
    (
        {"keyword": "Settings"},
        [
            {"text": "General Settings", "label": "btn1"},
            {"text": "Settings", "label": "btn2"},
        ],
        [{"text": "Settings", "label": "btn2"}],
    ),
    # 2. Multi-constraint: red button at top
    (
        {"subject": "button", "color": "red", "position": "top"},
        [
            {"text": "button", "color": "red", "location": [0.1, 0.5, 0.1, 0.1]},  # top-center
            {"text": "button", "color": "blue", "location": [0.1, 0.5, 0.1, 0.1]},
            {"text": "button", "color": "red", "location": [0.8, 0.5, 0.1, 0.1]},  # bottom
        ],
        [{"text": "button", "color": "red", "location": [0.1, 0.5, 0.1, 0.1]}],
    ),
    # 3. Spatial disambiguation: "right button"
    (
        {"subject": "button", "position": "right"},
        [
            {"text": "button", "location": [0.5, 0.1, 0.1, 0.1]},  # left
            {"text": "button", "location": [0.5, 0.8, 0.1, 0.1]},  # right
        ],
        [{"text": "button", "location": [0.5, 0.8, 0.1, 0.1]}],
    ),
    # 4. State filtering: disabled confirm button
    (
        {"subject": "confirm button", "state": "disabled"},
        [
            {"text": "confirm button", "state": "enabled"},
            {"text": "confirm button", "attributes": "disabled primary"},
        ],
        [{"text": "confirm button", "attributes": "disabled primary"}],
    ),
    # 5. False positive avoidance: unrelated elements
    (
        {"subject": "Submit"},
        [
            {"text": "Cancel", "name": "cancel_btn"},
            {"placeholder": "Username", "name": "user_input"},
        ],
        [],
    ),
    # 6. Negative verification intent (verify_negation logic)
    # Note: filter_elements_by_intent returns what IS found. If it returns [], negation is satisfied.
    (
        {"subject": "modal", "negated": True},
        [{"text": "Login Modal"}],
        [{"text": "Login Modal"}],  # It should find it if it's there
    ),
    # 7. Quoted mismatch with label/name
    (
        {"keyword": "Logout"},
        [
            {"label": "Sign Out", "name": "logout_btn"},
            {"text": "Logging out...", "name": "status"},
        ],
        [{"label": "Sign Out", "name": "logout_btn"}],
    ),
    # 8. Invisible elements (not really filtered by hidden by default, but check attributes)
    (
        {"subject": "popup", "state": "hidden"},
        [
            {"text": "popup", "state": "visible"},
            {"text": "popup", "attributes": "style: display:none; hidden"},
        ],
        [{"text": "popup", "attributes": "style: display:none; hidden"}],
    ),
    # 9. Substring confusion: Query 'Add' vs Element 'Address' (Exact keyword priority)
    (
        {"keyword": "Add"},
        [
            {"text": "Address Line 1"},
            {"text": "Add Item"},
        ],
        [{"text": "Add Item"}],
    ),
    # 10. Tricky overlapping dimensions: centered red icon
    (
        {"subject": "icon", "color": "red", "position": "center"},
        [
            {"text": "icon", "color": "blue", "location": [0.5, 0.5, 0.1, 0.1]},
            {"text": "icon", "color": "red", "location": [0.1, 0.1, 0.1, 0.1]},
            {"text": "icon", "color": "red", "location": [0.5, 0.5, 0.1, 0.1]},
        ],
        [{"text": "icon", "color": "red", "location": [0.5, 0.5, 0.1, 0.1]}],
    ),
]


@pytest.mark.parametrize("intent,elements,expected", FILTER_TEST_CASES)
def test_filter_elements_by_intent(intent, elements, expected):
    parser = SemanticParser()
    actual = parser.filter_elements_by_intent(intent, elements)

    # Simplified comparison of results
    actual_data = [
        {k: v for k, v in el.items() if k in ["text", "label", "name", "placeholder", "color", "state", "attributes"]}
        for el in actual
    ]
    expected_data = [
        {k: v for k, v in el.items() if k in ["text", "label", "name", "placeholder", "color", "state", "attributes"]}
        for el in expected
    ]

    assert len(actual) == len(expected), f"Length mismatch: got {len(actual)}, expected {len(expected)}"
    # Verify that the correct elements were picked (checking a few identifying keys)
    for a, e in zip(actual_data, expected_data):
        assert a == e
