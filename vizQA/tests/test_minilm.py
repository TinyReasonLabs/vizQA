"""
Tests for MiniLM class methods.

These tests require the actual model weights to be present at
``vizQA/weights/minilm/``.  When the weights are absent the entire module is
skipped gracefully so CI environments without downloaded weights are not
broken.
"""

import os

import pytest

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "weights", "minilm")
_model_available = os.path.exists(os.path.join(MODEL_DIR, "model.onnx")) and os.path.exists(
    os.path.join(MODEL_DIR, "tokenizer.json")
)

pytestmark = pytest.mark.skipif(
    not _model_available,
    reason="MiniLM weights not found — skipping MiniLM inference tests",
)


@pytest.fixture(scope="module")
def model():
    from vizQA.minilm import MiniLM  # pylint: disable=import-outside-toplevel

    return MiniLM(MODEL_DIR)


# ---------------------------------------------------------------------------
# semantic_match
# ---------------------------------------------------------------------------
class TestSemanticMatch:
    def test_identical_strings_match(self, model):
        matched = model.semantic_match("login button", ["login button", "cancel", "spinner"])
        assert 0 in matched

    def test_synonym_matches(self, model):
        matched = model.semantic_match("sign in button", ["login button", "header nav", "footer"])
        # "sign in button" should be semantically close to "login button"
        assert 0 in matched

    def test_unrelated_does_not_match(self, model):
        matched = model.semantic_match("submit form", ["loading spinner", "background image"], threshold=0.8)
        assert matched == []

    def test_threshold_respected(self, model):
        # High threshold — only exact or near-exact allowed
        matched_strict = model.semantic_match("login button", ["login button"], threshold=0.99)
        matched_loose = model.semantic_match("login button", ["login button"], threshold=0.50)
        # High threshold may not match, but loose must
        assert 0 in matched_loose

    def test_or_matches_any_clause(self, model):
        matched = model.semantic_match("login or cancel", ["login", "cancel", "spinner"])
        assert set(matched) == {0, 1}

    def test_and_requires_all_clauses(self, model):
        matched = model.semantic_match(
            "invalid and credentials", ["invalid credentials", "invalid", "credentials", "spinner"]
        )
        assert matched == [0]

    def test_mixed_or_and_respects_grouping(self, model):
        matched = model.semantic_match(
            "login or invalid and credentials",
            ["login", "invalid credentials", "invalid", "credentials", "spinner"],
        )
        assert set(matched) == {0, 1}

    def test_quoted_and_is_not_split_inside_phrase(self, model):
        matched = model.semantic_match(
            "'Verify and Continue' or resume",
            ["Verify and Continue", "resume", "Continue", "spinner"],
        )
        assert set(matched) == {0, 1}


# ---------------------------------------------------------------------------
# rank_candidates
# ---------------------------------------------------------------------------
class TestRankCandidates:
    def test_returns_sorted_descending(self, model):
        results = model.rank_candidates("login button", ["sign in button", "background", "submit form"])
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_threshold_filters(self, model):
        results = model.rank_candidates("login button", ["background noise", "random text"], threshold=0.9)
        # Very high threshold — likely no results for unrelated strings
        assert all(r["score"] >= 0.9 for r in results)

    def test_includes_index_and_text(self, model):
        candidates = ["login button", "cancel link"]
        results = model.rank_candidates("login button", candidates, threshold=0.0)
        for r in results:
            assert "index" in r and "text" in r and "score" in r


# ---------------------------------------------------------------------------
# classify_anchor_group / is_negation
# ---------------------------------------------------------------------------
class TestIntentClassification:
    @pytest.mark.parametrize(
        "word,expected_group",
        [
            ("red", "color"),
            ("blue", "color"),
            ("hidden", "state"),
            ("disabled", "state"),
            ("visible", "state"),
            ("top", "position"),
            ("bottom-right", "position"),
        ],
    )
    def test_classify_obvious_words(self, model, word, expected_group):
        result = model.classify_anchor_group(word, threshold=0.40)
        assert result == expected_group, f"{word!r} → {result!r}, expected {expected_group!r}"

    @pytest.mark.parametrize(
        "phrase,expected",
        [
            # --- CLEAR NEGATIONS ---
            ("the overlay should disappear", True),
            ("the element is no longer present", True),
            ("the spinner should be gone", True),
            ("the modal should close", True),
            ("the popup closes", True),
            ("the banner disappears", True),
            ("the tooltip should vanish", True),
            ("the dialog is removed", True),
            ("the element is absent", True),
            ("the loading indicator is no longer visible", True),
            ("the dropdown should collapse", True),
            ("the sidebar should be hidden", True),
            ("the notification disappears after a second", True),
            ("the modal is dismissed", True),
            ("the element gets removed from the DOM", True),
            # --- CLEAR POSITIVES ---
            ("the button should appear", False),
            ("the modal should appear in the center", False),
            ("the banner is visible", False),
            ("the element is present", False),
            ("the dropdown opens", False),
            ("the tooltip shows up", False),
            ("the dialog is displayed", False),
            ("the notification appears", False),
            ("the spinner is visible", False),
            ("the sidebar is shown", False),
            ("the modal becomes visible", False),
            ("the element is rendered", False),
            ("the button is shown", False),
            ("the popup stays", False),
            # --- IRRELEVANT UI ACTIONS ---
            ("click the submit button", False),
            ("tap the login link", False),
            ("scroll down the page", False),
            ("enter your email address", False),
            ("type the password", False),
            ("navigate to the dashboard", False),
            ("select an option from the dropdown", False),
            ("hover over the icon", False),
            ("press the confirm button", False),
            ("open the settings page", False),
            # --- TEMPORAL / STATE TRANSITIONS ---
            ("the loader disappears after completion", True),
            ("the modal closes after clicking outside", True),
            ("the popup is no longer shown", True),
            ("the banner eventually disappears", True),
            ("the spinner stops showing", True),
            ("the element is hidden", True),
            ("the modal appears after clicking the button", False),
            ("the dropdown becomes visible", False),
            ("the tooltip shows after hover", False),
            # --- TRICKY NEGATIONS ---
            ("the element is not visible", True),
            ("the element is not present", True),
            ("the modal is not shown", True),
            ("the popup is not displayed", True),
            ("the button is not visible anymore", True),
            # --- AMBIGUOUS / EDGE CASES ---
            ("the spinner should be done", True),  # borderline
            ("loading is finished", True),
            ("the process completes", True),
            ("the request is completed", True),
            ("'the operation finished successfully' should appear", False),
            # --- MIXED SENTENCES ---
            ("click the button and the modal disappears", True),
            ("submit the form and the spinner disappears", True),
            ("open the menu and the dropdown appears", False),
            ("click outside and the modal closes", True),
            # --- PASSIVE VOICE ---
            ("the modal is closed", True),
            ("the popup is hidden", True),
            ("the banner is removed", True),
            ("the modal is opened", False),
            ("the popup is displayed", False),
            # --- SHORT PHRASES ---
            ("disappears", True),
            ("vanishes", True),
            ("gone", True),
            ("removed", True),
            ("appears", False),
            ("visible", False),
            ("shown", False),
            # --- NOISE / RANDOM ---
            ("lorem ipsum dolor sit amet", False),
            ("hello world", False),
            ("this is a test", False),
            ("random text here", False),
            # --- SUBTLE UI LANGUAGE ---
            ("the element fades out", True),
            ("the modal fades away", True),
            ("the banner slides out of view", True),
            ("the element fades in", False),
            ("the modal slides into view", False),
            # --- CONDITIONALS ---
            ("if successful, the modal closes", True),
            ("once submitted, the spinner disappears", True),
            ("when clicked, the dropdown appears", False),
            # --- NEGATION WITH CONTEXT ---
            ("the error message is no longer displayed", True),
            ("the warning banner is no longer visible", True),
            ("the success message disappears", True),
            ("the success message is displayed", False),
            ("the error message appears", False),
        ],
    )
    def test_is_negation(self, model, phrase, expected):
        result = model.is_negation(phrase)
        assert result == expected, f"is_negation({phrase!r}) → {result}, expected {expected}"
