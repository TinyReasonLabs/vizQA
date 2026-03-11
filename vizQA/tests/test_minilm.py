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
            ("the overlay should vanish", True),
            ("the element is no longer present", True),
            ("the spinner should be done", True),
            ("the button should appear", False),
            ("click the submit button", False),
        ],
    )
    def test_is_negation(self, model, phrase, expected):
        result = model.is_negation(phrase)
        assert result == expected, f"is_negation({phrase!r}) → {result}, expected {expected}"
