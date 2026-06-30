"""
Tests for SemanticParser.filter_elements_by_intent.

Uses mock element dicts so that no perception API is required.
Covers: semantic baseline matching, non-destructive color/state stacking,
and the substring fallback path (no MiniLM).
"""

import os
from unittest.mock import MagicMock

import pytest

from vizQA.reasoning import MiniLM, SemanticParser

# ---------------------------------------------------------------------------
# Mock elements (shared across test cases)
# ---------------------------------------------------------------------------
ELEMENTS = [
    {"text": "Submit", "label": "submit-btn", "name": "submit", "color": "blue", "state": "enabled"},
    {"text": "Cancel", "label": "cancel-btn", "name": "cancel", "color": "red", "state": "enabled"},
    {"text": "Error: invalid credentials", "label": "error-msg", "name": "error", "color": "red", "state": ""},
    {"text": "Loading...", "label": "spinner", "name": "spinner", "color": "", "state": ""},
]


def _intent(keyword=None, color=None, position=None, state=None, negated=False, subject=""):
    return {
        "keyword": keyword,
        "color": color,
        "position": position,
        "state": state,
        "negated": negated,
        "subject": subject,
    }


# ---------------------------------------------------------------------------
#  Substring fallback (no MiniLM)
# ---------------------------------------------------------------------------
class TestSubstringFallback:
    def test_keyword_exact_match(self):
        parser = SemanticParser()
        result = parser.filter_elements_by_intent(_intent(keyword="Submit"), ELEMENTS)
        assert len(result) == 1
        assert result[0]["text"] == "Submit"

    def test_keyword_partial_match(self):
        parser = SemanticParser()
        result = parser.filter_elements_by_intent(_intent(keyword="Error"), ELEMENTS)
        # "Error: invalid credentials" contains "error" case-insensitively
        assert any("Error" in (el.get("placeholder") or el.get("text", "")) for el in result)

    def test_no_keyword_returns_all(self):
        parser = SemanticParser()
        result = parser.filter_elements_by_intent(_intent(), ELEMENTS)
        assert result == ELEMENTS

    def test_color_filter_non_destructive(self):
        """
        Color filter on 'Submit' (which is blue) shouldn't erase 'Submit'
        when asking for 'Submit' with color='red' — the semantic match takes
        priority and the empty color-filtered list is discarded.
        """
        parser = SemanticParser()
        result = parser.filter_elements_by_intent(_intent(keyword="Submit", color="red"), ELEMENTS)
        # Color filter would give nothing; baseline (Submit) must be kept
        assert any(el["text"] == "Submit" for el in result)

    def test_color_filter_applied_when_productive(self):
        """
        When color filter produces a non-empty result from the baseline,
        it should narrow the list.
        """
        parser = SemanticParser()
        # subject="error" matches element[2]; color="red" also matches it
        result = parser.filter_elements_by_intent(_intent(subject="Error", color="red"), ELEMENTS)
        assert all(el.get("color") == "red" for el in result)

    def test_empty_elements(self):
        parser = SemanticParser()
        result = parser.filter_elements_by_intent(_intent(keyword="Submit"), [])
        assert result == []

    def test_filter_target_candidates_prefers_exact_phrase(self):
        parser = SemanticParser()
        candidates = [
            {"text": "General Settings", "label": "settings-panel"},
            {"text": "Settings", "label": "settings-button"},
        ]

        result = parser.filter_target_candidates(_intent(keyword="Settings"), candidates)

        assert result == [candidates[1]]

    def test_filter_target_candidates_falls_back_to_overlap_without_stop_words(self):
        parser = SemanticParser()
        candidates = [
            {"text": "Debug Controls", "label": "debug-controls"},
            {"text": "Welcome Card", "label": "welcome-card"},
            {"text": "Section 1", "label": "section"},
        ]

        result = parser.filter_target_candidates(_intent(subject="debug controls section"), candidates)

        assert result == [candidates[0]]

    def test_filter_target_candidates_prefers_earlier_specific_terms_on_ties(self):
        parser = SemanticParser()
        candidates = [
            {"text": "Reviews", "label": "reviews-link"},
            {"text": "Section 1", "label": "section"},
        ]

        result = parser.filter_target_candidates(_intent(subject="reviews section"), candidates)

        assert result == [candidates[0]]


# ---------------------------------------------------------------------------
# With MiniLM
# ---------------------------------------------------------------------------
class TestWithMiniLM:
    def _make_parser(self):
        """Creates a SemanticParser backed by the real MiniLM weights."""
        model_dir = os.path.join("vizQA", "weights", "minilm")
        if not os.path.exists(model_dir):
            print("Model not found, skipping...")
            return

        model = MiniLM(model_dir, logger=MagicMock())  # Pass a mock logger to avoid initializing a real logger in tests
        return SemanticParser(minilm=model)

    def test_semantic_match_used(self):
        parser = self._make_parser()
        result = parser.filter_elements_by_intent(_intent(keyword="error or submit"), ELEMENTS)
        assert len(result) == 2
        assert {el["text"] for el in result} == {"Submit", "Error: invalid credentials"}

    def test_semantic_and_requires_both_concepts(self):
        parser = self._make_parser()
        result = parser.filter_elements_by_intent(_intent(keyword="error and credentials"), ELEMENTS)
        assert len(result) == 1
        assert result[0]["text"] == "Error: invalid credentials"

    def test_semantic_mixed_or_and_groups_clauses(self):
        parser = self._make_parser()
        result = parser.filter_elements_by_intent(_intent(keyword="submit or error and credentials"), ELEMENTS)
        assert len(result) == 2
        assert {el["text"] for el in result} == {"Submit", "Error: invalid credentials"}

    def test_semantic_boolean_query_preserves_quoted_phrase(self):
        parser = self._make_parser()
        elements = ELEMENTS + [
            {
                "text": "Verify and Continue",
                "label": "verify-and-continue-button",
                "name": "verify-continue",
                "color": "blue",
                "state": "enabled",
            }
        ]
        result = parser.filter_elements_by_intent(_intent(keyword="'Verify and Continue' or submit"), elements)
        texts = [el["text"] for el in result]
        assert "Submit" in texts
        assert "Verify and Continue" in texts

    def test_fallback_to_none_when_no_match(self):
        """When semantic_match returns nothing, filter should return empty list."""
        parser = self._make_parser()
        result = parser.filter_elements_by_intent(_intent(keyword="nonexistent"), ELEMENTS)
        assert result == []

    def test_color_filter_non_destructive_with_minilm(self):
        """Semantic baseline = [Submit]; color='red' gives nothing → keep [Submit]."""
        parser = self._make_parser()
        result = parser.filter_elements_by_intent(_intent(keyword="Submit", color="red"), ELEMENTS)
        assert any(el["text"] == "Submit" for el in result)
