"""
Tests for SemanticParser.filter_elements_by_intent.

Uses mock element dicts so that no perception API is required.
Covers: semantic baseline matching, non-destructive color/state stacking,
and the substring fallback path (no MiniLM).
"""

from unittest.mock import MagicMock

import pytest

from vizQA.parser import SemanticParser

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


# ---------------------------------------------------------------------------
# With a stubbed MiniLM
# ---------------------------------------------------------------------------
class TestWithMockMiniLM:
    def _make_parser(self, match_indices):
        """Creates a SemanticParser with a MiniLM stub that returns *match_indices*."""
        mock_model = MagicMock()
        mock_model.semantic_match.return_value = match_indices
        return SemanticParser(minilm=mock_model)

    def test_semantic_match_used(self):
        parser = self._make_parser([0, 2])  # Submit and Error
        result = parser.filter_elements_by_intent(_intent(keyword="error or submit"), ELEMENTS)
        assert len(result) == 2
        assert result[0]["text"] == "Submit"
        assert result[1]["text"] == "Error: invalid credentials"

    def test_fallback_to_none_when_no_match(self):
        """When semantic_match returns nothing, filter should return empty list."""
        parser = self._make_parser([])
        result = parser.filter_elements_by_intent(_intent(keyword="nonexistent"), ELEMENTS)
        assert result == []

    def test_color_filter_non_destructive_with_minilm(self):
        """Semantic baseline = [Submit]; color='red' gives nothing → keep [Submit]."""
        parser = self._make_parser([0])  # semantic match: Submit only
        result = parser.filter_elements_by_intent(_intent(keyword="Submit", color="red"), ELEMENTS)
        assert any(el["text"] == "Submit" for el in result)
