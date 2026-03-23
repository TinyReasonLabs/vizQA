from unittest.mock import MagicMock

import numpy as np
import pytest

from vizQA.parser import SemanticParser
from vizQA.ranking import MetadataGenerator, RankingEngine

ELEMENTS = [
    {
        "text": "Submit",
        "label": "submit-btn",
        "name": "submit",
        "color": "blue",
        "state": "enabled",
        "location": [0.8, 0.8, 0.1, 0.05],  # Bottom Right
        "salience": 0.9,
    },
    {
        "text": "Cancel",
        "label": "cancel-btn",
        "name": "cancel",
        "color": "red",
        "state": "enabled",
        "location": [0.8, 0.2, 0.1, 0.05],  # Bottom Left
        "salience": 0.4,
    },
    {
        "text": "Sidebar Item",
        "label": "nav-item",
        "name": "nav",
        "color": "gray",
        "state": "enabled",
        "location": [0.2, 0.1, 0.1, 0.1],  # Top Left
        "section": "Sidebar",
        "salience": 0.6,
    },
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


class TestAdvancedRanking:
    def test_metadata_generation(self):
        meta = MetadataGenerator.generate(ELEMENTS[2])
        assert "[Section: Sidebar]" in meta
        assert "[Location: Top Left]" in meta
        assert "[Color: gray]" in meta

    def test_ranking_engine_with_quotes(self):
        # Mock MiniLM to return predictable embeddings
        mock_minilm = MagicMock()
        mock_minilm.encode.return_value = np.zeros(384)
        mock_minilm.cosine_similarity.return_value = 0.5

        engine = RankingEngine(mock_minilm)
        # Quoted "Submit" should be top
        results = engine.rank("Submit", _intent(keyword="Submit"), ELEMENTS)
        assert results[0]["text"] == "Submit"
        assert results[0]["_ranking_score"] > 5.0

    def test_ranking_engine_with_salience(self):
        mock_minilm = MagicMock()
        mock_minilm.encode.return_value = np.zeros(384)
        mock_minilm.cosine_similarity.return_value = 0.5

        engine = RankingEngine(mock_minilm)
        # "main button" should boost high salience
        results = engine.rank("main button", _intent(subject="main button"), ELEMENTS)
        # Submit has salience 0.9, Cancel has 0.4
        # Multiplier for Submit: 1.0 + (0.9 - 0.5) * 2 = 1.8
        # Multiplier for Cancel: 1.0 + (0.4 - 0.5) * 2 = 0.8

        # Find index of Submit and Cancel in results to compare scores
        submit_score = next(el["_ranking_score"] for el in results if el["text"] == "Submit")
        cancel_score = next(el["_ranking_score"] for el in results if el["text"] == "Cancel")
        assert submit_score > cancel_score

    def test_settings_example_from_logs(self):
        elements = [
            {
                "id": "el_f5ddb878a921",
                "type": "input",
                "label": "Campaigns Settings",
                "placeholder": "Settings",
                "salience": 0.71,
                "spatial": {"position": "middle-left"},
            },
            {
                "id": "el_de8ed14d042b",
                "type": "input",
                "label": "Settings",
                "placeholder": "DashboardOverview",
                "salience": 0.51,
                "spatial": {"position": "top-left"},
            },
        ]

        mock_minilm = MagicMock()
        mock_minilm.encode.return_value = np.zeros(384)
        mock_minilm.cosine_similarity.return_value = 0.5

        engine = RankingEngine(mock_minilm)
        intent = {"keyword": "Settings", "subject": "Main page header"}
        results = engine.rank("Settings", intent, elements)

        assert len(results) == 2
        # el_de8ed14d042b has exact match in 'label', el_f5ddb878a921 has exact in 'placeholder'
        # Both should get +5.0 boost.
        assert all(el["_ranking_score"] > 5.0 for el in results)

    def test_exact_vs_synonym_priority(self):
        elements = [{"text": "Login", "id": "login"}, {"text": "Sign in", "id": "signin"}]
        mock_minilm = MagicMock()
        mock_minilm.encode.return_value = np.zeros(384)
        # Mock semantic similarity to be high for both
        mock_minilm.cosine_similarity.return_value = 0.8

        engine = RankingEngine(mock_minilm)
        # Searching for "Sign in"
        results = engine.rank("Sign in", _intent(keyword="Sign in"), elements)

        assert results[0]["id"] == "signin"
        assert results[1]["id"] == "login"
        assert results[0]["_ranking_score"] > results[1]["_ranking_score"] + 4.0

    def test_perception_similarity_influence(self):
        elements = [
            {"text": "Button A", "id": "a", "similarity": 10.0},
            {"text": "Button B", "id": "b", "similarity": 40.0},
        ]
        mock_minilm = MagicMock()
        mock_minilm.encode.return_value = np.zeros(384)
        mock_minilm.cosine_similarity.return_value = 0.5

        engine = RankingEngine(mock_minilm)
        results = engine.rank("button", _intent(keyword="button"), elements)

        # Button B should have higher score due to similarity (40/50 = 0.8 boost vs 10/50 = 0.2)
        score_a = next(el["_ranking_score"] for el in results if el["id"] == "a")
        score_b = next(el["_ranking_score"] for el in results if el["id"] == "b")
        assert score_b > score_a

    def test_parser_configurability(self):
        # Without advanced ranking, should use baseline
        parser_baseline = SemanticParser()
        parser_baseline.config.use_advanced_ranking = False
        # Should not have _ranking_score attached to filtered elements
        result = parser_baseline.filter_elements_by_intent(_intent(keyword="Submit"), ELEMENTS)
        assert "_ranking_score" not in result[0]

        # With advanced ranking (mock minilm needed)
        mock_minilm = MagicMock()
        mock_minilm.encode.return_value = np.zeros(384)
        mock_minilm.cosine_similarity.return_value = 0.5
        parser_adv = SemanticParser(minilm=mock_minilm)
        parser_adv.config.use_advanced_ranking = True
        result_adv = parser_adv.filter_elements_by_intent(_intent(keyword="Submit"), ELEMENTS)
        assert "_ranking_score" in result_adv[0]
