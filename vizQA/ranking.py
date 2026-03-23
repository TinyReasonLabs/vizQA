"""
Modular ranking pipeline for intent-element matching.
Implements a dual-vector strategy: label-based retrieval + metadata-based re-ranking.
"""

import re
from typing import Any, Dict, List, Optional

import numpy as np


# pylint: disable=too-few-public-methods
class MetadataGenerator:
    """
    Converts structured element data into natural language metadata strings.
    """

    @staticmethod
    def generate(element: Dict[str, Any]) -> str:
        """
        Generates a descriptive sentence about the element's physical state and context.

        :param element: The element to generate metadata for.
        :return: The metadata string.
        """
        parts = []

        # Section/Context
        section = element.get("section") or element.get("context")
        if section:
            parts.append(f"[Section: {section}]")

        # Location/Position
        loc = element.get("location")  # [y, x, w, h] normalized
        if loc and len(loc) >= 2:
            y, x = loc[0], loc[1]
            h_pos = "Left" if x < 0.33 else ("Right" if x > 0.66 else "Center")
            v_pos = "Top" if y < 0.33 else ("Bottom" if y > 0.66 else "Middle")
            parts.append(f"[Location: {v_pos} {h_pos}]")
        elif "spatial" in element and isinstance(element["spatial"], dict):
            pos = element["spatial"].get("position")
            if pos:
                parts.append(f"[Location: {pos.replace('-', ' ').title()}]")

        # Appearance/Color
        color = element.get("color")
        if color:
            parts.append(f"[Color: {color}]")

        # Importance/Salience
        salience = element.get("salience", 0.5)
        if salience > 0.7:
            parts.append("[Appearance: Prominent]")
        elif salience < 0.3:
            parts.append("[Appearance: Subtle]")

        # State
        state = element.get("state")
        if state:
            parts.append(f"[State: {state}]")

        return " ".join(parts)


# pylint: disable=too-few-public-methods
class SparseRanker:
    """
    Heuristic keyword-overlap ranker (fallback for BM25).
    """

    @staticmethod
    def score(query: str, candidates: List[str]) -> List[float]:
        """
        Calculates the score based on keyword overlap.

        :param query: The query string.
        :param candidates: The list of candidates.
        :return: The score.
        """
        if not query:
            return [0.0] * len(candidates)

        q_words = set(re.findall(r"\w+", query.lower()))
        scores = []
        for cand in candidates:
            if not cand:
                scores.append(0.0)
                continue
            c_words = set(re.findall(r"\w+", cand.lower()))
            overlap = len(q_words.intersection(c_words))
            # simple Jaccard-ish or just overlap count
            score = overlap / len(q_words) if q_words else 0.0
            scores.append(score)
        return scores


# pylint: disable=too-few-public-methods
class SemanticReRanker:
    """
    Phase 2: Re-ranks candidates based on metadata similarity to the query.
    """

    def __init__(self, minilm: Any):
        self.minilm = minilm

    def calculate_context_score(self, query_vec: np.ndarray, metadata_str: str) -> float:
        """
        Calculates the context score based on metadata similarity to the query.

        :param query_vec: The query vector.
        :param metadata_str: The metadata string.
        :return: The context score.
        """
        if not metadata_str:
            return 0.0
        meta_vec = self.minilm.encode(metadata_str)
        return self.minilm.cosine_similarity(query_vec, meta_vec)


# pylint: disable=too-few-public-methods
class SalienceScorer:
    """
    Phase 3: Adjusts scores based on visual salience and query modifiers.
    """

    def boost(self, query: str, element: Dict[str, Any], current_score: float) -> float:
        """
        Boosts the score based on salience and query modifiers.

        :param query: The query string.
        :param element: The element to rank.
        :param current_score: The current score.
        :return: The boosted score.
        """
        salience = element.get("salience", 0.5)
        q_lower = query.lower()

        multiplier = 1.0
        if any(w in q_lower for w in ["main", "prominent", "large", "primary"]):
            # Boost if salience is high
            multiplier += (salience - 0.5) * 2.0
        elif any(w in q_lower for w in ["subtle", "small", "secondary", "background"]):
            # Boost if salience is low
            multiplier += (0.5 - salience) * 2.0

        return current_score * max(0.5, multiplier)


# pylint: disable=too-few-public-methods
class QuoteScorer:
    """
    Phase 3: Pins exact quoted matches to the top.
    """

    def boost(self, intent: Dict[str, Any], element: Dict[str, Any], current_score: float) -> float:
        """
        Boosts the score if the element matches the keyword.

        :param intent: The intent dictionary.
        :param element: The element to rank.
        :param current_score: The current score.
        :return: The boosted score.
        """
        keyword = intent.get("keyword")
        if not keyword:
            return current_score

        # Check for exact or partial match in label fields
        label_fields = [
            str(element.get("text") or ""),
            str(element.get("label") or ""),
            str(element.get("name") or ""),
            str(element.get("placeholder") or ""),
        ]
        keyword_lower = keyword.lower()

        max_boost = 0.0
        for field in label_fields:
            if not field:
                continue
            field_lower = field.lower()
            if keyword_lower == field_lower:
                max_boost = max(max_boost, 5.0)  # Pin exact matches
            elif keyword_lower in field_lower:
                max_boost = max(max_boost, 2.0)  # Significant boost for partial matches

        return current_score + max_boost


# pylint: disable=too-few-public-methods
class RankingEngine:
    """
    Orchestrates the multi-phase ranking pipeline.
    """

    def __init__(self, minilm: Optional[Any] = None):
        self.minilm = minilm
        self.reranker = SemanticReRanker(minilm) if minilm else None
        self.salience_scorer = SalienceScorer()
        self.quote_scorer = QuoteScorer()

    # pylint: disable=too-many-locals
    def rank(self, query: str, intent: Dict[str, Any], elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Ranks elements based on query and intent.

        :param query: The query string.
        :param intent: The intent dictionary.
        :param elements: The list of elements to rank.
        :return: The ranked list of elements.
        """
        if not elements:
            return []

        # retrieval_query: Primary keyword or subject
        # context_query: modifiers and spatial context
        retrieval_query = intent.get("keyword") or intent.get("subject") or query
        context_query = intent.get("subject") or query

        # 1. Prepare candidates
        # We preserve the original values for display/metadata but use a cleaned version for matching
        labels = []
        for elem in elements:
            # Combine all text-like fields
            parts = [
                str(elem.get("placeholder") or ""),
                str(elem.get("text") or ""),
                str(elem.get("label") or ""),
                str(elem.get("name") or ""),
            ]
            labels.append(" ".join(filter(None, parts)))

        metadata_strs = [MetadataGenerator.generate(elem) for elem in elements]

        # 2. Phase 1: Retrieval (Dense + Sparse)
        scores = np.zeros(len(elements))

        if self.minilm:
            # Dense scores on retrieval query
            q_vec = self.minilm.encode(retrieval_query)
            c_vec = self.minilm.encode(context_query) if context_query != retrieval_query else q_vec

            for i, label in enumerate(labels):
                if label:
                    l_vec = self.minilm.encode(label)
                    # Dense similarity mapped to [0, 1] for stability
                    sim = self.minilm.cosine_similarity(q_vec, l_vec)
                    scores[i] += max(0, sim)

            # 3. Phase 2: Context Re-ranking
            # Use context_query against metadata
            for i, meta_str in enumerate(metadata_strs):
                if meta_str:
                    context_sim = self.reranker.calculate_context_score(c_vec, meta_str)
                    # Apply context multiplier (up to 2.0x)
                    multiplier = 1.0 + max(0, context_sim)
                    scores[i] *= multiplier

        # Sparse scores (Keyword overlap) always added to ensure hits on partial/keyword matches
        sparse_scores = SparseRanker.score(retrieval_query, labels)
        for i, s in enumerate(sparse_scores):
            scores[i] += s * 0.5

        # 4. Phase 3: Modular Heuristics
        for i, el in enumerate(elements):
            scores[i] = self.salience_scorer.boost(query, el, scores[i])
            scores[i] = self.quote_scorer.boost(intent, el, scores[i])

            # Incorporate perception similarity if available
            # Scale it: assume ~25.0-50.0 is a strong match from the model
            p_sim = el.get("similarity", 0.0)
            if p_sim > 0:
                scores[i] += min(1.0, p_sim / 50.0)

        # Attach scores and sort
        scored_elements = []
        for i, el in enumerate(elements):
            el_copy = el.copy()
            el_copy["_ranking_score"] = float(scores[i])
            scored_elements.append(el_copy)

        scored_elements.sort(key=lambda x: x["_ranking_score"], reverse=True)

        # Pruning: use a stricter threshold than just > 0 to avoid false positives
        # especially in verification tasks where we want high confidence.
        min_prune_threshold = intent.get("threshold", 0.4)
        final_elements = [el for el in scored_elements if el["_ranking_score"] > min_prune_threshold]

        return final_elements
