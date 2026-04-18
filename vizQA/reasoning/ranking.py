"""
Modular ranking pipeline for intent-element matching.
Implements a dual-vector strategy: label-based retrieval + metadata-based re-ranking.
"""

import re
from typing import Any, Dict, List, Optional

import numpy as np

from vizQA.app.config import CONFIG
from vizQA.reasoning.model_protocols import SemanticModel
from vizQA.reasoning.query_semantics import (
    is_boolean_query,
    lexical_term_score,
    normalize_boolean_term,
    split_boolean_query,
)


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

        section = element.get("section") or element.get("context")
        if section:
            parts.append(f"[Section: {section}]")

        loc = element.get("location")
        if loc and len(loc) >= 2:
            y, x = loc[0], loc[1]
            h_pos = "Left" if x < 0.33 else ("Right" if x > 0.66 else "Center")
            v_pos = "Top" if y < 0.33 else ("Bottom" if y > 0.66 else "Middle")
            parts.append(f"[Location: {v_pos} {h_pos}]")
        elif "spatial" in element and isinstance(element["spatial"], dict):
            pos = element["spatial"].get("position")
            if pos:
                parts.append(f"[Location: {pos.replace('-', ' ').title()}]")

        color = element.get("color")
        if color:
            parts.append(f"[Color: {color}]")

        salience = element.get("salience", 0.5)
        if salience > 0.7:
            parts.append("[Appearance: Prominent]")
        elif salience < 0.3:
            parts.append("[Appearance: Subtle]")

        state = element.get("state")
        if state:
            parts.append(f"[State: {state}]")

        return " ".join(parts)


class SparseRanker:
    """Heuristic keyword-overlap ranker (fallback for BM25)."""

    @staticmethod
    def score(query: str, candidates: List[str]) -> List[float]:
        """Return simple keyword-overlap scores for each candidate string."""
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
            score = overlap / len(q_words) if q_words else 0.0
            scores.append(score)
        return scores


class SemanticReRanker:
    """Phase 2: Re-ranks candidates based on metadata similarity to the query."""

    def __init__(self, minilm: Any):
        self.minilm = minilm

    def calculate_context_score(self, query_vec: np.ndarray, metadata_str: str) -> float:
        """Score metadata text against a precomputed query embedding."""
        if not metadata_str:
            return 0.0
        meta_vec = self.minilm.encode(metadata_str)
        return self.minilm.cosine_similarity(query_vec, meta_vec)


class SalienceScorer:
    """Phase 3: Adjusts scores based on visual salience and query modifiers."""

    def boost(self, query: str, element: Dict[str, Any], current_score: float) -> float:
        """Boost a score when query wording implies salience preferences."""
        salience = element.get("salience", 0.5)
        q_lower = query.lower()

        multiplier = 1.0
        if any(w in q_lower for w in ["main", "prominent", "large", "primary"]):
            multiplier += (salience - 0.5) * 2.0
        elif any(w in q_lower for w in ["subtle", "small", "secondary", "background"]):
            multiplier += (0.5 - salience) * 2.0

        return current_score * max(0.5, multiplier)


class QuoteScorer:
    """Phase 3: Pins exact quoted matches to the top."""

    def boost(self, intent: Dict[str, Any], element: Dict[str, Any], current_score: float) -> float:
        """Boost elements whose text-like fields match the quoted keyword closely."""
        keyword = intent.get("keyword")
        if not keyword:
            return current_score

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
                max_boost = max(max_boost, 5.0)
            elif keyword_lower in field_lower:
                max_boost = max(max_boost, 2.0)

        return current_score + max_boost


class RankingEngine:
    """Orchestrates the multi-phase ranking pipeline."""

    def __init__(self, minilm: Optional[SemanticModel] = None):
        self.minilm = minilm
        self.reranker = SemanticReRanker(minilm) if minilm else None
        self.salience_scorer = SalienceScorer()
        self.quote_scorer = QuoteScorer()

    def _best_boolean_dense_score(self, query: str, label: str) -> float:
        if not self.minilm or not label:
            return 0.0

        query_groups = split_boolean_query(query)
        boolean_query = is_boolean_query(query_groups)
        label_vec = self.minilm.encode(label)
        best_score = 0.0
        for group in query_groups:
            term_scores = []
            for term in group:
                lexical_score = lexical_term_score(term, label)
                normalized_term = normalize_boolean_term(term)
                semantic_score = max(0.0, self.minilm.cosine_similarity(self.minilm.encode(normalized_term), label_vec))
                if boolean_query:
                    if lexical_score > 0.0:
                        term_scores.append(1.0)
                    elif semantic_score >= CONFIG.semantic_match_threshold:
                        term_scores.append(semantic_score)
                    else:
                        term_scores = []
                        break
                else:
                    term_scores.append(max(lexical_score, semantic_score))
            if term_scores:
                best_score = max(best_score, min(term_scores))
        return best_score

    def _best_boolean_context_score(self, query: str, metadata_str: str) -> float:
        if not self.minilm or not metadata_str:
            return 0.0

        query_groups = split_boolean_query(query)
        boolean_query = is_boolean_query(query_groups)
        meta_vec = self.minilm.encode(metadata_str)
        best_score = 0.0
        for group in query_groups:
            term_scores = []
            for term in group:
                lexical_score = lexical_term_score(term, metadata_str)
                normalized_term = normalize_boolean_term(term)
                semantic_score = max(0.0, self.minilm.cosine_similarity(self.minilm.encode(normalized_term), meta_vec))
                if boolean_query:
                    if lexical_score > 0.0:
                        term_scores.append(1.0)
                    elif semantic_score >= CONFIG.semantic_match_threshold:
                        term_scores.append(semantic_score)
                    else:
                        term_scores = []
                        break
                else:
                    term_scores.append(max(lexical_score, semantic_score))
            if term_scores:
                best_score = max(best_score, min(term_scores))
        return best_score

    def _best_boolean_sparse_score(self, query: str, label: str) -> float:
        """Score lexical boolean matches, preserving non-boolean bag-of-words behavior."""
        query_groups = split_boolean_query(query)
        if not is_boolean_query(query_groups):
            scores = SparseRanker.score(query, [label])
            return scores[0] if scores else 0.0

        best_score = 0.0
        for group in query_groups:
            per_term_scores = [lexical_term_score(term, label) for term in group]
            if per_term_scores and all(score > 0.0 for score in per_term_scores):
                best_score = max(best_score, min(per_term_scores))
        return best_score

    # pylint: disable=too-many-locals
    def rank(self, query: str, intent: Dict[str, Any], elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank candidate elements using dense, sparse, and heuristic signals."""
        if not elements:
            return []

        retrieval_query = intent.get("keyword") or intent.get("subject") or query
        context_query = intent.get("subject") or query

        labels = []
        for elem in elements:
            parts = [
                str(elem.get("placeholder") or ""),
                str(elem.get("text") or ""),
                str(elem.get("label") or ""),
                str(elem.get("name") or ""),
            ]
            labels.append(" ".join(filter(None, parts)))

        metadata_strs = [MetadataGenerator.generate(elem) for elem in elements]
        scores = np.zeros(len(elements))

        if self.minilm:
            for i, label in enumerate(labels):
                scores[i] += self._best_boolean_dense_score(retrieval_query, label)

            for i, meta_str in enumerate(metadata_strs):
                if meta_str:
                    context_sim = self._best_boolean_context_score(context_query, meta_str)
                    multiplier = 1.0 + max(0, context_sim)
                    scores[i] *= multiplier

        for i, label in enumerate(labels):
            scores[i] += self._best_boolean_sparse_score(retrieval_query, label) * 0.5

        for i, el in enumerate(elements):
            scores[i] = self.salience_scorer.boost(query, el, scores[i])
            scores[i] = self.quote_scorer.boost(intent, el, scores[i])

            p_sim = el.get("similarity", 0.0)
            if p_sim > 0:
                scores[i] += min(1.0, p_sim / 50.0)

        scored_elements = []
        for i, el in enumerate(elements):
            el_copy = el.copy()
            el_copy["_ranking_score"] = float(scores[i])
            scored_elements.append(el_copy)

        scored_elements.sort(key=lambda x: x["_ranking_score"], reverse=True)

        min_prune_threshold = intent.get("threshold", CONFIG.semantic_match_threshold)
        final_elements = [el for el in scored_elements if el["_ranking_score"] > min_prune_threshold]

        return final_elements
