"""
Shared structural typing contracts for semantic models.
"""

from typing import Any, List, Optional, Protocol


class SemanticModel(Protocol):
    """Protocol for models used by parser and ranking components."""

    def encode(self, text: str) -> Any:
        """Embed a string into a vector-like object."""

    def cosine_similarity(self, a: Any, b: Any) -> float:
        """Return cosine similarity between two vector-like objects."""

    def split_boolean_query(self, query: str) -> List[List[str]]:
        """Split a query into OR groups of AND terms."""

    def normalize_boolean_term(self, term: str) -> str:
        """Normalize a boolean term before embedding."""

    def semantic_match(self, query: str, candidates: List[str], threshold: float = 0.7) -> List[int]:
        """Return candidate indices that semantically match a query."""

    def classify_anchor_group(
        self, text: str, threshold: float = 0.6, groups: Optional[List[str]] = None
    ) -> Optional[str]:
        """Classify text against known semantic anchor groups."""

    def is_negation(self, text: str, threshold: float = 0.4, logit_threshold: float = 0.7, delta: float = 0.02) -> bool:
        """Return whether the text expresses a negation-style intent."""
