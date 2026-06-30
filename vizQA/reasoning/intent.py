"""Canonical intent objects used by reasoning components."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional


@dataclass(frozen=True)
class IntentAttributes:
    """Normalized attributes extracted from verification or search text."""

    color: Optional[str] = None
    position: Optional[str] = None
    state: Optional[str] = None


# pylint: disable=too-many-instance-attributes
@dataclass(frozen=True)
class Intent:
    """Stable internal intent representation."""

    action: Optional[str] = None
    target: Optional[str] = None
    payload: Optional[str] = None
    keyword: Optional[str] = None
    subject: str = ""
    negated: bool = False
    attributes: IntentAttributes = field(default_factory=IntentAttributes)
    source: str = "rule"
    threshold: Optional[float] = None

    @property
    def color(self) -> Optional[str]:
        """Return the normalized color attribute, if any."""
        return self.attributes.color

    @property
    def position(self) -> Optional[str]:
        """Return the normalized position attribute, if any."""
        return self.attributes.position

    @property
    def state(self) -> Optional[str]:
        """Return the normalized state attribute, if any."""
        return self.attributes.state

    @property
    def query_text(self) -> str:
        """Return the normalized query text for matching purposes."""
        return self.keyword or self.subject or ""

    @property
    def normalized_subject(self) -> str:
        """Return the normalized subject text for matching purposes."""
        return (self.subject or self.keyword or "").strip()

    def has_targeting_clauses(self) -> bool:
        """Return True if the intent has any targeting clauses (keyword, color, or position)."""
        return any([self.keyword, self.color, self.position])

    def with_negated(self, negated: bool) -> "Intent":
        """Return a new Intent with the negated flag set to the given value."""
        return replace(self, negated=negated)

    def with_threshold(self, threshold: float) -> "Intent":
        """Return a new Intent with the threshold set to the given value."""
        return replace(self, threshold=threshold)
