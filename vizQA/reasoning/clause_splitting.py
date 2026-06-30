"""Shared clause-splitting helpers for reasoning parsers."""

from __future__ import annotations

import re
from typing import List

_VERIFY_PREDICATE_WORDS = {
    "it",
    "is",
    "are",
    "was",
    "were",
    "should",
    "must",
    "be",
    "appear",
    "appears",
    "visible",
    "hidden",
    "displayed",
    "present",
    "aligned",
    "center",
    "centered",
    "top",
    "bottom",
    "left",
    "right",
}
_VERIFY_ARTICLE_WORDS = {"the", "a", "an"}


def split_verify_conjunctions(clause: str) -> List[str]:
    """Split verification clauses on safe conjunction boundaries."""
    matches = list(re.finditer(r"\s+and\s+", clause, flags=re.IGNORECASE))
    if not matches:
        return [clause]

    parts: List[str] = []
    cursor = 0
    for match in matches:
        rhs = clause[match.end() :].strip()
        if not should_split_verify_rhs(rhs):
            continue
        lhs = clause[cursor : match.start()].strip()
        if lhs:
            parts.append(lhs)
        cursor = match.end()

    tail = clause[cursor:].strip()
    if tail:
        parts.append(tail)
    return parts or [clause]


def should_split_verify_rhs(rhs: str) -> bool:
    """Return True when the right-hand side looks like a fresh predicate."""
    if not rhs:
        return False

    words = rhs.split()
    if not words:
        return False

    first = words[0].lower()
    if first.startswith("__quote_") or first in _VERIFY_PREDICATE_WORDS:
        return True

    if first in _VERIFY_ARTICLE_WORDS:
        probe = [re.sub(r"[^a-z_]", "", word.lower()) for word in words[1:6]]
        return any(word in _VERIFY_PREDICATE_WORDS for word in probe if word)

    return False
