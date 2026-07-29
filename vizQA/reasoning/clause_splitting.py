"""Shared clause-splitting helpers for reasoning parsers."""

from __future__ import annotations

import re
from typing import List, Optional

from vizQA.reasoning.language import LanguagePack, alternation_pattern, default_language_pack


def split_verify_conjunctions(clause: str, language_pack: Optional[LanguagePack] = None) -> List[str]:
    """Split verification clauses on safe conjunction boundaries."""
    language_pack = language_pack or default_language_pack()
    conjunction_terms = alternation_pattern(language_pack.verify_conjunction_terms)
    predicate_terms = {term.lower() for term in language_pack.verify_conjunction_predicates if term}
    article_terms = {term.lower() for term in language_pack.articles if term}
    if not conjunction_terms or not predicate_terms:
        return [clause]

    matches = list(re.finditer(rf"\s+(?:{conjunction_terms})\s+", clause, flags=re.IGNORECASE))
    if not matches:
        return [clause]

    parts: List[str] = []
    cursor = 0
    for match in matches:
        rhs = clause[match.end() :].strip()
        if not _should_split_verify_rhs(rhs, predicate_terms, article_terms):
            continue
        lhs = clause[cursor : match.start()].strip()
        if lhs:
            parts.append(lhs)
        cursor = match.end()

    tail = clause[cursor:].strip()
    if tail:
        parts.append(tail)
    return parts or [clause]


def should_split_verify_rhs(rhs: str, language_pack: Optional[LanguagePack] = None) -> bool:
    """Return True when the right-hand side looks like a fresh predicate."""
    language_pack = language_pack or default_language_pack()
    predicate_terms = {term.lower() for term in language_pack.verify_conjunction_predicates if term}
    article_terms = {term.lower() for term in language_pack.articles if term}
    return _should_split_verify_rhs(rhs, predicate_terms, article_terms)


def _should_split_verify_rhs(rhs: str, predicate_terms: set[str], article_terms: set[str]) -> bool:
    if not rhs:
        return False

    words = rhs.split()
    if not words:
        return False

    first = words[0].lower()
    if first.startswith("__quote_") or first in predicate_terms:
        return True

    if first in article_terms:
        # A quoted UI label is a self-contained subject anchor.  It does not
        # need a nearby predicate to distinguish it from the preceding clause
        # (for example, ``and a 'Login Successful' alert should appear``).
        if len(words) > 1 and words[1].lower().startswith("__quote_"):
            return True
        probe = [re.sub(r"[^a-z_]", "", word.lower()) for word in words[1:6]]
        return any(word in predicate_terms for word in probe if word)

    return False
