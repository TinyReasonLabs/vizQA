"""
Deterministic parsing and lexical helpers for boolean-style UI queries.
"""

import re
from typing import List, Optional

from vizQA.reasoning.language import LanguagePack, alternation_pattern, default_language_pack


# pylint: disable=too-many-locals
def split_boolean_query(query: str, language_pack: Optional[LanguagePack] = None) -> List[List[str]]:
    """Split a query into OR groups of AND clauses while preserving quoted text."""
    language_pack = language_pack or default_language_pack()
    or_terms = alternation_pattern(language_pack.boolean_query_or_terms)
    and_terms = alternation_pattern(language_pack.boolean_query_and_terms)
    if not or_terms or not and_terms:
        return [[query]]

    local_quotes: List[str] = []

    def _replace(match: re.Match[str]) -> str:
        local_quotes.append(match.group(0))
        return f"__QUOTE_{len(local_quotes) - 1}__"

    protected = re.sub(r"(['\"])(.*?)\1", _replace, query)
    or_parts = [
        part.strip() for part in re.split(rf"\b(?:{or_terms})\b", protected, flags=re.IGNORECASE) if part.strip()
    ]

    groups: List[List[str]] = []
    for or_part in or_parts:
        and_parts = [
            part.strip() for part in re.split(rf"\b(?:{and_terms})\b", or_part, flags=re.IGNORECASE) if part.strip()
        ]
        restored_terms: List[str] = []
        for part in and_parts:
            restored = part
            for i, quote in enumerate(local_quotes):
                restored = restored.replace(f"__QUOTE_{i}__", quote)
            restored_terms.append(restored.strip())
        if restored_terms:
            groups.append(restored_terms)

    return groups or [[query]]


def normalize_boolean_term(term: str) -> str:
    """Unwrap quoted terms before matching or embedding."""
    term = term.strip()
    quoted_match = re.fullmatch(r"(['\"])(.*?)\1", term)
    if quoted_match:
        return quoted_match.group(2).strip()
    return term


def lexical_term_score(term: str, text: str) -> float:
    """Return a simple lexical match score for a term against candidate text."""
    normalized_term = normalize_boolean_term(term).lower()
    text_lower = text.lower().strip()
    if not normalized_term or not text_lower:
        return 0.0

    if " " in normalized_term:
        return 1.0 if normalized_term in text_lower else 0.0

    text_tokens = set(re.findall(r"\w+", text_lower))
    return 1.0 if normalized_term in text_tokens else 0.0


def is_boolean_query(groups: List[List[str]]) -> bool:
    """Return whether a parsed query contains explicit boolean structure."""
    if len(groups) > 1:
        return True
    return bool(groups and len(groups[0]) > 1)
