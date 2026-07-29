"""Shared vocabulary and regex helpers for parser and reasoning modules."""

import re

from vizQA.reasoning.language import default_language_pack

_DEFAULT_LANGUAGE_PACK = default_language_pack()


# pylint: disable=too-few-public-methods, duplicate-code
class ParserVocabulary:
    """Central repository for UI grounding vocabulary."""

    ACTION_VERBS = {name: spec.synonyms for name, spec in _DEFAULT_LANGUAGE_PACK.actions.items()}

    VERIFY_VERBS = _DEFAULT_LANGUAGE_PACK.verify_verbs
    VERIFY_BOILERPLATE = _DEFAULT_LANGUAGE_PACK.verify_boilerplate
    BOOLEAN_QUERY_OR_TERMS = _DEFAULT_LANGUAGE_PACK.boolean_query_or_terms
    BOOLEAN_QUERY_AND_TERMS = _DEFAULT_LANGUAGE_PACK.boolean_query_and_terms
    VERIFY_CONJUNCTION_TERMS = _DEFAULT_LANGUAGE_PACK.verify_conjunction_terms
    VERIFY_CONJUNCTION_PREDICATES = _DEFAULT_LANGUAGE_PACK.verify_conjunction_predicates

    COLORS = _DEFAULT_LANGUAGE_PACK.colors
    STATES = _DEFAULT_LANGUAGE_PACK.states
    POSITIONS = _DEFAULT_LANGUAGE_PACK.positions
    POSITION_ALIASES = _DEFAULT_LANGUAGE_PACK.position_aliases
    NEGATION_RE = re.compile(_DEFAULT_LANGUAGE_PACK.negation_regex.pattern, re.IGNORECASE)
