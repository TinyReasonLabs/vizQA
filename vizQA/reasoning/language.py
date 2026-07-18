"""Config-backed language pack loading for parser and semantic components."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass(frozen=True)
class ActionSpec:
    """Config for one canonical action."""

    synonyms: List[str]
    anchors: List[str]


# pylint: disable=too-many-instance-attributes
@dataclass(frozen=True)
class LanguagePack:
    """Normalized, typed language config used by reasoning modules."""

    language: str
    schema_version: int
    actions: Dict[str, ActionSpec]
    verify_verbs: List[str]
    verify_boilerplate: List[str]
    verify_prefixes: List[str]
    verify_trigger_terms: List[str]
    verify_subject_noise: List[str]
    verify_query_prefixes: List[str]
    keypress_prefixes: List[str]
    key_aliases: Dict[str, str]
    articles: List[str]
    pronouns: List[str]
    leading_prepositions: List[str]
    lead_in_noise_phrases: List[str]
    lead_in_noise_replacements: Dict[str, str]
    target_cleanup_phrases: List[str]
    clause_splitters: List[str]
    coordination_terms: List[str]
    coordination_punctuation: List[str]
    boolean_query_or_terms: List[str]
    boolean_query_and_terms: List[str]
    sequence_split_followers: List[str]
    rhs_noise_words: List[str]
    distributive_plural_nouns: List[str]
    wait_verbs: List[str]
    wait_condition_terms: List[str]
    wait_duration_units: Dict[str, float]
    verify_conjunction_terms: List[str]
    verify_conjunction_predicates: List[str]
    hold_action_verbs: List[str]
    hold_modifier_terms: List[str]
    bare_click_targets: List[str]
    noun_action_guards: Dict[str, List[str]]
    drag_target_connectors: List[str]
    select_source_connectors: List[str]
    type_target_connectors: List[str]
    position_aliases: Dict[str, str]
    position_terms: Dict[str, List[str]]
    colors: List[str]
    states: List[str]
    positions: List[str]
    negation_regex_terms: List[str]
    positive_regex_terms: List[str]
    negation_anchors: List[str]
    positive_anchors: List[str]
    generic_scope_terms: List[str]
    salience_prominent_terms: List[str]
    salience_subtle_terms: List[str]
    target_anchors: List[str]
    conjunction_anchors: List[str]

    @property
    def negation_regex(self) -> re.Pattern[str]:
        """Return the compiled negation regex for fast-path matching."""
        escaped = [re.escape(term) for term in self.negation_regex_terms if term]
        return re.compile(r"\b(" + "|".join(escaped) + r")\b", re.IGNORECASE)

    @property
    def positive_regex(self) -> re.Pattern[str]:
        """Return the compiled positive regex for fast-path matching."""
        escaped = [re.escape(term) for term in self.positive_regex_terms if term]
        return re.compile(r"\b(" + "|".join(escaped) + r")\b", re.IGNORECASE)


def _languages_dir() -> Path:
    return Path(__file__).resolve().parent / "languages"


def _required_list(data: Dict[str, Any], key: str) -> List[str]:
    value = data.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) and item.strip() for item in value):
        raise ValueError(f"Language pack field '{key}' must be a non-empty string list")
    return [item.strip() for item in value]


def _required_mapping(data: Dict[str, Any], key: str) -> Dict[str, str]:
    value = data.get(key)
    if not isinstance(value, dict) or not value:
        raise ValueError(f"Language pack field '{key}' must be a non-empty string mapping")
    normalized: Dict[str, str] = {}
    for raw_key, raw_value in value.items():
        if not isinstance(raw_key, str) or not raw_key.strip() or not isinstance(raw_value, str):
            raise ValueError(f"Language pack field '{key}' must contain non-empty string keys and string values")
        normalized[raw_key.strip()] = raw_value.strip()
    return normalized


def _optional_mapping(data: Dict[str, Any], key: str) -> Dict[str, str]:
    value = data.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Language pack field '{key}' must be a mapping when provided")
    normalized: Dict[str, str] = {}
    for raw_key, raw_value in value.items():
        if not isinstance(raw_key, str) or not raw_key.strip() or not isinstance(raw_value, str):
            raise ValueError(f"Language pack field '{key}' must contain non-empty string keys and string values")
        normalized[raw_key.strip()] = raw_value.strip()
    return normalized


def _required_list_mapping(data: Dict[str, Any], key: str) -> Dict[str, List[str]]:
    value = data.get(key)
    if not isinstance(value, dict) or not value:
        raise ValueError(f"Language pack field '{key}' must be a non-empty mapping of string lists")

    normalized: Dict[str, List[str]] = {}
    for raw_key, raw_values in value.items():
        if not isinstance(raw_key, str) or not raw_key.strip():
            raise ValueError(f"Language pack field '{key}' must contain non-empty string keys")
        if not isinstance(raw_values, list) or not raw_values:
            raise ValueError(f"Language pack field '{key}' must map '{raw_key}' to a non-empty string list")
        cleaned: List[str] = []
        for raw_value in raw_values:
            if not isinstance(raw_value, str) or not raw_value.strip():
                raise ValueError(f"Language pack field '{key}' must map '{raw_key}' to non-empty strings")
            cleaned.append(raw_value.strip())
        normalized[raw_key.strip()] = cleaned
    return normalized


def _required_number_mapping(data: Dict[str, Any], key: str) -> Dict[str, float]:
    value = data.get(key)
    if not isinstance(value, dict) or not value:
        raise ValueError(f"Language pack field '{key}' must be a non-empty numeric mapping")
    normalized: Dict[str, float] = {}
    for raw_key, raw_value in value.items():
        if not isinstance(raw_key, str) or not raw_key.strip() or not isinstance(raw_value, (int, float)):
            raise ValueError(f"Language pack field '{key}' must contain non-empty string keys and numeric values")
        if raw_value <= 0:
            raise ValueError(f"Language pack field '{key}' values must be positive")
        normalized[raw_key.strip()] = float(raw_value)
    return normalized


def _load_action_specs(raw_actions: Dict[str, Any]) -> Dict[str, ActionSpec]:
    if not isinstance(raw_actions, dict) or not raw_actions:
        raise ValueError("Language pack field 'actions' must be a non-empty mapping")

    actions: Dict[str, ActionSpec] = {}
    seen_synonyms: Dict[str, str] = {}
    for action_name, raw_spec in raw_actions.items():
        if not isinstance(raw_spec, dict):
            raise ValueError(f"Action spec for '{action_name}' must be a mapping")
        synonyms = _required_list(raw_spec, "synonyms")
        anchors = _required_list(raw_spec, "anchors")
        for synonym in synonyms:
            prior = seen_synonyms.get(synonym.lower())
            if prior and prior != action_name:
                raise ValueError(f"Duplicate action synonym '{synonym}' defined for '{prior}' and '{action_name}'")
            seen_synonyms[synonym.lower()] = action_name
        actions[action_name] = ActionSpec(synonyms=synonyms, anchors=anchors)
    return actions


def _load_pack_data(raw: Dict[str, Any]) -> LanguagePack:
    negation = raw.get("negation")
    if not isinstance(negation, dict):
        raise ValueError("Language pack field 'negation' must be a mapping")

    grammar = raw.get("grammar")
    if not isinstance(grammar, dict):
        raise ValueError("Language pack field 'grammar' must be a mapping")

    parser = raw.get("parser")
    if not isinstance(parser, dict):
        raise ValueError("Language pack field 'parser' must be a mapping")

    wait = raw.get("wait")
    if not isinstance(wait, dict):
        raise ValueError("Language pack field 'wait' must be a mapping")

    verify = raw.get("verify")
    if not isinstance(verify, dict):
        raise ValueError("Language pack field 'verify' must be a mapping")

    salience = raw.get("salience")
    if not isinstance(salience, dict):
        raise ValueError("Language pack field 'salience' must be a mapping")

    semantic = raw.get("semantic")
    if not isinstance(semantic, dict):
        raise ValueError("Language pack field 'semantic' must be a mapping")

    keypress = raw.get("keypress")
    if not isinstance(keypress, dict):
        raise ValueError("Language pack field 'keypress' must be a mapping")

    language = raw.get("language")
    schema_version = raw.get("schema_version")
    if not isinstance(language, str) or not language.strip():
        raise ValueError("Language pack field 'language' must be a non-empty string")
    if not isinstance(schema_version, int):
        raise ValueError("Language pack field 'schema_version' must be an integer")

    return LanguagePack(
        language=language.strip(),
        schema_version=schema_version,
        actions=_load_action_specs(raw.get("actions", {})),
        verify_verbs=_required_list(raw, "verify_verbs"),
        verify_boilerplate=_required_list(raw, "verify_boilerplate"),
        verify_prefixes=_required_list(verify, "prefixes"),
        verify_trigger_terms=_required_list(verify, "trigger_terms"),
        verify_subject_noise=_required_list(verify, "subject_noise"),
        verify_query_prefixes=_required_list(verify, "query_prefixes"),
        keypress_prefixes=_required_list(keypress, "prefixes"),
        key_aliases=_required_mapping(keypress, "aliases"),
        articles=_required_list(grammar, "articles"),
        pronouns=_required_list(grammar, "pronouns"),
        leading_prepositions=_required_list(grammar, "leading_prepositions"),
        lead_in_noise_phrases=_required_list(parser, "lead_in_noise_phrases"),
        lead_in_noise_replacements=_required_mapping(parser, "lead_in_noise_replacements"),
        target_cleanup_phrases=_required_list(parser, "target_cleanup_phrases"),
        clause_splitters=_required_list(parser, "clause_splitters"),
        coordination_terms=_required_list(parser, "coordination_terms"),
        coordination_punctuation=_required_list(parser, "coordination_punctuation"),
        boolean_query_or_terms=_required_list(semantic, "boolean_query_or_terms"),
        boolean_query_and_terms=_required_list(semantic, "boolean_query_and_terms"),
        sequence_split_followers=_required_list(parser, "sequence_split_followers"),
        rhs_noise_words=_required_list(parser, "rhs_noise_words"),
        distributive_plural_nouns=_required_list(parser, "distributive_plural_nouns"),
        wait_verbs=_required_list(parser, "wait_verbs"),
        wait_condition_terms=_required_list(parser, "wait_condition_terms"),
        wait_duration_units=_required_number_mapping(wait, "duration_units"),
        verify_conjunction_terms=_required_list(parser, "verify_conjunction_terms"),
        verify_conjunction_predicates=_required_list(parser, "verify_conjunction_predicates"),
        hold_action_verbs=_required_list(parser, "hold_action_verbs"),
        hold_modifier_terms=_required_list(parser, "hold_modifier_terms"),
        bare_click_targets=_required_list(parser, "bare_click_targets"),
        noun_action_guards=_required_list_mapping(parser, "noun_action_guards"),
        drag_target_connectors=_required_list(parser, "drag_target_connectors"),
        select_source_connectors=_required_list(parser, "select_source_connectors"),
        type_target_connectors=_required_list(parser, "type_target_connectors"),
        position_aliases=_optional_mapping(semantic, "position_aliases"),
        position_terms=_required_list_mapping(semantic, "position_terms"),
        colors=_required_list(raw, "colors"),
        states=_required_list(raw, "states"),
        positions=_required_list(raw, "positions"),
        negation_regex_terms=_required_list(negation, "regex_terms"),
        positive_regex_terms=_required_list(negation, "positive_regex_terms"),
        negation_anchors=_required_list(negation, "anchors"),
        positive_anchors=_required_list(negation, "positive_anchors"),
        generic_scope_terms=_required_list(raw, "generic_scope_terms"),
        salience_prominent_terms=_required_list(salience, "prominent_terms"),
        salience_subtle_terms=_required_list(salience, "subtle_terms"),
        target_anchors=_required_list(semantic, "target_anchors"),
        conjunction_anchors=_required_list(semantic, "conjunction_anchors"),
    )


@lru_cache(maxsize=None)
def load_language_pack(language: str = "en") -> LanguagePack:
    """Load a language pack by id from the bundled YAML files."""
    path = _languages_dir() / f"{language}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Language pack not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Language pack at {path} must decode to a mapping")
    return _load_pack_data(raw)


def default_language_pack() -> LanguagePack:
    """Return the default bundled English language pack."""
    return load_language_pack("en")


def language_pack_from_data(raw: Dict[str, Any]) -> LanguagePack:
    """Build a language pack from in-memory data for tests or overrides."""
    return _load_pack_data(raw)


def match_prefixed_payload(text: str, prefixes: List[str]) -> str | None:
    """Return the trailing payload for the first matching command prefix."""
    stripped = text.strip()
    for prefix in sorted((item.strip() for item in prefixes if item.strip()), key=len, reverse=True):
        pattern = rf"^\s*{re.escape(prefix)}(?:\b|\s)\s*(.*)$"
        match = re.match(pattern, stripped, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def alternation_pattern(terms: List[str]) -> str:
    """Return a length-sorted escaped alternation pattern for term lists."""
    normalized = [term.strip() for term in terms if term and term.strip()]
    normalized.sort(key=len, reverse=True)
    return "|".join(re.escape(term) for term in normalized)
