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
    colors: List[str]
    states: List[str]
    positions: List[str]
    negation_regex_terms: List[str]
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


def _languages_dir() -> Path:
    return Path(__file__).resolve().parent / "languages"


def _required_list(data: Dict[str, Any], key: str) -> List[str]:
    value = data.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) and item.strip() for item in value):
        raise ValueError(f"Language pack field '{key}' must be a non-empty string list")
    return [item.strip() for item in value]


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

    salience = raw.get("salience")
    if not isinstance(salience, dict):
        raise ValueError("Language pack field 'salience' must be a mapping")

    semantic = raw.get("semantic")
    if not isinstance(semantic, dict):
        raise ValueError("Language pack field 'semantic' must be a mapping")

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
        colors=_required_list(raw, "colors"),
        states=_required_list(raw, "states"),
        positions=_required_list(raw, "positions"),
        negation_regex_terms=_required_list(negation, "regex_terms"),
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
