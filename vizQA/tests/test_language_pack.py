"""Tests for config-backed language packs and parser/provider decoupling."""

from dataclasses import replace

from vizQA.reasoning import ParserVocabulary, SemanticParser
from vizQA.reasoning.clause_splitting import split_verify_conjunctions
from vizQA.reasoning.language import ActionSpec, LanguagePack, load_language_pack
from vizQA.reasoning.query_semantics import split_boolean_query


def test_default_english_language_pack_loads_expected_sections():
    pack = load_language_pack("en")

    assert pack.language == "en"
    assert pack.schema_version == 1
    assert "click" in pack.actions
    assert "click" in pack.actions["click"].synonyms
    assert "press-key" in pack.actions
    assert "press key" in pack.actions["press-key"].synonyms
    assert "red" in pack.colors
    assert "disabled" in pack.states
    assert "top right" in pack.positions
    assert "verify that" in pack.verify_prefixes
    assert "please navigate ahead and" in pack.target_cleanup_phrases
    assert pack.lead_in_noise_replacements["right then"] == "then"
    assert "it" in pack.pronouns
    assert "buttons" in pack.distributive_plural_nouns
    assert "into" in pack.type_target_connectors
    assert "appear" in pack.positive_regex_terms
    assert pack.noun_action_guards["input"] == ["the input"]
    assert pack.wait_condition_terms == ["until"]


def test_parser_vocabulary_proxies_default_language_pack():
    pack = load_language_pack("en")

    assert ParserVocabulary.ACTION_VERBS["click"] == pack.actions["click"].synonyms
    assert ParserVocabulary.COLORS == pack.colors
    assert ParserVocabulary.STATES == pack.states
    assert ParserVocabulary.POSITIONS == pack.positions
    assert ParserVocabulary.BOOLEAN_QUERY_OR_TERMS == pack.boolean_query_or_terms
    assert ParserVocabulary.POSITION_ALIASES == pack.position_aliases


def test_parser_uses_language_pack_synonyms_for_actions():
    base_pack = load_language_pack("en")
    custom_pack = replace(
        base_pack,
        actions={
            **base_pack.actions,
            "click": ActionSpec(
                synonyms=["poke"],
                anchors=["poke the button"],
            ),
        },
    )

    parser = SemanticParser(language_pack=custom_pack)
    nodes = parser.parse("Poke the launch control")

    assert [(node.type, node.value) for node in nodes] == [("FIND", "launch control"), ("DO", "poke")]


def test_parser_uses_language_pack_verify_prefixes_for_cleanup():
    base_pack = load_language_pack("en")
    custom_pack = replace(
        base_pack,
        verify_verbs=["confirm"],
        verify_prefixes=["confirm that", "confirm the", "confirm"],
    )

    parser = SemanticParser(language_pack=custom_pack)
    nodes = parser.parse("Confirm that the launch control is visible")

    assert [(node.type, node.value) for node in nodes] == [("VERIFY", "launch control is visible")]


def test_parser_uses_language_pack_drag_connectors():
    base_pack = load_language_pack("en")
    custom_pack = replace(
        base_pack,
        drag_target_connectors=["toward"],
    )

    parser = SemanticParser(language_pack=custom_pack)
    nodes = parser.parse("Drag the token toward the vault")

    assert [(node.type, node.value) for node in nodes] == [
        ("FIND", "token"),
        ("DO", "drag"),
        ("FIND", "vault"),
        ("DO", "drop"),
    ]


def test_parser_uses_language_pack_coordination_and_wait_condition_terms():
    base_pack = load_language_pack("en")
    custom_pack = replace(
        base_pack,
        coordination_terms=["plus"],
        wait_condition_terms=["pending"],
    )

    parser = SemanticParser(language_pack=custom_pack)

    action_nodes = parser.parse("Click submit plus cancel buttons")
    wait_nodes, _, _ = parser._parse_atomic_action(
        "Wait pending the loader disappears", "element"
    )  # pylint: disable=protected-access

    assert [(node.type, node.value) for node in action_nodes] == [
        ("FIND", "submit button"),
        ("DO", "click"),
        ("FIND", "cancel buttons"),
        ("DO", "click"),
    ]
    assert [(node.type, node.value) for node in wait_nodes] == [("VERIFY", "loader disappears")]


def test_boolean_query_split_uses_language_pack_terms():
    base_pack = load_language_pack("en")
    custom_pack = replace(
        base_pack,
        boolean_query_or_terms=["either"],
        boolean_query_and_terms=["plus"],
    )

    assert split_boolean_query("alpha either beta plus gamma", custom_pack) == [["alpha"], ["beta", "gamma"]]


def test_verify_clause_splitting_uses_language_pack_terms():
    base_pack = load_language_pack("en")
    custom_pack = replace(
        base_pack,
        verify_conjunction_terms=["plus"],
        verify_conjunction_predicates=["visible"],
    )

    assert split_verify_conjunctions("button plus visible", custom_pack) == ["button", "visible"]


def test_verify_clause_splitting_recognizes_article_then_quoted_subject():
    pack = load_language_pack("en")

    assert split_verify_conjunctions(
        "modal should close and a __QUOTE_0__ alert or state change should occur", pack
    ) == ["modal should close", "a __QUOTE_0__ alert or state change should occur"]


def test_parser_uses_language_pack_position_aliases():
    base_pack = load_language_pack("en")
    custom_pack = replace(base_pack, position_aliases={"centered": "center"})

    parser = SemanticParser(language_pack=custom_pack)
    intent = parser.parse_verify_intent("The modal should be centered")

    assert intent.position == "center"


def test_parser_uses_language_pack_coordination_punctuation():
    base_pack = load_language_pack("en")
    parser = SemanticParser(language_pack=replace(base_pack, coordination_punctuation=[";"]))

    nodes = parser.parse("Click first; click second")

    assert [(node.type, node.value) for node in nodes] == [
        ("FIND", "first"),
        ("DO", "click"),
        ("FIND", "second"),
        ("DO", "click"),
    ]


class _ClassifierOnlyProvider:
    provider_id = "fake-provider"
    revision = "test"

    def encode(self, text):
        raise AssertionError("encode should not be called in this parser test")

    def cosine_similarity(self, a, b):
        raise AssertionError("cosine_similarity should not be called in this parser test")

    def split_boolean_query(self, query):
        return [[query]]

    def normalize_boolean_term(self, term):
        return term

    def semantic_match(self, query, candidates, threshold=0.7):
        return []

    def classify_anchor_group(self, text, threshold=0.6, groups=None):
        assert groups is not None
        if "activate" in text.lower():
            return "click"
        return None

    def is_negation(self, text, threshold=0.4, logit_threshold=0.7, delta=0.02):
        return False


def test_parser_can_use_provider_without_minilm_private_action_groups():
    parser = SemanticParser(semantic_provider=_ClassifierOnlyProvider())

    nodes = parser.parse("Activate the launch control")

    assert [(node.type, node.value) for node in nodes] == [("FIND", "launch control"), ("DO", "click")]
