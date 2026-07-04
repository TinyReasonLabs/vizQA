"""Tests for config-backed language packs and parser/provider decoupling."""

from dataclasses import replace

from vizQA.reasoning import ParserVocabulary, SemanticParser
from vizQA.reasoning.language import ActionSpec, LanguagePack, load_language_pack


def test_default_english_language_pack_loads_expected_sections():
    pack = load_language_pack("en")

    assert pack.language == "en"
    assert pack.schema_version == 1
    assert "click" in pack.actions
    assert "click" in pack.actions["click"].synonyms
    assert "red" in pack.colors
    assert "disabled" in pack.states
    assert "top right" in pack.positions


def test_parser_vocabulary_proxies_default_language_pack():
    pack = load_language_pack("en")

    assert ParserVocabulary.ACTION_VERBS["click"] == pack.actions["click"].synonyms
    assert ParserVocabulary.COLORS == pack.colors
    assert ParserVocabulary.STATES == pack.states
    assert ParserVocabulary.POSITIONS == pack.positions


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
