import pytest

from vizQA.reasoning import SemanticParser


def test_semantic_history_matching():
    """
    Verifies that resolve_historical_target can semantically match
    negated intents to slightly different subjects in history.
    """
    # No MiniLM in this basic test, checking direct matching first
    parser = SemanticParser()

    modal_target = {"id": "m1", "text": "Login Modal", "label": "modal"}
    history_metadata = {
        "history": {"login modal": {"target": modal_target, "elements": [modal_target, {"text": "Background"}]}}
    }

    # 1. Exact match (normalized)
    intent_exact = {"subject": "the Login Modal", "negated": True}
    target, elements = parser.resolve_historical_target(intent_exact, history_metadata)
    assert target == modal_target
    assert len(elements) == 2

    # 2. Match with "the" and different casing
    intent_the = {"subject": "THE login modal", "negated": True}
    target, elements = parser.resolve_historical_target(intent_the, history_metadata)
    assert target == modal_target


def test_negation_with_history():
    """
    Verifies verify_negation uses the provided history_metadata.
    """
    parser = SemanticParser()

    target = {"text": "Gone", "label": "target"}
    history = {"history": {"target": {"target": target, "elements": [target]}}}

    # Target is present in 'after' (should fail negation)
    assert parser.verify_negation([target], "target", history_metadata=history) is False

    # Target is gone in 'after' (should pass negation)
    assert parser.verify_negation([], "target", history_metadata=history) is True


def test_negation_mix_up_repro():
    """
    Reproduces the issue where multiple matching elements cause negation to fail
    even if the intended one is gone.
    """
    parser = SemanticParser()

    # Intended element (Modal 1) and another similar element (Modal 2)
    modal_1 = {"id": "m1", "text": "Close me", "label": "modal", "name": "modal"}
    modal_2 = {"id": "m2", "text": "I stay here", "label": "modal", "name": "modal", "location": [0.8, 0.8, 0.1, 0.1]}
    other = {"id": "o1", "text": "Background", "label": "bg", "name": "bg"}

    # State A: Both modals are present
    before_elements = [modal_1, modal_2, other]

    # State B: Modal 1 is gone, Modal 2 stays
    after_elements = [modal_2, other]

    intent = {"subject": "modal", "negated": True}

    # Now, this should return True because Modal 1 (the likely target) is gone,
    # and the logic handles that specifically if Modal 1 was the last seen element.
    # To simulate the Automator behavior, we'd pass modal_1 as target.
    result = parser.verify_negation(after_elements, intent, target=modal_1)

    print(f"\nRepro Result (with target): {result}")
    assert result is True, "Negation should pass if the specific target is gone"

    # Also test without target but with before_elements (collective check)
    result_collective = parser.verify_negation(after_elements, intent, before_elements=before_elements)
    print(f"Repro Result (collective): {result_collective}")
    # Currently, collective check returns False if Modal 2 still exists.
    # This is correct as a fallback, but the target-based check is what fixes the user's issue.
    assert result_collective is False
