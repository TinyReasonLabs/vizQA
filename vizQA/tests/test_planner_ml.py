from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vizQA.app.exceptions import TestDefinitionError
from vizQA.app.memory import StepStatus
from vizQA.planning import StepPlanner
from vizQA.reasoning import MiniLM, SemanticParser
from vizQA.reasoning.language import load_language_pack


@pytest.fixture
def mock_minilm():
    with (
        patch("vizQA.reasoning.minilm.ort.InferenceSession") as mock_session,
        patch("vizQA.reasoning.minilm.Tokenizer.from_file") as mock_tokenizer,
    ):

        # Mock tokenizer
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.return_value = mock_tokenizer_instance
        mock_tokenizer_instance.encode.return_value.ids = [1, 2, 3]
        mock_tokenizer_instance.encode.return_value.attention_mask = [1, 1, 1]
        mock_tokenizer_instance.decode.return_value = (
            '[{"type": "FIND", "value": "username field"}, {"type": "DO", "value": "type admin"}]'
        )

        # Mock ONNX session
        mock_session_instance = MagicMock()
        mock_session.return_value = mock_session_instance
        mock_session_instance.get_inputs.return_value = [MagicMock(name="input_ids")]

        # Mock run output (tensor with shape (1, seq_len, hidden_dim))
        # e.g., (1, 22, 384) as reported by the user
        import numpy as np

        mock_session_instance.run.return_value = [np.zeros((1, 22, 384))]

        yield mock_tokenizer_instance, mock_session_instance


def test_planner_decomposition_with_minilm_fallback(mock_minilm):
    """Verifies that an encoder-only model (3D tensor) triggers the heuristic fallback."""
    _, mock_session = mock_minilm

    planner = StepPlanner(model_name="minilm", logger=MagicMock())  # Use a mock logger to suppress output
    raw_steps = [{"action": "Click Login"}]

    steps = planner.decompose(raw_steps)

    # 2 sub-steps from instruction fallback
    assert len(steps[0].sub_steps) == 2
    assert steps[0].sub_steps[0].instruction == "FIND: Login"
    assert steps[0].sub_steps[1].instruction == "DO: click"


def test_planner_decomposition_with_minilm_generative(mock_minilm):
    """Verifies that a generative model (2D tensor) uses the decoded JSON."""
    mock_tokenizer, mock_session = mock_minilm

    # Mock 2D tensor output
    import numpy as np

    mock_session.run.return_value = [np.array([[1, 2, 3]])]
    mock_tokenizer.decode.return_value = '[{"type": "FIND", "value": "button"}]'

    planner = StepPlanner(model_name="minilm", logger=MagicMock())  # Use a mock logger to suppress output
    steps = planner.decompose([{"action": "test"}])

    assert len(steps[0].sub_steps) == 1
    assert steps[0].sub_steps[0].instruction == "FIND: button"


def test_explicit_press_key_bypasses_minilm_prediction(mock_minilm):
    """Explicit direct key commands should not be reinterpreted by MiniLM."""
    mock_tokenizer, mock_session = mock_minilm

    import numpy as np

    mock_session.run.return_value = [np.array([[1, 2, 3]])]
    mock_tokenizer.decode.return_value = '[{"type": "FIND", "value": "button"}]'

    planner = StepPlanner(model_name="minilm", logger=MagicMock())
    steps = planner.decompose([{"action": "Press keys Ctrl+C"}])

    assert [sub.instruction for sub in steps[0].sub_steps] == ["DO: press-key Ctrl+C"]


def test_minilm_deserialization_error(mock_minilm):
    mock_tokenizer, mock_session = mock_minilm

    # Mock 2D to trigger strict parsing
    import numpy as np

    mock_session.run.return_value = [np.array([[1, 2, 3]])]
    mock_tokenizer.decode.return_value = "invalid json"

    planner = StepPlanner(model_name="minilm", logger=MagicMock())  # Use a mock logger to suppress output
    with pytest.raises(TestDefinitionError) as exc:
        planner.decompose([{"action": "test"}])
    assert "not valid JSON" in str(exc.value.internal_detail)


def test_minilm_malformed_step_error(mock_minilm):
    mock_tokenizer, mock_session = mock_minilm

    # Mock 2D to trigger strict parsing
    import numpy as np

    mock_session.run.return_value = [np.array([[1, 2, 3]])]
    mock_tokenizer.decode.return_value = '[{"type": "FIND"}]'  # Missing "value"

    planner = StepPlanner(model_name="minilm", logger=MagicMock())  # Use a mock logger to suppress output
    with pytest.raises(TestDefinitionError) as exc:
        planner.decompose([{"action": "test"}])
    assert "malformed or missing keys" in str(exc.value.internal_detail)


def test_minilm_semantic_dissection_uses_language_pack_coordination_terms(mock_minilm):
    """MiniLM semantic dissection should honor configured coordination vocabulary."""
    del mock_minilm
    base_pack = load_language_pack("en")
    custom_pack = replace(
        base_pack,
        coordination_terms=["plus"],
        hold_modifier_terms=["hold"],
    )
    model_dir = Path(__file__).resolve().parents[1] / "weights" / "minilm"

    with patch("vizQA.reasoning.minilm.default_language_pack", return_value=custom_pack):
        minilm = MiniLM(str(model_dir), logger=MagicMock())

    steps = minilm._semantic_dissection("Click submit plus cancel buttons")
    assert steps == [
        {"type": "FIND", "value": "submit buttons"},
        {"type": "DO", "value": "click"},
        {"type": "FIND", "value": "cancel buttons"},
        {"type": "DO", "value": "click"},
    ]

    hold_steps = minilm._semantic_dissection("press plus hold the button")
    assert hold_steps == [
        {"type": "FIND", "value": "button"},
        {"type": "DO", "value": "press-and-hold"},
    ]


def test_minilm_semantic_dissection_uses_language_pack_wait_condition_terms(mock_minilm):
    """MiniLM semantic dissection should honor configured wait-condition vocabulary."""
    del mock_minilm
    base_pack = load_language_pack("en")
    custom_pack = replace(base_pack, wait_condition_terms=["pending"])
    model_dir = Path(__file__).resolve().parents[1] / "weights" / "minilm"

    with patch("vizQA.reasoning.minilm.default_language_pack", return_value=custom_pack):
        minilm = MiniLM(str(model_dir), logger=MagicMock())

    steps = minilm._semantic_dissection("Wait pending the loader disappears")

    assert steps == [{"type": "VERIFY", "value": "loader disappears"}]


def test_planner_expectations_do_not_inject_english_verify_prefix():
    base_pack = load_language_pack("en")
    parser = SemanticParser(language_pack=replace(base_pack, verify_prefixes=["confirma"], verify_verbs=["confirma"]))
    planner = StepPlanner(model_name="rule", parser=parser)

    steps = planner.decompose([{"action": "Click save", "expect": "dialog visible"}])

    assert [step.instruction for step in steps[0].sub_steps][-1] == "VERIFY: dialog visible"
