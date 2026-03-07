from unittest.mock import MagicMock, patch

import pytest

from vizQA.memory import StepStatus
from vizQA.minilm import MiniLM
from vizQA.planner import StepPlanner


@pytest.fixture
def mock_minilm():
    with (
        patch("vizQA.minilm.ort.InferenceSession") as mock_session,
        patch("vizQA.minilm.Tokenizer.from_file") as mock_tokenizer,
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

    planner = StepPlanner(model_name="minilm")
    raw_steps = [{"action": "Click Login"}]

    steps = planner.decompose(raw_steps)

    # 2 sub-steps from instruction fallback
    assert len(steps[0].sub_steps) == 2
    assert steps[0].sub_steps[0].instruction == "FIND: target element"


def test_planner_decomposition_with_minilm_generative(mock_minilm):
    """Verifies that a generative model (2D tensor) uses the decoded JSON."""
    mock_tokenizer, mock_session = mock_minilm

    # Mock 2D tensor output
    import numpy as np

    mock_session.run.return_value = [np.array([[1, 2, 3]])]
    mock_tokenizer.decode.return_value = '[{"type": "FIND", "value": "button"}]'

    planner = StepPlanner(model_name="minilm")
    steps = planner.decompose([{"action": "test"}])

    assert len(steps[0].sub_steps) == 1
    assert steps[0].sub_steps[0].instruction == "FIND: button"


def test_minilm_deserialization_error(mock_minilm):
    mock_tokenizer, mock_session = mock_minilm

    # Mock 2D to trigger strict parsing
    import numpy as np

    mock_session.run.return_value = [np.array([[1, 2, 3]])]
    mock_tokenizer.decode.return_value = "invalid json"

    planner = StepPlanner(model_name="minilm")

    with pytest.raises(RuntimeError, match="not valid JSON"):
        planner.decompose([{"action": "test"}])


def test_minilm_malformed_step_error(mock_minilm):
    mock_tokenizer, mock_session = mock_minilm

    # Mock 2D to trigger strict parsing
    import numpy as np

    mock_session.run.return_value = [np.array([[1, 2, 3]])]
    mock_tokenizer.decode.return_value = '[{"type": "FIND"}]'  # Missing "value"

    planner = StepPlanner(model_name="minilm")

    with pytest.raises(RuntimeError, match="malformed or missing keys"):
        planner.decompose([{"action": "test"}])
