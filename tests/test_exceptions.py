from unittest.mock import MagicMock, patch

import httpx
import pytest

from vizQA.client import PerceptionClient
from vizQA.exceptions import (
    ActionExecutionError,
    ElementNotFoundError,
    PerceptionServiceError,
    TestDefinitionError,
    UserFacingException,
)
from vizQA.memory import StepStatus, TestStep
from vizQA.planner import StepPlanner


def test_exception_hierarchy():
    """Verify that all custom exceptions inherit from UserFacingException."""
    assert issubclass(PerceptionServiceError, UserFacingException)
    assert issubclass(TestDefinitionError, UserFacingException)
    assert issubclass(ElementNotFoundError, UserFacingException)
    assert issubclass(ActionExecutionError, UserFacingException)


def test_perception_service_error_wrapping():
    """Verify that PerceptionClient wraps httpx errors correctly."""
    client = PerceptionClient(base_url="http://invalid")

    with patch("httpx.AsyncClient.post", side_effect=httpx.ConnectError("Connection refused")):
        with patch("builtins.open", MagicMock()):

            async def run_perc():
                await client.perceive("fake.jpg", "query")

            with pytest.raises(PerceptionServiceError) as excinfo:
                import asyncio

                asyncio.run(run_perc())

        assert "visual perception service" in excinfo.value.user_message.lower()
        assert "Connection refused" in excinfo.value.internal_detail


def test_planner_line_reporting():
    """Verify that StepPlanner reports line numbers from YAML metadata."""
    planner = StepPlanner()
    raw_steps = [{"action": "drag onto unknown", "__line__": 42}]

    # "drag onto" without a source element should raise ValueError in minilm,
    # which planner should wrap in TestDefinitionError with line 42.
    with pytest.raises(TestDefinitionError) as excinfo:
        planner.decompose(raw_steps)

    assert "YAML line 42" in excinfo.value.user_message
    assert "source element" in excinfo.value.internal_detail.lower()


def test_minilm_deterministic_failure():
    """Verify that minilm raises ValueError for ambiguous 'onto' drag without sources."""
    import os

    from vizQA.minilm import MiniLM

    model_dir = os.path.join(os.path.dirname(__file__), "..", "vizQA", "weights", "minilm")
    if not os.path.exists(model_dir):
        pytest.skip("MiniLM weights not found, skipping deterministic failure test")

    m = MiniLM(model_dir)
    with pytest.raises(ValueError) as excinfo:
        # "drag onto" with nothing before "onto"
        m.predict("drag onto button")

    assert "source element" in str(excinfo.value).lower()
