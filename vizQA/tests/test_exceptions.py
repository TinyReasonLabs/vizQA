import os
from io import BytesIO
from unittest.mock import MagicMock, patch

import httpx
import pytest

from vizQA.app import PerceptionClient
from vizQA.app.exceptions import (
    ActionExecutionError,
    ElementNotFoundError,
    PerceptionServiceError,
    TestDefinitionError,
    UserFacingException,
)
from vizQA.app.memory import StepStatus, TestStep
from vizQA.planning import StepPlanner
from vizQA.reasoning import MiniLM


def test_exception_hierarchy():
    """Verify that all custom exceptions inherit from UserFacingException."""
    assert issubclass(PerceptionServiceError, UserFacingException)
    assert issubclass(TestDefinitionError, UserFacingException)
    assert issubclass(ElementNotFoundError, UserFacingException)
    assert issubclass(ActionExecutionError, UserFacingException)


def test_perception_service_error_wrapping():
    """Verify that PerceptionClient wraps httpx errors correctly."""
    client = PerceptionClient(base_url="http://invalid", logger=MagicMock())  # Use a mock logger to suppress output

    with patch("httpx.AsyncClient.post", side_effect=httpx.ConnectError("Connection refused")):
        with patch("builtins.open", MagicMock()):

            async def run_perc():
                await client.perceive("fake.jpg", "query")

            with pytest.raises(PerceptionServiceError) as excinfo:
                import asyncio

                asyncio.run(run_perc())

        assert "visual perception service" in excinfo.value.user_message.lower()
        assert "Connection refused" in excinfo.value.internal_detail


def test_perceive_without_scope_does_not_send_previous_session_id(tmp_path):
    """Verify that fresh screenshots stay independent when no scope is provided."""
    image_path = tmp_path / "fake.jpg"
    image_path.write_bytes(b"fake image bytes")

    client = PerceptionClient(base_url="http://example.test", logger=MagicMock())
    seen_payloads = []

    class _FakeResponse:
        def __init__(self, session_id: str):
            self._session_id = session_id

        def raise_for_status(self):
            return None

        def json(self):
            return {"session_id": self._session_id, "elements": [], "top_matches": []}

    async def _fake_post(_url, files=None, data=None):
        seen_payloads.append(dict(data or {}))
        return _FakeResponse(f"session-{len(seen_payloads)}")

    with patch("httpx.AsyncClient.post", side_effect=_fake_post):

        async def run_perc():
            await client.perceive(str(image_path), "first")
            await client.perceive(str(image_path), "second")

        import asyncio

        asyncio.run(run_perc())

    assert client.session_id == "session-2"
    assert seen_payloads[0]["query"] == "first"
    assert seen_payloads[1]["query"] == "second"
    assert "session_id" not in seen_payloads[0]
    assert "session_id" not in seen_payloads[1]


def test_perceive_reuses_session_id_within_same_scope(tmp_path):
    """Verify that the backend session is reused only for the same caller-provided scope."""
    image_path = tmp_path / "fake.jpg"
    image_path.write_bytes(b"fake image bytes")

    client = PerceptionClient(base_url="http://example.test", logger=MagicMock())
    seen_payloads = []

    class _FakeResponse:
        def __init__(self, session_id: str):
            self._session_id = session_id

        def raise_for_status(self):
            return None

        def json(self):
            return {"session_id": self._session_id, "elements": [], "top_matches": []}

    async def _fake_post(_url, files=None, data=None):
        seen_payloads.append(dict(data or {}))
        return _FakeResponse(f"session-{len(seen_payloads)}")

    with patch("httpx.AsyncClient.post", side_effect=_fake_post):

        async def run_perc():
            await client.perceive(str(image_path), "first", session_scope="test|page-a|y=0")
            await client.perceive(str(image_path), "second", session_scope="test|page-a|y=0")
            await client.perceive(str(image_path), "third", session_scope="test|page-a|y=400")

        import asyncio

        asyncio.run(run_perc())

    assert "session_id" not in seen_payloads[0]
    assert seen_payloads[1]["session_id"] == "session-1"
    assert "session_id" not in seen_payloads[2]


def test_perceive_accepts_in_memory_bytes():
    """Verify that PerceptionClient can send bytes without a filesystem path."""
    client = PerceptionClient(base_url="http://example.test", logger=MagicMock())

    class _FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"session_id": "bytes-session", "elements": [], "top_matches": []}

    async def _fake_post(_url, files=None, data=None):
        upload = files["file"]
        assert upload.read() == b"fake image bytes"
        assert data["query"] == "bytes-query"
        return _FakeResponse()

    with patch("httpx.AsyncClient.post", side_effect=_fake_post):

        async def run_perc():
            return await client.perceive(image_bytes=b"fake image bytes", query="bytes-query")

        import asyncio

        payload = asyncio.run(run_perc())

    assert payload["session_id"] == "bytes-session"


def test_perceive_accepts_file_objects():
    """Verify that PerceptionClient can send an already-open file object."""
    client = PerceptionClient(base_url="http://example.test", logger=MagicMock())

    class _FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"session_id": "file-session", "elements": [], "top_matches": []}

    async def _fake_post(_url, files=None, data=None):
        upload = files["file"]
        assert upload.read() == b"fake image bytes"
        assert data["query"] == "file-query"
        return _FakeResponse()

    with patch("httpx.AsyncClient.post", side_effect=_fake_post):

        async def run_perc():
            file_obj = BytesIO(b"fake image bytes")
            file_obj.name = "fake.jpg"
            return await client.perceive(image_file=file_obj, query="file-query")

        import asyncio

        payload = asyncio.run(run_perc())

    assert payload["session_id"] == "file-session"


def test_perceive_requires_exactly_one_image_source():
    """Verify that callers must provide exactly one image source."""
    client = PerceptionClient(base_url="http://example.test", logger=MagicMock())

    with pytest.raises(ValueError, match="exactly one"):
        import asyncio

        asyncio.run(client.perceive(query="missing-source"))

    with pytest.raises(ValueError, match="exactly one"):
        import asyncio

        asyncio.run(client.perceive("fake.jpg", image_bytes=b"fake image bytes", query="too-many"))


def test_planner_line_reporting():
    """Verify that StepPlanner reports line numbers from YAML metadata."""
    planner = StepPlanner(logger=MagicMock())  # Use a mock logger to suppress output
    raw_steps = [{"action": "drag onto unknown", "__line__": 42}]

    # "drag onto" without a source element should raise ValueError in minilm,
    # which planner should wrap in TestDefinitionError with line 42.
    with pytest.raises(TestDefinitionError) as excinfo:
        planner.decompose(raw_steps)

    assert "YAML line 42" in excinfo.value.user_message
    assert "source element" in excinfo.value.internal_detail.lower()


def test_minilm_deterministic_failure():
    """Verify that minilm raises ValueError for ambiguous 'onto' drag without sources."""
    model_dir = os.path.join(os.path.dirname(__file__), "..", "vizQA", "weights", "minilm")
    if not os.path.exists(model_dir):
        pytest.skip("MiniLM weights not found, skipping deterministic failure test")

    m = MiniLM(model_dir, logger=MagicMock())  # Use a mock logger to suppress output
    with pytest.raises(ValueError) as excinfo:
        # "drag onto" with nothing before "onto"
        m.predict("drag onto button")

    assert "source element" in str(excinfo.value).lower()
