import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vizQA.app.core import Automator
from vizQA.app.memory import FailureType, StepStatus, TestSession, TestStep


@pytest.fixture
def session():
    return TestSession(id="test_wait_session", test_name="Wait Command Test", url="http://localhost")


def _attach_scroll_minilm(automator):
    mock_minilm = MagicMock()

    def semantic_match(query, candidates, threshold=0.7):
        normalized = str(query).strip().lower()
        generic_terms = {
            "page",
            "screen",
            "view",
            "viewport",
            "document",
            "whole page",
            "entire screen",
            "current view",
        }
        return [0] if normalized in generic_terms else []

    mock_minilm.semantic_match.side_effect = semantic_match
    automator.minilm = mock_minilm
    automator.parser.minilm = mock_minilm
    return mock_minilm


def test_wait_for_seconds(session):
    automator = Automator(perception_client=MagicMock())
    step = TestStep(id="s1", instruction="DO: wait for 2.5 seconds")

    with patch("vizQA.app.core.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        success = asyncio.run(automator._execute_do(session, step, "wait for 2.5 seconds"))
        assert success is True
        assert step.status == StepStatus.PASSED
        mock_sleep.assert_called_once_with(2.5)


def test_wait_for_minutes(session):
    automator = Automator(perception_client=MagicMock())
    step = TestStep(id="s2", instruction="DO: wait 2m")

    with patch("vizQA.app.core.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        success = asyncio.run(automator._execute_do(session, step, "wait 2m"))
        assert success is True
        mock_sleep.assert_called_once_with(120.0)


def test_wait_for_element_polls_until_visible(session):
    mock_client = MagicMock()
    mock_client.perceive = AsyncMock(
        side_effect=[{"elements": []}, {"elements": [{"text": "success toast"}], "top_matches": []}]
    )
    automator = Automator(perception_client=mock_client)
    automator.page = MagicMock()
    automator.page.screenshot = AsyncMock()
    automator.page.evaluate = AsyncMock(return_value=0)
    automator.page.url = "http://localhost"
    automator.parser.config.wait_for_timeout_seconds = 5
    automator.parser.config.wait_for_poll_interval_seconds = 1.0

    step = TestStep(id="s2a", instruction="DO: wait for the success toast")

    with patch("vizQA.app.core.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        success = asyncio.run(automator._execute_do(session, step, "wait for the success toast"))

    assert success is True
    assert step.status == StepStatus.PASSED
    assert mock_client.perceive.call_count == 2
    mock_sleep.assert_called_once_with(1.0)


def test_wait_for_element_does_not_pass_on_unrelated_visible_elements(session):
    mock_client = MagicMock()
    mock_client.perceive = AsyncMock(return_value={"elements": [{"text": "sidebar link"}], "top_matches": []})

    automator = Automator(perception_client=mock_client)
    automator.page = MagicMock()
    automator.page.screenshot = AsyncMock()
    automator.page.evaluate = AsyncMock(return_value=0)
    automator.page.url = "http://localhost"
    automator.parser.config.wait_for_timeout_seconds = 2
    automator.parser.config.wait_for_poll_interval_seconds = 1.0

    step = TestStep(id="s2aa", instruction="DO: wait for the success toast")

    with (
        patch("vizQA.app.core.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        patch("vizQA.app.core.datetime") as mock_datetime,
    ):
        start = datetime.now()
        end = start + timedelta(seconds=3)
        mock_datetime.now.side_effect = [start, start, end, end]
        success = asyncio.run(automator._execute_do(session, step, "wait for the success toast"))

    assert success is False
    assert step.status == StepStatus.FAILED
    assert step.failure_type == FailureType.TIMEOUT
    mock_sleep.assert_called_once_with(1.0)


def test_wait_for_element_ignores_unrelated_top_match(session):
    mock_client = MagicMock()
    mock_client.perceive = AsyncMock(
        return_value={
            "top_matches": [{"text": "welcome card"}],
            "elements": [{"text": "success toast"}],
        }
    )

    automator = Automator(perception_client=mock_client)
    automator.page = MagicMock()
    automator.page.screenshot = AsyncMock()
    automator.page.evaluate = AsyncMock(return_value=0)
    automator.page.url = "http://localhost"
    automator.parser.config.wait_for_timeout_seconds = 2
    automator.parser.config.wait_for_poll_interval_seconds = 1.0

    step = TestStep(id="s2ab", instruction="DO: wait for the success toast")

    success = asyncio.run(automator._execute_do(session, step, "wait for the success toast"))

    assert success is True
    assert step.status == StepStatus.PASSED


def test_wait_for_element_times_out_with_configured_timeout(session):
    mock_client = MagicMock()
    mock_client.perceive = AsyncMock(return_value={"elements": []})

    automator = Automator(perception_client=mock_client)
    automator.page = MagicMock()
    automator.page.screenshot = AsyncMock()
    automator.page.evaluate = AsyncMock(return_value=0)
    automator.page.url = "http://localhost"
    automator.parser.config.wait_for_timeout_seconds = 2
    automator.parser.config.wait_for_poll_interval_seconds = 1.0

    step = TestStep(id="s2b", instruction="DO: wait for the success toast")

    with (
        patch("vizQA.app.core.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        patch("vizQA.app.core.datetime") as mock_datetime,
    ):
        start = datetime.now()
        end = start + timedelta(seconds=3)
        mock_datetime.now.side_effect = [start, start, end, end]
        success = asyncio.run(automator._execute_do(session, step, "wait for the success toast"))

    assert success is False
    assert step.status == StepStatus.FAILED
    assert step.failure_type == FailureType.TIMEOUT
    mock_sleep.assert_called_once_with(1.0)


def test_wait_until_polling(session):
    mock_client = MagicMock()
    mock_client.perceive = AsyncMock(
        side_effect=[{"elements": []}, {"elements": [{"text": "success indicator"}], "top_matches": []}]
    )
    automator = Automator(perception_client=mock_client)
    automator.page = MagicMock()
    automator.page.screenshot = AsyncMock()
    automator.page.evaluate = AsyncMock(return_value=0)
    automator.page.url = "http://localhost"

    step = TestStep(id="s3", instruction="VERIFY: success indicator")

    with patch("vizQA.app.core.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        success = asyncio.run(automator._execute_verify(session, step, "success indicator", timeout=5))
        assert success is True
        assert step.status == StepStatus.PASSED
        assert mock_client.perceive.call_count == 2
        mock_sleep.assert_called_once_with(1.0)


def test_wait_until_timeout(session):
    mock_client = MagicMock()
    mock_client.perceive = AsyncMock(return_value={"elements": []})

    automator = Automator(perception_client=mock_client)
    automator.page = MagicMock()
    automator.page.screenshot = AsyncMock()
    automator.page.evaluate = AsyncMock(return_value=0)
    automator.page.url = "http://localhost"

    step = TestStep(id="s4", instruction="VERIFY: success indicator")

    with (
        patch("vizQA.app.core.asyncio.sleep", new_callable=AsyncMock),
        patch("vizQA.app.core.datetime") as mock_datetime,
    ):

        start = datetime.now()
        end = start + timedelta(seconds=6)

        mock_datetime.now.side_effect = [
            start,
            start,
            start,
            start,
            end,
            end,
        ]

        success = asyncio.run(automator._execute_verify(session, step, "success indicator", timeout=5))
        assert success is False
        assert step.status == StepStatus.FAILED


def test_execute_interaction_uses_configured_step_delay():
    automator = Automator(perception_client=MagicMock())
    automator.page = MagicMock()
    automator.page.mouse = MagicMock()
    automator.page.mouse.click = AsyncMock()
    automator.parser.config.step_delay_seconds = 0.2

    with patch("vizQA.app.core.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        asyncio.run(automator._execute_interaction("click", 10, 20, ""))

    automator.page.mouse.click.assert_awaited_once_with(10, 20)
    mock_sleep.assert_called_once_with(0.2)


def test_execute_action_uses_configured_step_delay_without_persistent_artifacts(session):
    automator = Automator(perception_client=MagicMock(), artifact_dir=None)
    automator.page = MagicMock()
    automator.page.screenshot = AsyncMock()
    automator.parser.config.step_delay_seconds = 0.3

    step = TestStep(
        id="legacy",
        instruction="Click the login button",
        perception_result={"viewport": {"width": 1280, "height": 720}, "top_matches": [{"bounds": [1, 2, 3, 4]}]},
    )

    with patch("vizQA.app.core.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        asyncio.run(automator._execute_action(session, step))

    mock_sleep.assert_called_once_with(0.3)


class _FakeScrollPage:
    def __init__(self, *, start_top: int, viewport_height: int = 1000, scroll_height: int = 3000):
        self.scroll_top = start_top
        self.viewport_height = viewport_height
        self.scroll_height = scroll_height
        self.screenshot = AsyncMock()
        self.scroll_history = [start_top]

    async def evaluate(self, script, arg=None):
        if "document.scrollingElement" in script:
            return {
                "scrollTop": self.scroll_top,
                "maxScrollTop": max(0, self.scroll_height - self.viewport_height),
                "viewportHeight": self.viewport_height,
            }
        if "window.scrollY || document.documentElement.scrollTop || document.body.scrollTop || 0" in script:
            return self.scroll_top
        if "window.scrollBy" in script:
            self.scroll_top = max(0, min(max(0, self.scroll_height - self.viewport_height), self.scroll_top + int(arg)))
            self.scroll_history.append(self.scroll_top)
            return None
        if "window.scrollTo" in script:
            self.scroll_top = max(0, min(max(0, self.scroll_height - self.viewport_height), int(arg)))
            self.scroll_history.append(self.scroll_top)
            return None
        return None


def test_scroll_to_element_sweeps_down_then_restarts_from_top(session):
    mock_client = MagicMock()

    fake_page = _FakeScrollPage(start_top=1000)

    async def perceive_side_effect(*_args, **_kwargs):
        if fake_page.scroll_top == 0:
            return {
                "viewport": {"width": 1280, "height": 1000},
                "top_matches": [{"text": "reviews section", "bounds": [100, 0, 500, 100]}],
                "elements": [{"text": "reviews section", "bounds": [100, 0, 500, 100]}],
            }
        return {"viewport": {"width": 1280, "height": 1000}, "elements": [], "top_matches": []}

    mock_client.perceive = AsyncMock(side_effect=perceive_side_effect)
    automator = Automator(perception_client=mock_client)
    _attach_scroll_minilm(automator)
    automator.page = fake_page
    automator.parser.config.scroll_center_band_min = 0.35
    automator.parser.config.scroll_center_band_max = 0.65

    step = TestStep(id="s5", instruction="DO: scroll to the reviews section")

    success = asyncio.run(automator._execute_do(session, step, "scroll to the reviews section"))

    assert success is True
    assert step.status == StepStatus.PASSED
    assert 0 in fake_page.scroll_history
    assert fake_page.scroll_history[0] == 1000
    assert fake_page.scroll_history[-1] == 0


def test_scroll_to_element_fails_when_target_never_appears(session):
    mock_client = MagicMock()

    fake_page = _FakeScrollPage(start_top=1000)
    mock_client.perceive = AsyncMock(return_value={"viewport": {"width": 1280, "height": 1000}, "elements": []})

    automator = Automator(perception_client=mock_client)
    _attach_scroll_minilm(automator)
    automator.page = fake_page
    automator.parser.config.scroll_center_band_min = 0.35
    automator.parser.config.scroll_center_band_max = 0.65

    step = TestStep(id="s6", instruction="DO: scroll to the reviews section")

    success = asyncio.run(automator._execute_do(session, step, "scroll to the reviews section"))

    assert success is False
    assert step.status == StepStatus.FAILED
    assert step.failure_type == FailureType.PERCEPTION_MISMATCH
    assert len(fake_page.scroll_history) > 1


def test_scroll_to_element_does_not_pass_on_unrelated_visible_elements(session):
    mock_client = MagicMock()

    fake_page = _FakeScrollPage(start_top=1000)
    mock_client.perceive = AsyncMock(
        return_value={"viewport": {"width": 1280, "height": 1000}, "elements": [{"text": "welcome card"}]}
    )

    automator = Automator(perception_client=mock_client)
    _attach_scroll_minilm(automator)
    automator.page = fake_page
    automator.parser.config.scroll_center_band_min = 0.35
    automator.parser.config.scroll_center_band_max = 0.65

    step = TestStep(id="s7", instruction="DO: scroll to the debug controls section")

    success = asyncio.run(automator._execute_do(session, step, "scroll to the debug controls section"))

    assert success is False
    assert step.status == StepStatus.FAILED
    assert step.failure_type == FailureType.PERCEPTION_MISMATCH


def test_scroll_to_element_ignores_unrelated_top_match(session):
    mock_client = MagicMock()
    fake_page = _FakeScrollPage(start_top=1000)

    async def perceive_side_effect(*_args, **_kwargs):
        if fake_page.scroll_top == 0:
            return {
                "viewport": {"width": 1280, "height": 1000},
                "top_matches": [{"text": "welcome card", "bounds": [100, 350, 500, 100]}],
                "elements": [{"text": "debug controls", "bounds": [100, 350, 500, 100]}],
            }
        return {
            "viewport": {"width": 1280, "height": 1000},
            "top_matches": [{"text": "welcome card", "bounds": [100, 350, 500, 100]}],
            "elements": [{"text": "welcome card", "bounds": [100, 350, 500, 100]}],
        }

    mock_client.perceive = AsyncMock(side_effect=perceive_side_effect)
    automator = Automator(perception_client=mock_client)
    _attach_scroll_minilm(automator)
    automator.page = fake_page
    automator.parser.config.scroll_center_band_min = 0.35
    automator.parser.config.scroll_center_band_max = 0.65

    step = TestStep(id="s8", instruction="DO: scroll to the debug controls section")

    success = asyncio.run(automator._execute_do(session, step, "scroll to the debug controls section"))

    assert success is True
    assert step.status == StepStatus.PASSED
    assert 0 in fake_page.scroll_history


def test_scroll_top_reads_plain_values_from_page_evaluate():
    automator = Automator(perception_client=MagicMock())
    automator.page = MagicMock()
    automator.page.evaluate = AsyncMock(return_value=321)

    assert asyncio.run(automator._get_scroll_top()) == 321.0


def test_scroll_target_detection_is_not_tied_to_to_the_prefix():
    automator = Automator(perception_client=MagicMock())
    _attach_scroll_minilm(automator)

    assert automator._is_scroll_target_query("reviews section") is True
    assert automator._is_scroll_target_query("debug controls section") is True
    assert automator._is_scroll_target_query("to the reviews section") is True
    assert automator._is_scroll_target_query("top") is False
    assert automator._is_scroll_target_query("bottom of the page") is False
