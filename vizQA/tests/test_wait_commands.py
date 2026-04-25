import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vizQA.app.core import Automator
from vizQA.app.memory import StepStatus, TestSession, TestStep


@pytest.fixture
def session():
    return TestSession(id="test_wait_session", test_name="Wait Command Test", url="http://localhost")


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


def test_wait_until_polling(session):
    mock_client = MagicMock()
    mock_client.perceive = AsyncMock(
        side_effect=[{"elements": []}, {"elements": [{"text": "success indicator"}], "top_matches": []}]
    )
    automator = Automator(perception_client=mock_client)
    automator.page = MagicMock()
    automator.page.screenshot = AsyncMock()

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
