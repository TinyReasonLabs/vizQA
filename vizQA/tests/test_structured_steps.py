"""Regression coverage for deterministic YAML v2 planning and execution."""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from vizQA.app.core import Automator
from vizQA.app.exceptions import TestDefinitionError
from vizQA.app.memory import FailureType, StepStatus, TestSession
from vizQA.planning import StepPlanner
from vizQA.planning.structured import parse_structured_steps


def test_v2_planning_bypasses_semantic_dependencies():
    parser = MagicMock()
    planner = StepPlanner(model_name="rule", parser=parser)

    steps = planner.decompose(
        [{"click": {"target": "Sign in button"}}, {"type": {"target": "email", "text": "a@b.test"}}],
        schema_version=2,
    )

    assert [sub_step.instruction for sub_step in steps[0].sub_steps] == ["FIND: Sign in button", "DO: click"]
    assert steps[0].sub_steps[0].ranked_only is True
    assert [sub_step.instruction for sub_step in steps[1].sub_steps] == ["FIND: email", "DO: type a@b.test"]
    assert [sub_step.id for sub_step in steps[1].sub_steps] == ["step_01.01", "step_01.02"]
    assert not hasattr(steps[0], "operation")
    assert parser.method_calls == []


def test_v2_allows_legacy_and_structured_steps_in_one_flow():
    parser = MagicMock()
    parser.parse_direct_action.return_value = []
    planner = StepPlanner(model_name="rule", parser=parser)

    steps = planner.decompose(
        [{"click": {"target": "Sign in button"}}, {"action": "Click the Settings button"}],
        schema_version=2,
    )

    assert [sub_step.instruction for sub_step in steps[0].sub_steps] == ["FIND: Sign in button", "DO: click"]
    assert steps[0].sub_steps[0].ranked_only is True
    assert steps[1].instruction == "Click the Settings button"
    parser.parse_direct_action.assert_called_once_with("Click the Settings button")


def test_v2_accepts_every_documented_operation():
    steps = parse_structured_steps(
        [
            {"click": {"target": "button"}},
            {"right_click": {"target": "item"}},
            {"hover": {"target": "menu"}},
            {"type": {"target": "input", "text": "value"}},
            {"clear": {"target": "input"}},
            {"press_key": {"key": "Enter"}},
            {"check": {"target": "checkbox"}},
            {"uncheck": {"target": "checkbox"}},
            {"select": {"target": "dropdown", "option": "Option A"}},
            {"drag": {"source": "card", "target": "column"}},
            {"upload": {"target": "upload area", "file": "{file}"}},
            {"scroll": {"position": "bottom"}},
            {"wait": {"seconds": 0}},
            {"assert_visible": {"target": "toast"}},
            {"assert_not_visible": {"target": "dialog"}},
        ]
    )
    assert len(steps) == 15


@pytest.mark.parametrize(
    "step, message",
    [
        ({"click": {}}, "requires a non-empty target"),
        ({"type": {"target": "email"}}, "requires text"),
        ({"wait": {"seconds": 1, "target": "toast"}}, "either seconds or target"),
        ({"unknown": {}}, "Invalid v2 operation"),
    ],
)
def test_v2_validation_is_line_aware(step, message):
    step["__line__"] = 17
    with pytest.raises(TestDefinitionError, match=message) as error:
        parse_structured_steps([step])
    assert "17" in error.value.user_message


def _automator_with_match():
    client = MagicMock()
    client.perceive = AsyncMock(
        return_value={
            "top_matches": [{"text": "Sign in", "bounds": [10, 20, 110, 60]}],
            "viewport": {"width": 300, "height": 200},
        }
    )
    page = MagicMock()
    page.url = "https://example.test"
    page.screenshot = AsyncMock(return_value=b"image")
    page.evaluate = AsyncMock(return_value=0)
    page.mouse = MagicMock()
    page.mouse.click = AsyncMock()
    page.mouse.move = AsyncMock()
    page.mouse.down = AsyncMock()
    page.mouse.up = AsyncMock()
    page.keyboard = MagicMock()
    page.keyboard.press = AsyncMock()
    page.keyboard.type = AsyncMock()
    return Automator(perception_client=client, page=page, artifact_dir=None), client, page


def test_v2_type_uses_ranked_perception_target_and_playwright():
    automator, client, page = _automator_with_match()
    session = TestSession(id="s", test_name="v2", url="https://example.test")
    step = parse_structured_steps([{"type": {"target": "username field", "text": "admin"}}])[0]

    assert asyncio.run(automator._run_step_recursive(session, step, None)) is True
    assert step.status == StepStatus.PASSED
    assert client.perceive.await_args.kwargs["query"] == "username field"
    page.mouse.click.assert_awaited_with(60.0, 40.0)
    page.keyboard.type.assert_awaited_once_with("admin")


def test_ranked_find_rejects_fallback_candidates_and_skips_semantic_history():
    automator, client, _page = _automator_with_match()
    automator.parser.normalize_subject = MagicMock()
    client.perceive.return_value = {"top_matches": [], "elements": [{"text": "Fallback"}]}
    session = TestSession(id="s", test_name="v2", url="https://example.test")
    step = parse_structured_steps([{"click": {"target": "submit"}}])[0].sub_steps[0]

    assert asyncio.run(automator._execute_atomic_step(session, step)) is False
    assert step.failure_type == FailureType.PERCEPTION_MISMATCH
    automator.parser.normalize_subject.assert_not_called()


def test_ranked_find_enforces_configured_similarity_threshold():
    automator, client, _page = _automator_with_match()
    automator.parser.config.perception_match_threshold = 0.8
    client.perceive.return_value = {"top_matches": [{"text": "Submit", "similarity": 0.5}], "elements": []}
    session = TestSession(id="s", test_name="v2", url="https://example.test")
    step = parse_structured_steps([{"click": {"target": "submit"}}])[0].sub_steps[0]

    assert asyncio.run(automator._execute_atomic_step(session, step)) is False
    assert step.failure_type == FailureType.PERCEPTION_MISMATCH


def test_ranked_verify_bypasses_intent_parsing():
    automator, _client, _page = _automator_with_match()
    automator.parser.parse_verify_intent = MagicMock()
    session = TestSession(id="s", test_name="v2", url="https://example.test")
    step = parse_structured_steps([{"assert_visible": {"target": "success toast"}}])[0]

    assert asyncio.run(automator._run_step_recursive(session, step, None)) is True
    automator.parser.parse_verify_intent.assert_not_called()


def test_v2_absence_assertion_and_missing_target_have_expected_statuses():
    automator, client, _page = _automator_with_match()
    client.perceive.return_value = {"top_matches": [], "elements": []}
    session = TestSession(id="s", test_name="v2", url="https://example.test")
    absent = parse_structured_steps([{"assert_not_visible": {"target": "error toast", "timeout": 1}}])[0]
    missing = parse_structured_steps([{"click": {"target": "submit"}}])[0]

    assert asyncio.run(automator._run_step_recursive(session, absent, None)) is True
    assert asyncio.run(automator._run_step_recursive(session, missing, None)) is False
    assert missing.failure_type == FailureType.PERCEPTION_MISMATCH


def test_v2_executes_every_operation_with_playwright_primitives():
    automator, client, page = _automator_with_match()
    automator.parser.config.step_delay_seconds = 0
    file_input = MagicMock()
    file_input.count = AsyncMock(return_value=1)
    page.locator.return_value = file_input
    page.set_input_files = AsyncMock()
    session = TestSession(
        id="s",
        test_name="v2",
        url="https://example.test",
        artifacts={"fixture": {"type": "file", "value": "/tmp/fixture.txt"}},
    )
    operations = [
        {"click": {"target": "button"}},
        {"right_click": {"target": "item"}},
        {"hover": {"target": "menu"}},
        {"type": {"target": "input", "text": "value"}},
        {"clear": {"target": "input"}},
        {"press_key": {"key": "Enter"}},
        {"check": {"target": "checkbox"}},
        {"uncheck": {"target": "checkbox"}},
        {"select": {"target": "dropdown", "option": "Option A"}},
        {"drag": {"source": "card", "target": "column"}},
        {"upload": {"target": "upload area", "file": "{fixture}"}},
        {"scroll": {"position": "bottom"}},
        {"scroll": {"target": "footer"}},
        {"wait": {"seconds": 0}},
        {"assert_visible": {"target": "toast"}},
    ]
    for step in parse_structured_steps(operations):
        assert asyncio.run(automator._run_step_recursive(session, step, None)) is True
        assert step.status == StepStatus.PASSED

    page.keyboard.press.assert_awaited()
    page.mouse.down.assert_awaited_once()
    page.set_input_files.assert_awaited_once_with("input[type=file]", "/tmp/fixture.txt")
    assert client.perceive.await_count >= 13


def test_json_schema_is_valid_json_and_advertises_all_operations():
    schema = json.loads((Path(__file__).parents[2] / "schemas" / "vizqa-test.schema.json").read_text())
    rendered = json.dumps(schema)
    for action in (
        "click",
        "right_click",
        "hover",
        "type",
        "clear",
        "press_key",
        "check",
        "uncheck",
        "select",
        "drag",
        "upload",
        "scroll",
        "wait",
        "assert_visible",
        "assert_not_visible",
    ):
        assert f'"{action}"' in rendered
