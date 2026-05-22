# pylint:disable=W9015
"""Public library API for embedding vizQA into Playwright tests."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from playwright.async_api import Page

from vizQA.app.client import PerceptionClient
from vizQA.app.core import Automator
from vizQA.app.logger import NullLogger, get_logger
from vizQA.app.memory import TestSession, TestStep
from vizQA.planning import StepPlanner


@dataclass(slots=True)
class StepResult:
    """Structured result returned from a library API step execution.

    :param success: Whether the step completed successfully.
    :param instruction: The instruction that was executed.
    :param matched_element: The best matched element metadata, if available.
    :param artifacts: Persistent artifact paths captured for the step.
    :param duration: Execution duration in seconds.
    :param raw: Additional low-level metadata for debugging and reporting.
    """

    success: bool
    instruction: str
    matched_element: Optional[Dict[str, Any]]
    artifacts: Dict[str, str]
    duration: float
    raw: Dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        """Return the step success state for truthiness checks.

        :return: ``True`` when the step succeeded, otherwise ``False``.
        """
        return self.success


class VizQASession:
    """Session wrapper for executing vizQA steps on an existing Playwright page.

    The session reuses a caller-owned Playwright :class:`Page` and keeps the
    browser lifecycle under the caller's control.
    """

    def __init__(
        self,
        page: Page,
        *,
        perception_backend: Optional[str] = None,
        verbosity: int = 0,
        debug_dir: Optional[str] = None,
    ):
        """Initialize a reusable vizQA library session.

        :param page: The Playwright page to attach vizQA to.
        :param perception_backend: Optional override for the perception backend URL.
        :param verbosity: Verbosity level for runtime diagnostics.
        :param debug_dir: Optional directory for persistent step artifacts.
        """
        self.page = page
        self.debug_dir = debug_dir
        logger = get_logger("library") if debug_dir else NullLogger()
        self.client = PerceptionClient(base_url=perception_backend, logger=logger)
        self._automator = Automator(
            perception_client=self.client,
            verbosity=verbosity,
            page=page,
            logger=logger,
            artifact_dir=debug_dir,
        )
        self._planner = StepPlanner(
            model_name="minilm",
            parser=self._automator.parser,
            minilm=self._automator.minilm,
            logger=self._automator.logger,
        )

    async def close(self) -> None:
        """Release vizQA-managed resources without closing the attached page."""
        await self._automator.stop()

    async def run_step(self, instruction: str, **_options: Any) -> StepResult:
        """Run a single natural-language instruction against the attached page.

        :param instruction: A user-facing instruction such as
            ``"Click the sign in button"``.
        :return: The resulting :class:`StepResult`.
        """
        planned_step = self._planner.decompose([{"action": instruction}])[0]
        session = self._make_test_session([planned_step])
        success = await self._automator.run_session(session, preserve_page=True)
        return self._step_result(planned_step, session, success)

    async def run_steps(self, instructions: Iterable[str], **_options: Any) -> List[StepResult]:
        """Run multiple natural-language instructions in sequence.

        :param instructions: An iterable of natural-language instructions.
        :return: A list of :class:`StepResult` values in execution order.
        """
        results = []
        for instruction in instructions:
            results.append(await self.run_step(instruction))
        return results

    async def click(self, target: str, **options: Any) -> StepResult:
        """Click a target identified by visual or semantic description.

        :param target: The target description, such as ``"Sign in button"``.
        :return: The resulting :class:`StepResult`.
        """
        return await self.run_step(f"Click {target}", **options)

    async def type(self, target: str, text: str, **options: Any) -> StepResult:
        """Type text into a described input target.

        :param target: The input target description.
        :param text: The text to type.
        :return: The resulting :class:`StepResult`.
        """
        return await self.run_step(f"Type '{text}' into {target}", **options)

    async def verify(self, assertion: str, **_options: Any) -> StepResult:
        """Run a visual verification against the current page state.

        :param assertion: The assertion to verify, such as
            ``"Overview dashboard"``.
        :return: The resulting :class:`StepResult`.
        """
        step = TestStep(id="step_00", instruction=f"VERIFY: {assertion}")
        session = self._make_test_session([step])
        success = await self._automator.run_session(session, preserve_page=True)
        return self._step_result(step, session, success)

    def _make_test_session(self, steps: List[TestStep]) -> TestSession:
        """Create an internal test session for library execution.

        :param steps: The planned steps to execute.
        :return: A :class:`TestSession` configured for library usage.
        """
        current_url = getattr(self.page, "url", "") or ""
        return TestSession(
            id=str(uuid.uuid4())[:8],
            test_name="library_session",
            file_stem="library_session",
            url=current_url,
            steps=steps,
            metadata={"debug_dir": self.debug_dir} if self.debug_dir else {},
        )

    def _step_result(self, step: TestStep, session: TestSession, success: bool) -> StepResult:
        """Convert runtime state into a high-signal library result.

        :param step: The executed step.
        :param session: The session that produced the step outcome.
        :param success: Whether the step succeeded.
        :return: A normalized :class:`StepResult`.
        """
        matched_element = session.metadata.get("target")
        raw = {
            "status": step.status.value,
            "failure_reason": step.failure_reason,
            "error": step.error,
            "metadata": session.metadata,
        }
        return StepResult(
            success=success,
            instruction=step.instruction,
            matched_element=matched_element if isinstance(matched_element, dict) else None,
            artifacts=_collect_artifacts(step),
            duration=_step_duration(step),
            raw=raw,
        )


def attach(
    page: Page,
    *,
    perception_backend: Optional[str] = None,
    verbosity: int = 0,
    debug_dir: Optional[str] = None,
) -> VizQASession:
    """Attach vizQA to an existing Playwright page.

    :param page: The Playwright page to attach to.
    :param perception_backend: Optional override for the perception backend URL.
    :param verbosity: Verbosity level for runtime diagnostics.
    :param debug_dir: Optional directory for persistent step artifacts.
    :return: A reusable :class:`VizQASession`.
    """
    return VizQASession(
        page,
        perception_backend=perception_backend,
        verbosity=verbosity,
        debug_dir=debug_dir,
    )


async def run_step(page: Page, instruction: str, **options: Any) -> StepResult:
    """Run a single instruction using a short-lived attached session.

    :param page: The Playwright page to act on.
    :param instruction: The natural-language instruction to execute.
    :param options: Optional attach-time configuration such as
        ``perception_backend``, ``verbosity``, or ``debug_dir``.
    :return: The resulting :class:`StepResult`.
    """
    return await attach(page, **_attach_options(options)).run_step(instruction, **options)


async def run_steps(page: Page, instructions: Iterable[str], **options: Any) -> List[StepResult]:
    """Run several instructions using a short-lived attached session.

    :param page: The Playwright page to act on.
    :param instructions: The natural-language instructions to execute.
    :param options: Optional attach-time configuration.
    :return: A list of :class:`StepResult` values.
    """
    return await attach(page, **_attach_options(options)).run_steps(instructions, **options)


async def click(page: Page, target: str, **options: Any) -> StepResult:
    """Click a described target using a short-lived attached session.

    :param page: The Playwright page to act on.
    :param target: The visual or semantic target description.
    :param options: Optional attach-time configuration.
    :return: The resulting :class:`StepResult`.
    """
    return await attach(page, **_attach_options(options)).click(target, **options)


# pylint:disable=redefined-builtin
async def type(page: Page, target: str, text: str, **options: Any) -> StepResult:
    """Type text into a described target using a short-lived attached session.

    :param page: The Playwright page to act on.
    :param target: The input target description.
    :param text: The text to enter.
    :param options: Optional attach-time configuration.
    :return: The resulting :class:`StepResult`.
    """
    return await attach(page, **_attach_options(options)).type(target, text, **options)


async def verify(page: Page, assertion: str, **options: Any) -> StepResult:
    """Verify a visual assertion using a short-lived attached session.

    :param page: The Playwright page to inspect.
    :param assertion: The visual assertion to check.
    :param options: Optional attach-time configuration.
    :return: The resulting :class:`StepResult`.
    """
    return await attach(page, **_attach_options(options)).verify(assertion, **options)


def _attach_options(options: Dict[str, Any]) -> Dict[str, Any]:
    """Extract attach-relevant keyword arguments from a call.

    :param options: Arbitrary keyword arguments passed to a helper.
    :return: A dictionary containing only attach-time options.
    """
    return {
        "perception_backend": options.get("perception_backend"),
        "verbosity": options.get("verbosity", 0),
        "debug_dir": options.get("debug_dir"),
    }


def _collect_artifacts(step: TestStep) -> Dict[str, str]:
    """Recursively collect artifact paths from a step tree.

    :param step: The step whose artifacts should be gathered.
    :return: A mapping of artifact role to filesystem path.
    """
    artifacts: Dict[str, str] = {}
    if step.screenshot_before:
        artifacts["before"] = step.screenshot_before
    if step.action_screenshot:
        artifacts["action"] = step.action_screenshot
    if step.screenshot_after:
        artifacts["after"] = step.screenshot_after

    if artifacts:
        return artifacts

    for sub_step in step.sub_steps:
        artifacts.update(_collect_artifacts(sub_step))

    return artifacts


def _step_duration(step: TestStep) -> float:
    """Calculate the best available duration for a step.

    :param step: The step whose duration should be measured.
    :return: The step duration in seconds, or ``0.0`` if unavailable.
    """
    if step.start_time and step.end_time:
        return (step.end_time - step.start_time).total_seconds()

    for sub_step in step.sub_steps:
        duration = _step_duration(sub_step)
        if duration > 0:
            return duration

    return 0.0
