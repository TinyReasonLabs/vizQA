"""Public library API for embedding vizQA into Playwright tests."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from playwright.async_api import Page

from vizQA.app.client import PerceptionClient
from vizQA.app.core import Automator
from vizQA.app.logger import get_default_logger, get_logger, wrap_logger
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
    :param message: Human-readable step outcome details, if available.
    """

    success: bool
    instruction: str
    matched_element: Optional[Dict[str, Any]]
    artifacts: Dict[str, str]
    duration: float
    message: Optional[str] = None

    def __bool__(self) -> bool:
        """Return the step success state for truthiness checks.

        :return: ``True`` when the step succeeded, otherwise ``False``.
        """
        return self.success


# pylint: disable=too-many-instance-attributes
@dataclass(slots=True, init=False)
class ElementMatch:
    """Normalized UI element metadata returned by the search layer.

    Attributes:
        id: Backend-provided element identifier when available.
        type: Backend element type such as ``link`` or ``button``.
        label: Primary human-facing text for the element.
        location: Backend location tuple. When sourced from ``location``,
            values are normalized to the viewport using
            ``(x, y, width, height)`` in the ``0..1`` range. When sourced
            from legacy ``bounds``, the tuple is carried through as
            pixel-space ``(left, top, right, bottom)``.
        center: Pixel-space center point derived from the available
            coordinates. This is usually the most convenient value for
            automation callers.
        rank: 1-based candidate rank within the returned result set.
        confidence: Backend confidence or legacy score value.
        salience: Backend visual prominence score when available.
        similarity: Backend semantic similarity score when available.
        attributes: Remaining backend metadata that is not promoted to a
            first-class field.
    """

    id: str
    type: str
    label: str
    location: Tuple[float, float, float, float]
    center: Tuple[float, float]
    rank: int
    confidence: Optional[float] = None
    salience: Optional[float] = None
    similarity: Optional[float] = None
    attributes: Dict[str, Any] = field(default_factory=dict)

    def __init__(self, data: Dict[str, Any], rank: int):
        """Initialize a public element match from one backend candidate payload."""
        location = _location_tuple(data)
        self.id = str(data.get("id") or "")
        self.type = str(data.get("type") or data.get("role") or "")
        self.label = str(data.get("label") or data.get("text") or "")
        self.location = location
        self.center = _location_center(data, location)
        self.rank = rank
        self.confidence = _optional_float(data.get("confidence", data.get("score")))
        self.salience = _optional_float(data.get("salience"))
        self.similarity = _optional_float(data.get("similarity"))
        self.attributes = {
            key: value
            for key, value in data.items()
            if key
            not in {
                "id",
                "type",
                "role",
                "label",
                "text",
                "location",
                "bounds",
                "confidence",
                "score",
                "salience",
                "similarity",
            }
        }

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """Backward-compatible alias for the backend location tuple."""
        return self.location

    @property
    def text(self) -> str:
        """Backward-compatible alias for the primary element label."""
        return self.label

    @property
    def role(self) -> str:
        """Backward-compatible alias for the backend element type."""
        return self.type

    @property
    def score(self) -> Optional[float]:
        """Backward-compatible alias for backend confidence."""
        return self.confidence


@dataclass(slots=True, init=False)
class SearchResult:
    """Structured result returned from a library API search call.

    Attributes:
        query: The caller-provided search string.
        viewport: Backend viewport metadata.
        session_id: Backend perception session identifier.
        best_match: The top-ranked match, if any.
        matches: Ranked candidate matches.
        artifacts: Persistent local artifact paths.
        metadata: Additional backend fields.
    """

    query: str
    viewport: Dict[str, Any]
    session_id: Optional[str]
    best_match: Optional[ElementMatch]
    matches: List[ElementMatch]
    artifacts: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __init__(self, query: str, perception: Dict[str, Any], artifacts: Optional[Dict[str, str]] = None):
        """Initialize a public search result from one backend perception payload."""
        top_matches = perception.get("top_matches") or []
        elements = perception.get("elements") or []
        selected_rank_source = top_matches if top_matches else elements
        selected = selected_rank_source[0] if selected_rank_source else None

        self.query = query
        self.viewport = perception.get("viewport") or {}
        self.session_id = perception.get("session_id")
        self.best_match = ElementMatch(selected, rank=1) if isinstance(selected, dict) else None
        self.matches = [
            ElementMatch(candidate, rank=index)
            for index, candidate in enumerate(selected_rank_source, start=1)
            if isinstance(candidate, dict)
        ]
        self.artifacts = artifacts or {}
        self.metadata = {
            key: value
            for key, value in perception.items()
            if key not in {"session_id", "viewport", "top_matches", "elements"}
        }


class VizQASession:
    """Session wrapper for executing vizQA steps on an existing Playwright page.

    The session reuses a caller-owned Playwright :class:`Page` and keeps the
    browser lifecycle under the caller's control.
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        page: Page,
        *,
        perception_backend: Optional[str] = None,
        verbosity: int = 0,
        debug_dir: Optional[str] = None,
        logger: Optional[Any] = None,
    ):
        """Initialize a reusable vizQA library session.

        :param page: The Playwright page to attach vizQA to.
        :param perception_backend: Optional override for the perception backend URL.
        :param verbosity: Verbosity level for runtime diagnostics.
        :param debug_dir: Optional directory for persistent step artifacts.
        """
        self.page = page
        self.debug_dir = debug_dir
        normalized_logger = wrap_logger(logger) if logger is not None else None
        self.logger = normalized_logger or (get_logger("library") if debug_dir else get_default_logger())
        self.client = PerceptionClient(base_url=perception_backend, logger=self.logger)
        self._automator = Automator(
            perception_client=self.client,
            verbosity=verbosity,
            page=page,
            logger=self.logger,
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

    # pylint: disable=W0613,W9015
    async def run_step(self, instruction: str, **options: Any) -> StepResult:
        """Run a single natural-language instruction against the attached page.

        :param instruction: A user-facing instruction such as
            ``"Click the sign in button"``.
        :return: The resulting :class:`StepResult`.
        """
        planned_step = self._planner.decompose([{"action": instruction}])[0]
        session = self._make_test_session([planned_step])
        success = await self._automator.run_session(session, preserve_page=True)
        return self._step_result(planned_step, session, success)

    async def run_steps(self, instructions: Iterable[str], **options: Any) -> List[StepResult]:
        """Run multiple natural-language instructions in sequence.

        :param instructions: An iterable of natural-language instructions.
        :param options: Optional keyword arguments for runtime configuration.
        :return: A list of :class:`StepResult` values in execution order.
        """
        results = []
        for instruction in instructions:
            results.append(await self.run_step(instruction, **options))
        return results

    async def search(self, query: str, **options: Any) -> SearchResult:
        """Search the current page state and return typed element metadata.

        :param query: The visual or semantic description to search for.
        :param options: Optional keyword arguments for runtime configuration.
        :return: A structured :class:`SearchResult`.
        """
        session = self._make_test_session([])
        search_step = TestStep(id="search_00", instruction=f"SEARCH: {query}")
        test_slug = self._session_slug(session)
        image_input, persistent = await self._automator._capture_perception_input(  # pylint: disable=protected-access
            test_slug, f"{search_step.id}_before"
        )
        if persistent:
            search_step.screenshot_before = image_input["image_path"]

        perception_scope = await self._automator._build_perception_scope(session)  # pylint: disable=protected-access
        perception = await self.client.perceive(query=query, session_scope=perception_scope, **image_input)

        best_candidate = self._automator._select_perception_target(perception)  # pylint: disable=protected-access
        self._automator._log_perception_summary(  # pylint: disable=protected-access
            search_step.id, query, perception, selected=best_candidate
        )

        artifacts = {"before": search_step.screenshot_before} if search_step.screenshot_before else {}
        return _search_result(query, perception, artifacts=artifacts)

    async def click(self, target: str, **options: Any) -> StepResult:
        """Click a target identified by visual or semantic description.

        :param target: The target description, such as ``"Sign in button"``.
        :param options: Optional keyword arguments for runtime configuration.
        :return: The resulting :class:`StepResult`.
        """
        return await self.run_step(f"Click {target}", **options)

    async def type(self, target: str, text: str, **options: Any) -> StepResult:
        """Type text into a described input target.

        :param target: The input target description.
        :param text: The text to type.
        :param options: Optional keyword arguments for runtime configuration.
        :return: The resulting :class:`StepResult`.
        """
        return await self.run_step(f"Type '{text}' into {target}", **options)

    # pylint: disable=W0613,W9015
    async def verify(self, assertion: str, **options: Any) -> StepResult:
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

    @staticmethod
    def _session_slug(session: TestSession) -> str:
        """Return the filesystem-safe slug used for library artifacts."""
        return session.file_stem or session.test_name or session.id

    def _step_result(self, step: TestStep, session: TestSession, success: bool) -> StepResult:
        """Convert runtime state into a high-signal library result.

        :param step: The executed step.
        :param session: The session that produced the step outcome.
        :param success: Whether the step succeeded.
        :return: A normalized :class:`StepResult`.
        """
        matched_element = session.metadata.get("target")
        message = step.failure_reason or step.error
        return StepResult(
            success=success,
            instruction=step.instruction,
            matched_element=matched_element if isinstance(matched_element, dict) else None,
            artifacts=_collect_artifacts(step),
            duration=_step_duration(step),
            message=message,
        )


def attach(
    page: Page,
    *,
    perception_backend: Optional[str] = None,
    verbosity: int = 0,
    debug_dir: Optional[str] = None,
    logger: Optional[Any] = None,
) -> VizQASession:
    """Attach vizQA to an existing Playwright page.

    :param page: The Playwright page to attach to.
    :param perception_backend: Optional override for the perception backend URL.
    :param verbosity: Verbosity level for runtime diagnostics.
    :param debug_dir: Optional directory for persistent step artifacts.
    :param logger: Optional application logger to receive vizQA diagnostics.
        Can be a ``logging.Logger`` or any object exposing the vizQA logger methods.
    :return: A reusable :class:`VizQASession`.
    """
    return VizQASession(
        page,
        perception_backend=perception_backend,
        verbosity=verbosity,
        debug_dir=debug_dir,
        logger=logger,
    )


async def run_step(page: Page, instruction: str, **options: Any) -> StepResult:
    """Run a single instruction using a short-lived attached session.

    :param page: The Playwright page to act on.
    :param instruction: The natural-language instruction to execute.
    :param options: Optional attach-time configuration such as
        ``perception_backend``, ``verbosity``, or ``debug_dir``.
    :return: The resulting :class:`StepResult`.
    """
    return await attach(page, **_attachoptions(options)).run_step(instruction, **options)


async def run_steps(page: Page, instructions: Iterable[str], **options: Any) -> List[StepResult]:
    """Run several instructions using a short-lived attached session.

    :param page: The Playwright page to act on.
    :param instructions: The natural-language instructions to execute.
    :param options: Optional attach-time configuration.
    :return: A list of :class:`StepResult` values.
    """
    return await attach(page, **_attachoptions(options)).run_steps(instructions, **options)


async def click(page: Page, target: str, **options: Any) -> StepResult:
    """Click a described target using a short-lived attached session.

    :param page: The Playwright page to act on.
    :param target: The visual or semantic target description.
    :param options: Optional attach-time configuration.
    :return: The resulting :class:`StepResult`.
    """
    return await attach(page, **_attachoptions(options)).click(target, **options)


async def search(page: Page, query: str, **options: Any) -> SearchResult:
    """Search the current page using a short-lived attached session.

    :param page: The Playwright page to inspect.
    :param query: The visual or semantic target description.
    :param options: Optional attach-time configuration.
    :return: A structured :class:`SearchResult`.
    """
    return await attach(page, **_attachoptions(options)).search(query, **options)


# pylint:disable=redefined-builtin
async def type(page: Page, target: str, text: str, **options: Any) -> StepResult:
    """Type text into a described target using a short-lived attached session.

    :param page: The Playwright page to act on.
    :param target: The input target description.
    :param text: The text to enter.
    :param options: Optional attach-time configuration.
    :return: The resulting :class:`StepResult`.
    """
    return await attach(page, **_attachoptions(options)).type(target, text, **options)


async def verify(page: Page, assertion: str, **options: Any) -> StepResult:
    """Verify a visual assertion using a short-lived attached session.

    :param page: The Playwright page to inspect.
    :param assertion: The visual assertion to check.
    :param options: Optional attach-time configuration.
    :return: The resulting :class:`StepResult`.
    """
    return await attach(page, **_attachoptions(options)).verify(assertion, **options)


def _attachoptions(options: Dict[str, Any]) -> Dict[str, Any]:
    """Extract attach-relevant keyword arguments from a call.

    :param options: Arbitrary keyword arguments passed to a helper.
    :return: A dictionary containing only attach-time options.
    """
    return {
        "perception_backend": options.get("perception_backend"),
        "verbosity": options.get("verbosity", 0),
        "debug_dir": options.get("debug_dir"),
        "logger": options.get("logger"),
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


def _search_result(query: str, perception: Dict[str, Any], artifacts: Optional[Dict[str, str]] = None) -> SearchResult:
    """Normalize a raw perception payload into the public search result shape."""
    return SearchResult(query, perception, artifacts=artifacts)


def _element_match(candidate: Dict[str, Any], rank: int) -> ElementMatch:
    """Normalize one perception candidate into the public element shape."""
    return ElementMatch(candidate, rank=rank)


def _location_tuple(candidate: Dict[str, Any]) -> Tuple[float, float, float, float]:
    """Normalize location-like candidate coordinates into a strict 4-float tuple."""
    location = candidate.get("location")
    if isinstance(location, (list, tuple)) and len(location) == 4:
        return tuple(float(value) for value in location)

    bounds = candidate.get("bounds")
    if isinstance(bounds, (list, tuple)) and len(bounds) == 4:
        return tuple(float(value) for value in bounds)

    return (0.0, 0.0, 0.0, 0.0)


def _location_center(candidate: Dict[str, Any], location: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """Compute element center using the same coordinate semantics as the source payload."""
    if "location" in candidate:
        left, top, width, height = location
        return (left + (width / 2), top + (height / 2))

    left, top, right, bottom = location
    return ((left + right) / 2, (top + bottom) / 2)


def _optional_float(value: Any) -> Optional[float]:
    """Convert optional numeric-like values into floats when available."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
