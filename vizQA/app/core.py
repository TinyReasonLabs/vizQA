"""
Core execution engine for vision-driven UI automation.
"""

# pylint: disable=invalid-name, too-many-lines
import asyncio
import copy
import inspect
import os
import re
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from playwright.async_api import Browser, Page, async_playwright

from vizQA.app.client import PerceptionClient
from vizQA.app.exceptions import (
    ActionExecutionError,
    ArtifactError,
    BrowserError,
    ElementNotFoundError,
    UserFacingException,
    VerificationError,
)
from vizQA.app.logger import get_logger
from vizQA.app.memory import FailureType, StepStatus, TestSession, TestStep
from vizQA.app.support.weights import get_model_dir
from vizQA.reasoning import Intent, MiniLM, SemanticParser

if TYPE_CHECKING:
    from vizQA.app.logger import SessionLogger
    from vizQA.app.viewport import ViewportSpec


# pylint: disable=too-many-instance-attributes
class Automator:
    """
    Main controller for browser automation and perception-integrated execution.
    """

    # pylint: disable=too-many-arguments, too-many-positional-arguments
    def __init__(
        self,
        perception_client: PerceptionClient,
        verbosity: int = 0,
        debug_logging: Optional[bool] = None,
        headless: bool = True,
        viewport: Optional["ViewportSpec"] = None,
        logger: Optional["SessionLogger"] = None,
        page: Optional[Page] = None,
        artifact_dir: Optional[str] = ".vizQA",
    ):
        """
        Initialises the Automator.

        :param perception_client: The perception client.
        :param verbosity: The verbosity level.
        :param headless: Whether the browser should be headless.
        """
        self.client = perception_client
        self.verbosity = verbosity
        self.debug_logging = debug_logging
        self.headless = headless
        self.viewport = viewport
        self.playwright_mgr: Optional[Any] = None
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = page
        self._owns_browser = page is None
        self.artifact_dir = artifact_dir
        self.logger = logger or get_logger(viewport.slug if viewport else None)
        if self.artifact_dir:
            os.makedirs(self.artifact_dir, exist_ok=True)

        model_dir = os.fspath(get_model_dir())
        try:
            self.minilm: Optional[MiniLM] = MiniLM(model_dir, logger=self.logger)
        except (FileNotFoundError, RuntimeError):
            self.minilm = None
            self.logger.log_warning("init", "MiniLM model not found — semantic matching degraded to substring.")

        # Single shared parser instance wired to the model
        self.parser = SemanticParser(semantic_provider=self.minilm, logger=self.logger)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self):
        """Initialises the Playwright browser and page."""
        if self.page:
            if self.artifact_dir:
                os.makedirs(self.artifact_dir, exist_ok=True)
            return
        self.playwright_mgr = await async_playwright().start()
        self.browser = await self.playwright_mgr.chromium.launch(headless=self.headless)
        if self.viewport:
            self.page = await self.browser.new_page(
                viewport={"width": self.viewport.width, "height": self.viewport.height}
            )
        else:
            self.page = await self.browser.new_page()
        if self.artifact_dir:
            os.makedirs(self.artifact_dir, exist_ok=True)

    async def stop(self):
        """Closes the browser and stops the Playwright manager."""
        if self.page and self._owns_browser:
            await self.page.close()
        if self.browser and self._owns_browser:
            await self.browser.close()
        if self.playwright_mgr and self._owns_browser:
            await self.playwright_mgr.stop()

    # ------------------------------------------------------------------
    # Browser state persistence
    # ------------------------------------------------------------------

    async def capture_browser_state(self) -> Dict[str, Any]:
        """
        Captures the current browser state including localStorage, sessionStorage, and cookies.

        :return: Dictionary containing browser state with keys: 'localStorage', 'sessionStorage', 'cookies'
        """
        if not self.page:
            return {}

        state = {}

        try:
            # Capture localStorage
            local_storage = await self.page.evaluate(
                """() => {
                    const items = {};
                    for (let i = 0; i < localStorage.length; i++) {
                        const key = localStorage.key(i);
                        items[key] = localStorage.getItem(key);
                    }
                    return items;
                }"""
            )
            state["localStorage"] = local_storage
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self.logger.log_warning("capture_state", f"Failed to capture localStorage: {exc}")
            state["localStorage"] = {}

        try:
            # Capture sessionStorage
            session_storage = await self.page.evaluate(
                """() => {
                    const items = {};
                    for (let i = 0; i < sessionStorage.length; i++) {
                        const key = sessionStorage.key(i);
                        items[key] = sessionStorage.getItem(key);
                    }
                    return items;
                }"""
            )
            state["sessionStorage"] = session_storage
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self.logger.log_warning("capture_state", f"Failed to capture sessionStorage: {exc}")
            state["sessionStorage"] = {}

        try:
            # Capture cookies
            cookies = await self.page.context.cookies()
            state["cookies"] = cookies
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self.logger.log_warning("capture_state", f"Failed to capture cookies: {exc}")
            state["cookies"] = []

        return state

    async def restore_browser_state(self, state: Dict[str, Any]) -> None:
        """
        Restores previously captured browser state (localStorage, sessionStorage, cookies) to the current page.

        :param state: Dictionary with keys 'localStorage', 'sessionStorage', 'cookies'
        """
        if not self.page:
            return

        try:
            # Restore localStorage
            local_storage = state.get("localStorage", {})
            if local_storage:
                await self.page.evaluate(
                    """(items) => {
                        localStorage.clear();
                        for (const [key, value] of Object.entries(items)) {
                            localStorage.setItem(key, value);
                        }
                    }""",
                    local_storage,
                )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self.logger.log_warning("restore_state", f"Failed to restore localStorage: {exc}")

        try:
            # Restore sessionStorage
            session_storage = state.get("sessionStorage", {})
            if session_storage:
                await self.page.evaluate(
                    """(items) => {
                        sessionStorage.clear();
                        for (const [key, value] of Object.entries(items)) {
                            sessionStorage.setItem(key, value);
                        }
                    }""",
                    session_storage,
                )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self.logger.log_warning("restore_state", f"Failed to restore sessionStorage: {exc}")

        try:
            # Restore cookies
            cookies = state.get("cookies", [])
            if cookies:
                await self.page.context.add_cookies(cookies)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self.logger.log_warning("restore_state", f"Failed to restore cookies: {exc}")

    # ------------------------------------------------------------------
    # Session runner
    # ------------------------------------------------------------------

    async def navigate_to_session_url(self, session: TestSession) -> None:
        """Navigate to the session URL and raise a clean error if the site is unreachable."""
        if not self.page:
            raise RuntimeError("Browser page is not initialized.")

        try:
            await self.page.goto(session.url)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            test_label = session.file_stem or session.test_name
            raise BrowserError(
                f"Site URL is not reachable for test '{test_label}': {session.url}",
                internal_detail=str(exc),
            ) from exc

    async def run_session(
        self, session: TestSession, on_step_update: Optional[Any] = None, preserve_page: bool = False
    ) -> bool:
        """Main execution loop for a test session."""
        if not self.page:
            await self.start()

        self.logger.log_session(session.id, "start", f"url={session.url!r}")
        self.logger.log_debug("", f"Steps: {[x.sub_steps for x in session.steps]}")
        if session.headers:
            await self.page.set_extra_http_headers(session.headers)

        # Skip navigation if this test has dependencies (page is already at the right location)
        if not preserve_page and not session.dependency_results:
            await self.navigate_to_session_url(session)

        failed = False
        for step in session.steps:
            if failed:
                await self._skip_step_recursive(step, on_step_update)
                continue

            step_passed = await self._run_step_recursive(session, step, on_step_update)
            if not step_passed:
                failed = True

        session.end_time = datetime.now()
        self.logger.log_session(session.id, "end", f"duration={session.duration:.1f}s")
        return not failed

    # ------------------------------------------------------------------
    # Step dispatcher
    # ------------------------------------------------------------------

    async def _run_step_recursive(self, session: TestSession, step: TestStep, on_step_update: Optional[Any]) -> bool:
        """Executes a step and its sub-steps recursively."""
        step.status = StepStatus.RUNNING
        if on_step_update:
            await on_step_update(step)

        step.start_time = datetime.now()
        success = True

        try:
            if not step.sub_steps:
                success = await self._execute_atomic_step(session, step)
                if step.status == StepStatus.FAILED:
                    success = False
            else:
                success = await self._run_container_step(session, step, on_step_update)
        except (ActionExecutionError, UserFacingException) as exc:
            success = self._handle_user_facing_exception(step, exc)
        except (RuntimeError, ValueError) as exc:
            success = self._handle_system_exception(step, exc)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            success = self._handle_system_exception(step, exc)
        finally:
            step.end_time = datetime.now()
            instr_prefix = step.instruction.split(":")[0] if ":" in step.instruction else "STEP"
            self.logger.log_step(step.id, instr_prefix, step.status, step.failure_reason)
            if on_step_update:
                await on_step_update(step)

        return success

    async def _execute_atomic_step(self, session: TestSession, step: TestStep) -> bool:
        """Executes a single atomic step (FIND, DO, VERIFY, or legacy)."""
        instr = step.instruction
        if instr.startswith("FIND:"):
            return await self._execute_find(session, step, instr.replace("FIND:", "").strip())
        if instr.startswith("DO:"):
            return await self._execute_do(session, step, instr.replace("DO:", "").strip())
        if instr.startswith("VERIFY:"):
            return await self._execute_verify(session, step, instr.replace("VERIFY:", "").strip())
        return await self._execute_legacy(session, step)

    async def _run_container_step(self, session: TestSession, step: TestStep, on_step_update: Optional[Any]) -> bool:
        """Executes a step containing sub-steps."""
        sub_failed = False
        for sub_step in step.sub_steps:
            if sub_failed:
                await self._skip_step_recursive(sub_step, on_step_update)
                continue

            if not await self._run_step_recursive(session, sub_step, on_step_update):
                sub_failed = True

        if sub_failed:
            step.status = StepStatus.FAILED
            for sub_step in step.sub_steps:
                if sub_step.status == StepStatus.FAILED:
                    step.failure_type = sub_step.failure_type
                    step.failure_reason = sub_step.failure_reason
                    break
            return False

        step.status = StepStatus.PASSED
        return True

    def _handle_user_facing_exception(self, step: TestStep, exc: UserFacingException) -> bool:
        """Handles expected user-facing exceptions with proper mapping to FailureType."""
        self.logger.log_debug(step.id, f"User-facing error: {exc.user_message}")
        if exc.internal_detail:
            self.logger.log_debug(step.id, f"Detail: {exc.internal_detail}")

        step.status = StepStatus.FAILED
        if isinstance(exc, ElementNotFoundError):
            step.failure_type = FailureType.PERCEPTION_MISMATCH
        elif isinstance(exc, ActionExecutionError):
            step.failure_type = FailureType.ACTION_ERROR
        elif isinstance(exc, VerificationError):
            step.failure_type = FailureType.TIMEOUT
        else:
            step.failure_type = FailureType.SYSTEM_ERROR

        step.failure_reason = exc.user_message
        return False

    def _handle_system_exception(self, step: TestStep, exc: Exception) -> bool:
        """Handles unexpected system exceptions."""
        self.logger.log_exception(step.id, exc)
        step.status = StepStatus.FAILED
        step.failure_type = FailureType.SYSTEM_ERROR
        step.failure_reason = "An unexpected error occurred during step execution."
        step.error = str(exc)
        return False

    # ------------------------------------------------------------------
    # Atomic step executors
    # ------------------------------------------------------------------

    async def _execute_find(self, session: TestSession, step: TestStep, query: str) -> bool:
        """Handles a FIND: sub-step — perceives the page and stores the target.

        :param session: The test session.
        :param step: The step to execute.
        :param query: The query to execute.
        :return: True if the step was executed successfully, False otherwise.
        """
        test_slug = _test_slug(session)
        path, persistent = self._artifact_path(test_slug, f"{step.id}_before")
        await self.page.screenshot(path=path, type="jpeg")
        if persistent:
            step.screenshot_before = path

        # Check if the query refers to an artifact
        artifact_match = re.fullmatch(r"\{([a-zA-Z0-9_]+)\}", query)
        if artifact_match:
            art_name = artifact_match.group(1)
            if art_name in session.artifacts:
                artifact = session.artifacts[art_name]
                session.metadata["target"] = {"type": "artifact", "name": art_name, "value": artifact}
                step.status = StepStatus.PASSED
                return True
            self.logger.log_warning(step.id, f"Artifact '{art_name}' not found in session.")
            raise ValueError(f"Artifact '{art_name}' not found in session.")

        try:
            perception_scope = await self._build_perception_scope(session)
            perception = await self.client.perceive(path, query=query, session_scope=perception_scope)
            step.perception_result = perception
        finally:
            if not persistent:
                self._cleanup_temporary_artifact(path)

        target = None
        if perception.get("top_matches"):
            target = perception["top_matches"][0]
        elif perception.get("elements"):
            target = perception["elements"][0]

        self._log_perception_summary(step.id, query, perception, selected=target)

        if target:
            session.metadata["target"] = target
            session.metadata["last_perception"] = perception

            # Decoupled history update: just store raw data under normalized subject
            norm_subj = self.parser.normalize_subject(query)
            history = session.metadata.setdefault("history", {})
            if norm_subj not in history:
                history[norm_subj] = {
                    "target": copy.deepcopy(target),
                    "elements": copy.deepcopy(perception.get("elements", [])),
                }
                self.logger.log_debug(step.id, f"Stored FIRST appearance for '{norm_subj}'")

            step.status = StepStatus.PASSED
            return True

        step.status = StepStatus.FAILED
        step.failure_type = FailureType.PERCEPTION_MISMATCH
        step.failure_reason = self._failure_details("FIND", query, perception, "Element not found")
        return False

    async def _execute_do(self, session: TestSession, step: TestStep, action_cmd: str) -> bool:
        """Handles a DO: sub-step — resolves coords and fires the interaction.

        :param session: The current TestSession context.
        :param step: The DO step being executed.
        :param action_cmd: The raw command string (e.g. "click 'Login'").
        :return: True if the action was successfully performed.
        """
        parts = action_cmd.split(" ", 1)
        action = parts[0].lower()
        payload = self._resolve_payload(parts[1] if len(parts) > 1 else "", session)

        if action == "wait":
            return await self._handle_wait_action(session, step, payload)
        if action == "scroll":
            return await self._handle_scroll_action(session, step, payload)

        target = session.metadata.get("target")
        if not target:
            step.status = StepStatus.FAILED
            step.failure_reason = "No target element found. FIND step must precede DO step."
            return False

        if target.get("type") == "artifact":
            return await self._handle_artifact_action(session, step, action, target)

        drag_source = session.metadata.get("drag_source")
        if action == "drop" and drag_source and drag_source.get("type") == "artifact":
            success = await self._execute_artifact_drop(session, step, drag_source, target)
            if success:
                session.metadata.pop("drag_source", None)
            return success

        # Resolve coordinates from perception data
        last_perc = session.metadata.get("last_perception", {})
        viewport = last_perc.get("viewport", {"width": 1280, "height": 720})
        rect = _resolve_coords(target, viewport)  # [x, y, w, h]

        await self._capture_action_screenshot(session, step, action, rect, viewport)
        await self._execute_interaction(action, rect[0] + rect[2] / 2, rect[1] + rect[3] / 2, payload)

        step.status = StepStatus.PASSED
        return True

    # pylint: disable=too-many-locals
    async def _poll_for_intent_match(
        self,
        session: TestSession,
        step: TestStep,
        query: str,
        timeout: float,
        poll_interval: float,
        screenshot_suffix: str,
        require_element_match: bool = False,
    ) -> tuple[bool, Dict[str, Any], str]:
        """Polls perception until the semantic intent matches or the timeout expires."""
        start_wait = datetime.now()
        test_slug = _test_slug(session)

        intent = self.parser.parse_verify_intent(query)
        self.logger.log_debug(step.id, f"Parsed intent: {intent}")
        keyword = str(intent.keyword or "")
        subject = str(intent.subject or "")
        position = str(intent.position or "")
        if keyword:
            perc_query = f"'{keyword}' {subject} {position}"
        elif subject or position:
            perc_query = f"{subject} {position}"
        else:
            perc_query = query

        perception: Dict[str, Any] = {}
        reasoning = ""

        while (datetime.now() - start_wait).total_seconds() < timeout:
            path, persistent = self._artifact_path(test_slug, f"{step.id}_{screenshot_suffix}")
            await self.page.screenshot(path=path, type="jpeg")
            if persistent:
                step.screenshot_after = path

            try:
                perception_scope = await self._build_perception_scope(session)
                perception = await self.client.perceive(path, query=perc_query, session_scope=perception_scope)
                step.perception_result = perception
            except Exception:  # pylint: disable=broad-exception-caught
                perception = {}
            finally:
                if not persistent:
                    self._cleanup_temporary_artifact(path)

            if perception:
                self._log_perception_summary(step.id, perc_query, perception, selected=None)
                if require_element_match:
                    match_found, reasoning = self._check_wait_for_match(intent, perception)
                else:
                    match_found, reasoning = self._check_verification_match(session, intent, perception)
                if match_found:
                    return True, perception, reasoning

            await asyncio.sleep(poll_interval)

        return False, perception, reasoning

    async def _execute_verify(
        self, session: TestSession, step: TestStep, query: str, timeout: Optional[int] = None
    ) -> bool:
        """Handles a VERIFY: sub-step — semantically evaluates the UI state with polling."""
        timeout = timeout or self.parser.config.verification_timeout
        start_wait = datetime.now()

        success, perception, reasoning = await self._poll_for_intent_match(session, step, query, timeout, 1.0, "verify")
        if success:
            step.status = StepStatus.PASSED
            return True

        step.status = StepStatus.FAILED
        wait_time = (datetime.now() - start_wait).total_seconds()
        reasoning = reasoning or f"Verification failed after {wait_time:.1f}s"
        step.failure_reason = self._failure_details("VERIFY", query, perception, reasoning)
        return False

    async def _handle_wait_action(self, session: TestSession, step: TestStep, payload: str) -> bool:
        """Handles waiting for a specific amount of time or a semantic UI condition."""
        wait_time = self._parse_wait_duration(payload)
        if wait_time is not None:
            self.logger.log_debug(step.id, f"Waiting for {wait_time}s (parsed from '{payload}')")
            await asyncio.sleep(wait_time)
            step.status = StepStatus.PASSED
            return True

        query = payload.strip()
        if not query:
            self.logger.log_debug(step.id, "Waiting with no payload; falling back to the legacy short pause.")
            await asyncio.sleep(self.parser.config.step_delay_seconds)
            step.status = StepStatus.PASSED
            return True

        query = self.parser.normalize_verify_query(query)
        timeout = self.parser.config.wait_for_timeout_seconds
        poll_interval = self.parser.config.wait_for_poll_interval_seconds
        success, perception, reasoning = await self._poll_for_intent_match(
            session, step, query, timeout, poll_interval, "wait_for", require_element_match=True
        )
        if success:
            step.status = StepStatus.PASSED
            return True

        step.status = StepStatus.FAILED
        step.failure_type = FailureType.TIMEOUT
        timeout_reason = reasoning or f"Wait-for target not matched within {timeout:.1f}s."
        step.failure_reason = self._failure_details("WAIT", query, perception, timeout_reason)
        return False

    async def _handle_scroll_action(self, session: TestSession, step: TestStep, payload: str) -> bool:
        """Handles explicit scroll commands, including target-seeking scrolls."""
        query = payload.strip()
        normalized_query = self.parser.normalize_verify_query(query)

        # should be the usual case
        if self._is_scroll_target_query(query):
            return await self._scroll_to_target(session, step, query)

        viewport = await self._get_scroll_metrics()
        motion = self._classify_scroll_motion(normalized_query)

        # scroll to top or botton intents
        if motion == "absolute_top":
            await self._scroll_to_position(0)
        elif motion == "absolute_bottom":
            await self._scroll_to_position(int(viewport["max_scroll_top"]))
        else:
            # failed to parse intent
            step.status = StepStatus.FAILED
            step.failure_type = FailureType.ACTION_ERROR
            reason = "Failed to identify scroll intent."
            step.failure_reason = self._failure_details("SCROLL", query, {}, reason)
            return False

        step.status = StepStatus.PASSED
        return True

    def _parse_wait_duration(self, payload: str) -> Optional[float]:
        """Parses a duration payload into seconds when the wait is time-based."""
        wait_time = 0.5
        msg = payload.lower()

        match = re.search(r"([\d\.]+)\s*(s|sec|seconds?|ms|m|mins?|minutes?)", msg)
        if not match:
            return None

        val = float(match.group(1))
        unit = match.group(2)
        if unit.startswith("m") and unit != "ms":
            wait_time = val * 60
        elif unit == "ms":
            wait_time = val / 1000.0
        else:
            wait_time = val
        return wait_time

    def _is_scroll_target_query(self, payload: str) -> bool:
        """Returns whether a scroll payload looks like a target-seeking command."""
        query = self.parser.normalize_verify_query(payload)
        if not query:
            return False

        if self._classify_scroll_motion(query):
            return False

        return self.parser.has_specific_target_subject(query)

    def _classify_scroll_motion(self, payload: str) -> Optional[str]:
        """Semantically classifies non-target scroll intent into absolute or relative motion."""
        query = self.parser.normalize_verify_query(payload)
        if not query:
            return None

        intent = self.parser.parse_verify_intent(query)
        subject = intent.query_text.strip().lower()
        generic_scope_values = {"", "page", "screen", "view", "viewport", "document"}

        if intent.position == "top" and subject in generic_scope_values:
            return "absolute_top"
        if intent.position == "bottom" and subject in generic_scope_values:
            return "absolute_bottom"

        return None

    async def _get_scroll_metrics(self) -> Dict[str, float]:
        """Fetches the current document scroll metrics."""
        if not self.page:
            return {"scroll_top": 0.0, "max_scroll_top": 0.0, "viewport_height": 720.0}

        metrics = self.page.evaluate(
            """() => {
                const root = document.scrollingElement || document.documentElement;
                const viewportHeight = window.innerHeight || root.clientHeight || 720;
                const scrollTop = root.scrollTop || window.scrollY || 0;
                const maxScrollTop = Math.max(0, root.scrollHeight - root.clientHeight);
                return { scrollTop, maxScrollTop, viewportHeight };
            }"""
        )
        if inspect.isawaitable(metrics):
            metrics = await metrics
        if not isinstance(metrics, dict):
            metrics = {}
        return {
            "scroll_top": float(metrics.get("scrollTop", 0.0)),
            "max_scroll_top": float(metrics.get("maxScrollTop", 0.0)),
            "viewport_height": float(metrics.get("viewportHeight", 720.0)),
        }

    async def _get_scroll_top(self) -> float:
        """Reads the current vertical scroll offset."""
        if not self.page:
            return 0.0

        value = self.page.evaluate(
            "() => window.scrollY || document.documentElement.scrollTop || document.body.scrollTop || 0"
        )
        if inspect.isawaitable(value):
            value = await value
        try:
            return float(value or 0.0)
        except (TypeError, ValueError):
            return 0.0

    async def _build_perception_scope(self, session: TestSession) -> Optional[str]:
        """Builds a narrow backend-session scope for a stable page position within one testcase."""
        if not self.page:
            return None

        page_url = getattr(self.page, "url", None)
        if not isinstance(page_url, str) or not page_url:
            page_url = session.url
        scroll_top = int(round(await self._get_scroll_top()))
        return f"{session.id}|{page_url}|y={scroll_top}"

    async def _scroll_by_pixels(self, delta: int) -> bool:
        """Scrolls the page by a pixel delta."""
        if not self.page:
            return False
        before = await self._get_scroll_top()
        await self.page.evaluate("(amount) => window.scrollBy(0, amount)", delta)
        after = await self._get_scroll_top()
        return int(after) != int(before)

    async def _scroll_to_position(self, top: int) -> bool:
        """Scrolls the page to an absolute vertical position."""
        if not self.page:
            return False
        before = await self._get_scroll_top()
        await self.page.evaluate("(pos) => window.scrollTo(0, pos)", top)
        after = await self._get_scroll_top()
        return int(after) != int(before)

    # pylint: disable=too-many-locals, too-many-return-statements, too-many-branches, too-many-statements
    async def _scroll_to_target(self, session: TestSession, step: TestStep, payload: str) -> bool:
        """Scrolls until a target enters the center band or the page range is exhausted."""
        query = self.parser.normalize_verify_query(payload)
        if not query:
            step.status = StepStatus.FAILED
            step.failure_type = FailureType.PERCEPTION_MISMATCH
            step.failure_reason = "Scroll target was empty."
            return False

        test_slug = _test_slug(session)
        band_min = self.parser.config.scroll_center_band_min
        band_max = self.parser.config.scroll_center_band_max
        screenshot_name = f"{step.id}_scroll"
        scroll_state = await self._get_scroll_metrics()
        viewport_height = int(scroll_state["viewport_height"])
        step_size = max(1, int(viewport_height * 0.5))
        max_scroll_top = int(scroll_state["max_scroll_top"])
        max_iterations = max(8, int(max_scroll_top / max(1, step_size)) * 2 + 8)
        phase = 1
        reasoning = ""
        perception: Dict[str, Any] = {}

        for _ in range(max_iterations):
            path, persistent = self._artifact_path(test_slug, screenshot_name)
            await self.page.screenshot(path=path, type="jpeg")
            if persistent:
                step.screenshot_after = path

            try:
                perception_scope = await self._build_perception_scope(session)
                perception = await self.client.perceive(path, query=query, session_scope=perception_scope)
                step.perception_result = perception
            except Exception:  # pylint: disable=broad-exception-caught
                perception = {}
            finally:
                if not persistent:
                    self._cleanup_temporary_artifact(path)

            self._log_perception_summary(step.id, query, perception, selected=None)
            target = None
            if perception:
                intent = self.parser.parse_verify_intent(query)
                target = self._select_grounded_target(intent, perception)
            if target:
                metrics = perception.get("viewport", {}) or {}
                viewport = {
                    "width": metrics.get("width", 1280),
                    "height": metrics.get("height", viewport_height),
                }
                target_rect = _resolve_coords(target, viewport)
                target_center_y = target_rect[1] + (target_rect[3] / 2)
                band_top = viewport["height"] * band_min
                band_bottom = viewport["height"] * band_max

                session.metadata["target"] = target
                session.metadata["last_perception"] = perception

                if band_top <= target_center_y <= band_bottom:
                    step.status = StepStatus.PASSED
                    return True

                current_scroll = await self._get_scroll_metrics()
                if target_center_y < band_top:
                    if int(current_scroll["scroll_top"]) <= 0:
                        step.status = StepStatus.PASSED
                        return True
                    changed = await self._scroll_by_pixels(-step_size)
                    if not changed:
                        step.status = StepStatus.PASSED
                        return True
                    continue

                if int(current_scroll["scroll_top"]) >= max_scroll_top:
                    step.status = StepStatus.PASSED
                    return True
                changed = await self._scroll_by_pixels(step_size)
                if not changed:
                    step.status = StepStatus.PASSED
                    return True
                continue

            current_scroll = await self._get_scroll_metrics()
            current_top = int(current_scroll["scroll_top"])
            max_top = int(current_scroll["max_scroll_top"])

            if current_top < max_top:
                changed = await self._scroll_by_pixels(step_size)
                if changed:
                    continue

            if phase == 1:
                phase = 2
                if current_top != 0:
                    await self._scroll_to_position(0)
                    continue
                if max_top > 0:
                    await self._scroll_by_pixels(step_size)
                    continue

            reasoning = f"Scroll target '{query}' was not found after consuming the full page range."
            break

        step.status = StepStatus.FAILED
        step.failure_type = FailureType.PERCEPTION_MISMATCH
        step.failure_reason = self._failure_details(
            "SCROLL", query, perception, reasoning or "Unable to locate target."
        )
        return False

    def _select_grounded_target(self, intent: Intent, perception: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Selects a real target, preferring backend-ranked matches when they truly match."""
        top_matches = perception.get("top_matches") or []
        filtered_top_matches = self.parser.filter_target_candidates(intent, top_matches)
        if filtered_top_matches:
            return filtered_top_matches[0]

        elements = perception.get("elements") or []
        filtered_elements = self.parser.filter_target_candidates(intent, elements)
        if filtered_elements:
            return filtered_elements[0]

        return None

    def _resolve_payload(self, payload: str, session: TestSession) -> str:
        """Resolves placeholders in the payload string."""
        if "{" in payload and "}" in payload:
            for art_name, art_data in session.artifacts.items():
                placeholder = f"{{{art_name}}}"
                if placeholder in payload:
                    payload = payload.replace(placeholder, str(art_data["value"]))
        return payload

    async def _handle_artifact_action(
        self, _session: TestSession, step: TestStep, action: str, target: Dict[str, Any]
    ) -> bool:
        """Handles actions where the target is an artifact."""
        if action == "drag":
            _session.metadata["drag_source"] = target
            step.status = StepStatus.PASSED
            return True
        step.status = StepStatus.FAILED
        step.failure_reason = f"Action '{action}' is not supported directly on an artifact. Did you mean to 'drag' it?"
        return False

    # pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
    async def _capture_action_screenshot(
        self,
        session: TestSession,
        step: TestStep,
        action: str,
        rect: Tuple[float, float, float, float],
        viewport: Dict[str, Any],
    ):
        """Captures a screenshot at the point of action for debugging.

        :param session: current TestSession.
        :param step: active TestStep.
        :param action: keyword for the action being performed.
        :param rect: [x, y, width, height] of the target.
        :param viewport: viewport dimensions dictionary.
        """
        action_path, persistent = self._artifact_path(_test_slug(session), f"{step.id}_{action}")
        if not persistent:
            return
        vw, vh = viewport.get("width", 1280), viewport.get("height", 720)

        # Only clip if we have a valid bounding box
        if rect[2] > 0 and rect[3] > 0:
            pad_x, pad_y = rect[2] * 0.15, rect[3] * 0.15
            cx, cy = max(0, rect[0] - pad_x), max(0, rect[1] - pad_y)
            clip = {
                "x": cx,
                "y": cy,
                "width": min(vw - cx, rect[2] + 2 * pad_x),
                "height": min(vh - cy, rect[3] + 2 * pad_y),
            }
            try:
                await self.page.screenshot(path=action_path, type="jpeg", clip=clip)
                step.action_screenshot = action_path
            except Exception as e:  # pylint: disable=broad-exception-caught
                # Fallback to full page screenshot if clipping fails
                self.logger.log_warning(session.id, f"Failed to capture action screenshot (clipped): {e}")
                await self.page.screenshot(path=action_path, type="jpeg")
                step.action_screenshot = action_path
        else:
            # Full page screenshot fallback relative to current viewport
            await self.page.screenshot(path=action_path, type="jpeg")
            step.action_screenshot = action_path

    async def _execute_artifact_drop(
        self, session: TestSession, step: TestStep, source: Dict[str, Any], target: Dict[str, Any]
    ) -> bool:
        """Handles dropping an artifact (e.g. a file) onto a UI element.

        :param session: The test session.
        :param step: The step to execute.
        :param source: The source artifact.
        :param target: The target element.
        :return: True if the step was executed successfully, False otherwise.
        """
        art_data = source["value"]
        art_type = art_data.get("type")

        if art_type != "file":
            step.status = StepStatus.FAILED
            step.failure_reason = f"Drop action not yet implemented for artifact type: {art_type}"
            return False

        last_perc = session.metadata.get("last_perception", {})
        viewport = last_perc.get("viewport", {"width": 1280, "height": 720})
        px, py, pw, ph = _resolve_coords(target, viewport)
        cx, cy = px + pw / 2, py + ph / 2

        self.logger.log_debug(
            step.id, f"Artifact drop target: {target.get('text') or target.get('label')} at ({cx}, {cy})"
        )
        return await self._perform_file_upload(session, step, art_data["value"])

    async def _perform_file_upload(self, session: TestSession, step: TestStep, file_path: str) -> bool:
        """Helper to perform Playwright file upload."""
        self.logger.log_debug(step.id, f"Uploading file artifact: {file_path}")
        try:
            input_selector = "input[type=file]"
            count = await self.page.locator(input_selector).count()
            if count == 0:
                step.status = StepStatus.FAILED
                step.failure_reason = "No file input element found on the page to receive the artifact."
                return False

            await self.page.set_input_files(input_selector, file_path)
            await self.page.evaluate(
                """([sel, fname]) => {
                const el = document.querySelector(sel);
                if (el) el.dispatchEvent(new Event('change', { bubbles: true }));
            }""",
                [input_selector, file_path],
            )

            step.status = StepStatus.PASSED
            return True
        except Exception as e:  # pylint: disable=broad-exception-caught
            self.logger.log_warning(session.id, f"File upload failed: {e}")
            raise ArtifactError(f"Failed to upload file artifact for '{file_path}'", internal_detail=str(e)) from e

    def _check_verification_match(
        self, session: TestSession, intent: Intent, perception: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Checks if the current perception matches the verification intent."""
        all_elements = perception.get("elements", [])
        match_found = False
        reasoning = ""

        if intent.negated:
            match_found = self.parser.verify_negation(all_elements, intent, history_metadata=session.metadata)
            if not match_found:
                reasoning = "Negation failure: Element remains in the view"
        else:
            match_found = self._is_positive_match(intent, all_elements, perception)

        if match_found and not intent.negated:
            session.metadata["last_perception"] = perception
            self._update_historical_target(session, intent, all_elements)

        return match_found, reasoning

    def _check_wait_for_match(self, intent: Intent, perception: Dict[str, Any]) -> Tuple[bool, str]:
        """Checks whether the target element is actually present for wait-for polling."""
        target = self._select_grounded_target(intent, perception)
        if target:
            return True, "Target element became visible."
        return False, "Target element not yet visible."

    def _is_positive_match(
        self, intent: Intent, all_elements: List[Dict[str, Any]], perception: Dict[str, Any]
    ) -> bool:
        """Determines if a positive (non-negated) verification is successful."""
        filtered = self.parser.filter_elements_by_intent(intent, all_elements)
        if filtered and intent.has_targeting_clauses():
            return True
        if perception.get("elements") and not intent.has_targeting_clauses():
            return True
        return False

    def _update_historical_target(self, session: TestSession, intent: Intent, all_elements: List[Dict[str, Any]]):
        """Updates the session history with the matched target."""
        subj = intent.normalized_subject.lower().strip()
        if not subj:
            return

        norm_subj = self.parser.normalize_subject(subj)
        history = session.metadata.setdefault("history", {})
        matches = self.parser.filter_elements_by_intent(intent, all_elements)
        if matches:
            session.metadata["target"] = matches[0]
            if norm_subj not in history:
                history[norm_subj] = {"target": matches[0], "elements": all_elements}

    async def _execute_legacy(self, session: TestSession, step: TestStep) -> bool:
        """Legacy execution path for non-decomposed steps.

        :param session: The test session.
        :param step: The step to execute.
        :return: True if the step was executed successfully, False otherwise.
        """
        test_slug = _test_slug(session)
        before_path, persistent = self._artifact_path(test_slug, f"{step.id}_before")
        await self.page.screenshot(path=before_path, type="jpeg")
        if persistent:
            step.screenshot_before = before_path

        try:
            perception_scope = await self._build_perception_scope(session)
            perception = await self.client.perceive(before_path, query=step.instruction, session_scope=perception_scope)
            step.perception_result = perception
        finally:
            if not persistent:
                self._cleanup_temporary_artifact(before_path)

        self._log_perception_summary(
            step.id, step.instruction, perception, selected=self._select_perception_target(perception)
        )
        await self._execute_action(session, step)

        if step.expectation:
            step.status = await self._verify_expectation(session, step)
        else:
            step.status = StepStatus.PASSED

        return step.status == StepStatus.PASSED

    # ------------------------------------------------------------------
    # Failure reporting
    # ------------------------------------------------------------------

    def _failure_details(self, _stage: str, query: str, perception: Dict[str, Any], base_message: str) -> str:
        """Generates a concise failure reason string for CLI display.

        :param _stage: The stage of the test.
        :param query: The query to execute.
        :param perception: The perception result.
        :param base_message: The base message.
        :return: The failure reason.
        """
        reason = f"{base_message} for query: '{query}'"

        if self.verbosity >= 1:
            elements = perception.get("elements", [])
            if elements:
                visible_texts = list(
                    {
                        (el.get("text") or el.get("placeholder") or el.get("label"))
                        for el in elements
                        if (el.get("text") or el.get("placeholder") or el.get("label"))
                    }
                )[:10]
                if visible_texts:
                    reason += f"\n[Context] Elements visible on screen: {visible_texts}"
            else:
                reason += "\n[Context] No readable elements detected on the page."

        if self.verbosity >= 2:
            top_matches = perception.get("top_matches", [])
            if top_matches:
                matches_info = [
                    f"'{m.get('text', 'unnamed')}' (similarity: {m.get('similarity', 0.0):.2f})"
                    for m in top_matches[:3]
                ]
                reason += f"\n[Detail] Top candidates: {', '.join(matches_info)}"
            elif perception.get("elements"):
                reason += "\n[Detail] Semantic matching failed. No elements closely resembled the query."

        return reason

    # ------------------------------------------------------------------
    # Playwright interaction helpers
    # ------------------------------------------------------------------

    async def _execute_interaction(self, action: str, x: float, y: float, payload: str):
        """Performs actual Playwright interactions at the specified pixel coordinates.

        :param action: The action to perform.
        :param x: The x-coordinate.
        :param y: The y-coordinate.
        :param payload: The payload to use for the action.
        """
        # Strip quotes if present
        clean_payload = payload.strip("'\"")

        # Normalize action name (handle spaces/underscores/hyphens)
        norm_action = action.lower().replace(" ", "-").replace("_", "-")

        if self.verbosity >= 1:
            print(f"  [DO] {norm_action} {clean_payload!r} at ({x:.1f}, {y:.1f})")

        if norm_action == "click":
            await self.page.mouse.click(x, y)
        elif norm_action == "right-click":
            await self.page.mouse.click(x, y, button="right")
        elif any(verb in norm_action for verb in ["type", "enter", "input"]):
            # Focus and clear
            await self.page.mouse.click(x, y)
            # Use Meta on macOS, Control elsewhere
            modifier = "Meta" if sys.platform == "darwin" else "Control"
            await self.page.keyboard.press(f"{modifier}+A")
            await self.page.keyboard.press("Backspace")

            # Type the actual text
            await self.page.keyboard.type(clean_payload)

            # If it was "enter", maybe press Enter key?
            # In many CLI flows, "enter 'admin'" implies submitting the field,
            # but usually we follow with a "Click Submit".
            # Let's just stick to typing for now to avoid side effects.
        elif action == "hover":
            await self.page.mouse.move(x, y)
        elif action == "drag":
            await self.page.mouse.move(x, y)
            await self.page.mouse.down()
        elif action == "drop":
            await self.page.mouse.move(x, y)
            await self.page.mouse.up()
        elif action == "scroll":
            # Move mouse to target first to ensure it's grounded on the right element
            await self.page.mouse.move(x, y)
            # Basic scroll - we could use mouse.wheel if payload specified delta
            await self.page.evaluate(f"window.scrollTo({{top: {y}, behavior: 'smooth'}})")

        await asyncio.sleep(self.parser.config.step_delay_seconds)

    async def _skip_step_recursive(self, step: TestStep, on_step_update: Optional[Any]):
        """Recursively marks a step and its sub-steps as skipped.

        :param step: The step to skip.
        :param on_step_update: A callback to update the step status.
        """
        step.status = StepStatus.SKIPPED
        if on_step_update:
            await on_step_update(step)
        for sub_step in step.sub_steps:
            await self._skip_step_recursive(sub_step, on_step_update)

    # pylint: disable=too-many-locals, broad-exception-caught
    async def _execute_action(self, session: TestSession, step: TestStep):
        """Legacy: simulates finding an element and interacting with it.

        :param session: The test session.
        :param step: The step to execute.
        :return: True if the step was executed successfully, False otherwise.
        """
        action = self._parse_action(step.instruction)
        test_slug = _test_slug(session)

        clip = None
        if step.perception_result:
            result = step.perception_result
            viewport = result.get("viewport", {"width": 1280, "height": 720})
            vw, vh = viewport.get("width", 1280), viewport.get("height", 720)

            if result.get("top_matches"):
                match = result["top_matches"][0]
                bounds = match.get("bounds")
                if bounds:
                    x1, y1, x2, y2 = bounds
                    clip = {"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1}
            elif result.get("elements"):
                el = result["elements"][0]
                loc = el.get("location")
                if loc and len(loc) == 4:
                    nx, ny, nw, nh = loc
                    clip = {"x": nx * vw, "y": ny * vh, "width": nw * vw, "height": nh * vh}

        if clip:
            action_path, persistent = self._artifact_path(test_slug, f"{step.id}_{action}")
            if not persistent:
                await asyncio.sleep(self.parser.config.step_delay_seconds)
                return
            try:
                await self.page.screenshot(path=action_path, type="jpeg", clip=clip)
                step.action_screenshot = action_path
            except Exception as e:
                self.logger.log_warning(session.id, f"Failed to capture action screenshot (legacy): {e}")

        await asyncio.sleep(self.parser.config.step_delay_seconds)

    def _parse_action(self, instruction: str) -> str:
        """Heuristic to determine the action type from an instruction string.

        :param instruction: The instruction to parse.
        :return: The action type.
        """
        instr = instruction.lower()
        if "click" in instr or "tap" in instr:
            return "click"
        if "hover" in instr or "move" in instr:
            return "hover"
        if "type" in instr or "enter" in instr or "input" in instr:
            return "type"
        if "select" in instr:
            return "select"
        return "interact"

    # pylint: disable=broad-exception-caught
    async def _verify_expectation(
        self, session: TestSession, step: TestStep, timeout: Optional[int] = None
    ) -> StepStatus:
        """Polls the Perception API until the expectation is met or timed out.

        :param session: The test session.
        :param step: The step to execute.
        :param timeout: The timeout to wait for the verification.
        :return: The status of the step.
        """
        timeout = timeout or self.parser.config.verification_timeout
        start_wait = datetime.now()
        test_slug = _test_slug(session)

        while (datetime.now() - start_wait).total_seconds() < timeout:
            path, persistent = self._artifact_path(test_slug, f"{step.id}_after")
            await self.page.screenshot(path=path, type="jpeg")

            try:
                perception_scope = await self._build_perception_scope(session)
                result = await self.client.perceive(path, query=step.expectation, session_scope=perception_scope)
                self._log_perception_summary(step.id, step.expectation, result, selected=None)

                match_found = False
                q = step.expectation.lower()
                for el in result.get("elements", []):
                    text = (el.get("placeholder") or el.get("text", "")).lower()
                    label = el.get("label", "").lower()
                    name = el.get("name", "").lower()
                    if (q in text or q in label or q in name) or any(
                        word in text or word in label or word in name for word in q.split() if len(word) > 3
                    ):
                        match_found = True
                        break

                if match_found:
                    if persistent:
                        step.screenshot_after = path
                    return StepStatus.PASSED
            except Exception:
                pass
            finally:
                if not persistent:
                    self._cleanup_temporary_artifact(path)

            await asyncio.sleep(1)

        step.failure_type = FailureType.TIMEOUT
        reason = f"Expectation '{step.expectation}' not met visually within {timeout}s."
        if self.verbosity >= 1:
            reason += " (Timed out while polling Perception API)"
        step.failure_reason = reason
        return StepStatus.FAILED

    def _log_perception_summary(
        self,
        step_id: str,
        query: str,
        perception: Dict[str, Any],
        selected: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Logs compact perception details only at the highest verbosity."""
        should_log = self.debug_logging if self.debug_logging is not None else self.verbosity >= 2
        if should_log:
            self.logger.log_perception(step_id, query, perception, selected=selected)

    @staticmethod
    def _select_perception_target(perception: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Returns the runtime-selected candidate from a perception result."""
        if perception.get("top_matches"):
            return perception["top_matches"][0]
        if perception.get("elements"):
            return perception["elements"][0]
        return None

    def _artifact_path(self, test_slug: str, name: str) -> Tuple[str, bool]:
        """Returns a screenshot path and whether it should be kept as an artifact."""
        if self.artifact_dir:
            return str(Path(self.artifact_dir) / f"{test_slug}_{name}.jpg"), True

        temp_file = tempfile.NamedTemporaryFile(  # pylint: disable=consider-using-with
            prefix=f"vizqa_{test_slug}_{name}_", suffix=".jpg", delete=False
        )
        temp_file.close()
        return temp_file.name, False

    def _cleanup_temporary_artifact(self, path: str) -> None:
        """Removes a non-persistent screenshot used only for perception."""
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


# ------------------------------------------------------------------
# Module-level pure helpers
# ------------------------------------------------------------------


def _test_slug(session: TestSession) -> str:
    """Returns a filesystem-safe slug for the test session."""
    base = session.file_stem if session.file_stem else session.test_name
    base_slug = base.replace(" ", "_").replace(":", "_").lower()
    if session.viewport_slug:
        return f"{session.viewport_slug}__{base_slug}"
    return base_slug


def _resolve_coords(target: Dict[str, Any], viewport: Dict[str, Any]) -> Tuple[float, float, float, float]:
    """
    Extracts pixel coordinates from a perception target dict.

    Returns ``(x, y, width, height)`` in pixels.  Returns ``(0, 0, 0, 0)``
    when no usable coordinate data is found.

    :param target: The target to extract coordinates from.
    :param viewport: The viewport to use for coordinate calculations.
    :return: The coordinates of the target.
    """
    vw = viewport.get("width", 1280)
    vh = viewport.get("height", 720)

    if "bounds" in target:
        x1, y1, x2, y2 = target["bounds"]
        return float(x1), float(y1), float(x2 - x1), float(y2 - y1)

    if "location" in target:
        loc = target["location"]
        # Normalised [y, x, w, h] convention from the perception API
        return float(loc[1] * vw), float(loc[0] * vh), float(loc[2] * vw), float(loc[3] * vh)

    return 0.0, 0.0, 0.0, 0.0
