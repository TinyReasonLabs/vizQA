"""
Core execution engine for vision-driven UI automation.
"""

# pylint: disable=invalid-name, too-many-lines
import asyncio
import os
import re
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from playwright.async_api import Browser, Page, async_playwright

from vizQA.client import PerceptionClient
from vizQA.exceptions import (
    ActionExecutionError,
    ArtifactError,
    ElementNotFoundError,
    UserFacingException,
    VerificationError,
)
from vizQA.logger import get_logger
from vizQA.memory import FailureType, StepStatus, TestSession, TestStep
from vizQA.minilm import MiniLM
from vizQA.parser import SemanticParser


class Automator:
    """
    Main controller for browser automation and perception-integrated execution.
    """

    def __init__(self, perception_client: PerceptionClient, verbosity: int = 0, headless: bool = True):
        self.client = perception_client
        self.verbosity = verbosity
        self.headless = headless
        self.playwright_mgr: Optional[Any] = None
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None
        self._logger = get_logger()

        model_dir = os.path.join(os.path.dirname(__file__), "weights", "minilm")
        try:
            self.minilm: Optional[MiniLM] = MiniLM(model_dir)
        except (FileNotFoundError, RuntimeError):
            self.minilm = None
            self._logger.log_warning("init", "MiniLM model not found — semantic matching degraded to substring.")

        # Single shared parser instance wired to the model
        use_adv = os.environ.get("VIZQA_ADVANCED_RANKING", "1") == "1"
        intent_threq = float(os.environ.get("VIZQA_INTENT_THRESHOLD", "0.6"))
        action_threq = float(os.environ.get("VIZQA_ACTION_THRESHOLD", "0.52"))
        semantic_threq = float(os.environ.get("VIZQA_SEMANTIC_THRESHOLD", "0.70"))

        self.parser = SemanticParser(
            minilm=self.minilm,
            use_advanced_ranking=use_adv,
            intent_threshold=intent_threq,
            action_threshold=action_threq,
            semantic_match_threshold=semantic_threq,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self):
        """Initialises the Playwright browser and page."""
        self.playwright_mgr = await async_playwright().start()
        self.browser = await self.playwright_mgr.chromium.launch(headless=self.headless)
        self.page = await self.browser.new_page()
        os.makedirs(".vizQA", exist_ok=True)

    async def stop(self):
        """Closes the browser and stops the Playwright manager."""
        if self.page:
            await self.page.close()
        if self.browser:
            await self.browser.close()
        if self.playwright_mgr:
            await self.playwright_mgr.stop()

    # ------------------------------------------------------------------
    # Session runner
    # ------------------------------------------------------------------

    async def run_session(self, session: TestSession, on_step_update: Optional[Any] = None) -> bool:
        """Main execution loop for a test session."""
        if not self.page:
            await self.start()

        self._logger.log_session(session.id, "start", f"url={session.url!r}")
        await self.page.goto(session.url)

        failed = False
        for step in session.steps:
            if failed:
                await self._skip_step_recursive(step, on_step_update)
                continue

            step_passed = await self._run_step_recursive(session, step, on_step_update)
            if not step_passed:
                failed = True

        session.end_time = datetime.now()
        self._logger.log_session(session.id, "end", f"duration={session.duration:.1f}s")
        return not failed

    # ------------------------------------------------------------------
    # Step dispatcher
    # ------------------------------------------------------------------

    # pylint: disable=too-many-branches, broad-exception-caught
    async def _run_step_recursive(self, session: TestSession, step: TestStep, on_step_update: Optional[Any]) -> bool:
        """Executes a step and its sub-steps recursively."""
        step.status = StepStatus.RUNNING
        if on_step_update:
            await on_step_update(step)

        step.start_time = datetime.now()
        success = True

        try:
            if not step.sub_steps:
                instr = step.instruction

                if instr.startswith("FIND:"):
                    query = instr.replace("FIND:", "").strip()
                    success = await self._execute_find(session, step, query)

                elif instr.startswith("DO:"):
                    action_cmd = instr.replace("DO:", "").strip()
                    success = await self._execute_do(session, step, action_cmd)

                elif instr.startswith("VERIFY:"):
                    query = instr.replace("VERIFY:", "").strip()
                    success = await self._execute_verify(session, step, query)

                else:
                    # Legacy path for non-decomposed steps
                    success = await self._execute_legacy(session, step)

                if step.status == StepStatus.FAILED:
                    success = False

            else:
                # Container step — run sub-steps recursively
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
                    success = False
                else:
                    step.status = StepStatus.PASSED

        except UserFacingException as exc:
            # Report specific user-facing error message
            self._logger.log_debug(session.id, f"User-facing error: {exc.user_message}")
            if exc.internal_detail:
                self._logger.log_debug(session.id, f"Detail: {exc.internal_detail}")

            step.status = StepStatus.FAILED
            # Try to map exception type to FailureType if possible
            if isinstance(exc, ElementNotFoundError):
                step.failure_type = FailureType.PERCEPTION_MISMATCH
            elif isinstance(exc, ActionExecutionError):
                step.failure_type = FailureType.ACTION_ERROR
            elif isinstance(exc, VerificationError):
                step.failure_type = FailureType.TIMEOUT
            else:
                step.failure_type = FailureType.SYSTEM_ERROR

            step.failure_reason = exc.user_message
            success = False
        except Exception as exc:
            self._logger.log_exception(step.id, exc)
            step.status = StepStatus.FAILED
            step.failure_type = FailureType.SYSTEM_ERROR
            step.failure_reason = "An unexpected error occurred during step execution."
            step.error = str(exc)
            success = False
        finally:
            step.end_time = datetime.now()
            self._logger.log_step(
                step.id, instr.split(":")[0] if ":" in step.instruction else "STEP", step.status, step.failure_reason
            )
            if on_step_update:
                await on_step_update(step)

        return success

    # ------------------------------------------------------------------
    # Atomic step executors
    # ------------------------------------------------------------------

    async def _execute_find(self, session: TestSession, step: TestStep, query: str) -> bool:
        """Handles a FIND: sub-step — perceives the page and stores the target."""
        test_slug = _test_slug(session)
        path = f".vizQA/{test_slug}_{step.id}_before.jpg"
        await self.page.screenshot(path=path, type="jpeg")
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
            else:
                self._logger.log_warning(step.id, f"Artifact '{art_name}' not found in session.")
                # TODO error reporting
                raise ValueError(f"Artifact '{art_name}' not found in session.")

        perception = await self.client.perceive(path, query=query)
        step.perception_result = perception
        self._logger.log_perception(step.id, query, perception)

        target = None
        if perception.get("top_matches"):
            target = perception["top_matches"][0]
        elif perception.get("elements"):
            target = perception["elements"][0]

        if target:
            session.metadata["target"] = target
            session.metadata["last_perception"] = perception

            # Decoupled history update: just store raw data under normalized subject
            norm_subj = self.parser.normalize_subject(query)
            history = session.metadata.setdefault("history", {})
            if norm_subj not in history:
                history[norm_subj] = {"target": target, "elements": perception.get("elements", [])}
                self._logger.log_debug(step.id, f"Stored FIRST appearance for '{norm_subj}'")

            step.status = StepStatus.PASSED
            return True

        step.status = StepStatus.FAILED
        step.failure_type = FailureType.PERCEPTION_MISMATCH
        step.failure_reason = self._failure_details("FIND", query, perception, "Element not found")
        return False

    async def _execute_do(self, session: TestSession, step: TestStep, action_cmd: str) -> bool:
        """Handles a DO: sub-step — resolves coords and fires the interaction."""
        parts = action_cmd.split(" ", 1)
        action = parts[0].lower()
        payload = parts[1] if len(parts) > 1 else ""

        # Resolve placeholders in payload
        if "{" in payload and "}" in payload:
            for art_name, art_data in session.artifacts.items():
                placeholder = f"{{{art_name}}}"
                if placeholder in payload:
                    payload = payload.replace(placeholder, str(art_data["value"]))

        target = session.metadata.get("target")
        if not target:
            step.status = StepStatus.FAILED
            step.failure_reason = "No target element found. FIND step must precede DO step."
            return False

        # Handle specialized artifact actions
        if target.get("type") == "artifact":
            if action == "drag":
                session.metadata["drag_source"] = target
                step.status = StepStatus.PASSED
                return True
            else:
                step.status = StepStatus.FAILED
                step.failure_reason = (
                    f"Action '{action}' is not supported directly on an artifact. Did you mean to 'drag' it?"
                )
                return False

        # Specific handling for drop if the source was an artifact
        drag_source = session.metadata.get("drag_source")
        if action == "drop" and drag_source and drag_source.get("type") == "artifact":
            success = await self._execute_artifact_drop(session, step, drag_source, target)
            if success:
                session.metadata.pop("drag_source", None)
            return success

        last_perc = session.metadata.get("last_perception", {})
        viewport = last_perc.get("viewport", {"width": 1280, "height": 720})
        px, py, pw, ph = _resolve_coords(target, viewport)

        # Capture adaptive-crop action snapshot
        test_slug = _test_slug(session)
        action_path = f".vizQA/{test_slug}_{step.id}_{action}.jpg"
        vw = viewport.get("width", 1280)
        vh = viewport.get("height", 720)

        if pw > 0 and ph > 0:
            px_pad = pw * 0.15
            py_pad = ph * 0.15
            cx = max(0, px - px_pad)
            cy = max(0, py - py_pad)
            cw = min(vw - cx, pw + 2 * px_pad)
            ch = min(vh - cy, ph + 2 * py_pad)
            try:
                await self.page.screenshot(
                    path=action_path, type="jpeg", clip={"x": cx, "y": cy, "width": cw, "height": ch}
                )
                step.action_screenshot = action_path
            except Exception as e:  # pylint: disable=broad-exception-caught
                # Fallback to full page screenshot if clipping fails
                self._logger.log_warning(session.id, f"Failed to capture action screenshot (clipped): {e}")
                await self.page.screenshot(path=action_path, type="jpeg")
                step.action_screenshot = action_path
        else:
            # No target, but we still want a screenshot of where we thought we'd act
            await self.page.screenshot(path=action_path, type="jpeg")
            step.action_screenshot = action_path

        await self._execute_interaction(action, px + pw / 2, py + ph / 2, payload)
        step.status = StepStatus.PASSED
        return True

    async def _execute_artifact_drop(
        self, session: TestSession, step: TestStep, source: Dict[str, Any], target: Dict[str, Any]
    ) -> bool:
        """Handles dropping an artifact (e.g. a file) onto a UI element."""
        art_data = source["value"]
        art_type = art_data.get("type")

        last_perc = session.metadata.get("last_perception", {})
        viewport = last_perc.get("viewport", {"width": 1280, "height": 720})
        px, py, pw, ph = _resolve_coords(target, viewport)
        cx, cy = px + pw / 2, py + ph / 2

        self._logger.log_debug(
            step.id, f"Artifact drop target: {target.get('text') or target.get('label')} at ({cx}, {cy})"
        )

        if art_type == "file":
            file_path = art_data["value"]
            self._logger.log_debug(step.id, f"Uploading file artifact: {file_path}")

            try:
                input_selector = "input[type=file]"

                # Check if there are any file inputs
                count = await self.page.locator(input_selector).count()
                self._logger.log_debug(step.id, f"Found {count} file inputs on page")

                if count == 0:
                    step.status = StepStatus.FAILED
                    step.failure_reason = "No file input element found on the page to receive the artifact."
                    return False

                # Use Playwright's set_input_files
                await self.page.set_input_files(input_selector, file_path)
                self._logger.log_debug(step.id, f"set_input_files called for {file_path}")

                # Trigger a change event if needed
                await self.page.evaluate(
                    """([sel, fname]) => {
                    const el = document.querySelector(sel);
                    if (el) {
                        console.log("Triggering change for", fname);
                        el.dispatchEvent(new Event('change', { bubbles: true }));
                    }
                }""",
                    [input_selector, file_path],
                )

                step.status = StepStatus.PASSED
                return True
            except Exception as e:
                self._logger.log_warning(session.id, f"File upload failed: {e}")
                raise ArtifactError(f"Failed to upload file artifact for '{art_name}'", internal_detail=str(e))

        # Handle other artifact types (content, string) as drops?
        # Maybe just type them?

        step.status = StepStatus.FAILED
        step.failure_reason = f"Drop action not yet implemented for artifact type: {art_type}"
        return False

    async def _execute_verify(self, session: TestSession, step: TestStep, query: str, timeout: int = 0.2) -> bool:
        """Handles a VERIFY: sub-step — semantically evaluates the UI state with polling."""
        start_wait = datetime.now()
        test_slug = _test_slug(session)

        # Parse intent once at the beginning
        intent = self.parser.parse_verify_intent(query)
        self._logger.log_debug(step.id, f"verify intent={intent}")
        perception_query = f"'{intent['keyword']}' {intent['subject'] or ''} {intent['position'] or ''}"
        self._logger.log_debug(step.id, f"perception_query={perception_query}")

        while (datetime.now() - start_wait).total_seconds() < timeout:
            path = f".vizQA/{test_slug}_{step.id}_verify.jpg"
            await self.page.screenshot(path=path, type="jpeg")
            step.screenshot_after = path

            perception = await self.client.perceive(path, query=perception_query)
            step.perception_result = perception
            self._logger.log_debug(step.id, f"perception_result={perception}")
            all_elements = perception.get("elements", [])

            match_found = False
            reasoning = ""
            if intent.get("negated"):
                # Delegated negation path — uses history_metadata for match
                match_found = self.parser.verify_negation(all_elements, intent, history_metadata=session.metadata)
                if not match_found:  # if the negation is not met
                    reasoning = "Negation failure: Element remains in the view"
            else:
                filtered = self.parser.filter_elements_by_intent(intent, all_elements)
                self._logger.log_debug(step.id, f"filtered by intent={filtered}")

                # Success criteria:
                # 1. Intent markers (keyword/color/position) matched in filtered list
                # 2. Or high-confidence top_match from the API
                if filtered and (intent.get("keyword") or intent.get("color") or intent.get("position")):
                    match_found = True
                    self._logger.log_debug(step.id, "match found from keyword/color/position")
                elif (
                    perception.get("elements")
                    and not intent.get("keyword")
                    and not intent.get("color")
                    and not intent.get("position")
                ):
                    match_found = True
                    self._logger.log_debug(step.id, "match found from elements")
                elif not intent.get("keyword") and not intent.get("color") and not intent.get("position"):
                    # Only match if we have at least SOME high-confidence elements detected
                    if all_elements:
                        match_found = True
                        self._logger.log_debug(step.id, "found generic match from elements")

            if match_found:
                # Update perception history for subsequent steps/negations
                self._logger.log_debug(step.id, f"match_found={match_found}")
                if not intent.get("negated"):
                    session.metadata["last_perception"] = perception

                    # Update history if not already set (maintains "first" appearance)
                    subj = (intent.get("subject") or intent.get("keyword") or query).lower().strip()
                    norm_subj = self.parser.normalize_subject(subj)
                    history = session.metadata.setdefault("history", {})

                    # If we matched something specific, update target
                    matches = self.parser.filter_elements_by_intent(intent, all_elements)
                    if matches:
                        session.metadata["target"] = matches[0]
                        if norm_subj not in history:
                            history[norm_subj] = {"target": matches[0], "elements": all_elements}
                            self._logger.log_debug(step.id, f"identified target for {subj}: {matches[0]}")
                else:
                    reasoning = "Negation failure: Element remains in the view"

                step.status = StepStatus.PASSED
                return True
            await asyncio.sleep(1.0)

        step.status = StepStatus.FAILED
        wait_time = (datetime.now() - start_wait).total_seconds()
        if not reasoning:
            reasoning = f"Verification failed after {wait_time:.1f}s"
        step.failure_reason = self._failure_details("VERIFY", query, perception, reasoning)
        return False

    async def _execute_legacy(self, session: TestSession, step: TestStep) -> bool:
        """Legacy execution path for non-decomposed steps."""
        test_slug = _test_slug(session)
        before_path = f".vizQA/{test_slug}_{step.id}_before.jpg"
        await self.page.screenshot(path=before_path, type="jpeg")
        step.screenshot_before = before_path

        perception = await self.client.perceive(before_path, query=step.instruction)
        step.perception_result = perception
        self._logger.log_perception(step.id, step.instruction, perception)

        await self._execute_action(session, step)

        if step.expectation:
            step.status = await self._verify_expectation(session, step)
        else:
            step.status = StepStatus.PASSED

        return step.status == StepStatus.PASSED

    # ------------------------------------------------------------------
    # Failure reporting
    # ------------------------------------------------------------------

    def _failure_details(self, stage: str, query: str, perception: Dict[str, Any], base_message: str) -> str:
        """Generates a concise failure reason string for CLI display."""
        reason = f"{base_message} for query: '{query}'"

        if self.verbosity >= 1:
            elements = perception.get("elements", [])
            if elements:
                visible_texts = list(
                    {
                        (el.get("placeholder") or el.get("label"))
                        for el in elements
                        if (el.get("placeholder") or el.get("label"))
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
        """Performs actual Playwright interactions at the specified pixel coordinates."""
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
            await self.page.keyboard.down("Control")
            await self.page.keyboard.press("a")
            await self.page.keyboard.up("Control")
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

        await asyncio.sleep(0.5)

    async def _skip_step_recursive(self, step: TestStep, on_step_update: Optional[Any]):
        """Recursively marks a step and its sub-steps as skipped."""
        step.status = StepStatus.SKIPPED
        if on_step_update:
            await on_step_update(step)
        for sub_step in step.sub_steps:
            await self._skip_step_recursive(sub_step, on_step_update)

    # pylint: disable=too-many-locals, broad-exception-caught
    async def _execute_action(self, session: TestSession, step: TestStep):
        """Legacy: simulates finding an element and interacting with it."""
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
            action_path = f".vizQA/{test_slug}_{step.id}_{action}.jpg"
            try:
                await self.page.screenshot(path=action_path, type="jpeg", clip=clip)
                step.action_screenshot = action_path
            except Exception as e:
                self._logger.log_warning(session.id, f"Failed to capture action screenshot (legacy): {e}")

        await asyncio.sleep(0.5)

    def _parse_action(self, instruction: str) -> str:
        """Heuristic to determine the action type from an instruction string."""
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
    async def _verify_expectation(self, session: TestSession, step: TestStep, timeout: int = 10) -> StepStatus:
        """Polls the Perception API until the expectation is met or timed out."""
        start_wait = datetime.now()
        test_slug = _test_slug(session)

        while (datetime.now() - start_wait).total_seconds() < timeout:
            path = f".vizQA/{test_slug}_{step.id}_after.jpg"
            await self.page.screenshot(path=path, type="jpeg")

            try:
                result = await self.client.perceive(path, query=step.expectation)
                self._logger.log_perception(step.id, step.expectation, result)

                match_found = False
                if result.get("top_matches") or result.get("salience", 0) > 0.7:
                    match_found = True
                else:
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
                    step.screenshot_after = path
                    return StepStatus.PASSED
            except Exception:
                pass

            await asyncio.sleep(1)

        step.failure_type = FailureType.TIMEOUT
        reason = f"Expectation '{step.expectation}' not met visually within {timeout}s."
        if self.verbosity >= 1:
            reason += " (Timed out while polling Perception API)"
        step.failure_reason = reason
        return StepStatus.FAILED


# ------------------------------------------------------------------
# Module-level pure helpers
# ------------------------------------------------------------------


def _test_slug(session: TestSession) -> str:
    """Returns a filesystem-safe slug for the test session."""
    base = session.file_stem if session.file_stem else session.test_name
    return base.replace(" ", "_").replace(":", "_").lower()


def _resolve_coords(target: Dict[str, Any], viewport: Dict[str, Any]) -> Tuple[float, float, float, float]:
    """
    Extracts pixel coordinates from a perception target dict.

    Returns ``(x, y, width, height)`` in pixels.  Returns ``(0, 0, 0, 0)``
    when no usable coordinate data is found.
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
