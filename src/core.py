"""
Core execution engine for vision-driven UI automation.
"""

# pylint: disable=invalid-name, too-many-lines
import asyncio
import os
from datetime import datetime
from typing import Any, Dict, Optional

from playwright.async_api import Browser, Page, async_playwright

from .client import PerceptionClient
from .memory import FailureType, StepStatus, TestSession, TestStep
from .planner import StepPlanner


class Automator:
    """
    Main controller for browser automation and perception-integrated execution.
    """

    def __init__(self, perception_client: PerceptionClient, verbosity: int = 0):
        self.client = perception_client
        self.planner = StepPlanner()
        self.verbosity = verbosity
        self.playwright_mgr: Optional[Any] = None
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None

    async def start(self):
        """Initializes the Playwright browser and page."""
        self.playwright_mgr = await async_playwright().start()
        self.browser = await self.playwright_mgr.chromium.launch(headless=True)
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

    async def run_session(self, session: TestSession, on_step_update: Optional[Any] = None):
        """Main execution loop for a test session."""
        if not self.page:
            await self.start()

        await self.page.goto(session.url)

        failed = False
        for step in session.steps:
            if failed:
                await self._skip_step_recursive(step, on_step_update)
                continue

            # Execute the main step and its sub-steps
            step_passed = await self._run_step_recursive(session, step, on_step_update)
            if not step_passed:
                failed = True

        session.end_time = datetime.now()

    # pylint: disable=too-many-locals, too-many-branches, too-many-statements, too-many-nested-blocks, broad-exception-caught
    async def _run_step_recursive(self, session: TestSession, step: TestStep, on_step_update: Optional[Any]) -> bool:
        """Executes a step and its sub-steps recursively."""
        step.status = StepStatus.RUNNING
        if on_step_update:
            await on_step_update(step)

        step.start_time = datetime.now()
        success = True

        try:
            if not step.sub_steps:
                # --- ATOMIC STEP EXECUTION ---
                instr = step.instruction
                test_slug = session.test_name.replace(" ", "_").replace(":", "_").lower()

                if instr.startswith("FIND:"):
                    query = instr.replace("FIND:", "").strip()
                    # 1. Capture Full-Page Screenshot for Perception
                    path = f".vizQA/{test_slug}_{step.id}_before.jpg"
                    await self.page.screenshot(path=path, type="jpeg")
                    step.screenshot_before = path

                    # 2. Perceive
                    perception = await self.client.perceive(path, query=query)
                    step.perception_result = perception

                    # 3. Store match in session context
                    target = None
                    if perception.get("top_matches"):
                        target = perception["top_matches"][0]
                    elif perception.get("elements"):
                        target = perception["elements"][0]

                    if target:
                        session.metadata["target"] = target
                        session.metadata["last_perception"] = perception  # Cache for DO step
                        step.status = StepStatus.PASSED
                    else:
                        step.status = StepStatus.FAILED
                        step.failure_reason = self._get_failure_details("FIND", query, perception, "Element not found")

                elif instr.startswith("DO:"):
                    action_cmd = instr.replace("DO:", "").strip()
                    parts = action_cmd.split(" ", 1)
                    action = parts[0]
                    payload = parts[1] if len(parts) > 1 else ""

                    target = session.metadata.get("target")
                    if not target:
                        step.status = StepStatus.FAILED
                        step.failure_reason = "No target element found. FIND step must precede DO step."
                    else:
                        # 1. Capture Action Snapshot (Adaptive Crop)
                        action_path = f".vizQA/{test_slug}_{step.id}_{action}.jpg"
                        last_perc = session.metadata.get("last_perception", {})
                        viewport = last_perc.get("viewport", {"width": 1280, "height": 720})
                        vw, vh = viewport.get("width", 1280), viewport.get("height", 720)

                        # Get raw pixel coordinates
                        px, py, pw, ph = 0, 0, 0, 0
                        if "bounds" in target:
                            # Assuming bounds are [x1, y1, x2, y2] in pixels
                            x1, y1, x2, y2 = target["bounds"]
                            px, py, pw, ph = x1, y1, x2 - x1, y2 - y1
                        elif "location" in target:
                            # Normalized [y, x, w, h]
                            loc = target["location"]
                            px = loc[1] * vw
                            py = loc[0] * vh
                            pw = loc[2] * vw
                            ph = loc[3] * vh

                        if pw > 0 and ph > 0:
                            # 15% padding for width and height
                            px_pad = pw * 0.15
                            py_pad = ph * 0.15

                            cx = max(0, px - px_pad)
                            cy = max(0, py - py_pad)
                            cw = min(vw - cx, pw + 2 * px_pad)
                            ch = min(vh - cy, ph + 2 * py_pad)

                            clip = {"x": cx, "y": cy, "width": cw, "height": ch}

                            try:
                                await self.page.screenshot(path=action_path, type="jpeg", clip=clip)
                                step.action_screenshot = action_path
                            except Exception:
                                pass

                        # 2. Execute Interaction
                        await self._execute_interaction(action, px + pw / 2, py + ph / 2, payload)
                        step.status = StepStatus.PASSED

                elif instr.startswith("VERIFY:"):
                    query = instr.replace("VERIFY:", "").strip()
                    # VERIFY requires a fresh screenshot
                    path = f".vizQA/{test_slug}_{step.id}_verify.jpg"
                    await self.page.screenshot(path=path, type="jpeg")
                    step.screenshot_after = path

                    perception = await self.client.perceive(path, query=query)
                    step.perception_result = perception

                    match_found = False
                    if perception.get("top_matches"):
                        match_found = True
                    else:
                        # Semantic fallback: check element text content
                        q = query.lower()
                        for el in perception.get("elements", []):
                            text = (el.get("text") or "").lower()
                            label = (el.get("label") or "").lower()
                            name = (el.get("name") or "").lower()

                            # Check if query matches any descriptive field
                            if (q in text or q in label or q in name) or any(
                                word in text or word in label or word in name for word in q.split() if len(word) > 3
                            ):
                                match_found = True
                                break

                    if match_found:
                        step.status = StepStatus.PASSED
                    else:
                        step.status = StepStatus.FAILED
                        step.failure_reason = self._get_failure_details(
                            "VERIFY", query, perception, "Verification failed"
                        )

                else:
                    # Legacy execution for non-decomposed steps
                    test_slug = session.test_name.replace(" ", "_").replace(":", "_").lower()
                    before_path = f".vizQA/{test_slug}_{step.id}_before.jpg"
                    await self.page.screenshot(path=before_path, type="jpeg")
                    step.screenshot_before = before_path
                    perception = await self.client.perceive(before_path, query=step.instruction)
                    step.perception_result = perception
                    await self._execute_action(session, step)

                    if step.expectation:
                        step.status = await self._verify_expectation(session, step)
                    else:
                        step.status = StepStatus.PASSED
            else:
                # --- CONTAINER STEP EXECUTION ---
                sub_failed = False
                for sub_step in step.sub_steps:
                    if sub_failed:
                        await self._skip_step_recursive(sub_step, on_step_update)
                        continue

                    if not await self._run_step_recursive(session, sub_step, on_step_update):
                        sub_failed = True

                # Propagate failure details from child to parent
                if sub_failed:
                    step.status = StepStatus.FAILED
                    # Find the first failed sub-step to propagate reason
                    for sub_step in step.sub_steps:
                        if sub_step.status == StepStatus.FAILED:
                            step.failure_type = sub_step.failure_type
                            step.failure_reason = sub_step.failure_reason
                            break
                else:
                    step.status = StepStatus.PASSED

            if step.status == StepStatus.FAILED:
                success = False

        except Exception as e:
            step.status = StepStatus.FAILED
            step.failure_type = FailureType.ACTION_ERROR
            step.failure_reason = str(e)
            step.error = str(e)
            success = False
        finally:
            step.end_time = datetime.now()
            if on_step_update:
                await on_step_update(step)

        return success

    def _get_failure_details(self, stage: str, query: str, perception: Dict[str, Any], base_message: str) -> str:
        """Generates a detailed failure reason based on verbosity and perception results."""
        reason = f"{base_message} for query: '{query}'"

        if self.verbosity >= 1:
            # Level 1: List detected elements to show context
            elements = perception.get("elements", [])
            if elements:
                visible_texts = [el.get("text") for el in elements if el.get("text")]
                visible_texts = list(set(visible_texts))[:10]  # Deduplicate and limit
                if visible_texts:
                    reason += f"\n[Context] Elements visible on screen: {visible_texts}"
            else:
                reason += "\n[Context] No readable elements detected on the page."

        if self.verbosity >= 2:
            # Level 2: Comparison/Assertion details
            top_matches = perception.get("top_matches", [])
            if top_matches:
                matches_info = []
                for m in top_matches[:3]:
                    text = m.get("text", "unnamed")
                    sim = m.get("similarity", 0.0)
                    matches_info.append(f"'{text}' (similarity: {sim:.2f})")
                reason += f"\n[Detail] Top candidates: {', '.join(matches_info)}"
            elif perception.get("elements"):
                # If no top_matches, maybe show why elements didn't match
                reason += "\n[Detail] Semantic matching failed. No elements closely resembled the query."

        return reason

    async def _execute_interaction(self, action: str, x: float, y: float, payload: str):
        """Performs actual Playwright interactions at the specified pixel coordinates."""
        if action == "click":
            await self.page.mouse.click(x, y)
        elif "type" in action:
            # Click first to focus
            await self.page.mouse.click(x, y)
            # Clear field: Ctrl+A -> Backspace
            await self.page.keyboard.down("Control")
            await self.page.keyboard.press("a")
            await self.page.keyboard.up("Control")
            await self.page.keyboard.press("Backspace")
            # Type new content
            await self.page.keyboard.type(payload)
        elif action == "hover":
            await self.page.mouse.move(x, y)

        # Small wait for animation or state change
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
        """Simulates finding an element and interacting with it."""
        action = self._parse_action(step.instruction)
        test_slug = session.test_name.replace(" ", "_").replace(":", "_").lower()

        # Try to get bounds from perception result
        clip = None
        if step.perception_result:
            result = step.perception_result
            viewport = result.get("viewport", {"width": 1280, "height": 720})
            vw, vh = viewport.get("width", 1280), viewport.get("height", 720)

            if result.get("top_matches"):
                match = result["top_matches"][0]
                bounds = match.get("bounds")  # [x1, y1, x2, y2] in pixels
                if bounds:
                    x1, y1, x2, y2 = bounds
                    clip = {"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1}

            elif result.get("elements"):
                # Fallback to the most salient/first element if no top_matches
                el = result["elements"][0]
                loc = el.get("location")  # [nx, ny, nw, nh] normalized
                if loc and len(loc) == 4:
                    nx, ny, nw, nh = loc
                    clip = {"x": nx * vw, "y": ny * vh, "width": nw * vw, "height": nh * vh}

        if clip:
            # Capture action screenshot (cropped to element)
            action_path = f".vizQA/{test_slug}_{step.id}_{action}.jpg"
            try:
                await self.page.screenshot(path=action_path, type="jpeg", clip=clip)
                step.action_screenshot = action_path
            except Exception:
                # Fallback if clip is out of bounds or other error
                pass

        # Simulate interaction
        await asyncio.sleep(0.5)

    def _parse_action(self, instruction: str) -> str:
        """Heuristic to determine the action type from instruction."""
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
        """Polls the Perception API until the expectation is met or timeout."""
        start_wait = datetime.now()
        test_slug = session.test_name.replace(" ", "_").lower()

        while (datetime.now() - start_wait).total_seconds() < timeout:
            # Capture snapshot for verification (Deterministic 'after' name)
            path = f".vizQA/{test_slug}_{step.id}_after.jpg"
            await self.page.screenshot(path=path, type="jpeg")

            # Ask Perception API if expectation matches
            try:
                result = await self.client.perceive(path, query=step.expectation)
                # Success criteria: high salience or top matches found, or semantic match
                match_found = False
                if result.get("top_matches") or result.get("salience", 0) > 0.7:
                    match_found = True
                else:
                    q = step.expectation.lower()
                    for el in result.get("elements", []):
                        text = el.get("text", "").lower()
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
                # Keep polling even if perception call fails briefly
                pass

            await asyncio.sleep(1)

        step.failure_type = FailureType.TIMEOUT
        reason = f"Expectation '{step.expectation}' not met visually within {timeout}s."
        if self.verbosity >= 1:
            reason += " (Timed out while polling Perception API)"
        step.failure_reason = reason
        return StepStatus.FAILED
