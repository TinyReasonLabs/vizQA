import asyncio
from pathlib import Path

import pytest
import yaml
from playwright._impl._errors import TargetClosedError
from playwright.async_api import async_playwright

from vizQA import attach, click, run_step, run_steps
from vizQA import type as type_text
from vizQA import verify
from vizQA.app.client import PerceptionClient
from vizQA.app.memory import TestStep
from vizQA.library import _collect_artifacts

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_URL = (REPO_ROOT / "demo_site" / "dependency_auth_lab.html").resolve().as_uri()
PASSWORD_FLOW = yaml.safe_load((REPO_ROOT / "tests" / "dependency_login_password.yaml").read_text(encoding="utf-8"))
MFA_FLOW = yaml.safe_load((REPO_ROOT / "tests" / "dependency_login_mfa.yaml").read_text(encoding="utf-8"))


async def _fake_perceive(self, _image_path: str, query: str | None = None):
    self.session_id = self.session_id or "local-test-session"
    return await PerceptionClient._test_page.evaluate(
        """(query) => {
            const normalize = (value) => (value || "").toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
            const isVisible = (element) => {
                const style = window.getComputedStyle(element);
                const rect = element.getBoundingClientRect();
                return style.visibility !== "hidden" && style.display !== "none" && rect.width > 0 && rect.height > 0;
            };
            const textFor = (element) => {
                if (element.tagName === "INPUT" || element.tagName === "TEXTAREA") {
                    return "";
                }
                return (element.innerText || element.textContent || "").replace(/\\s+/g, " ").trim();
            };
            const labelFor = (element) => {
                if (!(element instanceof HTMLInputElement || element instanceof HTMLTextAreaElement || element instanceof HTMLSelectElement)) {
                    return "";
                }
                const wrapped = element.closest("label");
                if (wrapped) {
                    return wrapped.innerText.replace(/\\s+/g, " ").trim();
                }
                if (element.id) {
                    const explicit = document.querySelector(`label[for="${element.id}"]`);
                    if (explicit) {
                        return explicit.innerText.replace(/\\s+/g, " ").trim();
                    }
                }
                return "";
            };
            const nodes = Array.from(
                document.querySelectorAll("button, input, textarea, label, h1, h2, h3, h4, p, span, strong, td, [role='button']")
            ).filter(isVisible);
            const queryText = normalize(query);
            const queryTokens = queryText.split(/\\s+/).filter(Boolean);
            const elements = nodes.map((element) => {
                const rect = element.getBoundingClientRect();
                const text = textFor(element);
                const placeholder = element.getAttribute("placeholder") || "";
                const label = labelFor(element);
                const role = element.getAttribute("role") || "";
                const type = element.getAttribute("type") || "";
                const id = element.id || "";
                const extra = element.tagName === "INPUT" || element.tagName === "TEXTAREA" ? " field input" : "";
                const haystack = normalize([text, placeholder, label, role, type, id, extra].join(" "));
                let score = 0;
                if (!queryText) {
                    score = 1;
                } else {
                    if (haystack.includes(queryText)) {
                        score += 10;
                    }
                    for (const token of queryTokens) {
                        if (haystack.includes(token)) {
                            score += 1;
                        }
                    }
                }

                return {
                    text,
                    placeholder,
                    label,
                    role,
                    bounds: [rect.left, rect.top, rect.right, rect.bottom],
                    score,
                };
            });

            const ranked = elements
                .filter((element) => element.score > 0)
                .sort((left, right) => right.score - left.score);

            return {
                session_id: "local-test-session",
                viewport: { width: window.innerWidth, height: window.innerHeight },
                elements: ranked.length ? ranked : elements,
                top_matches: ranked.slice(0, 5),
            };
        }""",
        query or "",
    )


async def _with_page(callback):
    async with async_playwright() as playwright:
        try:
            browser = await playwright.chromium.launch(headless=True)
        except TargetClosedError as exc:
            pytest.skip(f"Playwright browser launch unavailable in this environment: {exc}")
        page = await browser.new_page(viewport={"width": 1440, "height": 1200})
        try:
            await page.goto(DEMO_URL)
            await callback(page)
        finally:
            await browser.close()


def _artifact_value(flow: dict, name: str) -> str:
    return flow["artifacts"][name]


def test_collect_artifacts_merges_find_and_do_substep_artifacts():
    step = TestStep(
        id="step_00",
        instruction="Click the 'Sign In' button",
        sub_steps=[
            TestStep(id="step_00.01", instruction="FIND: Sign In", screenshot_before="before.jpg"),
            TestStep(id="step_00.02", instruction="DO: click", action_screenshot="action.jpg"),
        ],
    )

    assert _collect_artifacts(step) == {"before": "before.jpg", "action": "action.jpg"}


def test_step_result_truthiness_reflects_success():
    success_result = run_step_result = None
    failure_result = None

    from vizQA.library import StepResult

    success_result = StepResult(
        success=True,
        instruction="VERIFY: dashboard",
        matched_element=None,
        artifacts={},
        duration=0.1,
        raw={},
    )
    failure_result = StepResult(
        success=False,
        instruction="VERIFY: dashboard",
        matched_element=None,
        artifacts={},
        duration=0.1,
        raw={"failure_reason": "Dashboard not found"},
    )

    assert bool(success_result) is True
    assert bool(failure_result) is False


def test_attach_defaults_to_no_persistent_artifact_directory():
    session = attach(page=object())

    assert session.debug_dir is None
    assert session._automator.artifact_dir is None


def test_attach_uses_debug_dir_as_artifact_directory(tmp_path):
    session = attach(page=object(), debug_dir=str(tmp_path))

    assert session.debug_dir == str(tmp_path)
    assert session._automator.artifact_dir == str(tmp_path)


def test_top_level_library_helpers_preserve_real_page_state(monkeypatch):
    async def run_test(page):
        monkeypatch.setattr(PerceptionClient, "perceive", _fake_perceive)
        PerceptionClient._test_page = page

        original_url = page.url
        username = _artifact_value(PASSWORD_FLOW, "username")
        password = _artifact_value(PASSWORD_FLOW, "password")
        mfa_code = _artifact_value(MFA_FLOW, "mfa_code")

        open_result = await run_step(page, PASSWORD_FLOW["steps"][0]["action"])
        assert open_result.success is True
        assert open_result.instruction == PASSWORD_FLOW["steps"][0]["action"]
        assert page.url == original_url
        assert await page.locator("#username-field").is_visible()
        assert open_result.matched_element["text"] == "Sign In"
        assert open_result.artifacts == {}

        username_result = await type_text(page, "username field", username)
        assert username_result.success is True
        assert await page.locator("#username-field").input_value() == username

        password_result = await type_text(page, "password field", password)
        assert password_result.success is True
        assert await page.locator("#password-field").input_value() == password

        verify_result = await verify(page, "Continue to MFA")
        assert verify_result.success is True

        continue_result = await click(page, "Continue to MFA")
        assert continue_result.success is True

        mfa_result = await type_text(page, "one-time code field", mfa_code)
        assert mfa_result.success is True
        assert await page.locator("#mfa-code-field").input_value() == mfa_code

        submit_result = await run_step(page, MFA_FLOW["steps"][1]["action"])
        assert submit_result.success is True

        role_result = await verify(page, "Analyst")
        assert role_result.success is True
        assert await page.locator("#role-label").inner_text() == "Analyst"
        assert await page.locator("#view-title").inner_text() == "Overview Dashboard"
        assert role_result.raw["status"] == "passed"
        assert role_result.raw["metadata"]["target"]["text"] == "Role: Analyst"

    asyncio.run(_with_page(run_test))


def test_attached_session_run_steps_reuses_the_same_page_and_leaves_it_open(monkeypatch):
    async def run_test(page):
        monkeypatch.setattr(PerceptionClient, "perceive", _fake_perceive)
        PerceptionClient._test_page = page

        username = _artifact_value(PASSWORD_FLOW, "username")
        password = _artifact_value(PASSWORD_FLOW, "password")

        session = attach(page, debug_dir="test-debug")
        results = await session.run_steps(
            [
                PASSWORD_FLOW["steps"][0]["action"],
                f"Type '{username}' into username field",
                f"Type '{password}' into password field",
            ]
        )

        assert len(results) == 3
        assert all(result.success for result in results)
        assert session.page is page
        assert session.client.session_id == "local-test-session"
        assert await page.locator("#username-field").input_value() == username
        assert await page.locator("#password-field").input_value() == password
        assert await page.locator("#modal-card h2").inner_text() == "Sign In"
        assert all("before" in result.artifacts for result in results)
        assert all(Path(result.artifacts["before"]).exists() for result in results)
        assert all(Path(result.artifacts["before"]).parent == Path("test-debug") for result in results)

        await session.close()

        assert await page.locator("#password-field").is_visible()
        await page.fill("#password-field", "")
        assert await page.locator("#password-field").input_value() == ""

    asyncio.run(_with_page(run_test))


def test_top_level_run_steps_can_finish_a_real_yaml_backed_flow(monkeypatch):
    async def run_test(page):
        monkeypatch.setattr(PerceptionClient, "perceive", _fake_perceive)
        PerceptionClient._test_page = page

        username = _artifact_value(PASSWORD_FLOW, "username")
        password = _artifact_value(PASSWORD_FLOW, "password")
        mfa_code = _artifact_value(MFA_FLOW, "mfa_code")

        initial_results = await run_steps(
            page,
            [
                PASSWORD_FLOW["steps"][0]["action"],
                f"Type '{username}' into username field",
                f"Type '{password}' into password field",
                "Click Continue to MFA",
                f"Type '{mfa_code}' into one-time code field",
                MFA_FLOW["steps"][1]["action"],
            ],
        )

        assert len(initial_results) == 6
        assert all(result.success for result in initial_results)
        assert await page.locator("#role-label").inner_text() == "Analyst"
        user_chip_text = await page.locator("#user-chip").inner_text()
        assert "Avery Analyst" in user_chip_text
        assert "Analyst" in user_chip_text
        assert await page.locator("#route-label").inner_text() == "overview"
        assert initial_results[-1].instruction == MFA_FLOW["steps"][1]["action"]
        assert initial_results[-1].raw["metadata"]["target"]["text"] == "Verify and Continue"

    asyncio.run(_with_page(run_test))
