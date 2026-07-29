import asyncio
import io
import logging
from pathlib import Path

import pytest
import yaml
from playwright._impl._errors import TargetClosedError
from playwright.async_api import async_playwright

import vizQA
from vizQA import StepResult, VizQASession, attach, click, key_input, run_step, run_steps
from vizQA import type as type_text
from vizQA import verify
from vizQA.app.client import PerceptionClient
from vizQA.app.core import Automator
from vizQA.app.logger import reset_logger
from vizQA.app.memory import StepStatus
from vizQA.search import ElementMatch, SearchResult, search

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_URL = (REPO_ROOT / "demo_site" / "dependency_auth_lab.html").resolve().as_uri()
PASSWORD_FLOW = yaml.safe_load((REPO_ROOT / "tests" / "dependency_login_password.yaml").read_text(encoding="utf-8"))
MFA_FLOW = yaml.safe_load((REPO_ROOT / "tests" / "dependency_login_mfa.yaml").read_text(encoding="utf-8"))


def _artifact_value(flow: dict, name: str) -> str:
    return flow["artifacts"][name]


def _legacy_candidate(**overrides):
    candidate = {
        "text": "Sign In",
        "label": "",
        "role": "button",
        "bounds": [485.0, 690.5, 655.0, 747.5],
        "score": 12.0,
        "placeholder": "",
        "color": "#facc15",
    }
    candidate.update(overrides)
    return candidate


def _normalized_candidate(**overrides):
    candidate = {
        "id": "el_68e88b2dd83a",
        "type": "link",
        "label": "Jobs",
        "location": [0.01, 0.64, 0.09, 0.03],
        "confidence": 0.36,
        "salience": 0.61,
        "similarity": 0.0,
        "parent_id": "sec_632fb9a9f7c2",
        "stable_id": "sid_68e88b2dd83a",
        "spatial": {
            "top": None,
            "bottom": "el_b73c131b86c9",
            "left": "el_91c7558550bd",
            "right": "el_facc69beee65",
            "position": "top-right",
        },
    }
    candidate.update(overrides)
    return candidate


def _perception_payload(*, session_id="test-session-123", top_matches=None, elements=None, **extra):
    payload = {
        "session_id": session_id,
        "viewport": {"width": 1440, "height": 1200},
        "top_matches": top_matches or [],
        "elements": elements or [],
    }
    payload.update(extra)
    return payload


class _FakePage:
    url = "https://example.test"

    def __init__(self):
        self.calls = []

    async def screenshot(self, *, path=None, type="png", **kwargs):
        self.calls.append({"path": path, "type": type, **kwargs})
        if path is None:
            return b"fake-jpeg-bytes"
        Path(path).write_bytes(b"fake-image")
        return None

    async def evaluate(self, _script):
        return 0


async def _fake_perceive(
    self,
    _image_path: str | None = None,
    query: str | None = None,
    session_scope: str | None = None,
    *,
    image_bytes: bytes | None = None,
    image_file=None,
):
    del image_bytes, image_file
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


async def _open_password_modal(page):
    await page.click("#open-sign-in")
    await page.locator("#username-field").wait_for()


# Public imports and namespace boundaries


def test_root_package_exports_only_documented_high_level_library_api():
    expected_exports = {
        "StepResult",
        "VizQASession",
        "attach",
        "click",
        "key_input",
        "run_step",
        "run_steps",
        "type",
        "verify",
    }

    assert set(vizQA.__all__) == expected_exports
    for name in expected_exports:
        assert hasattr(vizQA, name)

    assert not hasattr(vizQA, "SearchResult")
    assert not hasattr(vizQA, "ElementMatch")
    assert "search" not in vizQA.__all__


def test_search_namespace_exports_documented_low_level_search_api():
    import vizQA.search as search_module

    assert set(search_module.__all__) == {"ElementMatch", "SearchResult", "search"}
    assert search_module.search is search
    assert search_module.ElementMatch is ElementMatch
    assert search_module.SearchResult is SearchResult


# Search result model contract


def test_element_match_exposes_documented_fields_and_aliases():
    candidate = _normalized_candidate()

    match = ElementMatch(candidate, rank=1)

    assert match.id == candidate["id"]
    assert match.type == candidate["type"]
    assert match.label == candidate["label"]
    assert match.location == tuple(candidate["location"])
    assert match.center == (0.055, 0.655)
    assert match.rank == 1
    assert match.confidence == candidate["confidence"]
    assert match.salience == candidate["salience"]
    assert match.similarity == candidate["similarity"]
    assert match.attributes["parent_id"] == candidate["parent_id"]
    assert match.attributes["stable_id"] == candidate["stable_id"]
    assert match.attributes["spatial"]["position"] == "top-right"
    assert match.bounds == match.location
    assert match.text == match.label
    assert match.role == match.type
    assert match.score == match.confidence
    assert not hasattr(match, "raw")


def test_search_result_normalizes_legacy_backend_payload_and_preserves_metadata():
    perception = _perception_payload(
        top_matches=[_legacy_candidate()],
        backend_note="legacy-payload",
    )

    result = SearchResult("sign in button", perception, artifacts={"before": "/tmp/before.png"})

    assert result.query == "sign in button"
    assert result.viewport == {"width": 1440, "height": 1200}
    assert result.session_id == "test-session-123"
    assert result.best_match is not None
    assert result.best_match.label == "Sign In"
    assert result.best_match.location == (485.0, 690.5, 655.0, 747.5)
    assert result.best_match.confidence == 12.0
    assert result.best_match.attributes["placeholder"] == ""
    assert len(result.matches) == 1
    assert result.artifacts["before"] == "/tmp/before.png"
    assert result.metadata["backend_note"] == "legacy-payload"
    assert not hasattr(result, "raw")


def test_search_result_falls_back_to_elements_and_keeps_extra_metadata():
    candidate = _legacy_candidate(text="", label="Username", similarity=0.97)
    perception = _perception_payload(
        session_id="fallback-session",
        top_matches=[],
        elements=[candidate],
        backend_note="fallback-elements",
    )

    result = SearchResult("username field", perception)

    assert result.session_id == "fallback-session"
    assert result.best_match is not None
    assert result.best_match.label == "Username"
    assert result.best_match.similarity == 0.97
    assert result.metadata["backend_note"] == "fallback-elements"
    assert not hasattr(result, "raw")


# Step result model contract


def test_step_result_truthiness_reflects_success():
    success_result = StepResult(
        success=True,
        instruction="VERIFY: dashboard",
        matched_element=None,
        artifacts={},
        duration=0.1,
    )
    failure_result = StepResult(
        success=False,
        instruction="VERIFY: dashboard",
        matched_element=None,
        artifacts={},
        duration=0.1,
    )

    assert bool(success_result) is True
    assert bool(failure_result) is False
    assert success_result.message is None
    assert failure_result.message is None
    assert not hasattr(success_result, "raw")
    assert not hasattr(failure_result, "raw")


# Session lifecycle and page ownership


def test_attach_returns_reusable_session_with_documented_defaults():
    session = attach(page=object())

    assert isinstance(session, VizQASession)
    assert session.debug_dir is None
    assert session._automator.artifact_dir is None


def test_session_close_releases_vizqa_resources_without_closing_page(monkeypatch):
    async def run_test(page):
        monkeypatch.setattr(PerceptionClient, "perceive", _fake_perceive)
        PerceptionClient._test_page = page

        session = attach(page)
        await _open_password_modal(page)
        await session.verify("Sign In")
        await session.close()

        assert await page.locator("#password-field").is_visible()
        await page.fill("#password-field", "")
        assert await page.locator("#password-field").input_value() == ""

    asyncio.run(_with_page(run_test))


def test_attached_session_run_steps_preserves_page_state_between_steps(monkeypatch, tmp_path):
    async def run_test(page):
        username = _artifact_value(PASSWORD_FLOW, "username")
        password = _artifact_value(PASSWORD_FLOW, "password")

        await _open_password_modal(page)
        session = attach(page, debug_dir=str(tmp_path))

        async def fake_run_step(self, instruction, **options):
            del self, options
            artifact_path = tmp_path / f"{instruction.split()[0].lower()}_before.jpg"
            artifact_path.write_bytes(b"artifact")
            if "username field" in instruction:
                await page.fill("#username-field", username)
            elif "password field" in instruction:
                await page.fill("#password-field", password)
            return StepResult(
                success=True,
                instruction=instruction,
                matched_element={"text": instruction},
                artifacts={"before": str(artifact_path)},
                duration=0.01,
            )

        monkeypatch.setattr(VizQASession, "run_step", fake_run_step)
        results = await session.run_steps(
            [
                f"Type '{username}' into username field",
                f"Type '{password}' into password field",
            ]
        )

        assert len(results) == 2
        assert session.page is page
        assert await page.locator("#username-field").input_value() == username
        assert await page.locator("#password-field").input_value() == password
        assert await page.locator("#modal-card h2").inner_text() == "Sign In"
        assert all(result.success for result in results)
        assert all("before" in result.artifacts for result in results)
        assert all(Path(result.artifacts["before"]).exists() for result in results)

        await session.close()

    asyncio.run(_with_page(run_test))


# Artifact and logging behavior


def test_attach_without_debug_dir_does_not_create_persistent_run_logs(monkeypatch, tmp_path):
    monkeypatch.setattr("vizQA.app.logger._LOG_DIR", str(tmp_path))
    reset_logger()

    attach(page=object())

    assert list(tmp_path.glob("run_*.log")) == []
    reset_logger()


def test_attach_uses_debug_dir_as_artifact_directory(tmp_path):
    session = attach(page=object(), debug_dir=str(tmp_path))

    assert session.debug_dir == str(tmp_path)
    assert session._automator.artifact_dir == str(tmp_path)


def test_logger_injection_routes_messages_to_host_logger():
    stream = io.StringIO()
    host_logger = logging.getLogger("vizqa.library.test")
    handler = logging.StreamHandler(stream)
    host_logger.setLevel(logging.DEBUG)
    host_logger.addHandler(handler)

    try:
        session = attach(page=object(), logger=host_logger)
        session._automator.logger.log_warning("step-1", "custom logger test")
        session._automator.logger.log_debug("step-1", "debug message")
    finally:
        host_logger.removeHandler(handler)
        reset_logger()

    output = stream.getvalue()
    assert "custom logger test" in output
    assert "debug message" in output


def test_search_without_debug_dir_returns_no_persistent_artifacts(monkeypatch):
    page = _FakePage()

    async def mock_perceive(
        self,
        image_path: str | None = None,
        query: str | None = None,
        session_scope: str | None = None,
        *,
        image_bytes: bytes | None = None,
        image_file=None,
    ):
        del image_file, session_scope
        assert image_path is None
        assert image_bytes == b"fake-jpeg-bytes"
        assert query == "sign in button"
        self.session_id = "bytes-session-123"
        return _perception_payload(session_id="bytes-session-123", top_matches=[_legacy_candidate()])

    monkeypatch.setattr(PerceptionClient, "perceive", mock_perceive)

    result = asyncio.run(attach(page).search("sign in button"))

    assert result.session_id == "bytes-session-123"
    assert result.artifacts == {}
    assert len(page.calls) == 1
    assert page.calls[0]["path"] is None


def test_search_with_debug_dir_persists_artifacts(monkeypatch, tmp_path):
    page = _FakePage()

    async def mock_perceive(
        self,
        image_path: str | None = None,
        query: str | None = None,
        session_scope: str | None = None,
        *,
        image_bytes: bytes | None = None,
        image_file=None,
    ):
        del image_file, session_scope
        assert image_path is not None
        assert image_bytes is None
        assert query == "sign in button"
        self.session_id = "artifact-session"
        return _perception_payload(session_id="artifact-session", top_matches=[_legacy_candidate()])

    monkeypatch.setattr(PerceptionClient, "perceive", mock_perceive)

    result = asyncio.run(attach(page, debug_dir=str(tmp_path)).search("sign in button"))

    assert "before" in result.artifacts
    artifact_path = Path(result.artifacts["before"])
    assert artifact_path.exists()
    assert artifact_path.parent == tmp_path


# Top-level helper behavior


def test_attached_session_search_returns_typed_search_result(monkeypatch):
    async def run_test(page):
        async def mock_perceive(
            self,
            image_path: str | None = None,
            query: str | None = None,
            session_scope: str | None = None,
            *,
            image_bytes: bytes | None = None,
            image_file=None,
        ):
            del image_file, session_scope
            assert image_path is None
            assert isinstance(image_bytes, bytes)
            assert query == "jobs link"
            self.session_id = "typed-search-session"
            return _perception_payload(session_id="typed-search-session", top_matches=[_normalized_candidate()])

        monkeypatch.setattr(PerceptionClient, "perceive", mock_perceive)

        result = await attach(page).search("jobs link")

        assert isinstance(result, SearchResult)
        assert isinstance(result.best_match, ElementMatch)
        assert result.best_match is not None
        assert result.best_match.id == "el_68e88b2dd83a"
        assert result.best_match.type == "link"
        assert result.best_match.label == "Jobs"
        assert result.best_match.confidence == 0.36
        assert result.best_match.attributes["stable_id"] == "sid_68e88b2dd83a"

    asyncio.run(_with_page(run_test))


def test_top_level_helpers_return_step_results_and_preserve_page_state(monkeypatch):
    async def run_test(page):
        monkeypatch.setattr(PerceptionClient, "perceive", _fake_perceive)
        PerceptionClient._test_page = page

        username = _artifact_value(PASSWORD_FLOW, "username")
        password = _artifact_value(PASSWORD_FLOW, "password")
        mfa_code = _artifact_value(MFA_FLOW, "mfa_code")

        await _open_password_modal(page)

        open_result = await run_step(page, "Verify Sign In")
        assert isinstance(open_result, StepResult)
        assert open_result.success is True
        assert open_result.artifacts == {}

        username_result = await type_text(page, "username field", username)
        password_result = await type_text(page, "password field", password)
        assert await page.locator("#username-field").input_value() == username
        assert await page.locator("#password-field").input_value() == password

        verify_result = await verify(page, "Continue to MFA")
        continue_result = await click(page, "Continue to MFA")
        mfa_result = await type_text(page, "one-time code field", mfa_code)
        assert await page.locator("#mfa-code-field").input_value() == mfa_code

        submit_result = await run_step(page, MFA_FLOW["steps"][1]["action"])
        role_result = await verify(page, "Analyst")

        assert all(
            isinstance(result, StepResult)
            for result in [
                username_result,
                password_result,
                verify_result,
                continue_result,
                mfa_result,
                submit_result,
                role_result,
            ]
        )
        assert all(
            result.success
            for result in [
                username_result,
                password_result,
                verify_result,
                continue_result,
                mfa_result,
                submit_result,
                role_result,
            ]
        )
        assert await page.locator("#role-label").inner_text() == "Analyst"
        assert role_result.matched_element["text"] in {"Analyst", "Role: Analyst"}

    asyncio.run(_with_page(run_test))


def test_key_input_helpers_route_through_explicit_key_command(monkeypatch):
    captured = []

    async def fake_run_step(self, instruction, **options):
        del self, options
        captured.append(instruction)
        return StepResult(
            success=True,
            instruction=instruction,
            matched_element=None,
            artifacts={},
            duration=0.01,
        )

    monkeypatch.setattr(VizQASession, "run_step", fake_run_step)

    session_result = asyncio.run(attach(object()).key_input("Ctrl+C"))
    top_level_result = asyncio.run(key_input(object(), "Enter"))

    assert captured == ["Press key Ctrl+C", "Press key Enter"]
    assert session_result.success is True
    assert top_level_result.success is True


def test_failed_step_result_marks_failure_without_exposing_raw_payload(monkeypatch):
    async def fake_run_session(self, session, preserve_page=True):
        del self, preserve_page
        session.steps[0].status = StepStatus.FAILED
        session.steps[0].failure_reason = "Verification target was not found."
        return False

    monkeypatch.setattr(Automator, "run_session", fake_run_session)

    result = asyncio.run(verify(object(), "A success banner that does not exist"))

    assert isinstance(result, StepResult)
    assert result.success is False
    assert result.instruction == "VERIFY: A success banner that does not exist"
    assert result.message == "Verification target was not found."
    assert not hasattr(result, "raw")


def test_step_result_prefers_failure_reason_over_error_message(monkeypatch):
    async def fake_run_session(self, session, preserve_page=True):
        del self, preserve_page
        session.steps[0].status = StepStatus.FAILED
        session.steps[0].error = "Low-level error text"
        session.steps[0].failure_reason = "Readable failure reason"
        return False

    monkeypatch.setattr(Automator, "run_session", fake_run_session)

    result = asyncio.run(verify(object(), "Something failed"))

    assert result.message == "Readable failure reason"
    assert result.success is False


# End-to-end integration flows


def test_attached_session_hybrid_flow_matches_documented_library_usage(monkeypatch):
    async def run_test(page):
        monkeypatch.setattr(PerceptionClient, "perceive", _fake_perceive)
        PerceptionClient._test_page = page

        username = _artifact_value(PASSWORD_FLOW, "username")
        password = _artifact_value(PASSWORD_FLOW, "password")

        session = attach(page)
        await page.click("#open-sign-in")
        await page.get_by_label("Username").fill(username)
        await page.get_by_label("Password").fill(password)
        click_result = await session.click("Continue to MFA")
        verify_result = await session.verify("One-Time Code")

        assert click_result.success is True
        assert verify_result.success is True
        assert await page.locator("#mfa-code-field").is_visible()

    asyncio.run(_with_page(run_test))


def test_top_level_run_steps_can_finish_a_real_documented_flow(monkeypatch):
    async def run_test(page):
        monkeypatch.setattr(PerceptionClient, "perceive", _fake_perceive)
        PerceptionClient._test_page = page

        username = _artifact_value(PASSWORD_FLOW, "username")
        password = _artifact_value(PASSWORD_FLOW, "password")
        mfa_code = _artifact_value(MFA_FLOW, "mfa_code")

        results = await run_steps(
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

        assert len(results) == 6
        assert all(result.success for result in results)
        assert await page.locator("#role-label").inner_text() == "Analyst"
        assert await page.locator("#route-label").inner_text() == "overview"
        assert results[-1].instruction == MFA_FLOW["steps"][1]["action"]
        assert results[-1].matched_element is not None
        assert results[-1].matched_element["text"] == "Verify and Continue"

    asyncio.run(_with_page(run_test))
