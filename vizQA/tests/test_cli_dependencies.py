import asyncio
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import click
import pytest
from click.testing import CliRunner

from vizQA.app.cli import (
    _clean_run_artifacts,
    _collect_involved_test_stems,
    _count_top_level_results,
    _emit_report_event,
    _load_test_data,
    cli,
    run_single_test,
)
from vizQA.app.exceptions import TestDefinitionError
from vizQA.app.logger import reset_logger
from vizQA.app.memory import StepStatus, TestSession, TestStep
from vizQA.app.viewport import ViewportSpec
from vizQA.rendering.events import StepStartedEvent


def _make_test_file(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / f"{name}.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_load_test_data_expands_env_vars_in_yaml_strings(tmp_path, monkeypatch):
    test_path = _make_test_file(
        tmp_path,
        "env_vars",
        """
name: "${TEST_NAME}"
url: "${APP_URL}"
headers:
  Authorization: "Bearer ${API_TOKEN}"
steps:
  - action: "Open ${PAGE_NAME}"
    expect: "See ${PAGE_NAME}"
""".strip(),
    )

    monkeypatch.setenv("TEST_NAME", "Environment Variables Test")
    monkeypatch.setenv("APP_URL", "http://example.com/app")
    monkeypatch.setenv("API_TOKEN", "secret-token")
    monkeypatch.setenv("PAGE_NAME", "Dashboard")

    test_data = _load_test_data(test_path)

    assert test_data["name"] == "Environment Variables Test"
    assert test_data["url"] == "http://example.com/app"
    assert test_data["headers"]["Authorization"] == "Bearer secret-token"
    assert test_data["steps"][0]["action"] == "Open Dashboard"
    assert test_data["steps"][0]["expect"] == "See Dashboard"


def test_load_test_data_raises_for_missing_env_var(tmp_path, monkeypatch):
    test_path = _make_test_file(
        tmp_path,
        "missing_env_var",
        """
name: "Missing Env Var Test"
url: "${APP_URL}"
steps: []
""".strip(),
    )

    monkeypatch.delenv("APP_URL", raising=False)

    with pytest.raises(TestDefinitionError) as exc_info:
        _load_test_data(test_path)

    assert exc_info.value.user_message == f"Failed to load test file {test_path.name}"
    assert "APP_URL" in exc_info.value.internal_detail


def test_run_single_test_restores_dependency_state_and_merges_artifacts(tmp_path):
    async def run_test():
        test_path = _make_test_file(
            tmp_path,
            "main",
            """
name: "Main Test"
url: "http://example.com/app"
artifacts:
  local_only: "child"
steps: []
""".strip(),
        )

        automator = SimpleNamespace(
            parser=object(),
            minilm=None,
            logger=MagicMock(),
            restore_browser_state=AsyncMock(),
            page=SimpleNamespace(goto=AsyncMock(), set_extra_http_headers=AsyncMock()),
            run_session=AsyncMock(return_value=True),
            capture_browser_state=AsyncMock(return_value={"cookies": [], "localStorage": {}, "sessionStorage": {}}),
        )
        reporter = SimpleNamespace(
            register_session=MagicMock(),
            sessions=[],
            on_parent_step_start=MagicMock(),
            on_parent_step_done=MagicMock(),
            on_session_start=MagicMock(),
        )

        with (
            patch(
                "vizQA.app.cli._run_test_dependencies",
                new=AsyncMock(
                    return_value=(
                        True,
                        [
                            {
                                "name": "Login MFA",
                                "status": "passed",
                                "session_id": "dep-1",
                                "file_stem": "dependency_login_mfa",
                            }
                        ],
                        {
                            "username": {"type": "string", "value": "analyst.user"},
                            "mfa_code": {"type": "string", "value": "246810"},
                        },
                    )
                ),
            ),
            patch("vizQA.app.cli.StepPlanner") as mock_planner_cls,
            patch("vizQA.app.cli._load_config", return_value={}),
            patch("vizQA.app.cli.BrowserStateCache.load", return_value={"localStorage": {"auth": "ok"}}),
            patch("vizQA.app.cli.BrowserStateCache.cache") as mock_cache_state,
        ):
            mock_planner_cls.return_value.decompose.return_value = []

            result = await run_single_test(test_path, automator, reporter)

        assert result is True
        automator.restore_browser_state.assert_awaited_once_with({"localStorage": {"auth": "ok"}})
        automator.page.goto.assert_awaited_once_with("http://example.com/app")
        session = reporter.register_session.call_args.args[0]
        assert session.artifacts["username"]["value"] == "analyst.user"
        assert session.artifacts["mfa_code"]["value"] == "246810"
        assert session.artifacts["local_only"]["value"] == "child"
        mock_cache_state.assert_called_once_with("main", {"cookies": [], "localStorage": {}, "sessionStorage": {}})

    asyncio.run(run_test())


def test_run_single_test_caches_dependency_state(tmp_path):
    async def run_test():
        test_path = _make_test_file(
            tmp_path,
            "dependency_login_password",
            """
name: "Password Login"
url: "http://example.com/login"
steps: []
""".strip(),
        )

        automator = SimpleNamespace(
            parser=object(),
            minilm=None,
            logger=MagicMock(),
            restore_browser_state=AsyncMock(),
            run_session=AsyncMock(return_value=True),
            capture_browser_state=AsyncMock(
                return_value={"sessionStorage": {"route": "mfa"}, "localStorage": {}, "cookies": []}
            ),
        )
        reporter = SimpleNamespace(
            register_session=MagicMock(),
            sessions=[],
            on_parent_step_start=MagicMock(),
            on_parent_step_done=MagicMock(),
            on_session_start=MagicMock(),
        )

        with (
            patch("vizQA.app.cli.StepPlanner") as mock_planner_cls,
            patch("vizQA.app.cli._load_config", return_value={}),
            patch("vizQA.app.cli.BrowserStateCache.cache") as mock_cache_state,
        ):
            mock_planner_cls.return_value.decompose.return_value = []
            result = await run_single_test(test_path, automator, reporter, is_dependency=True)

        assert result is True
        mock_cache_state.assert_called_once_with(
            "dependency_login_password",
            {"sessionStorage": {"route": "mfa"}, "localStorage": {}, "cookies": []},
        )

    asyncio.run(run_test())


def test_run_single_test_uses_viewport_namespaced_browser_state_cache(tmp_path):
    async def run_test():
        test_path = _make_test_file(
            tmp_path,
            "dependency_login_password",
            """
name: "Password Login"
url: "http://example.com/login"
steps: []
""".strip(),
        )

        automator = SimpleNamespace(
            parser=object(),
            minilm=None,
            logger=MagicMock(),
            restore_browser_state=AsyncMock(),
            run_session=AsyncMock(return_value=True),
            capture_browser_state=AsyncMock(
                return_value={"sessionStorage": {"route": "mfa"}, "localStorage": {}, "cookies": []}
            ),
        )
        reporter = SimpleNamespace(
            register_session=MagicMock(),
            sessions=[],
            on_parent_step_start=MagicMock(),
            on_parent_step_done=MagicMock(),
            on_session_start=MagicMock(),
        )

        viewport = ViewportSpec(name="mobile", width=390, height=844)

        with (
            patch("vizQA.app.cli.StepPlanner") as mock_planner_cls,
            patch("vizQA.app.cli._load_config", return_value={}),
            patch("vizQA.app.cli.BrowserStateCache.cache") as mock_cache_state,
        ):
            mock_planner_cls.return_value.decompose.return_value = []
            result = await run_single_test(test_path, automator, reporter, is_dependency=True, viewport=viewport)

        assert result is True
        mock_cache_state.assert_called_once_with(
            "dependency_login_password",
            {"sessionStorage": {"route": "mfa"}, "localStorage": {}, "cookies": []},
            namespace="mobile",
        )

    asyncio.run(run_test())


def test_emit_report_event_passes_legacy_viewport_to_step_callbacks():
    step = TestStep(id="step-1", instruction="VERIFY: Dashboard", status=StepStatus.RUNNING)
    reporter = SimpleNamespace(
        sessions=[
            TestSession(
                id="session-1",
                test_name="Dashboard",
                file_stem="dashboard",
                url="http://example.com",
                steps=[],
                viewport_name="Desktop",
                viewport_width=1440,
                viewport_height=900,
            )
        ],
        on_parent_step_start=MagicMock(),
    )

    _emit_report_event(reporter, StepStartedEvent(session_id="session-1", step=step))

    viewport = reporter.on_parent_step_start.call_args.kwargs["viewport"]
    assert viewport == ViewportSpec(name="Desktop", width=1440, height=900)


def test_run_single_test_stops_when_dependency_fails(tmp_path):
    async def run_test():
        test_path = _make_test_file(
            tmp_path,
            "main",
            """
name: "Main Test"
url: "http://example.com/app"
steps: []
""".strip(),
        )

        automator = SimpleNamespace(
            parser=object(),
            minilm=None,
            logger=MagicMock(),
            restore_browser_state=AsyncMock(),
            run_session=AsyncMock(return_value=True),
            capture_browser_state=AsyncMock(return_value={}),
        )
        reporter = SimpleNamespace(
            register_session=MagicMock(),
            sessions=[],
            on_parent_step_start=MagicMock(),
            on_parent_step_done=MagicMock(),
            on_session_start=MagicMock(),
        )

        with (
            patch(
                "vizQA.app.cli._run_test_dependencies",
                new=AsyncMock(
                    return_value=(
                        False,
                        [
                            {
                                "name": "Login MFA",
                                "status": "failed",
                                "session_id": "dep-1",
                                "file_stem": "dependency_login_mfa",
                            }
                        ],
                        {},
                    )
                ),
            ),
            patch("vizQA.app.cli.StepPlanner") as mock_planner_cls,
        ):
            mock_planner_cls.return_value.decompose.return_value = []
            result = await run_single_test(test_path, automator, reporter)

        assert result is False
        automator.run_session.assert_not_called()
        reporter.on_session_start.assert_called_once()
        reporter.on_parent_step_start.assert_not_called()

    asyncio.run(run_test())


def test_run_single_test_calls_reporter_session_start_before_running_steps(tmp_path):
    async def run_test():
        test_path = _make_test_file(
            tmp_path,
            "main",
            """
name: "Main Test"
url: "http://example.com/app"
steps: []
""".strip(),
        )

        reporter = SimpleNamespace(
            register_session=MagicMock(),
            sessions=[],
            on_parent_step_start=MagicMock(),
            on_parent_step_done=MagicMock(),
            on_session_start=MagicMock(),
        )

        async def run_session(_session, on_step_update=None):
            reporter.on_session_start.assert_called_once()
            if on_step_update:
                await on_step_update(TestStep(id="s1", instruction="VERIFY: ok", status=StepStatus.PASSED))
            return True

        automator = SimpleNamespace(
            parser=object(),
            minilm=None,
            logger=MagicMock(),
            restore_browser_state=AsyncMock(),
            page=SimpleNamespace(goto=AsyncMock(), set_extra_http_headers=AsyncMock()),
            run_session=AsyncMock(side_effect=run_session),
            capture_browser_state=AsyncMock(return_value={}),
        )

        with (
            patch("vizQA.app.cli.StepPlanner") as mock_planner_cls,
            patch("vizQA.app.cli._load_config", return_value={}),
            patch("vizQA.app.cli.BrowserStateCache.cache"),
        ):
            mock_planner_cls.return_value.decompose.return_value = []
            result = await run_single_test(
                test_path,
                automator,
                reporter,
                on_step_update=AsyncMock(),
                viewport=ViewportSpec(name="mobile", width=390, height=844),
            )

        assert result is True

    asyncio.run(run_test())


def test_run_single_test_passes_viewport_to_dependency_runs(tmp_path):
    async def run_test():
        test_path = _make_test_file(
            tmp_path,
            "main",
            """
name: "Main Test"
url: "http://example.com/app"
requires:
  - dependency_login_password
steps: []
""".strip(),
        )

        viewport = ViewportSpec(name="desktop", width=1440, height=900)
        automator = SimpleNamespace(
            parser=object(),
            minilm=None,
            logger=MagicMock(),
            restore_browser_state=AsyncMock(),
            page=SimpleNamespace(goto=AsyncMock(), set_extra_http_headers=AsyncMock()),
            run_session=AsyncMock(return_value=True),
            capture_browser_state=AsyncMock(return_value={}),
        )
        reporter = SimpleNamespace(
            register_session=MagicMock(),
            sessions=[],
            on_parent_step_start=MagicMock(),
            on_parent_step_done=MagicMock(),
            on_session_start=MagicMock(),
        )

        dependency_calls = []

        async def fake_run_single_test(
            dep_path,
            _automator,
            _reporter,
            owner_key=None,
            on_step_update=None,
            interactive=False,
            is_dependency=False,
            viewport=None,
        ):
            dependency_calls.append(
                {
                    "dep_path": dep_path,
                    "owner_key": owner_key,
                    "interactive": interactive,
                    "is_dependency": is_dependency,
                    "viewport": viewport,
                    "has_callback": on_step_update is not None,
                }
            )
            return True

        with (
            patch("vizQA.app.cli._resolve_dependencies", return_value=[tmp_path / "dependency_login_password.yaml"]),
            patch("vizQA.app.cli._load_config", return_value={}),
            patch("vizQA.app.cli._load_artifacts", return_value={}),
            patch("vizQA.app.cli._load_test_data") as mock_load_test_data,
            patch("vizQA.app.cli.run_single_test", side_effect=fake_run_single_test),
            patch("vizQA.app.cli.StepPlanner") as mock_planner_cls,
            patch("vizQA.app.cli.BrowserStateCache.cache"),
        ):

            def fake_load_test_data(path):
                if path == test_path:
                    return {
                        "name": "Main Test",
                        "url": "http://example.com/app",
                        "requires": ["dependency_login_password"],
                        "steps": [],
                    }
                return {
                    "name": "Dependency Login Password",
                    "url": "http://example.com/login",
                    "steps": [],
                }

            mock_load_test_data.side_effect = fake_load_test_data
            mock_planner_cls.return_value.decompose.return_value = []

            await run_single_test(test_path, automator, reporter, viewport=viewport)

        assert len(dependency_calls) == 1
        assert dependency_calls[0]["is_dependency"] is True
        assert dependency_calls[0]["viewport"] == viewport
        assert dependency_calls[0]["has_callback"] is False

    asyncio.run(run_test())


def test_count_top_level_results_excludes_dependency_sessions():
    sessions = [
        TestSession(
            id="dep-1",
            test_name="Dependency Login",
            url="http://example.com",
            is_dependency=True,
            steps=[TestStep(id="s1", instruction="step", status=StepStatus.PASSED)],
        ),
        TestSession(
            id="top-1",
            test_name="Main Flow",
            url="http://example.com",
            is_dependency=False,
            steps=[TestStep(id="s2", instruction="step", status=StepStatus.PASSED)],
        ),
    ]

    passed, failed = _count_top_level_results(sessions, total_tests=1)

    assert passed == 1
    assert failed == 0


def test_count_top_level_results_requires_all_viewports_to_pass():
    sessions = [
        TestSession(
            id="desktop-1",
            test_name="Main Flow",
            file_stem="main",
            url="http://example.com",
            viewport_slug="desktop",
            steps=[TestStep(id="s1", instruction="step", status=StepStatus.PASSED)],
        ),
        TestSession(
            id="mobile-1",
            test_name="Main Flow",
            file_stem="main",
            url="http://example.com",
            viewport_slug="mobile",
            steps=[TestStep(id="s2", instruction="step", status=StepStatus.FAILED)],
        ),
    ]

    passed, failed = _count_top_level_results(sessions, total_tests=1, total_viewports=2)

    assert passed == 0
    assert failed == 1


def test_collect_involved_test_stems_includes_dependencies(tmp_path):
    main_test = _make_test_file(
        tmp_path,
        "main",
        """
name: "Main"
url: "http://example.com"
requires:
  - login
steps: []
""".strip(),
    )
    _make_test_file(
        tmp_path,
        "login",
        """
name: "Login"
url: "http://example.com/login"
steps: []
""".strip(),
    )

    stems = _collect_involved_test_stems([main_test])

    assert stems == {"main", "login"}


def test_clean_run_artifacts_removes_only_involved_test_files(tmp_path, monkeypatch):
    artifact_dir = tmp_path / ".vizQA"
    browser_state_dir = artifact_dir / "browser_states"
    browser_state_dir.mkdir(parents=True)

    (artifact_dir / "main_step_before.jpg").write_text("old", encoding="utf-8")
    (artifact_dir / "login_step_verify.jpg").write_text("old", encoding="utf-8")
    (artifact_dir / "other_step_before.jpg").write_text("keep", encoding="utf-8")
    (browser_state_dir / "main.json").write_text("{}", encoding="utf-8")
    (browser_state_dir / "other.json").write_text("{}", encoding="utf-8")
    (artifact_dir / "run_20240101_010101.log").write_text("old log", encoding="utf-8")

    monkeypatch.setattr("vizQA.app.cli._ARTIFACT_DIR", artifact_dir)
    monkeypatch.setattr("vizQA.app.cli.BrowserStateCache.CACHE_DIR", browser_state_dir)

    deleted = _clean_run_artifacts({"main", "login"})

    assert deleted == {"screenshots": 2, "browser_states": 1, "logs": 1}
    assert not (artifact_dir / "main_step_before.jpg").exists()
    assert not (artifact_dir / "login_step_verify.jpg").exists()
    assert not (browser_state_dir / "main.json").exists()
    assert not (artifact_dir / "run_20240101_010101.log").exists()
    assert (artifact_dir / "other_step_before.jpg").exists()
    assert (browser_state_dir / "other.json").exists()


def test_clean_run_artifacts_removes_namespaced_lane_files(tmp_path, monkeypatch):
    artifact_dir = tmp_path / ".vizQA"
    browser_state_dir = artifact_dir / "browser_states"
    browser_state_dir.mkdir(parents=True)

    (artifact_dir / "mobile__main_step_before.jpg").write_text("old", encoding="utf-8")
    (artifact_dir / "desktop__main_step_before.jpg").write_text("old", encoding="utf-8")
    (artifact_dir / "other__main_step_before.jpg").write_text("keep", encoding="utf-8")
    (browser_state_dir / "mobile__main.json").write_text("{}", encoding="utf-8")
    (browser_state_dir / "desktop__main.json").write_text("{}", encoding="utf-8")
    (browser_state_dir / "other__main.json").write_text("{}", encoding="utf-8")
    (artifact_dir / "run_20240101_010101_mobile.log").write_text("old log", encoding="utf-8")
    (artifact_dir / "run_20240101_010101_desktop.log").write_text("old log", encoding="utf-8")

    monkeypatch.setattr("vizQA.app.cli._ARTIFACT_DIR", artifact_dir)
    monkeypatch.setattr("vizQA.app.cli.BrowserStateCache.CACHE_DIR", browser_state_dir)

    deleted = _clean_run_artifacts({"main"}, viewport_slugs={"mobile", "desktop"})

    assert deleted == {"screenshots": 2, "browser_states": 2, "logs": 2}
    assert not (artifact_dir / "mobile__main_step_before.jpg").exists()
    assert not (artifact_dir / "desktop__main_step_before.jpg").exists()
    assert not (browser_state_dir / "mobile__main.json").exists()
    assert not (browser_state_dir / "desktop__main.json").exists()
    assert not (artifact_dir / "run_20240101_010101_mobile.log").exists()
    assert not (artifact_dir / "run_20240101_010101_desktop.log").exists()
    assert (artifact_dir / "other__main_step_before.jpg").exists()
    assert (browser_state_dir / "other__main.json").exists()


def test_run_shows_cleanup_message_in_debug_log_mode(tmp_path):
    test_path = _make_test_file(
        tmp_path,
        "main",
        """
name: "Main"
url: "http://example.com"
steps: []
""".strip(),
    )

    runner = CliRunner()
    with (
        patch("vizQA.app.cli._clean_run_artifacts", return_value={"screenshots": 2, "browser_states": 1, "logs": 1}),
        patch("vizQA.app.cli.get_logger"),
        patch("vizQA.app.cli.PerceptionClient"),
        patch("vizQA.app.cli.Automator") as mock_automator_cls,
        patch("vizQA.app.cli.run_single_test", new=AsyncMock(return_value=True)),
    ):
        automator = SimpleNamespace(start=AsyncMock(), stop=AsyncMock(), _logger=MagicMock())
        mock_automator_cls.return_value = automator

        result = runner.invoke(cli, ["run", str(test_path), "--debug-log"])

    assert result.exit_code == 0
    assert "Cleaned 2 screenshots, 1 browser states, 1 old run logs" in result.output


def test_run_hides_cleanup_message_in_silent_mode(tmp_path):
    test_path = _make_test_file(
        tmp_path,
        "main",
        """
name: "Main"
url: "http://example.com"
steps: []
""".strip(),
    )

    runner = CliRunner()
    with (
        patch("vizQA.app.cli._clean_run_artifacts", return_value={"screenshots": 2, "browser_states": 1, "logs": 1}),
        patch("vizQA.app.cli.get_logger"),
        patch("vizQA.app.cli.PerceptionClient"),
        patch("vizQA.app.cli.Automator") as mock_automator_cls,
        patch("vizQA.app.cli.run_single_test", new=AsyncMock(return_value=True)),
    ):
        automator = SimpleNamespace(start=AsyncMock(), stop=AsyncMock(), _logger=MagicMock())
        mock_automator_cls.return_value = automator

        result = runner.invoke(cli, ["run", "--silent", str(test_path)])

    assert result.exit_code == 0
    assert "Cleaned 2 screenshots" not in result.output


def test_run_rejects_removed_verbose_flags(tmp_path):
    test_path = _make_test_file(
        tmp_path,
        "main",
        """
name: "Main"
url: "http://example.com"
steps: []
""".strip(),
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["run", "-v", str(test_path)])

    assert result.exit_code != 0
    assert "No such option: -v" in result.output


def test_run_debug_log_configures_logger(tmp_path):
    test_path = _make_test_file(
        tmp_path,
        "main",
        """
name: "Main"
url: "http://example.com"
steps: []
""".strip(),
    )

    runner = CliRunner()
    with (
        patch("vizQA.app.cli._clean_run_artifacts", return_value={"screenshots": 0, "browser_states": 0, "logs": 0}),
        patch("vizQA.app.cli.configure_logging") as mock_configure_logging,
        patch("vizQA.app.cli.get_logger"),
        patch("vizQA.app.cli.PerceptionClient"),
        patch("vizQA.app.cli.Automator") as mock_automator_cls,
        patch("vizQA.app.cli.run_single_test", new=AsyncMock(return_value=True)),
    ):
        automator = SimpleNamespace(start=AsyncMock(), stop=AsyncMock(), _logger=MagicMock())
        mock_automator_cls.return_value = automator

        result = runner.invoke(cli, ["run", "--debug-log", str(test_path)])

    assert result.exit_code == 0
    mock_configure_logging.assert_called_once_with(debug_enabled=True)


def test_run_creates_one_automator_per_viewport_lane(tmp_path):
    test_path = _make_test_file(
        tmp_path,
        "main",
        """
name: "Main"
url: "http://example.com"
steps: []
""".strip(),
    )

    runner = CliRunner()
    automator_instances = [
        SimpleNamespace(start=AsyncMock(), stop=AsyncMock()),
        SimpleNamespace(start=AsyncMock(), stop=AsyncMock()),
    ]

    with (
        patch("vizQA.app.cli._clean_run_artifacts", return_value={"screenshots": 0, "browser_states": 0, "logs": 0}),
        patch("vizQA.app.cli.get_logger"),
        patch("vizQA.app.cli.PerceptionClient"),
        patch("vizQA.app.cli.load_viewport_config"),
        patch(
            "vizQA.app.cli.resolve_viewports",
            return_value=[
                ViewportSpec(name="desktop", width=1440, height=900),
                ViewportSpec(name="mobile", width=390, height=844),
            ],
        ),
        patch("vizQA.app.cli.Automator", side_effect=automator_instances) as mock_automator_cls,
        patch("vizQA.app.cli.run_single_test", new=AsyncMock(return_value=True)),
    ):
        result = runner.invoke(cli, ["run", "--viewport", "desktop", "--viewport", "mobile", str(test_path)])

    assert result.exit_code == 0
    assert mock_automator_cls.call_count == 2


def test_run_interactive_cancels_other_viewports_on_first_abort(tmp_path):
    test_path = _make_test_file(
        tmp_path,
        "main",
        """
name: "Main"
url: "http://example.com"
steps: []
""".strip(),
    )

    runner = CliRunner()
    automator_instances = [
        SimpleNamespace(start=AsyncMock(), stop=AsyncMock()),
        SimpleNamespace(start=AsyncMock(), stop=AsyncMock()),
    ]
    cancellation_seen = {"mobile": False}

    async def fake_run_single_test(
        _test_file,
        _automator,
        _reporter,
        owner_key=None,
        on_step_update=None,
        interactive=False,
        is_dependency=False,
        viewport=None,
    ):
        del owner_key
        del on_step_update
        del is_dependency
        assert interactive is True
        if viewport.name == "desktop":
            raise click.Abort()
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            cancellation_seen["mobile"] = True
            raise
        return True

    with (
        patch("vizQA.app.cli._clean_run_artifacts", return_value={"screenshots": 0, "browser_states": 0, "logs": 0}),
        patch("vizQA.app.cli.inspect_weight_state") as mock_weight_state,
        patch("vizQA.app.cli.PerceptionClient"),
        patch("vizQA.app.cli.get_logger", side_effect=[MagicMock(), MagicMock()]),
        patch("vizQA.app.cli.load_viewport_config"),
        patch(
            "vizQA.app.cli.resolve_viewports",
            return_value=[
                ViewportSpec(name="mobile", width=390, height=844),
                ViewportSpec(name="desktop", width=1440, height=900),
            ],
        ),
        patch("vizQA.app.cli.Automator", side_effect=automator_instances),
        patch("vizQA.app.cli.run_single_test", side_effect=fake_run_single_test),
    ):
        mock_weight_state.return_value = SimpleNamespace(
            installed_revision="0.1.0",
            expected_revision="0.1.0",
            status="aligned",
            assumed_revision=False,
        )
        started = time.perf_counter()
        result = runner.invoke(cli, ["run", "-x", "--viewport", "desktop", "--viewport", "mobile", str(test_path)])
        elapsed = time.perf_counter() - started

    assert result.exit_code != 0
    assert elapsed < 2
    assert cancellation_seen["mobile"] is True or elapsed < 2
    assert automator_instances[0].stop.await_count == 1
    assert automator_instances[1].stop.await_count == 1


def test_run_logs_playwright_errors_without_printing_them(tmp_path, monkeypatch):
    test_path = _make_test_file(
        tmp_path,
        "main",
        """
name: "Main"
url: "http://example.com"
steps: []
""".strip(),
    )

    artifact_dir = tmp_path / ".vizQA"
    monkeypatch.setattr("vizQA.app.cli._ARTIFACT_DIR", artifact_dir)
    monkeypatch.setattr("vizQA.app.logger._LOG_DIR", str(artifact_dir))
    reset_logger()

    class FakePlaywrightError(Exception):
        pass

    FakePlaywrightError.__module__ = "playwright._impl._errors"

    runner = CliRunner()
    with (
        patch("vizQA.app.cli._clean_run_artifacts", return_value={"screenshots": 0, "browser_states": 0, "logs": 0}),
        patch("vizQA.app.cli.inspect_weight_state") as mock_weight_state,
        patch("vizQA.app.cli.PerceptionClient"),
        patch("vizQA.app.cli.Automator") as mock_automator_cls,
        patch("vizQA.app.cli.run_single_test", new=AsyncMock(return_value=True)),
    ):
        mock_weight_state.return_value = SimpleNamespace(
            installed_revision="0.1.0",
            expected_revision="0.1.0",
            status="aligned",
            assumed_revision=False,
            package_version="0.1.0",
        )
        mock_automator_cls.return_value = SimpleNamespace(
            start=AsyncMock(side_effect=FakePlaywrightError("browser crashed")),
            stop=AsyncMock(),
        )

        result = runner.invoke(cli, ["run", str(test_path)])

    try:
        assert result.exit_code == 1
        assert "browser crashed" not in result.output
        assert "An unexpected error occurred during execution" not in result.output

        log_files = sorted(artifact_dir.glob("run_*.log"))
        assert len(log_files) == 1
        log_text = log_files[0].read_text(encoding="utf-8")
        assert "Playwright lane failure" in log_text
        assert "browser crashed" in log_text
    finally:
        reset_logger()
