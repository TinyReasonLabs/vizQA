import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from vizQA.cli import _count_top_level_results, run_single_test
from vizQA.memory import StepStatus, TestSession, TestStep


def _make_test_file(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / f"{name}.yaml"
    path.write_text(body, encoding="utf-8")
    return path


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
                "vizQA.cli._run_test_dependencies",
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
            patch("vizQA.cli.StepPlanner") as mock_planner_cls,
            patch("vizQA.cli._load_config", return_value={}),
            patch("vizQA.cli.BrowserStateCache.load", return_value={"localStorage": {"auth": "ok"}}),
            patch("vizQA.cli.BrowserStateCache.cache") as mock_cache_state,
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
            patch("vizQA.cli.StepPlanner") as mock_planner_cls,
            patch("vizQA.cli._load_config", return_value={}),
            patch("vizQA.cli.BrowserStateCache.cache") as mock_cache_state,
        ):
            mock_planner_cls.return_value.decompose.return_value = []
            result = await run_single_test(test_path, automator, reporter, is_dependency=True)

        assert result is True
        mock_cache_state.assert_called_once_with(
            "dependency_login_password",
            {"sessionStorage": {"route": "mfa"}, "localStorage": {}, "cookies": []},
        )

    asyncio.run(run_test())


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
                "vizQA.cli._run_test_dependencies",
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
            patch("vizQA.cli.StepPlanner") as mock_planner_cls,
        ):
            mock_planner_cls.return_value.decompose.return_value = []
            result = await run_single_test(test_path, automator, reporter)

        assert result is False
        automator.run_session.assert_not_called()
        reporter.on_session_start.assert_called_once()
        reporter.on_parent_step_start.assert_not_called()

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
