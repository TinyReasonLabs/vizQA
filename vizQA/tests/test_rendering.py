from unittest.mock import MagicMock

from rich.console import Console

from vizQA.app.memory import StepStatus, TestSession, TestStep
from vizQA.rendering.components import build_compact_run_row
from vizQA.rendering.events import (
    RunFinishedEvent,
    SessionBlockedEvent,
    SessionFinishedEvent,
    SessionStartedEvent,
    StepFinishedEvent,
    StepStartedEvent,
    TopLevelTestStartedEvent,
)
from vizQA.rendering.layout import compose_layout
from vizQA.rendering.models import DisplayMode, RunStatus
from vizQA.rendering.store import RunStateStore
from vizQA.rendering.terminal_reporter import TerminalReporter
from vizQA.rendering.theme import REPORT_GREEN, REPORT_RED, RUN_STATUS_STYLES, STEP_STATUS_STYLES, VERIFY_STYLE


def _session(
    session_id: str,
    test_name: str,
    *,
    file_stem: str,
    is_dependency: bool = False,
    viewport_name: str | None = None,
    viewport_slug: str | None = None,
    dependency_results: list[dict] | None = None,
) -> TestSession:
    return TestSession(
        id=session_id,
        test_name=test_name,
        file_stem=file_stem,
        url="http://example.com",
        is_dependency=is_dependency,
        viewport_name=viewport_name,
        viewport_slug=viewport_slug,
        dependency_results=dependency_results or [],
        steps=[],
    )


def _render_plain(renderable, *, width: int = 120) -> str:
    console = Console(record=True, width=width)
    console.print(renderable)
    return console.export_text()


def test_store_groups_dependencies_under_top_level_and_tracks_focus():
    store = RunStateStore(display_mode=DisplayMode.VERBOSE)
    store.handle(
        TopLevelTestStartedEvent(
            owner_key="checkout",
            test_name="Checkout",
            file_stem="checkout",
            display_path="tests/checkout.yaml",
            expected_dependency_total=1,
        )
    )
    store.handle(
        SessionStartedEvent(
            owner_key="checkout",
            session=_session("dep-1", "Login dependency", file_stem="login", is_dependency=True),
        )
    )
    store.handle(
        SessionStartedEvent(
            owner_key="checkout",
            session=_session(
                "main-1", "Checkout", file_stem="checkout", viewport_name="Desktop", viewport_slug="desktop"
            ),
        )
    )

    snapshot = store.snapshot()

    assert snapshot.focused_owner_key == "checkout"
    assert len(snapshot.top_level_runs) == 1
    top_level = snapshot.top_level_runs[0]
    assert top_level.display_path == "tests/checkout.yaml"
    assert [session.test_name for session in top_level.dependencies] == ["Login dependency"]
    assert [session.session_id for session in top_level.sessions] == ["main-1"]


def test_store_merges_shared_step_rows_across_viewports():
    store = RunStateStore(display_mode=DisplayMode.VERBOSE)
    store.handle(
        TopLevelTestStartedEvent(
            owner_key="settings",
            test_name="Settings",
            file_stem="settings",
            display_path="tests/settings.yaml",
            expected_dependency_total=0,
        )
    )
    store.handle(
        SessionStartedEvent(
            owner_key="settings",
            session=_session(
                "mobile", "Settings", file_stem="settings", viewport_name="Mobile", viewport_slug="mobile"
            ),
        )
    )
    store.handle(
        SessionStartedEvent(
            owner_key="settings",
            session=_session(
                "desktop", "Settings", file_stem="settings", viewport_name="Desktop", viewport_slug="desktop"
            ),
        )
    )

    mobile_step = TestStep(id="m1", instruction="VERIFY: Settings page", status=StepStatus.PASSED)
    desktop_step = TestStep(id="d1", instruction="VERIFY: Settings page", status=StepStatus.PASSED)
    store.handle(StepFinishedEvent(session_id="mobile", step=mobile_step))
    store.handle(StepFinishedEvent(session_id="desktop", step=desktop_step))

    merged = store.snapshot().top_level_runs[0].merged_step_rows

    assert len(merged) == 1
    assert merged[0].text == "VERIFY Settings page"
    assert merged[0].viewport_status["mobile"].complete is True
    assert merged[0].viewport_status["desktop"].complete is True


def test_reporting_theme_uses_custom_green_and_red_palette():
    assert REPORT_GREEN == "#38d9a9"
    assert REPORT_RED == "#ff5e74"
    assert STEP_STATUS_STYLES[StepStatus.PASSED] == ("✔", REPORT_GREEN)
    assert STEP_STATUS_STYLES[StepStatus.FAILED] == ("✘", REPORT_RED)
    assert RUN_STATUS_STYLES[RunStatus.PASSED] == ("✔", REPORT_GREEN)
    assert RUN_STATUS_STYLES[RunStatus.FAILED] == ("✘", REPORT_RED)
    assert RUN_STATUS_STYLES[RunStatus.BLOCKED] == ("✘", REPORT_RED)
    assert VERIFY_STYLE == f"bold {REPORT_GREEN}"


def test_store_tracks_cursor_on_latest_row_only():
    store = RunStateStore(display_mode=DisplayMode.VERBOSE)
    store.handle(
        TopLevelTestStartedEvent(
            owner_key="auth",
            test_name="Auth",
            file_stem="auth",
            display_path="tests/auth.yaml",
            expected_dependency_total=0,
        )
    )
    store.handle(
        SessionStartedEvent(
            owner_key="auth",
            session=_session("mobile", "Auth", file_stem="auth", viewport_name="Mobile", viewport_slug="mobile"),
        )
    )

    first = TestStep(id="s1", instruction="FIND: Continue", status=StepStatus.PASSED)
    second = TestStep(id="s2", instruction="DO: click", status=StepStatus.PASSED)
    store.handle(StepFinishedEvent(session_id="mobile", step=first))
    store.handle(StepFinishedEvent(session_id="mobile", step=second))

    merged = store.snapshot().top_level_runs[0].merged_step_rows

    assert merged[0].viewport_status["mobile"].active is False
    assert merged[1].viewport_status["mobile"].active is True


def test_store_keeps_shared_row_running_while_another_viewport_is_still_active_on_it():
    store = RunStateStore(display_mode=DisplayMode.VERBOSE)
    store.handle(
        TopLevelTestStartedEvent(
            owner_key="settings",
            test_name="Settings",
            file_stem="settings",
            display_path="tests/settings.yaml",
            expected_dependency_total=0,
        )
    )
    desktop_session = _session(
        "desktop",
        "Settings",
        file_stem="settings",
        viewport_name="Desktop",
        viewport_slug="desktop",
    )
    mobile_session = _session(
        "mobile",
        "Settings",
        file_stem="settings",
        viewport_name="Mobile",
        viewport_slug="mobile",
    )
    store.handle(SessionStartedEvent(owner_key="settings", session=desktop_session))
    store.handle(SessionStartedEvent(owner_key="settings", session=mobile_session))

    failed_step = TestStep(id="desktop-step", instruction="VERIFY: Save banner", status=StepStatus.FAILED)
    running_step = TestStep(id="mobile-step", instruction="VERIFY: Save banner", status=StepStatus.RUNNING)

    store.handle(StepFinishedEvent(session_id="desktop", step=failed_step))
    store.handle(StepFinishedEvent(session_id="mobile", step=running_step))

    merged_row = store.snapshot().top_level_runs[0].merged_step_rows[0]

    assert merged_row.status == RunStatus.RUNNING
    assert merged_row.viewport_status["mobile"].active is True
    assert merged_row.viewport_status["desktop"].complete is True


def test_focus_stays_on_earliest_active_top_level_run():
    store = RunStateStore(display_mode=DisplayMode.VERBOSE)
    store.handle(
        TopLevelTestStartedEvent(
            owner_key="a",
            test_name="Test A",
            file_stem="a",
            display_path="tests/a.yaml",
            expected_dependency_total=0,
        )
    )
    store.handle(
        SessionStartedEvent(
            owner_key="a",
            session=_session("a-mobile", "Test A", file_stem="a", viewport_name="Mobile", viewport_slug="mobile"),
        )
    )
    store.handle(
        TopLevelTestStartedEvent(
            owner_key="b",
            test_name="Test B",
            file_stem="b",
            display_path="tests/b.yaml",
            expected_dependency_total=0,
        )
    )
    store.handle(
        SessionStartedEvent(
            owner_key="b",
            session=_session("b-desktop", "Test B", file_stem="b", viewport_name="Desktop", viewport_slug="desktop"),
        )
    )

    late_step = TestStep(id="late-a", instruction="VERIFY: A still running", status=StepStatus.PASSED)
    store.handle(StepFinishedEvent(session_id="a-mobile", step=late_step))

    assert store.snapshot().focused_owner_key == "a"


def test_store_removes_substeps_after_parent_completes():
    store = RunStateStore(display_mode=DisplayMode.VERBOSE)
    store.handle(
        TopLevelTestStartedEvent(
            owner_key="flow",
            test_name="Flow",
            file_stem="flow",
            display_path="tests/flow.yaml",
            expected_dependency_total=0,
        )
    )
    session = TestSession(
        id="main-1",
        test_name="Flow",
        file_stem="flow",
        url="http://example.com",
        steps=[
            TestStep(
                id="parent-1",
                instruction="Click on X",
                expectation="Expect Y",
                sub_steps=[
                    TestStep(id="sub-1", instruction="FIND: Item"),
                    TestStep(id="sub-2", instruction="DO: Click"),
                ],
            )
        ],
    )
    store.handle(SessionStartedEvent(owner_key="flow", session=session))
    parent = session.steps[0]
    find_step = parent.sub_steps[0]
    do_step = parent.sub_steps[1]

    store.handle(StepStartedEvent(session_id="main-1", step=parent))
    find_step.status = StepStatus.PASSED
    do_step.status = StepStatus.PASSED
    store.handle(StepFinishedEvent(session_id="main-1", step=find_step))
    store.handle(StepFinishedEvent(session_id="main-1", step=do_step))

    running_rows = store.snapshot().top_level_runs[0].merged_step_rows
    assert [row.text for row in running_rows] == ["Click on X → Expect Y", "FIND Item", "DO Click"]

    parent.status = StepStatus.PASSED
    store.handle(StepFinishedEvent(session_id="main-1", step=parent))

    final_rows = store.snapshot().top_level_runs[0].merged_step_rows
    assert [row.text for row in final_rows] == ["Click on X → Expect Y"]


def test_verbose_layout_renders_dependency_panel_and_pinned_header():
    store = RunStateStore(display_mode=DisplayMode.VERBOSE)
    store.handle(
        TopLevelTestStartedEvent(
            owner_key="checkout",
            test_name="Checkout",
            file_stem="checkout",
            display_path="tests/dependency_login_mfa.yaml",
            expected_dependency_total=1,
        )
    )
    dependency = _session("dep-1", "Login dependency", file_stem="dependency_auth_seed", is_dependency=True)
    dependency_mobile = _session("dep-2", "Login dependency", file_stem="dependency_auth_seed", is_dependency=True)
    session = _session("main-1", "Checkout", file_stem="checkout", viewport_name="Desktop", viewport_slug="desktop")
    store.handle(SessionStartedEvent(owner_key="checkout", session=dependency))
    store.handle(SessionStartedEvent(owner_key="checkout", session=dependency_mobile))
    store.handle(SessionStartedEvent(owner_key="checkout", session=session))
    store.handle(
        StepStartedEvent(
            session_id="main-1",
            step=TestStep(id="parent-1", instruction="Open checkout", expectation="Checkout page"),
        )
    )
    store.handle(
        StepFinishedEvent(
            session_id="main-1",
            step=TestStep(id="atomic-1", instruction="VERIFY: Checkout page", status=StepStatus.PASSED),
        )
    )

    plain = _render_plain(compose_layout(store.snapshot(), height=18, width=100), width=100)

    assert "Running 1 dependency" in plain
    assert "dependency_auth_seed" in plain
    assert plain.count("dependency_auth_seed") == 1
    assert "tests/dependency_login_mfa.yaml" in plain
    assert "VERIFY Checkout page" in plain
    assert "╭" not in plain
    assert "│" not in plain
    assert "Results:" not in plain


def test_dependency_header_uses_expected_total_not_started_count():
    store = RunStateStore(display_mode=DisplayMode.VERBOSE)
    store.handle(
        TopLevelTestStartedEvent(
            owner_key="checkout",
            test_name="Checkout",
            file_stem="checkout",
            display_path="tests/dependency_login_mfa.yaml",
            expected_dependency_total=4,
        )
    )
    dependency = _session("dep-1", "Dependency one", file_stem="dependency_one", is_dependency=True)
    store.handle(SessionStartedEvent(owner_key="checkout", session=dependency))

    plain = _render_plain(compose_layout(store.snapshot(), height=14, width=100), width=100)

    assert "Running 4 dependencies..." in plain


def test_silent_layout_renders_compact_rows_only():
    store = RunStateStore(display_mode=DisplayMode.SILENT)
    store.handle(
        TopLevelTestStartedEvent(
            owner_key="search",
            test_name="Search",
            file_stem="search",
            display_path="tests/search.yaml",
            expected_dependency_total=0,
        )
    )
    store.handle(SessionStartedEvent(owner_key="search", session=_session("main-1", "Search", file_stem="search")))
    store.handle(
        StepFinishedEvent(
            session_id="main-1",
            step=TestStep(id="atomic-1", instruction="VERIFY: Results shown", status=StepStatus.PASSED),
        )
    )

    plain = _render_plain(compose_layout(store.snapshot(), height=10, width=80), width=80)

    assert "Dependency tests" not in plain
    assert "tests/search.yaml" in plain
    assert "Results shown" in plain


def test_silent_layout_shows_dependency_progress_placeholder_while_dependencies_run():
    store = RunStateStore(display_mode=DisplayMode.SILENT)
    store.handle(
        TopLevelTestStartedEvent(
            owner_key="checkout",
            test_name="Checkout",
            file_stem="checkout",
            display_path="tests/checkout.yaml",
            expected_dependency_total=2,
        )
    )
    dependency = TestSession(
        id="dep-1",
        test_name="Dependency login",
        file_stem="dependency_login",
        url="http://example.com",
        is_dependency=True,
        steps=[TestStep(id="dep-parent", instruction="Action name")],
    )
    store.handle(SessionStartedEvent(owner_key="checkout", session=dependency))
    store.handle(StepStartedEvent(session_id="dep-1", step=dependency.steps[0]))

    plain = _render_plain(compose_layout(store.snapshot(), height=10, width=80), width=80)

    assert "tests/checkout.yaml" in plain
    assert "running 2 dependencies..." in plain


def test_silent_layout_indents_substeps_more_than_parent_steps():
    store = RunStateStore(display_mode=DisplayMode.SILENT)
    session = TestSession(
        id="main-1",
        test_name="Main Test",
        file_stem="main_test",
        url="http://example.com",
        steps=[
            TestStep(
                id="parent-1",
                instruction="Click on X",
                sub_steps=[TestStep(id="sub-1", instruction="FIND: Item")],
            )
        ],
    )
    store.handle(
        TopLevelTestStartedEvent(
            owner_key="main",
            test_name="Main Test",
            file_stem="main_test",
            display_path="tests/main_test.yaml",
            expected_dependency_total=0,
        )
    )
    store.handle(SessionStartedEvent(owner_key="main", session=session))
    parent = session.steps[0]
    store.handle(StepStartedEvent(session_id="main-1", step=parent))
    session.steps[0].sub_steps[0].status = StepStatus.PASSED
    store.handle(StepFinishedEvent(session_id="main-1", step=session.steps[0].sub_steps[0]))

    plain = _render_plain(compose_layout(store.snapshot(), height=10, width=80), width=80)

    assert "  Click on X" in plain
    assert "    FIND Item" in plain


def test_finished_successful_run_collapses_to_compact_rows_only():
    store = RunStateStore(display_mode=DisplayMode.VERBOSE)

    store.handle(
        TopLevelTestStartedEvent(
            owner_key="auth",
            test_name="Auth Flow",
            file_stem="auth_flow",
            display_path="tests/auth_flow.yaml",
            expected_dependency_total=1,
        )
    )
    dependency = _session("dep-1", "Dependency Login", file_stem="dependency_login", is_dependency=True)
    auth_session = _session("auth-main", "Auth Flow", file_stem="auth_flow")
    store.handle(SessionStartedEvent(owner_key="auth", session=dependency))
    store.handle(SessionStartedEvent(owner_key="auth", session=auth_session))

    dependency_step = TestStep(id="dep-step", instruction="VERIFY: Dependency ready", status=StepStatus.PASSED)
    auth_step = TestStep(id="auth-step", instruction="VERIFY: User signed in", status=StepStatus.PASSED)
    store.handle(StepFinishedEvent(session_id="dep-1", step=dependency_step))
    dependency.steps = [dependency_step]
    store.handle(SessionFinishedEvent(session=dependency))
    store.handle(StepFinishedEvent(session_id="auth-main", step=auth_step))
    auth_session.steps = [auth_step]
    store.handle(SessionFinishedEvent(session=auth_session))

    store.handle(
        TopLevelTestStartedEvent(
            owner_key="approval",
            test_name="Manager Approval",
            file_stem="dependency_manager_approval",
            display_path="tests/dependency_manager_approval.yaml",
            expected_dependency_total=0,
        )
    )
    approval_session = _session(
        "approval-main",
        "Manager Approval",
        file_stem="dependency_manager_approval",
    )
    store.handle(SessionStartedEvent(owner_key="approval", session=approval_session))
    approval_step = TestStep(
        id="approval-step",
        instruction="Click the 'Approve Latest Request' button",
        expectation="'Request approved' should appear",
        status=StepStatus.PASSED,
    )
    store.handle(StepFinishedEvent(session_id="approval-main", step=approval_step))
    approval_session.steps = [approval_step]
    store.handle(SessionFinishedEvent(session=approval_session))
    store.handle(RunFinishedEvent())

    plain = _render_plain(compose_layout(store.snapshot(), height=16, width=100), width=100)

    assert "tests/auth_flow.yaml" in plain
    assert "tests/dependency_manager_approval.yaml" in plain
    assert "Running 1 dependency" not in plain
    assert "Dependency ready" not in plain
    assert "Request approved" not in plain


def test_finished_failed_run_also_collapses_to_compact_rows_only():
    store = RunStateStore(display_mode=DisplayMode.VERBOSE)

    store.handle(
        TopLevelTestStartedEvent(
            owner_key="auth",
            test_name="Auth Flow",
            file_stem="auth_flow",
            display_path="tests/auth_flow.yaml",
            expected_dependency_total=0,
        )
    )
    auth_session = _session("auth-main", "Auth Flow", file_stem="auth_flow")
    store.handle(SessionStartedEvent(owner_key="auth", session=auth_session))
    auth_step = TestStep(id="auth-step", instruction="VERIFY: User signed in", status=StepStatus.PASSED)
    store.handle(StepFinishedEvent(session_id="auth-main", step=auth_step))
    auth_session.steps = [auth_step]
    store.handle(SessionFinishedEvent(session=auth_session))

    store.handle(
        TopLevelTestStartedEvent(
            owner_key="auth-fail",
            test_name="Auth Flow Fail",
            file_stem="auth_flow_fail",
            display_path="tests/auth_flow_fail.yaml",
            expected_dependency_total=2,
        )
    )
    dependency = _session("dep-1", "Dependency Login", file_stem="dependency_login", is_dependency=True)
    failed_session = _session(
        "fail-main",
        "Auth Flow Fail",
        file_stem="auth_flow_fail",
        viewport_name="desktop",
        viewport_slug="desktop",
    )
    store.handle(SessionStartedEvent(owner_key="auth-fail", session=dependency))
    store.handle(SessionStartedEvent(owner_key="auth-fail", session=failed_session))

    dependency_step = TestStep(id="dep-step", instruction="VERIFY: Dependency ready", status=StepStatus.PASSED)
    failed_step = TestStep(
        id="failed-step",
        instruction="Click the Submit button",
        expectation="The 'Sign in' modal should close",
        status=StepStatus.FAILED,
    )
    store.handle(StepFinishedEvent(session_id="dep-1", step=dependency_step))
    dependency.steps = [dependency_step]
    store.handle(SessionFinishedEvent(session=dependency))
    store.handle(StepFinishedEvent(session_id="fail-main", step=failed_step))
    failed_session.steps = [failed_step]
    store.handle(SessionFinishedEvent(session=failed_session))
    store.handle(RunFinishedEvent())

    plain = _render_plain(compose_layout(store.snapshot(), height=16, width=100), width=100)

    assert "tests/auth_flow.yaml" in plain
    assert "tests/auth_flow_fail.yaml" in plain
    assert "[desktop] FAIL" in plain
    assert "Running 2 dependencies" not in plain
    assert "Dependency ready" not in plain
    assert "Click the Submit button" not in plain


def test_dependency_section_shows_active_dependency_actions_and_dotted_substeps():
    store = RunStateStore(display_mode=DisplayMode.VERBOSE)
    dependency = TestSession(
        id="dep-1",
        test_name="Dependency Login",
        file_stem="dependency_login",
        url="http://example.com",
        is_dependency=True,
        steps=[
            TestStep(
                id="dep-parent",
                instruction="Action name",
                sub_steps=[
                    TestStep(id="dep-sub-1", instruction="FIND: thing"),
                    TestStep(id="dep-sub-2", instruction="DO: click"),
                ],
            )
        ],
    )
    store.handle(
        TopLevelTestStartedEvent(
            owner_key="main",
            test_name="Main",
            file_stem="main",
            display_path="tests/main.yaml",
            expected_dependency_total=1,
        )
    )
    store.handle(SessionStartedEvent(owner_key="main", session=dependency))
    store.handle(StepStartedEvent(session_id="dep-1", step=dependency.steps[0]))
    dependency.steps[0].sub_steps[0].status = StepStatus.PASSED
    store.handle(StepFinishedEvent(session_id="dep-1", step=dependency.steps[0].sub_steps[0]))

    plain = _render_plain(compose_layout(store.snapshot(), height=16, width=90), width=90)

    assert "› dependency_login" in plain
    assert "Action name" in plain
    assert "FIND thing" not in plain
    assert "DO click" not in plain
    assert "." in plain


def test_main_section_keeps_indented_substeps_for_active_test():
    store = RunStateStore(display_mode=DisplayMode.VERBOSE)
    session = TestSession(
        id="main-1",
        test_name="Main Test",
        file_stem="main_test",
        url="http://example.com",
        viewport_name="Desktop",
        viewport_slug="desktop",
        steps=[
            TestStep(
                id="parent-1",
                instruction="Click on X",
                expectation="Expect Y",
                sub_steps=[
                    TestStep(id="sub-1", instruction="FIND: Item"),
                    TestStep(id="sub-2", instruction="DO: Click"),
                    TestStep(id="sub-3", instruction="VERIFY: Visible state"),
                ],
            )
        ],
    )
    store.handle(
        TopLevelTestStartedEvent(
            owner_key="main",
            test_name="Main Test",
            file_stem="main_test",
            display_path="tests/main_test.yaml",
            expected_dependency_total=0,
        )
    )
    store.handle(SessionStartedEvent(owner_key="main", session=session))
    parent = session.steps[0]
    store.handle(StepStartedEvent(session_id="main-1", step=parent))
    session.steps[0].sub_steps[0].status = StepStatus.PASSED
    session.steps[0].sub_steps[1].status = StepStatus.PASSED
    store.handle(StepFinishedEvent(session_id="main-1", step=session.steps[0].sub_steps[0]))
    store.handle(StepFinishedEvent(session_id="main-1", step=session.steps[0].sub_steps[1]))

    plain = _render_plain(compose_layout(store.snapshot(), height=16, width=90), width=90)

    assert "Click on X → Expect Y" in plain
    assert "  ✔ FIND Item" in plain
    assert "  ✔ DO Click" in plain


def test_main_section_only_expands_latest_active_parent_across_viewports():
    store = RunStateStore(display_mode=DisplayMode.VERBOSE)
    desktop_session = TestSession(
        id="desktop-main",
        test_name="Auth Flow",
        file_stem="auth_flow",
        url="http://example.com",
        viewport_name="Desktop",
        viewport_slug="desktop",
        steps=[
            TestStep(
                id="parent-1",
                instruction="Type wrong credentials",
                sub_steps=[
                    TestStep(id="p1-sub-1", instruction="FIND: username field"),
                    TestStep(id="p1-sub-2", instruction="DO: type 'wrong_user'"),
                ],
            ),
            TestStep(
                id="parent-2",
                instruction="Click the Submit button",
                expectation="Success state should occur",
                sub_steps=[
                    TestStep(id="p2-sub-1", instruction="FIND: Submit button"),
                    TestStep(id="p2-sub-2", instruction="DO: click"),
                ],
            ),
        ],
    )
    mobile_session = TestSession(
        id="mobile-main",
        test_name="Auth Flow",
        file_stem="auth_flow",
        url="http://example.com",
        viewport_name="Mobile",
        viewport_slug="mobile",
        steps=[
            TestStep(
                id="parent-1",
                instruction="Type wrong credentials",
                sub_steps=[
                    TestStep(id="p1-sub-1", instruction="FIND: username field"),
                    TestStep(id="p1-sub-2", instruction="DO: type 'wrong_user'"),
                ],
            ),
            TestStep(
                id="parent-2",
                instruction="Click the Submit button",
                expectation="Success state should occur",
                sub_steps=[
                    TestStep(id="p2-sub-1", instruction="FIND: Submit button"),
                    TestStep(id="p2-sub-2", instruction="DO: click"),
                ],
            ),
        ],
    )
    store.handle(
        TopLevelTestStartedEvent(
            owner_key="auth",
            test_name="Auth Flow",
            file_stem="auth_flow",
            display_path="tests/auth_flow.yaml",
            expected_dependency_total=0,
        )
    )
    store.handle(SessionStartedEvent(owner_key="auth", session=desktop_session))
    store.handle(SessionStartedEvent(owner_key="auth", session=mobile_session))

    desktop_parent_one = desktop_session.steps[0]
    desktop_parent_two = desktop_session.steps[1]
    mobile_parent_one = mobile_session.steps[0]

    store.handle(StepStartedEvent(session_id="desktop-main", step=desktop_parent_one))
    desktop_parent_one.sub_steps[0].status = StepStatus.PASSED
    desktop_parent_one.sub_steps[1].status = StepStatus.PASSED
    store.handle(StepFinishedEvent(session_id="desktop-main", step=desktop_parent_one.sub_steps[0]))
    store.handle(StepFinishedEvent(session_id="desktop-main", step=desktop_parent_one.sub_steps[1]))
    desktop_parent_one.status = StepStatus.PASSED
    store.handle(StepFinishedEvent(session_id="desktop-main", step=desktop_parent_one))

    store.handle(StepStartedEvent(session_id="mobile-main", step=mobile_parent_one))
    store.handle(StepStartedEvent(session_id="desktop-main", step=desktop_parent_two))
    desktop_parent_two.sub_steps[0].status = StepStatus.PASSED
    store.handle(StepFinishedEvent(session_id="desktop-main", step=desktop_parent_two.sub_steps[0]))

    plain = _render_plain(compose_layout(store.snapshot(), height=18, width=100), width=100)

    assert "› Type wrong credentials" in plain
    assert "› Click the Submit button → Success state should occur" in plain
    assert "  ✔ FIND Submit button" in plain
    assert "DO type 'wrong_user'" not in plain
    assert plain.count("FIND username field") == 0


def test_main_section_hides_child_rows_from_inactive_viewport_for_active_parent():
    store = RunStateStore(display_mode=DisplayMode.VERBOSE)
    desktop_session = TestSession(
        id="desktop-main",
        test_name="Auth Flow",
        file_stem="auth_flow",
        url="http://example.com",
        viewport_name="Desktop",
        viewport_slug="desktop",
        steps=[
            TestStep(
                id="parent-1",
                instruction="Click the Submit button",
                expectation="Success state should occur",
                sub_steps=[
                    TestStep(id="sub-1", instruction="FIND: Submit button"),
                    TestStep(id="sub-2", instruction="DO: click"),
                    TestStep(id="sub-3", instruction="VERIFY: Success state should occur"),
                ],
            )
        ],
    )
    mobile_session = TestSession(
        id="mobile-main",
        test_name="Auth Flow",
        file_stem="auth_flow",
        url="http://example.com",
        viewport_name="Mobile",
        viewport_slug="mobile",
        steps=[
            TestStep(
                id="parent-1",
                instruction="Click the Submit button",
                expectation="Success state should occur",
                sub_steps=[
                    TestStep(id="sub-1", instruction="FIND: Submit button"),
                    TestStep(id="sub-2", instruction="DO: click"),
                    TestStep(id="sub-3", instruction="VERIFY: Success state should occur"),
                ],
            )
        ],
    )
    store.handle(
        TopLevelTestStartedEvent(
            owner_key="auth",
            test_name="Auth Flow",
            file_stem="auth_flow",
            display_path="tests/auth_flow.yaml",
            expected_dependency_total=0,
        )
    )
    store.handle(SessionStartedEvent(owner_key="auth", session=desktop_session))
    store.handle(SessionStartedEvent(owner_key="auth", session=mobile_session))

    mobile_parent = mobile_session.steps[0]
    desktop_parent = desktop_session.steps[0]

    store.handle(StepStartedEvent(session_id="mobile-main", step=mobile_parent))
    mobile_parent.sub_steps[0].status = StepStatus.FAILED
    mobile_parent.sub_steps[1].status = StepStatus.SKIPPED
    mobile_parent.sub_steps[2].status = StepStatus.SKIPPED
    store.handle(StepFinishedEvent(session_id="mobile-main", step=mobile_parent.sub_steps[0]))
    store.handle(StepFinishedEvent(session_id="mobile-main", step=mobile_parent.sub_steps[1]))
    store.handle(StepFinishedEvent(session_id="mobile-main", step=mobile_parent.sub_steps[2]))
    mobile_parent.status = StepStatus.FAILED
    store.handle(StepFinishedEvent(session_id="mobile-main", step=mobile_parent))

    store.handle(StepStartedEvent(session_id="desktop-main", step=desktop_parent))

    plain = _render_plain(compose_layout(store.snapshot(), height=18, width=100), width=100)

    assert "› Click the Submit button → Success state should occur   [Desktop]" in plain
    assert "FIND Submit button" not in plain
    assert "DO click" not in plain
    assert "VERIFY Success state should occur" not in plain


def test_main_section_hides_skipped_future_rows_after_failure():
    store = RunStateStore(display_mode=DisplayMode.VERBOSE)
    session = TestSession(
        id="main-1",
        test_name="Auth Flow",
        file_stem="auth_flow_fail",
        url="http://example.com",
        viewport_name="Desktop",
        viewport_slug="desktop",
        steps=[
            TestStep(
                id="parent-1",
                instruction="Type 'wrong_user' into the username field",
                sub_steps=[
                    TestStep(id="p1-sub-1", instruction="FIND: username field"),
                    TestStep(id="p1-sub-2", instruction="DO: type 'wrong_user'"),
                ],
            ),
            TestStep(
                id="parent-2",
                instruction="Click the Submit button",
                expectation="The 'Sign in' modal should close and a 'Login Successful' alert or state change should occur",
                sub_steps=[
                    TestStep(id="p2-sub-1", instruction="FIND: Submit button"),
                    TestStep(id="p2-sub-2", instruction="DO: click"),
                    TestStep(id="p2-sub-3", instruction="VERIFY: 'Sign in' modal should close"),
                    TestStep(
                        id="p2-sub-4",
                        instruction="VERIFY: 'Login Successful' alert or state change should occur",
                    ),
                ],
            ),
            TestStep(
                id="parent-3",
                instruction="Open the dashboard",
                sub_steps=[
                    TestStep(id="p3-sub-1", instruction="FIND: dashboard link"),
                    TestStep(id="p3-sub-2", instruction="DO: click"),
                ],
            ),
        ],
    )
    store.handle(
        TopLevelTestStartedEvent(
            owner_key="auth-fail",
            test_name="Auth Flow",
            file_stem="auth_flow_fail",
            display_path="tests/auth_flow_fail.yaml",
            expected_dependency_total=0,
        )
    )
    store.handle(SessionStartedEvent(owner_key="auth-fail", session=session))

    first_parent = session.steps[0]
    second_parent = session.steps[1]
    future_parent = session.steps[2]

    store.handle(StepStartedEvent(session_id="main-1", step=first_parent))
    first_parent.sub_steps[0].status = StepStatus.PASSED
    first_parent.sub_steps[1].status = StepStatus.PASSED
    store.handle(StepFinishedEvent(session_id="main-1", step=first_parent.sub_steps[0]))
    store.handle(StepFinishedEvent(session_id="main-1", step=first_parent.sub_steps[1]))
    first_parent.status = StepStatus.PASSED
    store.handle(StepFinishedEvent(session_id="main-1", step=first_parent))

    store.handle(StepStartedEvent(session_id="main-1", step=second_parent))
    second_parent.sub_steps[0].status = StepStatus.PASSED
    second_parent.sub_steps[1].status = StepStatus.PASSED
    second_parent.sub_steps[2].status = StepStatus.FAILED
    second_parent.sub_steps[2].failure_reason = "Negation failure"
    second_parent.sub_steps[3].status = StepStatus.SKIPPED
    store.handle(StepFinishedEvent(session_id="main-1", step=second_parent.sub_steps[0]))
    store.handle(StepFinishedEvent(session_id="main-1", step=second_parent.sub_steps[1]))
    store.handle(StepFinishedEvent(session_id="main-1", step=second_parent.sub_steps[2]))
    store.handle(StepFinishedEvent(session_id="main-1", step=second_parent.sub_steps[3]))
    second_parent.status = StepStatus.FAILED
    second_parent.failure_reason = "Negation failure"
    store.handle(StepFinishedEvent(session_id="main-1", step=second_parent))

    future_parent.status = StepStatus.SKIPPED
    future_parent.sub_steps[0].status = StepStatus.SKIPPED
    future_parent.sub_steps[1].status = StepStatus.SKIPPED
    store.handle(StepFinishedEvent(session_id="main-1", step=future_parent))
    store.handle(StepFinishedEvent(session_id="main-1", step=future_parent.sub_steps[0]))
    store.handle(StepFinishedEvent(session_id="main-1", step=future_parent.sub_steps[1]))

    plain = _render_plain(compose_layout(store.snapshot(), height=20, width=120), width=120)

    assert "✔ Type 'wrong_user' into the username field" in plain
    assert "✘ Click the Submit button" in plain
    assert "✔ FIND Submit button" in plain
    assert "✔ DO click" in plain
    assert "✘ VERIFY 'Sign in' modal should close" in plain
    assert "VERIFY 'Login Successful' alert or state change should occur" not in plain
    assert "Open the dashboard" not in plain
    assert "dashboard link" not in plain


def test_finished_previous_run_stays_green_without_remaining_steps_while_newer_run_is_active():
    store = RunStateStore(display_mode=DisplayMode.VERBOSE)
    first_session = _session("first-main", "Auth Flow", file_stem="auth_flow")
    first_step = TestStep(id="first-step", instruction="VERIFY: Auth done", status=StepStatus.PASSED)
    first_session.steps = [first_step]

    store.handle(
        TopLevelTestStartedEvent(
            owner_key="auth",
            test_name="Auth Flow",
            file_stem="auth_flow",
            display_path="tests/auth_flow.yaml",
            expected_dependency_total=0,
        )
    )
    store.handle(SessionStartedEvent(owner_key="auth", session=first_session))
    store.handle(StepFinishedEvent(session_id="first-main", step=first_step))
    store.handle(SessionFinishedEvent(session=first_session))

    second_session = _session("second-main", "Checkout", file_stem="checkout")
    second_step = TestStep(id="second-step", instruction="VERIFY: Checkout page")
    second_session.steps = [second_step]
    store.handle(
        TopLevelTestStartedEvent(
            owner_key="checkout",
            test_name="Checkout",
            file_stem="checkout",
            display_path="tests/checkout.yaml",
            expected_dependency_total=1,
        )
    )
    store.handle(SessionStartedEvent(owner_key="checkout", session=second_session))
    store.handle(StepStartedEvent(session_id="second-main", step=second_step))

    snapshot = store.snapshot()
    previous_run = snapshot.top_level_runs[0]
    compact_row = build_compact_run_row(previous_run)

    assert previous_run.summary_status == RunStatus.PASSED
    assert previous_run.remaining_steps == 0
    assert compact_row.style == f"bold {REPORT_GREEN}"
    assert "steps remaining" not in compact_row.plain


def test_failed_previous_run_stays_red_in_compact_summary():
    store = RunStateStore(display_mode=DisplayMode.VERBOSE)
    failed_session = _session("failed-main", "Checkout", file_stem="checkout")
    failed_step = TestStep(id="failed-step", instruction="VERIFY: Checkout page", status=StepStatus.FAILED)
    failed_session.steps = [failed_step]

    store.handle(
        TopLevelTestStartedEvent(
            owner_key="checkout",
            test_name="Checkout",
            file_stem="checkout",
            display_path="tests/checkout.yaml",
            expected_dependency_total=0,
        )
    )
    store.handle(SessionStartedEvent(owner_key="checkout", session=failed_session))
    store.handle(StepFinishedEvent(session_id="failed-main", step=failed_step))
    store.handle(SessionFinishedEvent(session=failed_session))

    compact_row = build_compact_run_row(store.snapshot().top_level_runs[0])

    assert compact_row.style == f"bold {REPORT_RED}"


def test_failed_row_includes_viewport_tag_when_another_lane_continues():
    store = RunStateStore(display_mode=DisplayMode.VERBOSE)
    desktop_session = TestSession(
        id="desktop-main",
        test_name="Auth Flow",
        file_stem="auth_flow",
        url="http://example.com",
        viewport_name="Desktop",
        viewport_slug="desktop",
        steps=[TestStep(id="desktop-step", instruction="VERIFY: Final desktop state")],
    )
    mobile_session = TestSession(
        id="mobile-main",
        test_name="Auth Flow",
        file_stem="auth_flow",
        url="http://example.com",
        viewport_name="Mobile",
        viewport_slug="mobile",
        steps=[
            TestStep(
                id="mobile-failed-parent",
                instruction="Click the primary Login button in the header",
                expectation="A 'Sign In' modal should appear",
                status=StepStatus.FAILED,
            )
        ],
    )
    store.handle(
        TopLevelTestStartedEvent(
            owner_key="auth",
            test_name="Auth Flow",
            file_stem="auth_flow",
            display_path="tests/auth_flow.yaml",
            expected_dependency_total=0,
        )
    )
    store.handle(SessionStartedEvent(owner_key="auth", session=desktop_session))
    store.handle(SessionStartedEvent(owner_key="auth", session=mobile_session))
    store.handle(StepFinishedEvent(session_id="mobile-main", step=mobile_session.steps[0]))
    store.handle(SessionFinishedEvent(session=mobile_session))
    store.handle(StepStartedEvent(session_id="desktop-main", step=desktop_session.steps[0]))

    plain = _render_plain(compose_layout(store.snapshot(), height=16, width=100), width=100)

    assert "✘ Click the primary Login button in the header ➜ A 'Sign In' modal should appear   [Mobile]" in plain


def test_focused_header_includes_failed_viewport_badge_when_another_lane_continues():
    store = RunStateStore(display_mode=DisplayMode.VERBOSE)
    desktop_session = TestSession(
        id="desktop-main",
        test_name="Auth Flow",
        file_stem="auth_flow",
        url="http://example.com",
        viewport_name="Desktop",
        viewport_slug="desktop",
        steps=[TestStep(id="desktop-step", instruction="VERIFY: Final desktop state")],
    )
    mobile_session = TestSession(
        id="mobile-main",
        test_name="Auth Flow",
        file_stem="auth_flow",
        url="http://example.com",
        viewport_name="Mobile",
        viewport_slug="mobile",
        steps=[TestStep(id="mobile-step", instruction="VERIFY: Earlier mobile failure", status=StepStatus.FAILED)],
    )
    store.handle(
        TopLevelTestStartedEvent(
            owner_key="auth",
            test_name="Auth Flow",
            file_stem="auth_flow",
            display_path="tests/auth_flow.yaml",
            expected_dependency_total=0,
        )
    )
    store.handle(SessionStartedEvent(owner_key="auth", session=desktop_session))
    store.handle(SessionStartedEvent(owner_key="auth", session=mobile_session))
    store.handle(StepFinishedEvent(session_id="mobile-main", step=mobile_session.steps[0]))
    store.handle(SessionFinishedEvent(session=mobile_session))
    store.handle(StepStartedEvent(session_id="desktop-main", step=desktop_session.steps[0]))

    plain = _render_plain(compose_layout(store.snapshot(), height=12, width=100), width=100)

    assert "tests/auth_flow.yaml" in plain
    assert "[Mobile] FAIL" in plain


def test_newer_top_level_run_does_not_steal_focus_while_previous_owner_still_runs_in_another_lane():
    store = RunStateStore(display_mode=DisplayMode.VERBOSE)
    auth_desktop = _session(
        "auth-desktop",
        "Auth Flow",
        file_stem="auth_flow",
        viewport_name="Desktop",
        viewport_slug="desktop",
    )
    auth_mobile = _session(
        "auth-mobile",
        "Auth Flow",
        file_stem="auth_flow",
        viewport_name="Mobile",
        viewport_slug="mobile",
    )
    checkout_desktop = _session(
        "checkout-desktop",
        "Checkout",
        file_stem="checkout",
        viewport_name="Desktop",
        viewport_slug="desktop",
    )

    auth_desktop.steps = [
        TestStep(id="auth-desktop-step", instruction="VERIFY: Auth desktop", status=StepStatus.PASSED)
    ]
    auth_mobile.steps = [TestStep(id="auth-mobile-step", instruction="VERIFY: Auth mobile")]
    checkout_desktop.steps = [TestStep(id="checkout-desktop-step", instruction="VERIFY: Checkout desktop")]

    store.handle(
        TopLevelTestStartedEvent(
            owner_key="auth",
            test_name="Auth Flow",
            file_stem="auth_flow",
            display_path="tests/auth_flow.yaml",
            expected_dependency_total=0,
        )
    )
    store.handle(SessionStartedEvent(owner_key="auth", session=auth_desktop))
    store.handle(SessionStartedEvent(owner_key="auth", session=auth_mobile))
    store.handle(StepFinishedEvent(session_id="auth-desktop", step=auth_desktop.steps[0]))
    store.handle(SessionFinishedEvent(session=auth_desktop))
    store.handle(StepStartedEvent(session_id="auth-mobile", step=auth_mobile.steps[0]))

    store.handle(
        TopLevelTestStartedEvent(
            owner_key="checkout",
            test_name="Checkout",
            file_stem="checkout",
            display_path="tests/checkout.yaml",
            expected_dependency_total=1,
        )
    )
    store.handle(SessionStartedEvent(owner_key="checkout", session=checkout_desktop))
    store.handle(StepStartedEvent(session_id="checkout-desktop", step=checkout_desktop.steps[0]))

    snapshot = store.snapshot()

    assert snapshot.focused_owner_key == "auth"


def test_terminal_reporter_fallback_prints_without_live():
    console = Console(record=True, force_terminal=False, width=100)
    reporter = TerminalReporter(console=console, display_mode=DisplayMode.SILENT)

    reporter.handle(
        TopLevelTestStartedEvent(
            owner_key="settings",
            test_name="Settings",
            file_stem="settings",
            display_path="tests/settings.yaml",
            expected_dependency_total=0,
        )
    )
    reporter.handle(
        SessionStartedEvent(owner_key="settings", session=_session("main-1", "Settings", file_stem="settings"))
    )
    reporter.handle(
        StepFinishedEvent(
            session_id="main-1",
            step=TestStep(id="atomic-1", instruction="VERIFY: Settings page", status=StepStatus.PASSED),
        )
    )
    reporter.finalize()

    output = console.export_text()
    assert "tests/settings.yaml" in output
    assert "VERIFY Settings page" in output


def test_terminal_reporter_print_failures_uses_snapshot_sessions():
    console = MagicMock()
    reporter = TerminalReporter(console=console, display_mode=DisplayMode.VERBOSE)

    blocked_session = _session(
        "main-1",
        "Checkout",
        file_stem="checkout",
        viewport_name="mobile",
        viewport_slug="mobile",
        dependency_results=[
            {"name": "Login dependency", "status": "failed", "session_id": "dep-1", "file_stem": "login"}
        ],
    )

    reporter.handle(
        TopLevelTestStartedEvent(
            owner_key="checkout",
            test_name="Checkout",
            file_stem="checkout",
            display_path="tests/checkout.yaml",
            expected_dependency_total=0,
        )
    )
    reporter.handle(SessionStartedEvent(owner_key="checkout", session=blocked_session))
    reporter.handle(SessionBlockedEvent(session_id="main-1", reason="Required dependency failed: Login dependency"))
    reporter.handle(SessionFinishedEvent(session=blocked_session))
    reporter.handle(RunFinishedEvent())
    reporter.print_failures()

    printed = "\n".join(call.args[0] for call in console.print.call_args_list if call.args)
    assert "tests/checkout.yaml [mobile]" in printed
    assert "Required dependency failed: Login dependency" in printed


def test_terminal_reporter_print_failures_verbose_shows_parent_subset_reason_and_omits_skipped_steps():
    console = MagicMock()
    reporter = TerminalReporter(console=console, display_mode=DisplayMode.VERBOSE)

    parent = TestStep(
        id="parent-1",
        instruction="Click the primary Login button in the header",
        expectation="A 'Sign In' modal should appear in the center of the screen",
    )
    passed_substep = TestStep(id="sub-1", instruction="FIND: Login button", status=StepStatus.PASSED)
    failed_substep = TestStep(
        id="sub-2",
        instruction="VERIFY: 'Sign In' modal should close",
        status=StepStatus.FAILED,
        failure_reason="Verification failed for query: ''Sign In' modal should close'",
    )
    skipped_substep = TestStep(
        id="sub-3",
        instruction="VERIFY: 'Login Successful' alert or state change should occur",
        status=StepStatus.SKIPPED,
    )
    parent.sub_steps = [passed_substep, failed_substep, skipped_substep]
    parent.status = StepStatus.FAILED
    parent.failure_reason = failed_substep.failure_reason

    failed_session = TestSession(
        id="main-1",
        test_name="Auth Flow",
        file_stem="auth_flow",
        url="http://example.com",
        viewport_name="Mobile",
        viewport_slug="mobile",
        steps=[parent],
    )

    reporter.handle(
        TopLevelTestStartedEvent(
            owner_key="auth",
            test_name="Auth Flow",
            file_stem="auth_flow",
            display_path="tests/auth_flow.yaml",
            expected_dependency_total=0,
        )
    )
    reporter.handle(SessionStartedEvent(owner_key="auth", session=failed_session))
    reporter.handle(StepStartedEvent(session_id="main-1", step=parent))
    reporter.handle(StepFinishedEvent(session_id="main-1", step=passed_substep))
    reporter.handle(StepFinishedEvent(session_id="main-1", step=failed_substep))
    reporter.handle(StepFinishedEvent(session_id="main-1", step=skipped_substep))
    reporter.handle(StepFinishedEvent(session_id="main-1", step=parent))
    reporter.handle(SessionFinishedEvent(session=failed_session))
    reporter.handle(RunFinishedEvent())
    reporter.print_failures()

    printed = "\n".join(call.args[0] for call in console.print.call_args_list if call.args)
    assert "Failures" in printed
    assert "[bold]Step:[/] Click the primary Login button in the header" in printed
    assert "[bold]Failed on:[/] VERIFY 'Sign In' modal should close" in printed
    assert "[bold]Reason:[/] Verification failed for query:" in printed
    assert "tests/auth_flow.yaml [Mobile]" in printed
    assert "[bold]Failed on:[/] VERIFY 'Login Successful' alert or state change should occur" not in printed
    assert "====" not in printed


def test_terminal_reporter_print_failures_silent_is_compact_and_hides_reason():
    console = MagicMock()
    reporter = TerminalReporter(console=console, display_mode=DisplayMode.SILENT)

    failed_step = TestStep(
        id="step-1",
        instruction="VERIFY: Login successful",
        status=StepStatus.FAILED,
        failure_reason="Verification failed for query: 'Login successful'",
    )
    failed_session = TestSession(
        id="main-1",
        test_name="Login",
        file_stem="login",
        url="http://example.com",
        viewport_name="Mobile",
        viewport_slug="mobile",
        steps=[failed_step],
    )

    reporter.handle(
        TopLevelTestStartedEvent(
            owner_key="login",
            test_name="Login",
            file_stem="login",
            display_path="tests/login.yaml",
            expected_dependency_total=0,
        )
    )
    reporter.handle(SessionStartedEvent(owner_key="login", session=failed_session))
    reporter.handle(StepFinishedEvent(session_id="main-1", step=failed_step))
    reporter.handle(SessionFinishedEvent(session=failed_session))
    reporter.handle(RunFinishedEvent())
    reporter.print_failures()

    printed = "\n".join(call.args[0] for call in console.print.call_args_list if call.args)
    assert "tests/login.yaml [Mobile]" in printed
    assert "Reason:" not in printed
    assert "Failed on:" not in printed


def test_terminal_reporter_print_failures_includes_reason_for_interactive_active_failure():
    console = MagicMock()
    reporter = TerminalReporter(console=console, display_mode=DisplayMode.VERBOSE)

    parent = TestStep(
        id="parent-1",
        instruction="Click the Submit button",
        expectation="The 'Sign in' modal should close and a 'Login Successful' alert or state change should occur",
    )
    failed_substep = TestStep(
        id="sub-3",
        instruction="VERIFY: 'Sign in' modal should close",
        status=StepStatus.FAILED,
        failure_reason="Negation failure: Element remains in the view",
    )
    skipped_substep = TestStep(
        id="sub-4",
        instruction="VERIFY: 'Login Successful' alert or state change should occur",
        status=StepStatus.SKIPPED,
    )
    parent.sub_steps = [failed_substep, skipped_substep]
    parent.status = StepStatus.FAILED
    parent.failure_reason = failed_substep.failure_reason

    failed_session = TestSession(
        id="main-1",
        test_name="Auth Flow Fail",
        file_stem="auth_flow_fail",
        url="http://example.com",
        viewport_name="desktop",
        viewport_slug="desktop",
        steps=[parent],
    )

    reporter.handle(
        TopLevelTestStartedEvent(
            owner_key="auth-fail",
            test_name="Auth Flow Fail",
            file_stem="auth_flow_fail",
            display_path="tests/auth_flow_fail.yaml",
            expected_dependency_total=0,
        )
    )
    reporter.handle(SessionStartedEvent(owner_key="auth-fail", session=failed_session))
    reporter.handle(StepStartedEvent(session_id="main-1", step=parent))
    reporter.handle(StepFinishedEvent(session_id="main-1", step=failed_substep))
    reporter.handle(StepFinishedEvent(session_id="main-1", step=skipped_substep))
    reporter.handle(StepFinishedEvent(session_id="main-1", step=parent))

    reporter.print_failures()

    printed = "\n".join(call.args[0] for call in console.print.call_args_list if call.args)
    assert "tests/auth_flow_fail.yaml [desktop]" in printed
    assert "[bold]Step:[/] Click the Submit button" in printed
    assert "[bold]Failed on:[/] VERIFY 'Sign in' modal should close" in printed
    assert "[bold]Reason:[/] Negation failure: Element remains in the view" in printed
    assert "[bold]Failed on:[/] VERIFY 'Login Successful' alert or state change should occur" not in printed
