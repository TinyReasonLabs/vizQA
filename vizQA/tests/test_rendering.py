from unittest.mock import MagicMock

from rich.text import Text

from vizQA.app.memory import StepStatus, TestSession, TestStep
from vizQA.app.viewport import ViewportSpec
from vizQA.rendering import ProgressiveReporter, format_step_prefix, print_dependency_failure, print_session_header


def test_format_step_prefix_formats_verify_instruction():
    text = format_step_prefix("VERIFY: dashboard visible")

    assert isinstance(text, Text)
    assert text.plain == "VERIFY dashboard visible"


def test_print_session_header_includes_dependency_chain():
    console = MagicMock()
    session = TestSession(
        id="sess-1",
        test_name="Main Flow",
        url="http://example.com",
        dependency_results=[
            {"name": "Login", "status": "passed", "session_id": "dep-1"},
            {"name": "Resume", "status": "passed", "session_id": "dep-2"},
        ],
    )

    print_session_header(console, session)

    assert console.print.call_count == 2
    first_call = console.print.call_args_list[0].args[0]
    second_call = console.print.call_args_list[1].args[0]
    assert "Main Flow" in first_call
    assert "(sess-1)" in first_call
    assert "Login → Resume" in second_call


def test_print_dependency_failure_uses_standard_message():
    console = MagicMock()

    print_dependency_failure(console, "Login MFA")

    assert console.print.call_count == 1
    assert "Login MFA" in console.print.call_args.args[0]


def test_print_failures_marks_dependency_sessions_clearly():
    console = MagicMock()
    reporter = ProgressiveReporter(console=console)
    reporter.sessions = [
        TestSession(
            id="dep-1",
            test_name="Dependency Role Elevation",
            url="http://example.com",
            is_dependency=True,
            steps=[
                TestStep(
                    id="s1",
                    instruction="Click the 'Approve elevation' button",
                    status=StepStatus.FAILED,
                    sub_steps=[
                        TestStep(
                            id="s1-1",
                            instruction="VERIFY: role '{elevated_role}'",
                            status=StepStatus.FAILED,
                            failure_reason="Verification failed after 16.8s",
                        )
                    ],
                )
            ],
        )
    ]

    reporter.print_failures()

    printed = "\n".join(call.args[0] for call in console.print.call_args_list if call.args)
    assert "Dependency failure in Dependency Role Elevation" in printed


def test_print_failures_marks_dependency_sessions_with_viewport_context():
    console = MagicMock()
    reporter = ProgressiveReporter(console=console)
    reporter.sessions = [
        TestSession(
            id="dep-mobile-1",
            test_name="Dependency Password Login",
            url="http://example.com",
            is_dependency=True,
            viewport_name="mobile",
            viewport_slug="mobile",
            steps=[
                TestStep(
                    id="s1",
                    instruction="Click 'Continue to MFA'",
                    status=StepStatus.FAILED,
                    sub_steps=[
                        TestStep(
                            id="s1-1",
                            instruction="VERIFY: 'Complete Multi-factor Authentication' modal",
                            status=StepStatus.FAILED,
                            failure_reason="Verification failed after 6.2s",
                        )
                    ],
                )
            ],
        )
    ]

    reporter.print_failures()

    printed = "\n".join(call.args[0] for call in console.print.call_args_list if call.args)
    assert "Dependency failure in Dependency Password Login [mobile]" in printed
    assert "Viewport:" in printed
    assert "mobile" in printed


def test_progressive_reporter_moves_lane_tag_to_latest_line_only():
    console = MagicMock()
    reporter = ProgressiveReporter(console=console)
    mobile = ViewportSpec(name="mobile", width=390, height=844)
    reporter.on_session_start(
        TestSession(
            id="sess-mobile",
            test_name="Main Flow",
            file_stem="main",
            url="http://example.com",
            viewport_name="mobile",
            viewport_slug="mobile",
        )
    )

    reporter.on_parent_step_start(TestStep(id="s1", instruction="Open drawer"), viewport=mobile)
    reporter.on_step_done(
        TestStep(id="s2", instruction="VERIFY: menu visible", status=StepStatus.PASSED), viewport=mobile
    )

    assert len(reporter._renderable_lines) == 3  # pylint: disable=protected-access
    rendered_header = reporter._render_line(0).plain  # pylint: disable=protected-access
    rendered_first = reporter._render_line(1).plain  # pylint: disable=protected-access
    rendered_second = reporter._render_line(2).plain  # pylint: disable=protected-access
    assert "Main Flow" in rendered_header
    assert "[mobile]" not in rendered_first
    assert "VERIFY menu visible" in rendered_second
    assert "[mobile]" in rendered_second


def test_progressive_reporter_keeps_tags_on_current_lines_for_multiple_viewports():
    console = MagicMock()
    reporter = ProgressiveReporter(console=console)
    mobile = ViewportSpec(name="mobile", width=390, height=844)
    desktop = ViewportSpec(name="desktop", width=1440, height=900)
    reporter.on_session_start(
        TestSession(
            id="sess-mobile",
            test_name="Main Flow",
            file_stem="main",
            url="http://example.com",
            viewport_name="mobile",
            viewport_slug="mobile",
        )
    )
    reporter.on_session_start(
        TestSession(
            id="sess-desktop",
            test_name="Main Flow",
            file_stem="main",
            url="http://example.com",
            viewport_name="desktop",
            viewport_slug="desktop",
        )
    )

    reporter.on_parent_step_start(TestStep(id="s1", instruction="Click on 'Settings'"), viewport=mobile)
    reporter.on_parent_step_start(TestStep(id="s2", instruction="Click on 'Settings'"), viewport=desktop)
    reporter.on_step_done(
        TestStep(id="s3", instruction="VERIFY: Main page header 'Settings'", status=StepStatus.PASSED),
        viewport=mobile,
    )

    assert len(reporter._renderable_lines) == 3  # pylint: disable=protected-access
    rendered_lines = [reporter._render_line(idx).plain for idx in range(3)]  # pylint: disable=protected-access
    assert "[mobile]" not in rendered_lines[1]
    assert "[desktop]" in rendered_lines[1]
    assert "[mobile]" in rendered_lines[2]


def test_progressive_reporter_coalesces_same_step_for_multiple_viewports():
    console = MagicMock()
    reporter = ProgressiveReporter(console=console)
    mobile = ViewportSpec(name="mobile", width=390, height=844)
    desktop = ViewportSpec(name="desktop", width=1440, height=900)
    reporter.on_session_start(
        TestSession(
            id="sess-mobile",
            test_name="Main Flow",
            file_stem="main",
            url="http://example.com",
            viewport_name="mobile",
            viewport_slug="mobile",
        )
    )
    reporter.on_session_start(
        TestSession(
            id="sess-desktop",
            test_name="Main Flow",
            file_stem="main",
            url="http://example.com",
            viewport_name="desktop",
            viewport_slug="desktop",
        )
    )

    reporter.on_step_done(
        TestStep(id="m1", instruction="FIND: 'Settings' in the sidebar", status=StepStatus.PASSED), viewport=mobile
    )
    reporter.on_step_done(
        TestStep(id="d1", instruction="FIND: 'Settings' in the sidebar", status=StepStatus.PASSED),
        viewport=desktop,
    )

    assert len(reporter._renderable_lines) == 2  # pylint: disable=protected-access
    rendered = reporter._render_line(1).plain  # pylint: disable=protected-access
    assert "[mobile]" in rendered
    assert "[desktop]" in rendered


def test_progressive_reporter_allows_same_step_text_to_reappear_after_lanes_move_on():
    console = MagicMock()
    reporter = ProgressiveReporter(console=console)
    mobile = ViewportSpec(name="mobile", width=390, height=844)
    reporter.on_session_start(
        TestSession(
            id="sess-mobile",
            test_name="Main Flow",
            file_stem="main",
            url="http://example.com",
            viewport_name="mobile",
            viewport_slug="mobile",
        )
    )

    reporter.on_step_done(
        TestStep(id="m1", instruction="FIND: 'Settings' in the sidebar", status=StepStatus.PASSED), viewport=mobile
    )
    reporter.on_step_done(TestStep(id="m2", instruction="DO: click", status=StepStatus.PASSED), viewport=mobile)
    reporter.on_session_start(
        TestSession(
            id="sess-mobile-2",
            test_name="Main Flow Two",
            file_stem="main-two",
            url="http://example.com",
            viewport_name="mobile",
            viewport_slug="mobile",
        )
    )
    reporter.on_step_done(
        TestStep(id="m3", instruction="FIND: 'Settings' in the sidebar", status=StepStatus.PASSED), viewport=mobile
    )

    assert len(reporter._renderable_lines) == 5  # pylint: disable=protected-access


def test_progressive_reporter_prints_shared_session_header_once_for_multiple_viewports():
    console = MagicMock()
    reporter = ProgressiveReporter(console=console)

    reporter.on_session_start(
        TestSession(
            id="sess-mobile",
            test_name="Main Flow",
            file_stem="main",
            url="http://example.com",
            viewport_name="mobile",
            viewport_slug="mobile",
        )
    )
    reporter.on_session_start(
        TestSession(
            id="sess-desktop",
            test_name="Main Flow",
            file_stem="main",
            url="http://example.com",
            viewport_name="desktop",
            viewport_slug="desktop",
        )
    )

    assert len(reporter._renderable_lines) == 1  # pylint: disable=protected-access
    rendered = reporter._render_line(0).plain  # pylint: disable=protected-access
    assert "Main Flow" in rendered
    assert "[mobile]" in rendered
    assert "[desktop]" in rendered


def test_progressive_reporter_keeps_cursor_on_latest_atomic_step_after_parent_completes():
    console = MagicMock()
    reporter = ProgressiveReporter(console=console)
    mobile = ViewportSpec(name="mobile", width=390, height=844)

    reporter.on_session_start(
        TestSession(
            id="sess-mobile",
            test_name="Main Flow",
            file_stem="main",
            url="http://example.com",
            viewport_name="mobile",
            viewport_slug="mobile",
        )
    )

    parent = TestStep(id="p1", instruction="Click on 'Overview'", expectation="Dashboard Overview")
    child = TestStep(id="c1", instruction="VERIFY: Dashboard Overview", status=StepStatus.PASSED)

    reporter.on_parent_step_start(parent, viewport=mobile)
    reporter.on_step_done(child, viewport=mobile)
    parent.status = StepStatus.PASSED
    reporter.on_parent_step_done(parent, viewport=mobile)

    parent_line = reporter._render_line(1).plain  # pylint: disable=protected-access
    child_line = reporter._render_line(2).plain  # pylint: disable=protected-access

    assert "[mobile]" not in parent_line
    assert "[mobile]" in child_line


def test_print_failures_includes_viewport_context_for_top_level_failures():
    console = MagicMock()
    reporter = ProgressiveReporter(console=console)
    reporter.sessions = [
        TestSession(
            id="top-1",
            test_name="Settings Dark Mode",
            url="http://example.com",
            viewport_name="mobile",
            viewport_slug="mobile",
            steps=[
                TestStep(
                    id="s1",
                    instruction="Open settings",
                    status=StepStatus.FAILED,
                    failure_reason="Drawer not found",
                )
            ],
        )
    ]

    reporter.print_failures()

    printed = "\n".join(call.args[0] for call in console.print.call_args_list if call.args)
    assert "Settings Dark Mode [mobile]" in printed
