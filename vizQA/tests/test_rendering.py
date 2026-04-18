from unittest.mock import MagicMock

from rich.text import Text

from vizQA.app.memory import StepStatus, TestSession, TestStep
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
