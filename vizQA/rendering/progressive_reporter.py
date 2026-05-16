"""Backward-compatible shim for the new terminal reporter stack."""

from __future__ import annotations

from rich.console import Console

from vizQA.app.memory import TestSession, TestStep
from vizQA.rendering.events import SessionStartedEvent, StepFinishedEvent, StepStartedEvent
from vizQA.rendering.models import DisplayMode
from vizQA.rendering.terminal_reporter import TerminalReporter


def print_session_header(console: Console, session: TestSession) -> None:
    """Compatibility helper for legacy callers."""

    viewport_note = f" [{session.viewport_name}]" if session.viewport_name else ""
    console.print(f"\n[bold]● {session.test_name}{viewport_note}[/] [dim]({session.id})[/]")
    if session.dependency_results:
        dep_names = " → ".join([d["name"] for d in session.dependency_results])
        console.print(f"[dim]dependencies: {dep_names}[/]")


def print_dependency_failure(console: Console, dependency_name: str) -> None:
    """Compatibility helper for legacy callers."""

    console.print(f"[red]✘ Test skipped because required test failed: {dependency_name}[/]")


class ProgressiveReporter(TerminalReporter):
    """Compatibility wrapper around the new terminal reporter."""

    def __init__(self, console: Console, verbosity: int = 0):
        display_mode = DisplayMode.VERBOSE if verbosity else DisplayMode.SILENT
        super().__init__(console=console, display_mode=display_mode)

    def register_session(self, session: TestSession) -> None:
        """Legacy no-op retained for test doubles and older callers."""

        setattr(self, "_last_session_id", session.id)

    def on_session_start(self, session: TestSession) -> None:
        """Bridge legacy session-start callbacks into structured events."""

        owner_key = session.file_stem or session.test_name
        self.handle(SessionStartedEvent(owner_key=owner_key, session=session))

    def on_parent_step_start(self, step: TestStep, viewport=None) -> None:
        """Bridge legacy parent-step start callbacks into structured events."""

        del viewport
        session_id = getattr(self, "_last_session_id", "unknown")
        self.handle(StepStartedEvent(session_id=session_id, step=step))

    def on_step_done(self, step: TestStep, viewport=None) -> None:
        """Bridge legacy atomic-step completion callbacks into structured events."""

        del viewport
        session_id = getattr(self, "_last_session_id", "unknown")
        self.handle(StepFinishedEvent(session_id=session_id, step=step))

    def on_parent_step_done(self, step: TestStep, viewport=None) -> None:
        """Bridge legacy parent-step completion callbacks into structured events."""

        del viewport
        session_id = getattr(self, "_last_session_id", "unknown")
        self.handle(StepFinishedEvent(session_id=session_id, step=step))
