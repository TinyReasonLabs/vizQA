"""Terminal reporter facade for the Rich live UI."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from typing import Any

from rich.console import Console
from rich.live import Live

from vizQA.app.memory import FailureType
from vizQA.rendering.layout import compose_layout
from vizQA.rendering.models import DisplayMode, RunStatus
from vizQA.rendering.store import RunStateStore


class TerminalReporter:
    """Own the live renderer and reporting store."""

    def __init__(self, console: Console, display_mode: DisplayMode):
        self.console = console
        self.display_mode = display_mode
        self.store = RunStateStore(display_mode=display_mode)
        self.sessions = self.store._session_lookup  # pylint: disable=protected-access
        self._live: Live | None = None
        self._last_fallback_text = ""

    def handle(self, event: Any) -> None:
        """Consume an event and refresh the display."""

        self.store.handle(event)
        self._render()

    def finalize(self) -> None:
        """Stop live rendering and ensure the last snapshot is visible."""

        if self._live:
            self._live.stop()
            self._live = None
        else:
            self._render(force_fallback=True)

    def print_failures(self) -> None:
        """Print failure blocks after the run completes."""

        snapshot = self.store.snapshot()
        failed_entries: list[_FailureEntry] = []
        for run in snapshot.top_level_runs:
            for session in [*run.dependencies, *run.sessions]:
                if session.status not in (RunStatus.FAILED, RunStatus.BLOCKED):
                    continue
                failed_entries.append(self._build_failure_entry(run.display_path, session))

        if not failed_entries:
            return

        self.console.print("\n[bold red]Failures[/]")
        for entry in failed_entries:
            self.console.print("")
            self.console.print(f"[bold red]✘[/] {entry.label}")
            if self.display_mode == DisplayMode.SILENT:
                continue
            if entry.step:
                self.console.print(f"  [bold]Step:[/] {entry.step}")
            if entry.failed_on:
                self.console.print(f"  [bold]Failed on:[/] {entry.failed_on}")
            if entry.reason:
                self.console.print(f"  [bold]Reason:[/] {entry.reason}")

    def _build_failure_entry(self, display_path: str, session) -> "_FailureEntry":
        label = session.test_name if session.is_dependency else display_path
        if session.is_dependency:
            label = f"Dependency failure in {session.file_stem or label}"
        if session.viewport_name:
            label = f"{label} [{session.viewport_name}]"

        if session.blocked_reason:
            return _FailureEntry(label=label, step=None, failed_on=None, reason=session.blocked_reason)

        if session.failure_step_text:
            return _FailureEntry(
                label=label,
                step=session.failure_parent_text or session.failure_step_text,
                failed_on=session.failure_step_text if session.failure_parent_text else None,
                reason=session.failure_reason or session.failure_step_text,
            )

        failed_rows = [row for row in session.step_rows if row.status == RunStatus.FAILED]
        if not failed_rows:
            return _FailureEntry(label=label, step=None, failed_on=None, reason=str(FailureType.NONE))

        failed_row = max(failed_rows, key=lambda row: row.order)
        parent_row = None
        if failed_row.parent_step_id:
            parent_candidates = [row for row in session.step_rows if row.step_id == failed_row.parent_step_id]
            if parent_candidates:
                parent_row = max(parent_candidates, key=lambda row: row.order)

        step = parent_row.text if parent_row else failed_row.text
        failed_on = failed_row.text if parent_row and failed_row.text != parent_row.text else None
        reason = failed_row.failure_reason or (parent_row.failure_reason if parent_row else None) or failed_row.text
        return _FailureEntry(label=label, step=step, failed_on=failed_on, reason=reason)

    def _render(self, *, force_fallback: bool = False) -> None:
        if not isinstance(self.console, Console):
            return
        renderable = compose_layout(
            self.store.snapshot(),
            height=self._terminal_height(),
            width=getattr(self.console, "width", 120),
        )
        if force_fallback or not self.console.is_terminal:
            self._print_fallback(renderable)
            return
        if not self._live:
            self._live = Live(renderable, console=self.console, refresh_per_second=4, transient=False)
            self._live.start()
            return
        self._live.update(renderable)

    def _print_fallback(self, renderable) -> None:
        fallback_console = Console(record=True, width=self.console.width, force_terminal=False)
        fallback_console.print(renderable)
        text = fallback_console.export_text()
        if text and text != self._last_fallback_text:
            self.console.print(text.rstrip())
            self._last_fallback_text = text

    @staticmethod
    def _terminal_height() -> int:
        return max(12, shutil.get_terminal_size().lines - 4)


@dataclass(slots=True)
class _FailureEntry:
    label: str
    step: str | None
    failed_on: str | None
    reason: str
