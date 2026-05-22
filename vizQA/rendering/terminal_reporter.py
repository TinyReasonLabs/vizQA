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
from vizQA.rendering.theme import FAILURE_BOLD_STYLE


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
            failed_entries.extend(self._build_run_failure_entries(run))

        if not failed_entries:
            return

        self.console.print(f"\n[{FAILURE_BOLD_STYLE}]Failures[/]")
        for entry in failed_entries:
            self.console.print("")
            self.console.print(f"[{FAILURE_BOLD_STYLE}]✘[/] {entry.label}")
            if self.display_mode == DisplayMode.SILENT:
                continue
            if entry.step:
                self.console.print(f"  [bold]Step:[/] {entry.step}")
            if entry.failed_on:
                self.console.print(f"  [bold]Failed on:[/] {entry.failed_on}")
            if entry.reason:
                self.console.print(f"  [bold]Reason:[/] {entry.reason}")

    def _build_run_failure_entries(self, run) -> list["_FailureEntry"]:
        dependency_entries: dict[tuple[str, str | None, str | None, str], _FailureEntry] = {}
        ordered_entries: list[_FailureEntry] = []

        for session in run.dependencies:
            if not self._session_has_failure_details(session):
                continue
            entry = self._build_failure_entry(run.display_path, session, include_viewport=False)
            key = (entry.label, entry.step, entry.failed_on, entry.reason)
            if key not in dependency_entries:
                dependency_entries[key] = entry
                ordered_entries.append(entry)
            dependency_entries[key].viewport_names.update(self._session_viewport_names(session))

        seen_top_level_entries: set[tuple[str, str | None, str | None, str]] = set()
        for session in run.sessions:
            if self._should_skip_redundant_prerequisite_block(run, session):
                continue
            if not self._session_has_failure_details(session):
                continue
            entry = self._build_failure_entry(run.display_path, session)
            key = (entry.label, entry.step, entry.failed_on, entry.reason)
            if key in seen_top_level_entries:
                continue
            seen_top_level_entries.add(key)
            ordered_entries.append(entry)

        for entry in ordered_entries:
            entry.label = self._format_failure_label(entry)

        return ordered_entries

    def _build_failure_entry(self, display_path: str, session, *, include_viewport: bool = True) -> "_FailureEntry":
        label = session.test_name if session.is_dependency else display_path
        if session.is_dependency:
            label = f"Pre-requisite failure in {session.file_stem or label}"
        if include_viewport and session.viewport_name:
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

    @staticmethod
    def _session_has_failure_details(session) -> bool:
        has_failure_details = bool(session.blocked_reason or session.failure_step_text)
        return session.status in (RunStatus.FAILED, RunStatus.BLOCKED) or has_failure_details

    @staticmethod
    def _session_viewport_names(session) -> set[str]:
        return {session.viewport_name} if session.viewport_name else set()

    @staticmethod
    def _format_failure_label(entry: "_FailureEntry") -> str:
        if not entry.viewport_names:
            return entry.label
        viewport_list = ", ".join(sorted(entry.viewport_names))
        return f"{entry.label} [{viewport_list}]"

    @staticmethod
    def _should_skip_redundant_prerequisite_block(run, session) -> bool:
        if session.is_dependency or not session.blocked_reason:
            return False
        prefix = "Required pre-requisite failed: "
        if not session.blocked_reason.startswith(prefix):
            return False

        failed_dependency_name = session.blocked_reason[len(prefix) :]
        failed_results = {
            result.get("name")
            for result in session.dependency_results
            if str(result.get("status", "")).lower() in {"failed", "blocked"}
        }
        if failed_dependency_name not in failed_results:
            return False

        return any(dependency.status in (RunStatus.FAILED, RunStatus.BLOCKED) for dependency in run.dependencies)

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
    viewport_names: set[str] = None

    def __post_init__(self) -> None:
        if self.viewport_names is None:
            self.viewport_names = set()
