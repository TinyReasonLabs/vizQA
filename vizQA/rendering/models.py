"""Models for terminal reporting state and snapshots."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class DisplayMode(str, Enum):
    """Supported terminal display modes."""

    VERBOSE = "verbose"
    SILENT = "silent"


class RunStatus(str, Enum):
    """High-level reporting status for sessions and rows."""

    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    BLOCKED = "blocked"


TERMINAL_RUN_STATUSES = {
    RunStatus.PASSED,
    RunStatus.FAILED,
    RunStatus.SKIPPED,
    RunStatus.BLOCKED,
}


@dataclass(slots=True)
class ViewportProgressState:
    """Display state for a viewport marker on a merged row."""

    label: str
    status: RunStatus = RunStatus.PENDING
    active: bool = False
    partial: bool = False
    complete: bool = False


@dataclass(slots=True)
class StepRowState:  # pylint: disable=too-many-instance-attributes
    """One display row inside a session."""

    key: str
    step_id: str
    parent_step_id: str | None
    text: str
    instruction: str
    expectation: str | None
    failure_reason: str | None
    status: RunStatus
    kind: str
    order: int
    depth: int = 0
    viewport_status: dict[str, ViewportProgressState] = field(default_factory=dict)


@dataclass(slots=True)
class SessionViewState:  # pylint: disable=too-many-instance-attributes
    """Tracked reporting state for one executed session."""

    session_id: str
    test_name: str
    file_stem: str | None
    is_dependency: bool
    viewport_name: str | None
    viewport_slug: str | None
    dependency_results: list[dict] = field(default_factory=list)
    status: RunStatus = RunStatus.PENDING
    total_atomic_steps: int = 0
    completed_atomic_steps: int = 0
    blocked_reason: str | None = None
    failure_parent_text: str | None = None
    failure_step_text: str | None = None
    failure_reason: str | None = None
    latest_row_key: str | None = None
    step_rows: list[StepRowState] = field(default_factory=list)

    @property
    def remaining_steps(self) -> int:
        """Return how many atomic steps are still in flight for this session."""

        if self.status in TERMINAL_RUN_STATUSES:
            return 0
        return max(0, self.total_atomic_steps - self.completed_atomic_steps)

    @property
    def lane_key(self) -> str:
        """Return the stable lane identifier used for merged viewport rows."""

        if self.viewport_slug:
            return self.viewport_slug
        return self.session_id

    @property
    def lane_label(self) -> str:
        """Return the human-facing lane label for viewport badges."""

        if self.viewport_name:
            return self.viewport_name
        return "default"


@dataclass(slots=True)
class TopLevelRunIdentity:
    """Shared identifying metadata for a requested top-level test."""

    owner_key: str
    test_name: str
    file_stem: str | None
    display_path: str
    expected_dependency_total: int = 0


@dataclass(slots=True)
class TopLevelRunView(TopLevelRunIdentity):  # pylint: disable=too-many-instance-attributes
    """Snapshot of one requested top-level test and its related sessions."""

    dependencies: list[SessionViewState] = field(default_factory=list)
    sessions: list[SessionViewState] = field(default_factory=list)
    merged_step_rows: list[StepRowState] = field(default_factory=list)

    @property
    def status(self) -> RunStatus:
        """Return the aggregate status across dependency and top-level sessions."""

        statuses = [session.status for session in [*self.dependencies, *self.sessions]]
        if not statuses:
            return RunStatus.PENDING
        if any(status in (RunStatus.FAILED, RunStatus.BLOCKED) for status in statuses):
            return RunStatus.FAILED
        if any(status == RunStatus.RUNNING for status in statuses):
            return RunStatus.RUNNING
        if all(status == RunStatus.PASSED for status in statuses):
            return RunStatus.PASSED
        if any(status == RunStatus.SKIPPED for status in statuses):
            return RunStatus.SKIPPED
        return RunStatus.PENDING

    @property
    def summary_status(self) -> RunStatus:
        """Status used for the compact top-level row."""

        statuses = [session.status for session in self.sessions]
        if not statuses:
            return self.status

        result = RunStatus.PENDING
        if any(status in (RunStatus.FAILED, RunStatus.BLOCKED) for status in statuses):
            result = RunStatus.FAILED
        elif any(status == RunStatus.RUNNING for status in statuses):
            result = RunStatus.RUNNING
        elif all(status == RunStatus.PASSED for status in statuses):
            result = RunStatus.PASSED
        elif all(status in (RunStatus.PASSED, RunStatus.SKIPPED) for status in statuses):
            result = RunStatus.PASSED if any(status == RunStatus.PASSED for status in statuses) else RunStatus.SKIPPED
        elif any(status == RunStatus.SKIPPED for status in statuses):
            result = RunStatus.SKIPPED
        return result

    @property
    def dependency_completed(self) -> int:
        """Return how many dependency sessions reached a terminal state."""

        return sum(
            1
            for session in self.dependencies
            if session.status in (RunStatus.PASSED, RunStatus.FAILED, RunStatus.BLOCKED, RunStatus.SKIPPED)
        )

    @property
    def dependency_total(self) -> int:
        """Return the number of dependencies declared before execution started."""

        return self.expected_dependency_total

    @property
    def remaining_steps(self) -> int:
        """Return remaining atomic steps across top-level sessions only."""

        if self.summary_status in TERMINAL_RUN_STATUSES:
            return 0
        return sum(session.remaining_steps for session in self.sessions)


@dataclass(slots=True)
class RunSnapshot:
    """Full terminal render snapshot."""

    display_mode: DisplayMode
    top_level_runs: list[TopLevelRunView] = field(default_factory=list)
    focused_owner_key: str | None = None
    passed_top_level: int = 0
    failed_top_level: int = 0
    run_finished: bool = False
