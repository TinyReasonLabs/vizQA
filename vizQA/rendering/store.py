"""State store for structured terminal reporting."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from vizQA.app.memory import StepStatus, TestSession, TestStep
from vizQA.rendering.events import (
    RunFinishedEvent,
    SessionBlockedEvent,
    SessionFinishedEvent,
    SessionStartedEvent,
    StepFinishedEvent,
    StepStartedEvent,
    TopLevelTestStartedEvent,
)
from vizQA.rendering.models import (
    DisplayMode,
    RunSnapshot,
    RunStatus,
    SessionViewState,
    StepRowState,
    TopLevelRunIdentity,
    TopLevelRunView,
    ViewportProgressState,
)


def _count_atomic(step: TestStep) -> int:
    if step.sub_steps:
        return sum(_count_atomic(child) for child in step.sub_steps)
    return 1


def _count_session_atomic_steps(session: TestSession) -> int:
    return sum(_count_atomic(step) for step in session.steps)


def _step_to_run_status(status: StepStatus) -> RunStatus:
    if status == StepStatus.RUNNING:
        return RunStatus.RUNNING
    if status == StepStatus.PASSED:
        return RunStatus.PASSED
    if status == StepStatus.FAILED:
        return RunStatus.FAILED
    if status == StepStatus.SKIPPED:
        return RunStatus.SKIPPED
    return RunStatus.PENDING


def _session_to_run_status(session: TestSession) -> RunStatus:
    if not session.steps:
        return RunStatus.PASSED
    statuses = [step.status for step in session.steps]
    if any(status == StepStatus.FAILED for status in statuses):
        return RunStatus.FAILED
    if any(status == StepStatus.RUNNING for status in statuses):
        return RunStatus.RUNNING
    if all(status == StepStatus.PASSED for status in statuses):
        return RunStatus.PASSED
    if any(status == StepStatus.SKIPPED for status in statuses):
        return RunStatus.SKIPPED
    return RunStatus.PENDING


def _step_row_text(step: TestStep) -> str:
    instr = step.instruction
    if instr.startswith("FIND:"):
        label = f"FIND {instr[5:].strip()}"
    elif instr.startswith("DO:"):
        label = f"DO {instr[3:].strip()}"
    elif instr.startswith("VERIFY:"):
        label = f"VERIFY {instr[7:].strip()}"
    else:
        label = instr
    if step.expectation:
        label = f"{label} → {step.expectation}"
    return label


@dataclass(slots=True)
class _OwnerState:
    """Mutable bookkeeping for one top-level run while execution is active."""

    run: TopLevelRunView
    next_order: int = 0


class RunStateStore:  # pylint: disable=too-many-instance-attributes
    """Consumes events and produces render snapshots."""

    def __init__(self, display_mode: DisplayMode):
        self.display_mode = display_mode
        self._owners: dict[str, _OwnerState] = {}
        self._owner_order: list[str] = []
        self._owner_position: dict[str, int] = {}
        self._session_lookup: dict[str, SessionViewState] = {}
        self._session_owner: dict[str, str] = {}
        self._step_row_keys: dict[str, dict[str, str]] = defaultdict(dict)
        self._signature_counts: dict[str, dict[tuple[str, str], int]] = defaultdict(lambda: defaultdict(int))
        self._step_depths: dict[str, dict[str, int]] = defaultdict(dict)
        self._step_parents: dict[str, dict[str, str | None]] = defaultdict(dict)
        self._step_children: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
        self._run_finished = False
        self.focused_owner_key: str | None = None

    def handle(self, event: Any) -> None:
        """Apply one event to the store."""

        if isinstance(event, TopLevelTestStartedEvent):
            self._register_owner(event)
            if self.focused_owner_key is None:
                self.focused_owner_key = event.owner_key
            self._rebalance_focus()
            return
        if isinstance(event, SessionStartedEvent):
            self._handle_session_started(event)
            return
        if isinstance(event, StepStartedEvent):
            self._handle_step_started(event.session_id, event.step)
            return
        if isinstance(event, StepFinishedEvent):
            self._handle_step_finished(event.session_id, event.step)
            return
        if isinstance(event, SessionBlockedEvent):
            session = self._session_lookup[event.session_id]
            session.status = RunStatus.BLOCKED
            session.blocked_reason = event.reason
            session.completed_atomic_steps = session.total_atomic_steps
            self._rebalance_focus()
            return
        if isinstance(event, SessionFinishedEvent):
            session = self._session_lookup[event.session.id]
            session.status = RunStatus.BLOCKED if session.blocked_reason else _session_to_run_status(event.session)
            session.completed_atomic_steps = session.total_atomic_steps
            self._rebalance_focus()
            return
        if isinstance(event, RunFinishedEvent):
            self._run_finished = True

    def snapshot(self) -> RunSnapshot:
        """Build an immutable snapshot for rendering."""

        runs: list[TopLevelRunView] = []
        passed = 0
        failed = 0
        for owner_key in self._owner_order:
            owner = self._owners[owner_key]
            run = deepcopy(owner.run)
            run.merged_step_rows = self._merge_rows(run)
            if self._run_finished:
                if run.status == RunStatus.PASSED:
                    passed += 1
                elif run.status in (RunStatus.FAILED, RunStatus.BLOCKED):
                    failed += 1
            runs.append(run)
        return RunSnapshot(
            display_mode=self.display_mode,
            top_level_runs=runs,
            focused_owner_key=self.focused_owner_key,
            passed_top_level=passed,
            failed_top_level=failed,
            run_finished=self._run_finished,
        )

    def _register_owner(self, run_identity: TopLevelRunIdentity) -> _OwnerState:
        owner_key = run_identity.owner_key
        if owner_key not in self._owners:
            self._owners[owner_key] = _OwnerState(
                run=TopLevelRunView(
                    owner_key=run_identity.owner_key,
                    test_name=run_identity.test_name,
                    file_stem=run_identity.file_stem,
                    display_path=run_identity.display_path,
                    expected_dependency_total=run_identity.expected_dependency_total,
                )
            )
            self._owner_order.append(owner_key)
            self._owner_position[owner_key] = len(self._owner_order) - 1
        elif run_identity.expected_dependency_total:
            self._owners[owner_key].run.expected_dependency_total = run_identity.expected_dependency_total
        return self._owners[owner_key]

    def _handle_session_started(self, event: SessionStartedEvent) -> None:
        owner = self._owners.get(event.owner_key)
        if owner is None:
            owner = self._register_owner(
                TopLevelRunIdentity(
                    owner_key=event.owner_key,
                    test_name=event.session.test_name,
                    file_stem=event.session.file_stem,
                    display_path=event.session.file_stem or event.session.test_name,
                )
            )
        session_view = SessionViewState(
            session_id=event.session.id,
            test_name=event.session.test_name,
            file_stem=event.session.file_stem,
            is_dependency=event.session.is_dependency,
            viewport_name=event.session.viewport_name,
            viewport_slug=event.session.viewport_slug,
            dependency_results=list(event.session.dependency_results),
            status=RunStatus.RUNNING,
            total_atomic_steps=_count_session_atomic_steps(event.session),
        )
        if event.session.is_dependency:
            owner.run.dependencies.append(session_view)
        else:
            owner.run.sessions.append(session_view)
        self._session_lookup[event.session.id] = session_view
        self._session_owner[event.session.id] = event.owner_key
        self._index_session_steps(event.session)
        self._rebalance_focus()

    def _index_session_steps(self, session: TestSession) -> None:
        def walk(step: TestStep, depth: int, parent_id: str | None) -> None:
            self._step_depths[session.id][step.id] = depth
            self._step_parents[session.id][step.id] = parent_id
            if parent_id:
                self._step_children[session.id][parent_id].append(step.id)
            for child in step.sub_steps:
                walk(child, depth + 1, step.id)

        for top_level_step in session.steps:
            walk(top_level_step, 0, None)

    def _allocate_row_key(self, session_id: str, kind: str, text: str) -> str:
        signature = (kind, text)
        occurrence = self._signature_counts[session_id][signature]
        self._signature_counts[session_id][signature] += 1
        return f"{kind}:{occurrence}:{text}"

    def _get_or_create_row(
        self, session_id: str, step: TestStep, *, kind: str, final_status: RunStatus
    ) -> StepRowState:
        session = self._session_lookup[session_id]
        owner = self._owners[self._session_owner[session_id]]
        row_key = self._step_row_keys[session_id].get(step.id)
        depth = self._step_depths[session_id].get(step.id, 0)
        if row_key is None:
            row_key = self._allocate_row_key(session_id, kind, _step_row_text(step))
            self._step_row_keys[session_id][step.id] = row_key
            row = StepRowState(
                key=row_key,
                step_id=step.id,
                parent_step_id=self._step_parents[session_id].get(step.id),
                text=_step_row_text(step),
                instruction=step.instruction,
                expectation=step.expectation,
                failure_reason=step.failure_reason,
                status=final_status,
                kind=kind,
                order=owner.next_order,
                depth=depth,
            )
            owner.next_order += 1
            session.step_rows.append(row)
            return row
        for row in session.step_rows:
            if row.key == row_key:
                row.status = final_status
                row.parent_step_id = self._step_parents[session_id].get(step.id)
                row.instruction = step.instruction
                row.expectation = step.expectation
                row.failure_reason = step.failure_reason
                return row
        row = StepRowState(
            key=row_key,
            step_id=step.id,
            parent_step_id=self._step_parents[session_id].get(step.id),
            text=_step_row_text(step),
            instruction=step.instruction,
            expectation=step.expectation,
            failure_reason=step.failure_reason,
            status=final_status,
            kind=kind,
            order=owner.next_order,
            depth=depth,
        )
        owner.next_order += 1
        session.step_rows.append(row)
        return row

    def _handle_step_started(self, session_id: str, step: TestStep) -> None:
        session = self._session_lookup[session_id]
        session.status = RunStatus.RUNNING
        row = self._get_or_create_row(session_id, step, kind="parent", final_status=RunStatus.RUNNING)
        session.latest_row_key = row.key
        self._rebalance_focus()

    def _handle_step_finished(self, session_id: str, step: TestStep) -> None:
        session = self._session_lookup[session_id]
        session.status = RunStatus.RUNNING
        kind = "parent" if step.sub_steps else "atomic"
        row = self._get_or_create_row(session_id, step, kind=kind, final_status=_step_to_run_status(step.status))
        session.latest_row_key = row.key
        if row.status == RunStatus.FAILED:
            if row.depth > 0 or session.failure_step_text is None:
                parent_text = None
                if row.parent_step_id:
                    parent_row = next((item for item in session.step_rows if item.step_id == row.parent_step_id), None)
                    parent_text = parent_row.text if parent_row else None
                session.failure_parent_text = parent_text
                session.failure_step_text = row.text
                session.failure_reason = step.failure_reason or row.failure_reason or row.text
        if not step.sub_steps and step.status not in (StepStatus.RUNNING, StepStatus.PENDING):
            session.completed_atomic_steps += 1
        if step.sub_steps and step.status not in (StepStatus.RUNNING, StepStatus.PENDING):
            child_ids = self._step_children[session_id].get(step.id, [])
            child_row_keys = {self._step_row_keys[session_id].get(child_id) for child_id in child_ids}
            session.step_rows = [item for item in session.step_rows if item.key not in child_row_keys]
        self._rebalance_focus()

    def _merge_rows(self, run: TopLevelRunView) -> list[StepRowState]:
        merged: dict[str, StepRowState] = {}
        for session in run.sessions:
            for row in session.step_rows:
                merged_row = merged.setdefault(
                    row.key,
                    StepRowState(
                        key=row.key,
                        step_id=row.step_id,
                        parent_step_id=row.parent_step_id,
                        text=row.text,
                        instruction=row.instruction,
                        expectation=row.expectation,
                        failure_reason=row.failure_reason,
                        status=row.status,
                        kind=row.kind,
                        order=row.order,
                        depth=row.depth,
                    ),
                )
                merged_row.order = min(merged_row.order, row.order)
                merged_row.status = self._merge_status(merged_row.status, row.status)
                if row.failure_reason:
                    merged_row.failure_reason = row.failure_reason
                merged_row.viewport_status[session.lane_key] = ViewportProgressState(
                    label=session.lane_label,
                    status=row.status,
                    active=session.latest_row_key == row.key and session.status == RunStatus.RUNNING,
                    partial=True,
                    complete=row.status in (RunStatus.PASSED, RunStatus.FAILED, RunStatus.SKIPPED),
                )
        return sorted(merged.values(), key=lambda item: item.order)

    @staticmethod
    def _merge_status(current: RunStatus, incoming: RunStatus) -> RunStatus:
        order = {
            # Active work should stay visibly active even if another lane has
            # already finished the same shared row with a terminal result.
            RunStatus.RUNNING: 6,
            RunStatus.FAILED: 5,
            RunStatus.BLOCKED: 5,
            RunStatus.PASSED: 3,
            RunStatus.SKIPPED: 2,
            RunStatus.PENDING: 1,
        }
        return incoming if order[incoming] >= order[current] else current

    def _rebalance_focus(self) -> None:
        """Focus the earliest still-active top-level run."""

        for owner_key in self._owner_order:
            owner = self._owners[owner_key]
            sessions = [*owner.run.dependencies, *owner.run.sessions]
            if any(session.status == RunStatus.RUNNING for session in sessions):
                self.focused_owner_key = owner_key
                return
        if self.focused_owner_key not in self._owners and self._owner_order:
            self.focused_owner_key = self._owner_order[-1]
