"""Progressive rendering helpers for CLI reporting."""

import shutil
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from rich.console import Console, Group
from rich.live import Live
from rich.text import Text

from vizQA.app.memory import StepStatus, TestSession, TestStep

if TYPE_CHECKING:
    from vizQA.app.viewport import ViewportSpec

STEP_STATUS_STYLES = {
    StepStatus.RUNNING: ("▶", "yellow"),
    StepStatus.PASSED: ("✔", "green"),
    StepStatus.FAILED: ("✘", "red"),
    StepStatus.SKIPPED: ("○", "dim"),
    StepStatus.PENDING: ("○", "white"),
}


def format_step_prefix(instr: str) -> Text:
    """Returns a colored prefix Text for a step instruction string."""
    if instr.startswith("FIND:"):
        return Text.assemble(("FIND ", "bold cyan"), (instr[5:].strip(), "white"))
    if instr.startswith("DO:"):
        return Text.assemble(("DO ", "bold magenta"), (instr[3:].strip(), "white"))
    if instr.startswith("VERIFY:"):
        return Text.assemble(("VERIFY ", "bold green"), (instr[7:].strip(), "white"))
    return Text(instr, "white")


def print_session_header(console: Console, session: TestSession) -> None:
    """Prints a session header and dependency chain if present."""
    viewport_note = f" [{session.viewport_name}]" if session.viewport_name else ""
    console.print(f"\n[bold]● {session.test_name}{viewport_note}[/] [dim]({session.id})[/]")
    if session.dependency_results:
        dep_names = " → ".join([d["name"] for d in session.dependency_results])
        console.print(f"[dim]dependencies: {dep_names}[/]")


def print_dependency_failure(console: Console, dependency_name: str) -> None:
    """Prints the standard dependency failure message."""
    console.print(f"[red]✘ Test skipped because required test failed: {dependency_name}[/]")


def _deepest_failed(step: TestStep) -> TestStep:
    for sub in step.sub_steps:
        if sub.status == StepStatus.FAILED:
            return _deepest_failed(sub)
    return step


# pylint: disable=too-many-instance-attributes
class ProgressiveReporter:
    """
    Prints each step to the console as it completes, one line at a time.
    Uses rich.Live to allow in-place updates for parent steps.
    """

    def __init__(self, console: Console, verbosity: int = 0):
        self.console = console
        self.verbosity = verbosity
        self.sessions: List[TestSession] = []
        self._total_sub_steps = 0
        self._completed_sub_steps = 0
        self._live: Optional[Live] = None
        self._renderable_lines: List[Any] = []
        self._lane_positions: Dict[str, int] = {}
        self._lane_labels: Dict[str, str] = {}
        self._lane_scopes: Dict[str, tuple[Any, ...]] = {}
        self._lane_next_slot: Dict[str, int] = {}
        self._lane_parent_slots: Dict[str, Dict[str, int]] = {}
        self._scope_header_lines: Dict[tuple[Any, ...], int] = {}
        self._scope_slot_lines: Dict[tuple[Any, ...], Dict[int, int]] = {}

    def register_session(self, session: TestSession) -> None:
        """Register a session so the reporter can count total sub-steps."""
        self.sessions.append(session)
        for step in session.steps:
            self._total_sub_steps += self._count_atomic(step)

    def _count_atomic(self, step: TestStep) -> int:
        if step.sub_steps:
            return sum(self._count_atomic(s) for s in step.sub_steps)
        return 1

    def _remaining(self) -> int:
        return max(0, self._total_sub_steps - self._completed_sub_steps)

    def _get_footer(self) -> Text:
        remaining = self._remaining()
        if remaining > 0:
            return Text(f"  ▶ {remaining} step{'s' if remaining != 1 else ''} remaining", style="dim")
        return Text("")

    def _update_live(self) -> None:
        if not self._live:
            self._live = Live(
                Group(*self._get_visible_lines(), self._get_footer()),
                console=self.console,
                refresh_per_second=4,
                transient=False,
            )
            self._live.start()
        else:
            self._live.update(Group(*self._get_visible_lines(), self._get_footer()))

    def _get_visible_lines(self) -> List[Any]:
        """Returns a subset of lines if they exceed terminal height to simulate scrolling."""
        term_height = shutil.get_terminal_size().lines
        max_lines = max(5, term_height - 6)
        rendered = [self._render_line(idx) for idx in range(len(self._renderable_lines))]
        if len(rendered) > max_lines:
            return rendered[-max_lines:]
        return rendered

    def _lane_key(self, viewport: Optional["ViewportSpec"] = None, session: Optional[TestSession] = None) -> str:
        if viewport:
            return viewport.slug
        if session and session.viewport_slug:
            return session.viewport_slug
        return "default"

    def _lane_label(self, viewport: Optional["ViewportSpec"] = None, session: Optional[TestSession] = None) -> str:
        if viewport:
            return viewport.name
        if session and session.viewport_name:
            return session.viewport_name
        return "default"

    def _set_lane_position(self, lane_key: str, label: str, line_idx: int) -> None:
        self._lane_positions[lane_key] = line_idx
        self._lane_labels[lane_key] = label
        self._update_live()

    def _session_scope(self, session: TestSession) -> tuple[Any, ...]:
        dep_chain = tuple(dep.get("file_stem", dep.get("name")) for dep in session.dependency_results)
        return (
            session.file_stem or session.test_name,
            session.test_name,
            session.is_dependency,
            dep_chain,
        )

    def _ensure_scope_header(self, scope_key: tuple[Any, ...], title: Text) -> int:
        if scope_key in self._scope_header_lines:
            return self._scope_header_lines[scope_key]

        idx = len(self._renderable_lines)
        self._renderable_lines.append(title)
        self._scope_header_lines[scope_key] = idx
        self._scope_slot_lines[scope_key] = {}
        return idx

    def _get_or_create_scope_slot(self, scope_key: tuple[Any, ...], slot: int, line: Text) -> int:
        scope_lines = self._scope_slot_lines.setdefault(scope_key, {})
        if slot in scope_lines:
            idx = scope_lines[slot]
            self._renderable_lines[idx] = line
            return idx

        idx = len(self._renderable_lines)
        self._renderable_lines.append(line)
        scope_lines[slot] = idx
        return idx

    def _lane_scope(self, lane_key: str) -> tuple[Any, ...]:
        return self._lane_scopes[lane_key]

    def _render_line(self, line_idx: int) -> Text:
        base = Text(self._renderable_lines[line_idx].plain)
        if hasattr(self._renderable_lines[line_idx], "spans"):
            base = self._renderable_lines[line_idx].copy()

        lane_tags = [
            self._lane_labels[lane_key]
            for lane_key, current_idx in self._lane_positions.items()
            if current_idx == line_idx
        ]
        if lane_tags:
            for lane_tag in sorted(lane_tags):
                base.append(f"  [{lane_tag}]", style="bold cyan")
        return base

    def on_step_done(self, step: TestStep, viewport: Optional["ViewportSpec"] = None) -> None:
        """Called when an atomic step finishes."""
        if step.status in (StepStatus.RUNNING, StepStatus.PENDING):
            return

        self._completed_sub_steps += 1
        icon, color = STEP_STATUS_STYLES.get(step.status, ("?", "white"))
        prefix_text = format_step_prefix(step.instruction)
        lane_key = self._lane_key(viewport=viewport)
        scope_key = self._lane_scope(lane_key)
        slot = self._lane_next_slot.get(lane_key, 0)

        line = Text()
        line.append(f"{icon} ", style=color)
        line.append_text(prefix_text)
        if step.expectation:
            line.append(f" → {step.expectation}", style="dim")

        idx = self._get_or_create_scope_slot(scope_key, slot, line)
        self._lane_next_slot[lane_key] = slot + 1
        if viewport:
            self._set_lane_position(
                lane_key,
                self._lane_label(viewport=viewport),
                idx,
            )
        else:
            self._update_live()

    def on_parent_step_start(self, step: TestStep, viewport: Optional["ViewportSpec"] = None) -> None:
        """Called when a container step starts."""
        lane_key = self._lane_key(viewport=viewport)
        scope_key = self._lane_scope(lane_key)
        slot = self._lane_next_slot.get(lane_key, 0)

        line = Text()
        line.append("● ", style="white")
        line.append(step.instruction, style="bold white")
        if step.expectation:
            line.append(f" → {step.expectation}", style="dim")

        idx = self._get_or_create_scope_slot(scope_key, slot, line)
        self._lane_parent_slots.setdefault(lane_key, {})[step.id] = slot
        self._lane_next_slot[lane_key] = slot + 1
        if viewport:
            self._set_lane_position(
                lane_key,
                self._lane_label(viewport=viewport),
                idx,
            )
        else:
            self._update_live()

    def on_session_start(self, session: TestSession) -> None:
        """Called when a test session starts. Displays dependency chain if present."""
        lane_key = self._lane_key(session=session)
        scope_key = self._session_scope(session)
        self._lane_scopes[lane_key] = scope_key
        self._lane_next_slot[lane_key] = 0
        self._lane_parent_slots[lane_key] = {}
        self._lane_labels[lane_key] = self._lane_label(session=session)

        title = Text()
        title.append("● ", style="white")
        title.append(session.test_name, style="bold white")
        if session.dependency_results:
            dep_chain = " → ".join([d["name"] for d in session.dependency_results])
            title.append(f" [dependencies: {dep_chain}]", style="dim")

        idx = self._ensure_scope_header(scope_key, title)
        self._set_lane_position(lane_key, self._lane_label(session=session), idx)

    def on_parent_step_done(self, step: TestStep, viewport: Optional["ViewportSpec"] = None) -> None:
        """Called when a container step finishes — updates its line color in-place."""
        _, color = STEP_STATUS_STYLES.get(step.status, ("?", "white"))
        lane_key = self._lane_key(viewport=viewport)
        scope_key = self._lane_scope(lane_key)
        slot = self._lane_parent_slots.get(lane_key, {}).get(step.id)
        if slot is None:
            slot = self._lane_next_slot.get(lane_key, 0)
            self._lane_next_slot[lane_key] = slot + 1

        line = Text()
        line.append("● ", style=color)
        line.append(step.instruction, style=f"bold {color}")
        if step.expectation:
            line.append(f" → {step.expectation}", style=f"bold {color}")

        idx = self._get_or_create_scope_slot(scope_key, slot, line)
        self._lane_parent_slots.get(lane_key, {}).pop(step.id, None)

        if viewport:
            current_idx = self._lane_positions.get(lane_key)
            if current_idx is None or current_idx < idx:
                self._set_lane_position(
                    lane_key,
                    self._lane_label(viewport=viewport),
                    idx,
                )
            else:
                self._update_live()
        else:
            self._update_live()

    def finalize(self) -> None:
        """Stops the Live display and clears any trailing footer."""
        if self._live:
            self._live.stop()
            self._live = None

    def print_failures(self) -> None:
        """Prints detailed failure block after all sessions complete."""
        failed_sessions = [s for s in self.sessions if any(st.status == StepStatus.FAILED for st in s.steps)]
        if not failed_sessions:
            return

        self.console.print("\n[bold red]" + "=" * 20 + " FAILURES " + "=" * 20 + "[/]")
        for session in failed_sessions:
            for top_step in session.steps:
                if top_step.status == StepStatus.FAILED:
                    failed_step = _deepest_failed(top_step)
                    session_label = session.test_name
                    if session.viewport_name:
                        session_label = f"{session_label} [{session.viewport_name}]"
                    if session.is_dependency:
                        session_label = f"Dependency failure in {session.test_name}"
                        if session.viewport_name:
                            session_label = f"{session_label} [{session.viewport_name}]"

                    self.console.print(f"\n[bold red]FAILURE in {session_label} › {top_step.instruction}[/]")
                    if failed_step != top_step:
                        self.console.print(f"  [bold red]↳ Failed at:[/] {failed_step.instruction}")

                    if failed_step.failure_type and str(failed_step.failure_type) != "FailureType.NONE":
                        self.console.print(f"  [bold]Type:[/] {failed_step.failure_type}")

                    reason = failed_step.failure_reason or failed_step.error
                    if not reason and hasattr(failed_step, "user_message"):
                        reason = failed_step.user_message

                    self.console.print(f"  [bold]Reason:[/] {reason}")

        self.console.print("[bold red]" + "=" * 50 + "[/]")
