"""Progressive rendering helpers for CLI reporting."""

import shutil
from typing import Any, Dict, List, Optional

from rich.console import Console, Group
from rich.live import Live
from rich.text import Text

from vizQA.app.memory import StepStatus, TestSession, TestStep

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
    console.print(f"\n[bold]● {session.test_name}[/] [dim]({session.id})[/]")
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
        self._parent_map: Dict[str, int] = {}

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
        if len(self._renderable_lines) > max_lines:
            return self._renderable_lines[-max_lines:]
        return self._renderable_lines

    def on_step_done(self, step: TestStep, depth: int = 0) -> None:
        """Called when an atomic step finishes."""
        if step.status in (StepStatus.RUNNING, StepStatus.PENDING):
            return

        self._completed_sub_steps += 1
        icon, color = STEP_STATUS_STYLES.get(step.status, ("?", "white"))
        indent = "  " * depth
        prefix_text = format_step_prefix(step.instruction)

        line = Text()
        line.append(f"{indent}{icon} ", style=color)
        line.append_text(prefix_text)
        if step.expectation:
            line.append(f" → {step.expectation}", style="dim")

        if self.verbosity >= 1 and step.status == StepStatus.FAILED and step.failure_reason:
            line.append(f"\n{indent}  ↳ {step.failure_reason}", style="red dim")

        self._renderable_lines.append(line)
        self._update_live()

    def on_parent_step_start(self, step: TestStep) -> None:
        """Called when a container step starts."""
        line = Text()
        line.append("● ", style="white")
        line.append(step.instruction, style="bold white")
        if step.expectation:
            line.append(f" → {step.expectation}", style="dim")

        self._parent_map[step.id] = len(self._renderable_lines)
        self._renderable_lines.append(line)
        self._update_live()

    def on_session_start(self, session: TestSession) -> None:
        """Called when a test session starts. Displays dependency chain if present."""
        line = Text()
        line.append("● ", style="white")
        line.append(session.test_name, style="bold white")

        if session.dependency_results:
            dep_chain = " → ".join([d["name"] for d in session.dependency_results])
            line.append(f" [dependencies: {dep_chain}]", style="dim")

        self._renderable_lines.append(line)
        self._update_live()

    def on_parent_step_done(self, step: TestStep) -> None:
        """Called when a container step finishes — updates its line color in-place."""
        if step.id in self._parent_map:
            idx = self._parent_map[step.id]
            _, color = STEP_STATUS_STYLES.get(step.status, ("?", "white"))

            line = Text()
            line.append("● ", style=color)
            line.append(step.instruction, style=f"bold {color}")
            if step.expectation:
                line.append(f" → {step.expectation}", style=f"bold {color}")

            self._renderable_lines[idx] = line
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

                    self.console.print(f"\n[bold red]FAILURE in {session.test_name} › {top_step.instruction}[/]")
                    if failed_step != top_step:
                        self.console.print(f"  [bold red]↳ Failed at:[/] {failed_step.instruction}")

                    if failed_step.failure_type and str(failed_step.failure_type) != "FailureType.NONE":
                        self.console.print(f"  [bold]Type:[/] {failed_step.failure_type}")

                    reason = failed_step.failure_reason or failed_step.error
                    if not reason and hasattr(failed_step, "user_message"):
                        reason = failed_step.user_message

                    self.console.print(f"  [bold]Reason:[/] {reason}")

        self.console.print("[bold red]" + "=" * 50 + "[/]")
