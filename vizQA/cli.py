"""
Command-line interface for the UI testing framework.
"""

import asyncio
import shutil
import traceback
import uuid
from pathlib import Path
from typing import Any, List, Optional

import click
import yaml
from rich.console import Console, Group
from rich.live import Live
from rich.text import Text
from rich.tree import Tree

from vizQA.client import PerceptionClient
from vizQA.core import Automator
from vizQA.logger import get_logger
from vizQA.memory import StepStatus, TestSession, TestStep
from vizQA.planner import StepPlanner

console = Console(highlight=False)

# ---------------------------------------------------------------------------
# Status icon / color helpers
# ---------------------------------------------------------------------------

_STATUS_ICON = {
    StepStatus.RUNNING: ("▶", "yellow"),
    StepStatus.PASSED: ("✔", "green"),
    StepStatus.FAILED: ("✘", "red"),
    StepStatus.SKIPPED: ("○", "dim"),
    StepStatus.PENDING: ("○", "white"),
}


def _step_prefix(instr: str) -> Text:
    """Returns a coloured prefix Text for a step instruction string."""
    if instr.startswith("FIND:"):
        return Text.assemble(("FIND ", "bold cyan"), (instr[5:].strip(), "white"))
    if instr.startswith("DO:"):
        return Text.assemble(("DO ", "bold magenta"), (instr[3:].strip(), "white"))
    if instr.startswith("VERIFY:"):
        return Text.assemble(("VERIFY ", "bold green"), (instr[7:].strip(), "white"))
    return Text(instr, "white")


# ---------------------------------------------------------------------------
# Progressive renderer
# ---------------------------------------------------------------------------


class ProgressiveReporter:
    """
    Prints each step to the console as it completes, one line at a time.
    Uses rich.Live to allow in-place updates for parent steps.
    """

    def __init__(self, verbosity: int = 0):
        self.verbosity = verbosity
        self.sessions: List[TestSession] = []
        self._total_sub_steps = 0
        self._completed_sub_steps = 0
        self._live: Optional[rich.live.Live] = None
        self._renderable_lines: List[Any] = []
        self._parent_map: Dict[str, int] = {}  # maps step.id to line index

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

    def _update_live(self):
        if not self._live:
            self._live = Live(
                Group(*self._get_visible_lines(), self._get_footer()),
                console=console,
                refresh_per_second=4,
                transient=False,
            )
            self._live.start()
        else:
            self._live.update(Group(*self._get_visible_lines(), self._get_footer()))

    def _get_visible_lines(self) -> List[Any]:
        """Returns a subset of lines if they exceed terminal height to simulate scrolling."""
        term_height = shutil.get_terminal_size().lines
        # Reserve ~4 lines for header, footer, and padding
        max_lines = max(5, term_height - 6)

        if len(self._renderable_lines) > max_lines:
            return self._renderable_lines[-max_lines:]
        return self._renderable_lines

    def on_step_done(self, step: TestStep, depth: int = 0) -> None:
        """Called when an atomic step finishes."""
        if step.status == StepStatus.RUNNING or step.status == StepStatus.PENDING:
            return

        self._completed_sub_steps += 1
        icon, color = _STATUS_ICON.get(step.status, ("?", "white"))
        indent = "  " * depth
        prefix_text = _step_prefix(step.instruction)

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

    def on_parent_step_done(self, step: TestStep) -> None:
        """Called when a container step finishes — updates its line color in-place."""
        if step.id in self._parent_map:
            idx = self._parent_map[step.id]
            icon, color = _STATUS_ICON.get(step.status, ("?", "white"))

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

        console.print("\n[bold red]" + "=" * 20 + " FAILURES " + "=" * 20 + "[/]")
        for session in failed_sessions:
            for top_step in session.steps:
                if top_step.status == StepStatus.FAILED:
                    failed_step = _deepest_failed(top_step)

                    console.print(f"\n[bold red]FAILURE in {session.test_name} › {top_step.instruction}[/]")
                    if failed_step != top_step:
                        console.print(f"  [bold red]↳ Failed at:[/] {failed_step.instruction}")

                    if failed_step.failure_type and str(failed_step.failure_type) != "FailureType.NONE":
                        console.print(f"  [bold]Type:[/] {failed_step.failure_type}")

                    console.print(f"  [bold]Reason:[/] {failed_step.failure_reason or failed_step.error}")

                    # if failed_step.screenshot_before:
                    #     console.print(f"  [dim]Before screenshot:[/] {failed_step.screenshot_before}")
                    # if failed_step.screenshot_after:
                    #     console.print(f"  [dim]After screenshot:[/] {failed_step.screenshot_after}")
                    # if failed_step.action_screenshot:
                    #     console.print(f"  [dim]Action snapshot:[/] {failed_step.action_screenshot}")

        console.print("[bold red]" + "=" * 50 + "[/]")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _deepest_failed(step: TestStep) -> TestStep:
    for sub in step.sub_steps:
        if sub.status == StepStatus.FAILED:
            return _deepest_failed(sub)
    return step


def discover_test_files(paths: List[str]) -> List[Path]:
    """
    Discovers test files (YAML/YML) from a list of paths.
    Recursively searches directories.
    """
    test_files = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            console.print(f"[yellow]Warning: Path does not exist: {p}[/]")
            continue

        if path.is_file():
            if path.suffix.lower() in (".yaml", ".yml"):
                test_files.append(path)
            else:
                console.print(f"[yellow]Warning: Skipping non-YAML file: {p}[/]")
        elif path.is_dir():
            # Recursive discovery
            test_files.extend(list(path.rglob("*.yaml")) + list(path.rglob("*.yml")))

    # Remove duplicates while preserving order
    seen = set()
    unique_files = []
    for f in test_files:
        if f.absolute() not in seen:
            unique_files.append(f)
            seen.add(f.absolute())

    return unique_files


def _add_step_node(parent_node: Tree, step: TestStep, verbosity: int):
    icon, color = _STATUS_ICON.get(step.status, ("?", "white"))
    instr_text = _step_prefix(step.instruction)
    step_text = Text.assemble((f"{icon} ", color), instr_text)
    step_node = parent_node.add(step_text)
    for sub_step in step.sub_steps:
        _add_step_node(step_node, sub_step, verbosity)


def _load_artifacts(artifacts_data: dict[str, Any], base_path: Path) -> dict[str, Any]:
    """
    Loads artifacts from a dictionary, resolving paths and reading file contents if specified.
    Stores them with type metadata: {"type": "string|content|file|data", "value": ...}
    """
    loaded = {}
    for name, data in artifacts_data.items():
        if isinstance(data, str):
            loaded[name] = {"type": "string", "value": data}
        elif isinstance(data, dict):
            if "file" in data:
                file_path = base_path.parent / data["file"]
                if file_path.exists():
                    loaded[name] = {
                        "type": "content",
                        "value": file_path.read_text(encoding="utf-8"),
                        "source": str(file_path.absolute()),
                    }
                else:
                    console.print(f"[yellow]Warning: Artifact file not found: {file_path}[/]")
                    loaded[name] = None
            elif "path" in data:
                file_path = base_path.parent / data["path"]
                loaded[name] = {"type": "file", "value": str(file_path.absolute())}
            else:
                loaded[name] = {"type": "data", "value": data}
        else:
            loaded[name] = {"type": "data", "value": data}
    return loaded


# ---------------------------------------------------------------------------
# Session runner
# ---------------------------------------------------------------------------


async def run_single_test(
    test_path: Path,
    automator: Automator,
    reporter: ProgressiveReporter,
    on_step_update: Optional[Any] = None,
    interactive: bool = False,
):
    """Runs a single test file and updates the reporter."""
    try:
        test_data = yaml.safe_load(test_path.read_text())
    except Exception as err:  # pylint: disable=broad-exception-caught
        console.print(f"[red]Error loading {test_path}: {err}[/]")
        return

    planner = StepPlanner()
    steps = planner.decompose(test_data.get("steps", []))

    artifacts = _load_artifacts(test_data.get("artifacts", {}), test_path)

    session = TestSession(
        id=str(uuid.uuid4())[:8],
        test_name=test_data.get("name", test_path.stem),
        file_stem=test_path.stem,
        url=test_data.get("url", ""),
        steps=steps,
        artifacts=artifacts,
    )
    reporter.register_session(session)

    # Print session header
    console.print(f"\n[bold]● {session.test_name}[/] [dim]({session.id})[/]")

    result = await automator.run_session(session, on_step_update=on_step_update)
    if interactive and not result:
        reporter.finalize()
        raise click.Abort()


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------


@click.command()
@click.argument("paths", type=click.Path(exists=True), nargs=-1)
@click.option("--headless/--no-headless", default=True, help="Run browser in headless mode (default: True)")
@click.option("-v", "--verbose", count=True, help="Verbosity (-v steps, -vv timing/detail)")
@click.option(
    "-x",
    "--interactive",
    is_flag=True,
    default=False,
    help="Run in interactive mode, stops at the first failing test (default: False)",
)
def cli(paths, headless, verbose, interactive):
    """
    UI Testing Framework - Vision-Driven Automation

    Run UI tests from files or directories.

    :param paths: Paths to the test files or directories.
    :param headless: Whether to run the browser in headless mode.
    :param verbose: Verbosity level for output.
    """
    if not paths:
        ctx = click.get_current_context()
        click.echo(ctx.get_help())
        return

    reporter = ProgressiveReporter(verbosity=verbose)
    test_files = discover_test_files(paths)

    if not test_files:
        console.print("[yellow]No test files found.[/]")
        return

    async def main():
        client = PerceptionClient()
        automator = Automator(client, verbosity=verbose, headless=headless)
        logger = get_logger()

        try:
            await automator.start()

            for test_file in test_files:

                async def on_step_update(step: TestStep):
                    """Called by core after every status change."""
                    if step.sub_steps:
                        # Container step
                        if step.status == StepStatus.RUNNING:
                            reporter.on_parent_step_start(step)
                        elif step.status not in (StepStatus.RUNNING, StepStatus.PENDING):
                            reporter.on_parent_step_done(step)
                    else:
                        reporter.on_step_done(step, depth=1)

                await run_single_test(
                    test_file, automator, reporter, on_step_update=on_step_update, interactive=interactive
                )

        except Exception as err:  # pylint: disable=broad-exception-caught
            reporter.finalize()
            console.print(f"[red]Error during execution: {err}[/]")
            traceback.print_exc()
        finally:
            reporter.finalize()
            await automator.stop()

        reporter.print_failures()

        # Summary line
        total = len(paths)
        passed = sum(1 for s in reporter.sessions if all(st.status == StepStatus.PASSED for st in s.steps))
        failed = total - passed if not interactive else 1

        summary = Text.assemble(
            ("\nResults: ", "bold"),
            (f"{passed} passed", "green") if passed else "",
            (", " if passed and failed else ""),
            (f"{failed} failed", "red") if failed else "",
            (f" in {total} test{'s' if total != 1 else ''}", "dim"),
        )
        console.print(summary)
        console.print(f"[dim]Full log: {logger.log_path}[/]")

    asyncio.run(main())


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
