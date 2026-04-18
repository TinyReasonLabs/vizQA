"""
Command-line interface for the UI testing framework.
"""

import asyncio
import configparser
import shutil
import tomllib
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
import yaml
from rich.console import Console, Group
from rich.live import Live
from rich.text import Text
from rich.tree import Tree

from vizQA.client import PerceptionClient
from vizQA.core import Automator
from vizQA.dependency_resolver import DependencyResolver
from vizQA.exceptions import TestDefinitionError, UserFacingException
from vizQA.logger import get_logger
from vizQA.memory import StepStatus, TestSession, TestStep
from vizQA.planner import StepPlanner
from vizQA.utility_classes import BrowserStateCache, LineLoader

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
        self._live: Optional[Live] = None
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
        if step.status in (StepStatus.RUNNING, StepStatus.PENDING):
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
            _, color = _STATUS_ICON.get(step.status, ("?", "white"))

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

                    reason = failed_step.failure_reason or failed_step.error
                    if not reason and hasattr(failed_step, "user_message"):
                        reason = failed_step.user_message

                    console.print(f"  [bold]Reason:[/] {reason}")

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

    # Remove duplicates and filter out non-vizQA files
    seen = set()
    unique_files = []
    for f in test_files:
        if f.absolute() not in seen:
            try:
                content = f.read_text(encoding="utf-8")
                # Basic check for vizQA test indicators without full YAML parse if possible
                if "steps:" in content or "url:" in content:
                    unique_files.append(f)
                    seen.add(f.absolute())
            except Exception:  # pylint: disable=broad-exception-caught
                continue

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


def _load_config() -> dict[str, str]:
    """
    Loads global headers from pyproject.toml or .ini files.
    Priority:
    1. pyproject.toml [tool.vizqa.headers]
    2. [.ini files] [vizqa.headers] (pytest.ini, tox.ini, setup.cfg, vizqa.ini)
    """
    headers = {}
    cwd = Path.cwd()

    # 1. pyproject.toml
    pyproject = cwd / "pyproject.toml"
    if pyproject.exists():
        try:
            with open(pyproject, "rb") as f:
                data = tomllib.load(f)
                headers.update(data.get("tool", {}).get("vizqa", {}).get("headers", {}))
        except Exception:  # pylint: disable=broad-exception-caught
            pass

    # 2. .ini files
    ini_files = ["pytest.ini", "tox.ini", "setup.cfg", "vizqa.ini"]
    for ini_name in ini_files:
        ini_path = cwd / ini_name
        if ini_path.exists():
            try:
                config = configparser.ConfigParser()
                config.read(ini_path)
                if "vizqa.headers" in config.sections():
                    headers.update(dict(config["vizqa.headers"]))
            except Exception:  # pylint: disable=broad-exception-caught
                pass

    return {str(k): str(v) for k, v in headers.items()}


def _count_top_level_results(sessions: List[TestSession], total_tests: int) -> tuple[int, int]:
    """
    Count pass/fail totals for user-invoked tests only.

    Dependency sessions are excluded so a single top-level test with multiple
    passing prerequisites does not produce negative failure counts.
    """
    top_level_sessions = [session for session in sessions if not session.is_dependency]
    passed = sum(1 for session in top_level_sessions if all(step.status == StepStatus.PASSED for step in session.steps))
    failed = max(0, total_tests - passed)
    return passed, failed


def _load_test_data(test_path: Path) -> dict[str, Any]:
    """
    Load test data from a YAML file.

    :param test_path: Path to the test file
    :return: Parsed test data
    """
    try:
        test_data = yaml.load(test_path.read_text(), Loader=LineLoader)
    except Exception as err:  # pylint: disable=broad-exception-caught
        raise TestDefinitionError(f"Failed to load test file {test_path.name}", internal_detail=str(err)) from err
    return test_data


def _resolve_dependencies(test_path: Path) -> List[Path]:
    """
    Resolve dependencies for a test file.

    :param test_path: Path to the test file
    :return: List of dependency paths
    """
    try:
        resolver = DependencyResolver(test_path.parent)
        dependency_paths = resolver.resolve(test_path)
    except TestDefinitionError as err:
        console.print(f"[red]Error resolving dependencies: {err.user_message}[/]")
        if err.internal_detail:
            console.print(f"[dim]{err.internal_detail}[/]")
        raise
    return dependency_paths


async def _run_single_dependency(
    dep_path: Path,
    automator: Automator,
    reporter: ProgressiveReporter,
    on_step_update: Optional[Any],
    interactive: bool,
) -> tuple[Dict[str, Any], bool, Dict[str, Any]]:
    """
    Run a single dependency test and return its result.

    :param dep_path: Path to the dependency test file
    :param automator: Automator instance
    :param reporter: ProgressiveReporter instance
    :param on_step_update: Step update callback
    :param interactive: Interactive mode flag
    :return: Tuple of (dependency_result_dict, passed_bool, artifacts_dict)
    """
    dep_test_data = _load_test_data(dep_path)
    dep_name = dep_test_data.get("name", dep_path.stem)
    dep_artifacts = _load_artifacts(dep_test_data.get("artifacts", {}), dep_path)

    result = await run_single_test(
        dep_path,
        automator,
        reporter,
        on_step_update=on_step_update,
        interactive=interactive,
        is_dependency=True,
    )

    last_session = reporter.sessions[-1] if reporter.sessions else None
    dep_session_id = last_session.id if last_session else "unknown"
    dep_status = "passed" if result else "failed"

    dep_result = {
        "name": dep_name,
        "status": dep_status,
        "session_id": dep_session_id,
        "file_stem": dep_path.stem,
    }

    return dep_result, result, dep_artifacts


async def _run_test_dependencies(
    test_path: Path,
    automator: Automator,
    reporter: ProgressiveReporter,
    on_step_update: Optional[Any] = None,
    interactive: bool = False,
) -> tuple[bool, List[Dict[str, Any]], Dict[str, Any]]:
    """
    Resolve and run all dependencies for a test.

    :param test_path: Path to the test file
    :param automator: Automator instance
    :param reporter: ProgressiveReporter instance
    :param on_step_update: Step update callback
    :param interactive: Interactive mode flag
    :return: Tuple of (all_dependencies_passed, dependency_results_list, inherited_artifacts)
    """
    test_data = _load_test_data(test_path)
    requires = test_data.get("requires", [])
    if not requires:
        return True, [], {}

    dependency_paths = _resolve_dependencies(test_path)

    dependency_results: List[Dict[str, Any]] = []
    inherited_artifacts: Dict[str, Any] = {}
    all_passed = True

    # Execute each dependency in order
    for dep_path in dependency_paths:
        dep_result, result, dep_artifacts = await _run_single_dependency(
            dep_path, automator, reporter, on_step_update, interactive
        )
        inherited_artifacts.update(dep_artifacts)
        dependency_results.append(dep_result)

        if not result:
            all_passed = False
            console.print(f"[red]Required test '{dep_result['name']}' failed. Stopping test execution.[/]")
            break  # Stop on first dependency failure

    return all_passed, dependency_results, inherited_artifacts


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals,too-many-statements
async def run_single_test(
    test_path: Path,
    automator: Automator,
    reporter: ProgressiveReporter,
    on_step_update: Optional[Any] = None,
    interactive: bool = False,
    is_dependency: bool = False,
) -> bool:
    """
    Runs a single test file and updates the reporter.

    :param test_path: Path to the test file
    :param automator: Automator instance
    :param reporter: ProgressiveReporter instance
    :param on_step_update: Callback for step updates
    :param interactive: Whether to run in interactive mode
    :param is_dependency: Whether this test is running as a dependency
    :return: True if test passed, False otherwise
    """
    try:
        test_data = yaml.load(test_path.read_text(), Loader=LineLoader)
    except Exception as err:  # pylint: disable=broad-exception-caught
        raise TestDefinitionError(f"Failed to load test file {test_path.name}", internal_detail=str(err)) from err

    # Resolve and run dependencies (only if not already a dependency)
    dependency_results: List[Dict[str, Any]] = []
    inherited_artifacts: Dict[str, Any] = {}

    if not is_dependency:
        deps_passed, dependency_results, inherited_artifacts = await _run_test_dependencies(
            test_path, automator, reporter, on_step_update, interactive
        )
        if not deps_passed:
            # Mark this test as failed due to dependency failure
            planner = StepPlanner(model_name="minilm", parser=automator.parser, minilm=automator.minilm)
            steps = planner.decompose(test_data.get("steps", []))

            session = TestSession(
                id=str(uuid.uuid4())[:8],
                test_name=test_data.get("name", test_path.stem),
                file_stem=test_path.stem,
                url=test_data.get("url", ""),
                steps=steps,
                artifacts={},
                headers={},
                is_dependency=is_dependency,
                dependency_results=dependency_results,
            )
            reporter.register_session(session)

            # Print session header
            reporter.on_session_start(session)
            console.print(f"\n[bold]● {session.test_name}[/] [dim]({session.id})[/]")
            console.print(f"[red]✘ Test skipped because required test failed: {dependency_results[-1]['name']}[/]")

            if interactive:
                reporter.finalize()
                raise click.Abort()

            return False

    # Consolidate model choice: use the parser/minilm from automator
    planner = StepPlanner(model_name="minilm", parser=automator.parser, minilm=automator.minilm)

    steps = planner.decompose(test_data.get("steps", []))

    local_artifacts = _load_artifacts(test_data.get("artifacts", {}), test_path)
    artifacts = {**inherited_artifacts, **local_artifacts}

    # Load and merge headers
    global_headers = _load_config()
    test_headers = test_data.get("headers", {})
    # Test-specific headers override global headers
    merged_headers = {**global_headers, **test_headers}

    session = TestSession(
        id=str(uuid.uuid4())[:8],
        test_name=test_data.get("name", test_path.stem),
        file_stem=test_path.stem,
        url=test_data.get("url", ""),
        steps=steps,
        artifacts=artifacts,
        headers=merged_headers,
        is_dependency=is_dependency,
        dependency_results=dependency_results,
    )
    reporter.register_session(session)

    # Print session header with dependency info if present
    console.print(f"\n[bold]● {session.test_name}[/] [dim]({session.id})[/]")
    if dependency_results:
        dep_names = " → ".join([d["name"] for d in dependency_results])
        console.print(f"[dim]dependencies: {dep_names}[/]")

        latest_dependency = dependency_results[-1]
        latest_state = BrowserStateCache.load(latest_dependency["file_stem"])
        if latest_state:
            await automator.restore_browser_state(latest_state)
            if automator.page:
                if merged_headers:
                    await automator.page.set_extra_http_headers(merged_headers)
                # Rehydrate the page from restored storage/cookies so UI state driven by
                # localStorage/sessionStorage is visible to the next dependent test.
                await automator.page.goto(session.url)

    result = await automator.run_session(session, on_step_update=on_step_update)

    # Capture browser state after successful test
    if result:
        try:
            browser_state = await automator.capture_browser_state()
            session.browser_state = browser_state
            BrowserStateCache.cache(test_path.stem, browser_state)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            console.print(f"[yellow]Warning: Failed to capture browser state: {exc}[/]")

    if interactive and not result:
        reporter.finalize()
        raise click.Abort()

    return result


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# CLI Group Logic
# ---------------------------------------------------------------------------


class DefaultGroup(click.Group):
    """
    Click Group that allows a default command to be invoked if no subcommand is found.
    Positional arguments are passed to the default command.
    """

    def __init__(self, *args, **kwargs):
        self.default_command = kwargs.pop("default_command", "run")
        super().__init__(*args, **kwargs)

    def resolve_command(self, ctx, args):
        try:
            return super().resolve_command(ctx, args)
        except click.UsageError:
            # If command resolve fails, we assume the first arg belongs to the default command
            return self.default_command, self.get_command(ctx, self.default_command), args


@click.group(cls=DefaultGroup, default_command="run", invoke_without_command=True)
@click.version_option()
@click.pass_context
def cli(ctx):
    """
    UI Testing Framework - Vision-Driven Automation
    """
    if ctx.invoked_subcommand is None and not ctx.args:
        click.echo(ctx.get_help())


@cli.command()
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
@click.option(
    "--clean-cache",
    is_flag=True,
    default=False,
    help="Remove all cached browser states before running tests (default: False)",
)
def run(paths: tuple[str, ...], headless: bool, verbose: int, interactive: bool, clean_cache: bool):
    """
    Run UI tests from files or directories.

    :param paths: Paths to the test files or directories.
    :param headless: Whether to run the browser in headless mode.
    :param verbose: Verbosity level for output.
    :param interactive: Whether to run in interactive mode.
    :param clean_cache: Whether to clean cached browser states before running.
    """
    if not paths:
        ctx = click.get_current_context()
        click.echo(ctx.get_help())
        return

    if clean_cache:
        count = BrowserStateCache.clean()
        console.print(f"[dim]Cleaned {count} cached browser states[/]")

    reporter = ProgressiveReporter(verbosity=verbose)
    test_files = discover_test_files(list(paths))

    if not test_files:
        console.print("[yellow]No test files found.[/]")
        return

    async def main():
        client = PerceptionClient()
        automator = Automator(client, verbosity=verbose, headless=headless)
        get_logger()

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
                    test_file,
                    automator,
                    reporter,
                    on_step_update=on_step_update,
                    interactive=interactive,
                )

        except UserFacingException as err:
            reporter.finalize()
            console.print(f"\n[bold red]Error:[/] {err.user_message}")
            if verbose >= 1 and err.internal_detail:
                console.print(f"[dim]Details: {err.internal_detail}[/]")
        except Exception as err:  # pylint: disable=broad-exception-caught
            reporter.finalize()
            console.print(f"[red]An unexpected error occurred during execution: {err}[/]")
            if verbose >= 1:
                traceback.print_exc()
        finally:
            reporter.finalize()
            await automator.stop()

        reporter.print_failures()

        # Summary line
        total = len(test_files)
        passed, failed = _count_top_level_results(reporter.sessions, total)

        summary = Text.assemble(
            ("\nResults: ", "bold"),
            (f"{passed} passed", "green") if passed else "",
            (", " if passed and failed else ""),
            (f"{failed} failed", "red") if failed else "",
            (f" in {total} test{'s' if total != 1 else ''}", "dim"),
        )
        console.print(summary)

    asyncio.run(main())


@cli.command()
@click.option("--token", help="Hugging Face authentication token for private repositories.")
def install(token: Optional[str]):
    """
    Installs required browser binaries and model weights concurrently.
    """
    from vizQA.install import run_install  # pylint:disable=C0415

    asyncio.run(run_install(token=token))


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
