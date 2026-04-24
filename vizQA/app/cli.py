"""
Command-line interface for the UI testing framework.
"""

import asyncio
import configparser
import tomllib
import traceback
import uuid
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import click
import yaml
from rich.console import Console
from rich.text import Text
from rich.tree import Tree

from vizQA.app.client import PerceptionClient
from vizQA.app.core import Automator
from vizQA.app.exceptions import TestDefinitionError, UserFacingException
from vizQA.app.logger import get_logger
from vizQA.app.memory import StepStatus, TestSession, TestStep
from vizQA.app.support.weights import inspect_weight_state
from vizQA.app.viewport import ViewportSpec, load_viewport_config, resolve_viewports
from vizQA.planning import DependencyResolver, StepPlanner
from vizQA.rendering import STEP_STATUS_STYLES, ProgressiveReporter, format_step_prefix, print_dependency_failure
from vizQA.utils import BrowserStateCache, LineLoader

console = Console(highlight=False)
_ARTIFACT_DIR = Path(".vizQA")


def get_package_version() -> str:
    """Return the installed vizQA version, with a repo fallback for local runs."""
    try:
        return pkg_version("vizQA")
    except PackageNotFoundError:
        pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
        with open(pyproject, "rb") as fh:
            return tomllib.load(fh)["project"]["version"]


def _format_weights_version_line(state) -> str:
    if state.installed_revision is None:
        return "weights: missing"

    details = [state.status]
    if state.status != "aligned":
        details.append(f"expected {state.expected_revision}")
    if state.assumed_revision:
        details.append("assumed from missing metadata")
    return f"weights: {state.installed_revision} ({'; '.join(details)})"


def _show_version(ctx: click.Context, _param: click.Option, value: bool) -> None:
    if not value or ctx.resilient_parsing:
        return

    package_version = get_package_version()
    state = inspect_weight_state(package_version=package_version)
    click.echo(f"vizQA {package_version}")
    click.echo(_format_weights_version_line(state))
    ctx.exit()


def _warn_about_weight_state() -> None:
    state = inspect_weight_state(package_version=get_package_version())

    if state.installed_revision is None:
        console.print(
            "[yellow]Warning:[/] model weights were not found in `vizQA/weights/minilm`. "
            "Run `vizqa install` to install them."
        )
        return

    if state.assumed_revision:
        console.print(
            "[yellow]Warning:[/] assuming installed weights version "
            f"{state.installed_revision} because metadata was not found in `vizQA/weights`. "
            "Run `vizqa install` to refresh the metadata."
        )

    if state.status == "older than expected":
        console.print(
            "[yellow]Warning:[/] installed model weights are older than expected "
            f"(installed {state.installed_revision}; expected {state.expected_revision} for vizQA "
            f"{state.package_version}). Run `vizqa install` to align them."
        )
    elif state.status == "newer than expected":
        console.print(
            "[yellow]Warning:[/] installed model weights are newer than expected "
            f"(installed {state.installed_revision}; expected {state.expected_revision} for vizQA "
            f"{state.package_version}). Run `vizqa install` to align them."
        )


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
    icon, color = STEP_STATUS_STYLES.get(step.status, ("?", "white"))
    instr_text = format_step_prefix(step.instruction)
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


def _count_top_level_results(
    sessions: List[TestSession],
    total_tests: int,
    total_viewports: int = 1,
) -> tuple[int, int]:
    """
    Count pass/fail totals for user-invoked tests only.

    Dependency sessions are excluded so a single top-level test with multiple
    passing prerequisites does not produce negative failure counts.
    """
    top_level_sessions = [session for session in sessions if not session.is_dependency]
    grouped: Dict[str, List[TestSession]] = {}
    for session in top_level_sessions:
        key = session.file_stem or session.test_name
        grouped.setdefault(key, []).append(session)

    passed = sum(
        1
        for lane_sessions in grouped.values()
        if len(lane_sessions) == total_viewports
        and all(all(step.status == StepStatus.PASSED for step in session.steps) for session in lane_sessions)
    )
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


def _collect_involved_test_stems(test_files: List[Path]) -> Set[str]:
    """
    Collect all test stems that may produce artifacts in this run.

    Includes top-level tests plus any resolved dependencies.
    """
    stems = {test_path.stem for test_path in test_files}
    for test_path in test_files:
        for dependency_path in _resolve_dependencies(test_path):
            stems.add(dependency_path.stem)
    return stems


# pylint: disable=too-many-branches
def _clean_run_artifacts(test_stems: Set[str], viewport_slugs: Optional[Set[str]] = None) -> Dict[str, int]:
    """
    Remove stale screenshots, related browser-state caches, and prior run logs.

    Screenshot cleanup is scoped to the tests involved in the current run so
    unrelated artifacts remain available for debugging.
    """
    deleted = {"screenshots": 0, "browser_states": 0, "logs": 0}
    if not _ARTIFACT_DIR.exists():
        return deleted

    if viewport_slugs:
        for slug in viewport_slugs:
            for stem in test_stems:
                for screenshot_path in _ARTIFACT_DIR.glob(f"{slug}__{stem}_*.jpg"):
                    if screenshot_path.is_file():
                        screenshot_path.unlink()
                        deleted["screenshots"] += 1

                cache_key = BrowserStateCache.build_cache_key(stem, namespace=slug)
                cache_path = BrowserStateCache.CACHE_DIR / f"{cache_key}.json"
                if cache_path.exists():
                    cache_path.unlink()
                    deleted["browser_states"] += 1

            for log_path in _ARTIFACT_DIR.glob(f"run_*_{slug}.log"):
                if log_path.is_file():
                    log_path.unlink()
                    deleted["logs"] += 1
    else:
        for stem in test_stems:
            for screenshot_path in _ARTIFACT_DIR.glob(f"{stem}_*.jpg"):
                if screenshot_path.is_file():
                    screenshot_path.unlink()
                    deleted["screenshots"] += 1

            cache_path = BrowserStateCache.CACHE_DIR / f"{stem}.json"
            if cache_path.exists():
                cache_path.unlink()
                deleted["browser_states"] += 1

        for log_path in _ARTIFACT_DIR.glob("run_*.log"):
            if log_path.is_file():
                log_path.unlink()
                deleted["logs"] += 1

    return deleted


# pylint: disable=too-many-arguments,too-many-positional-arguments
async def _run_single_dependency(
    dep_path: Path,
    automator: Automator,
    reporter: ProgressiveReporter,
    on_step_update: Optional[Any],
    interactive: bool,
    viewport: Optional[ViewportSpec] = None,
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
        viewport=viewport,
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


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
async def _run_test_dependencies(
    test_path: Path,
    automator: Automator,
    reporter: ProgressiveReporter,
    on_step_update: Optional[Any] = None,
    interactive: bool = False,
    viewport: Optional[ViewportSpec] = None,
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
            dep_path, automator, reporter, on_step_update, interactive, viewport
        )
        inherited_artifacts.update(dep_artifacts)
        dependency_results.append(dep_result)

        if not result:
            all_passed = False
            console.print(f"[red]Required test '{dep_result['name']}' failed. Stopping test execution.[/]")
            break  # Stop on first dependency failure

    return all_passed, dependency_results, inherited_artifacts


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals,too-many-statements,too-many-branches
async def run_single_test(
    test_path: Path,
    automator: Automator,
    reporter: ProgressiveReporter,
    on_step_update: Optional[Any] = None,
    interactive: bool = False,
    is_dependency: bool = False,
    viewport: Optional[ViewportSpec] = None,
) -> bool:
    """
    Runs a single test file and updates the reporter.

    TODO refactor this to smaller functions, reduce nesting

    :param test_path: Path to the test file
    :param automator: Automator instance
    :param reporter: ProgressiveReporter instance
    :param on_step_update: Callback for step updates
    :param interactive: Whether to run in interactive mode
    :param is_dependency: Whether this test is running as a dependency
    :param viewport: ViewportSpec for this test run, if applicable
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
            test_path, automator, reporter, on_step_update, interactive, viewport
        )
        if not deps_passed:
            # Mark this test as failed due to dependency failure
            planner = StepPlanner(
                model_name="minilm", parser=automator.parser, minilm=automator.minilm, logger=automator.logger
            )
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
                viewport_name=viewport.name if viewport else None,
                viewport_slug=viewport.slug if viewport else None,
                viewport_width=viewport.width if viewport else None,
                viewport_height=viewport.height if viewport else None,
            )
            reporter.register_session(session)

            # In the dependency-failure path, prefer the reporter's richer
            # session line so the skipped top-level test appears only once.
            reporter.on_session_start(session)
            print_dependency_failure(console, dependency_results[-1]["name"])

            if interactive:
                reporter.finalize()
                raise click.Abort()

            return False

    # Consolidate model choice: use the parser/minilm from automator
    planner = StepPlanner(
        model_name="minilm", parser=automator.parser, minilm=automator.minilm, logger=automator.logger
    )

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
        viewport_name=viewport.name if viewport else None,
        viewport_slug=viewport.slug if viewport else None,
        viewport_width=viewport.width if viewport else None,
        viewport_height=viewport.height if viewport else None,
    )
    reporter.register_session(session)

    # Initialize the reporter's shared flow state before any step callbacks fire.
    reporter.on_session_start(session)
    if dependency_results:

        latest_dependency = dependency_results[-1]
        if viewport:
            latest_state = BrowserStateCache.load(latest_dependency["file_stem"], namespace=viewport.slug)
        else:
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
            if viewport:
                BrowserStateCache.cache(test_path.stem, browser_state, namespace=viewport.slug)
            else:
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
@click.option(
    "--version",
    is_flag=True,
    is_eager=True,
    expose_value=False,
    callback=_show_version,
    help="Show the vizQA package version and installed weights version.",
)
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
@click.option(
    "--viewport",
    "viewport",
    multiple=True,
    help="Viewport profile name or WIDTHxHEIGHT. Repeat to run a viewport matrix.",
)
def run(
    paths: tuple[str, ...],
    headless: bool,
    verbose: int,
    interactive: bool,
    clean_cache: bool,
    viewport: tuple[str, ...],
):
    """
    Run UI tests from files or directories.

    :param paths: Paths to the test files or directories.
    :param headless: Whether to run the browser in headless mode.
    :param verbose: Verbosity level for output.
    :param interactive: Whether to run in interactive mode.
    :param clean_cache: Whether to clean cached browser states before running.
    :param viewport: Viewport profile names or dimensions to run tests against.
    """
    if not paths:
        ctx = click.get_current_context()
        click.echo(ctx.get_help())
        return

    _warn_about_weight_state()

    if clean_cache:
        count = BrowserStateCache.clean()
        console.print(f"[dim]Cleaned {count} cached browser states[/]")

    viewport_config = load_viewport_config()
    viewports = resolve_viewports(list(viewport), viewport_config)
    reporter = ProgressiveReporter(console=console, verbosity=verbose)
    test_files = discover_test_files(list(paths))

    if not test_files:
        console.print("[yellow]No test files found.[/]")
        return

    involved_test_stems = _collect_involved_test_stems(test_files)
    cleanup_counts = _clean_run_artifacts(involved_test_stems, viewport_slugs={item.slug for item in viewports})

    if verbose >= 1 and any(cleanup_counts.values()):
        console.print(
            "[dim]Cleaned "
            f"{cleanup_counts['screenshots']} screenshots, "
            f"{cleanup_counts['browser_states']} browser states, "
            f"{cleanup_counts['logs']} old run logs[/]"
        )

    async def main():
        async def run_lane(viewport_spec: ViewportSpec) -> bool:
            logger = get_logger(viewport_spec.slug)
            client = PerceptionClient(logger=logger)
            automator = Automator(
                client,
                verbosity=verbose,
                headless=headless,
                viewport=viewport_spec,
                logger=logger,
            )
            lane_passed = True

            try:
                await automator.start()

                for test_file in test_files:

                    async def on_step_update(step: TestStep, lane_viewport: ViewportSpec = viewport_spec):
                        """Called by core after every status change."""
                        if step.sub_steps:
                            if step.status == StepStatus.RUNNING:
                                reporter.on_parent_step_start(step, viewport=lane_viewport)
                            elif step.status not in (StepStatus.RUNNING, StepStatus.PENDING):
                                reporter.on_parent_step_done(step, viewport=lane_viewport)
                        else:
                            reporter.on_step_done(step, viewport=lane_viewport)

                    result = await run_single_test(
                        test_file,
                        automator,
                        reporter,
                        on_step_update=on_step_update,
                        interactive=interactive,
                        viewport=viewport_spec,
                    )
                    lane_passed = lane_passed and result

            except click.Abort:
                lane_passed = False
            except UserFacingException as err:
                lane_passed = False
                console.print(f"\n[bold red]Error ({viewport_spec.name}):[/] {err.user_message}")
                if verbose >= 1 and err.internal_detail:
                    console.print(f"[dim]Details: {err.internal_detail}[/]")
            except Exception as err:  # pylint: disable=broad-exception-caught
                lane_passed = False
                console.print(f"[red]An unexpected error occurred during execution ({viewport_spec.name}): {err}[/]")
                if verbose >= 1:
                    traceback.print_exc()
            finally:
                await automator.stop()

            return lane_passed

        try:
            results = await asyncio.gather(*(run_lane(item) for item in viewports))
        finally:
            reporter.finalize()

        reporter.print_failures()

        # Summary line
        total = len(test_files)
        passed, failed = _count_top_level_results(reporter.sessions, total, len(viewports))

        summary = Text.assemble(
            ("\nResults: ", "bold"),
            (f"{passed} passed", "green") if passed else "",
            (", " if passed and failed else ""),
            (f"{failed} failed", "red") if failed else "",
            (f" in {total} test{'s' if total != 1 else ''}", "dim"),
        )
        console.print(summary)
        return all(results) and (not reporter.sessions or failed == 0)

    success = asyncio.run(main())
    if not success:
        raise click.exceptions.Exit(1)


@cli.command()
@click.option("--token", help="Hugging Face authentication token for private repositories.")
def install(token: Optional[str]):
    """
    Installs required browser binaries and model weights concurrently.
    """
    from vizQA.app.support.install import run_install  # pylint:disable=C0415

    asyncio.run(run_install(token=token))


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
