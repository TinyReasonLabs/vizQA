"""
Command-line interface for the UI testing framework.
"""

import asyncio
import traceback
import uuid
from pathlib import Path
from typing import Any, List, Optional

import click
import yaml
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text
from rich.tree import Tree

from vizQA.client import PerceptionClient
from vizQA.core import Automator
from vizQA.memory import StepStatus, TestSession, TestStep
from vizQA.planner import StepPlanner

console = Console()


class ModernReporter:
    """
    Handles pretty printing of test execution results to the console.
    """

    def __init__(self, verbosity: int = 0):
        self.sessions: List[TestSession] = []
        self.verbosity = verbosity

    def get_renderable(self):
        """Generates a rich Tree representing the current test sessions."""
        root = Tree("[bold]UI Testing Framework[/]")
        for session in self.sessions:
            steps_passed = all(s.status == StepStatus.PASSED for s in session.steps)
            steps_failed = any(s.status == StepStatus.FAILED for s in session.steps)

            status_color = "green" if steps_passed else "red" if steps_failed else "yellow"
            session_node = root.add(f"[{status_color}]●[/] [bold]{session.test_name}[/] [dim]({session.id})[/]")

            if self.verbosity >= 1:
                for step in session.steps:
                    self._add_step_node(session_node, step)

        return root

    def _add_step_node(self, parent_node: Tree, step: TestStep):
        """Helper to recursively add step nodes to the tree."""
        icon = "○"
        color = "white"
        if step.status == StepStatus.RUNNING:
            icon = "▶"
            color = "yellow"
        elif step.status == StepStatus.PASSED:
            icon = "✔"
            color = "green"
        elif step.status == StepStatus.FAILED:
            icon = "✘"
            color = "red"
        elif step.status == StepStatus.SKIPPED:
            icon = "○"
            color = "dim"

        instr = step.instruction
        if instr.startswith("FIND:"):
            instr_text = Text.assemble(("FIND: ", "bold cyan"), (instr.replace("FIND:", "").strip(), "white"))
        elif instr.startswith("DO:"):
            instr_text = Text.assemble(("DO: ", "bold magenta"), (instr.replace("DO:", "").strip(), "white"))
        elif instr.startswith("VERIFY:"):
            instr_text = Text.assemble(("VERIFY: ", "bold green"), (instr.replace("VERIFY:", "").strip(), "white"))
        else:
            instr_text = Text(instr, "white")

        step_text = Text.assemble(
            (f"{icon} ", color), instr_text, (f" -> {step.expectation}" if step.expectation else "", "dim")
        )

        if step.action_screenshot and self.verbosity >= 1:
            step_text.append(f" [dim][Action Snapshot: {Path(step.action_screenshot).name}][/]")

        step_node = parent_node.add(step_text)

        if self.verbosity >= 2 and step.perception_result:
            # Adding perception data as a JSON snippet under the step
            perception_data = step.perception_result

            # Request context
            request_info = {
                "request": {
                    "image": step.screenshot_before or step.screenshot_after or step.action_screenshot,
                    "query": step.instruction.replace("FIND:", "").replace("VERIFY:", "").strip(),
                }
            }

            # Merge or display request info first
            full_view = {**request_info, "response": perception_data}

            dumped = yaml.dump(full_view, default_flow_style=False).splitlines()
            if len(dumped) > 20:
                dumped = dumped[:20] + ["... (truncated for brevity)"]

            perception_text = Syntax(
                "\n".join(dumped),
                "yaml",
                theme="monokai",
                line_numbers=False,
                word_wrap=True,
            )
            step_node.add(Panel(perception_text, title="Perception Details (v2)", border_style="dim"))

        # Recursively add sub-steps
        for sub_step in step.sub_steps:
            self._add_step_node(step_node, sub_step)

    def print_failures(self):
        """Prints details of all failed steps to the console."""
        failed_sessions = [s for s in self.sessions if any(step.status == StepStatus.FAILED for step in s.steps)]
        if not failed_sessions:
            return

        console.print("\n[bold red]" + "=" * 20 + " FAILURES " + "=" * 20 + "[/]")
        for session in failed_sessions:
            for step in session.steps:
                if step.status == StepStatus.FAILED:
                    console.print(f"\n[bold red]FAILURE in {session.test_name} > {step.instruction}[/]")
                    console.print(f"[dim]Type:[/] {step.failure_type}")
                    console.print(f"[dim]Reason:[/] {step.failure_reason or step.error}")
                    if step.screenshot_before:
                        console.print(f"[dim]Before Screenshot:[/] {step.screenshot_before}")
                    if step.screenshot_after:
                        console.print(f"[dim]After Screenshot:[/] {step.screenshot_after}")
                    if step.action_screenshot:
                        console.print(f"[dim]Action Snapshot:[/] {step.action_screenshot}")

                    if self.verbosity >= 2 and step.perception_result:
                        console.print("[dim]Perception Data (Truncated):[/]")
                        import json

                        perc_str = json.dumps(step.perception_result, indent=2).splitlines()
                        if len(perc_str) > 20:
                            perc_str = perc_str[:20] + ["  ... (truncated)"]
                        console.print("\n".join(perc_str))
        console.print("[bold red]" + "=" * 50 + "[/]")


reporter = ModernReporter()


async def run_single_test(test_path: Path, automator: Automator, on_step_update: Optional[Any] = None):
    """Runs a single test file and updates the reporter."""
    try:
        test_data = yaml.safe_load(test_path.read_text())
    except Exception as err:  # pylint: disable=broad-exception-caught
        console.print(f"[red]Error loading {test_path}: {err}[/]")
        return

    planner = StepPlanner()
    steps = planner.decompose(test_data.get("steps", []))

    session = TestSession(
        id=str(uuid.uuid4())[:8],
        test_name=test_data.get("name", test_path.stem),
        url=test_data.get("url", ""),
        steps=steps,
    )
    reporter.sessions.append(session)

    await automator.run_session(session, on_step_update=on_step_update)


class DefaultGroup(click.Group):
    def parse_args(self, ctx, args):
        if args and args[0] not in self.commands and args[0] not in ("-h", "--help"):
            # Route unknown commands (like paths) to 'run' implicitly
            args.insert(0, "run")
        return super().parse_args(ctx, args)


@click.group(cls=DefaultGroup, invoke_without_command=True)
@click.pass_context
def cli(ctx):
    """UI Testing Framework - Vision-Driven Automation"""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--headless", is_flag=True, default=True, help="Run browser in headless mode")
@click.option("-v", "--verbose", count=True, help="Verbosity level (-v for steps, -vv for perception data)")
def run(path, headless, verbose):
    """
    Run UI tests from a file or directory.

    :param path: Path to the test file or directory.
    :param headless: Whether to run the browser in headless mode.
    :param verbose: Verbosity level for output.
    """
    reporter.verbosity = verbose
    target_path = Path(path)
    test_files = []

    if target_path.is_file():
        test_files = [target_path]
    else:
        test_files = list(target_path.glob("*.yaml")) + list(target_path.glob("*.yml"))

    if not test_files:
        console.print("[yellow]No test files found.[/]")
        return

    async def main():
        client = PerceptionClient()  # Defaults to localhost:8000
        automator = Automator(client, verbosity=reporter.verbosity)

        try:
            with Live(
                reporter.get_renderable(), refresh_per_second=4, transient=False, vertical_overflow="visible"
            ) as live:

                async def on_step_update(_):
                    live.update(reporter.get_renderable())

                await automator.start()
                try:
                    for test_file in test_files:
                        await run_single_test(test_file, automator, on_step_update=on_step_update)
                        live.update(reporter.get_renderable())
                finally:
                    await automator.stop()
        except Exception as err:
            console.print(f"[red]Error during execution: {err}[/]")
            traceback.print_exc()

        reporter.print_failures()

        # Final Summary
        total = len(reporter.sessions)
        passed = sum(1 for s in reporter.sessions if all(st.status == StepStatus.PASSED for st in s.steps))
        failed = total - passed

        summary = Text.assemble(
            ("\nResults: ", "bold"),
            (f"{passed} passed", "green") if passed else "",
            (", " if passed and failed else ""),
            (f"{failed} failed", "red") if failed else "",
            (f" in {total} tests", "dim"),
        )
        console.print(summary)

    asyncio.run(main())


if __name__ == "__main__":
    cli()
