"""Theme tokens and text helpers for terminal reporting."""

from __future__ import annotations

from rich.text import Text

from vizQA.app.memory import StepStatus
from vizQA.rendering.models import RunStatus

ACCENT_BLUE = "#00d7ff"
PROGRESS_STYLE = f"bold {ACCENT_BLUE}"
VIEWPORT_CURSOR_STYLE = f"bold {ACCENT_BLUE}"
PREREQUISITE_CURSOR_STYLE = "italic white"

STEP_STATUS_STYLES = {
    StepStatus.RUNNING: ("›", "white"),
    StepStatus.PASSED: ("✔", "green"),
    StepStatus.FAILED: ("✘", "red"),
    StepStatus.SKIPPED: ("○", "dim"),
    StepStatus.PENDING: ("○", "white"),
}

RUN_STATUS_STYLES = {
    RunStatus.PENDING: ("○", "white"),
    RunStatus.RUNNING: ("›", "white"),
    RunStatus.PASSED: ("✔", "green"),
    RunStatus.FAILED: ("✘", "red"),
    RunStatus.SKIPPED: ("○", "dim"),
    RunStatus.BLOCKED: ("✘", "red"),
}


def format_step_prefix(instr: str) -> Text:
    """Render FIND/DO/VERIFY prefixes consistently."""

    if instr.startswith("FIND:"):
        return Text.assemble(("FIND ", PROGRESS_STYLE), (instr[5:].strip(), "white"))
    if instr.startswith("DO:"):
        return Text.assemble(("DO ", "bold magenta"), (instr[3:].strip(), "white"))
    if instr.startswith("VERIFY:"):
        return Text.assemble(("VERIFY ", "bold green"), (instr[7:].strip(), "white"))
    return Text(instr, "white")


def step_text(instr: str, expectation: str | None = None) -> Text:
    """Return a formatted step label with optional expectation."""

    text = format_step_prefix(instr)
    if expectation:
        text.append(f" → {expectation}", style="dim")
    return text


def status_icon(status: RunStatus) -> tuple[str, str]:
    """Return icon and style for a run status."""

    return RUN_STATUS_STYLES.get(status, ("?", "white"))
