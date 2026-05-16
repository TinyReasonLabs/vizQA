"""Theme tokens and text helpers for terminal reporting."""

from __future__ import annotations

from rich.text import Text

from vizQA.app.memory import StepStatus
from vizQA.rendering.models import RunStatus

ACCENT_BLUE = "#00d7ff"
REPORT_GREEN = "#38d9a9"
REPORT_RED = "#ff5e74"
PROGRESS_STYLE = f"bold {ACCENT_BLUE}"
VIEWPORT_CURSOR_STYLE = "bold white"
PREREQUISITE_CURSOR_STYLE = "italic white"
VERIFY_STYLE = f"bold {REPORT_GREEN}"
FAILURE_STYLE = REPORT_RED
SUCCESS_STYLE = REPORT_GREEN
FAILURE_BOLD_STYLE = f"bold {REPORT_RED}"
SUCCESS_BOLD_STYLE = f"bold {REPORT_GREEN}"

STEP_STATUS_STYLES = {
    StepStatus.RUNNING: ("›", "white"),
    StepStatus.PASSED: ("✔", SUCCESS_STYLE),
    StepStatus.FAILED: ("✘", FAILURE_STYLE),
    StepStatus.SKIPPED: ("○", "dim"),
    StepStatus.PENDING: ("○", "white"),
}

RUN_STATUS_STYLES = {
    RunStatus.PENDING: ("○", "white"),
    RunStatus.RUNNING: ("›", "white"),
    RunStatus.PASSED: ("✔", SUCCESS_STYLE),
    RunStatus.FAILED: ("✘", FAILURE_STYLE),
    RunStatus.SKIPPED: ("○", "dim"),
    RunStatus.BLOCKED: ("✘", FAILURE_STYLE),
}


def format_step_prefix(instr: str) -> Text:
    """Render FIND/DO/VERIFY prefixes consistently."""

    if instr.startswith("FIND:"):
        return Text.assemble(("FIND ", PROGRESS_STYLE), (instr[5:].strip(), "white"))
    if instr.startswith("DO:"):
        return Text.assemble(("DO ", "bold magenta"), (instr[3:].strip(), "white"))
    if instr.startswith("VERIFY:"):
        return Text.assemble(("VERIFY ", VERIFY_STYLE), (instr[7:].strip(), "white"))
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
