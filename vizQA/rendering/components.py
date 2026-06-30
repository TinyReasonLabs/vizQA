"""Reusable Rich components for terminal reporting."""

from __future__ import annotations

from rich.console import Group
from rich.text import Text

from vizQA.rendering.models import RunSnapshot, RunStatus, SessionViewState, StepRowState, TopLevelRunView
from vizQA.rendering.theme import (
    FAILURE_BOLD_STYLE,
    FAILURE_STYLE,
    PREREQUISITE_CURSOR_STYLE,
    PROGRESS_STYLE,
    SUCCESS_BOLD_STYLE,
    SUCCESS_STYLE,
    VIEWPORT_CURSOR_STYLE,
    format_step_prefix,
    status_icon,
)


def build_progress_bar(completed: int, total: int, *, width: int) -> Text:
    """Build a full-width text progress bar."""

    width = max(8, width)
    if total <= 0:
        return Text("─" * width, style="bright_black")
    filled = min(width, round((completed / total) * width))
    return Text.assemble(("━" * filled, PROGRESS_STYLE), ("─" * (width - filled), "bright_black"))


def build_viewport_markers(row: StepRowState) -> Text:
    """Render moving viewport cursors for the active row only."""

    active = [viewport for viewport in row.viewport_status.values() if viewport.active]
    if not active:
        return Text("")

    text = Text()
    for viewport in active:
        text.append(f"   [{viewport.label}]", style=VIEWPORT_CURSOR_STYLE)
    return text


def build_failed_viewport_markers(row: StepRowState) -> Text:
    """Render viewport labels for terminal failures on merged rows."""

    failed = [
        viewport
        for viewport in row.viewport_status.values()
        if viewport.status in (RunStatus.FAILED, RunStatus.BLOCKED)
    ]
    if not failed:
        return Text("")

    text = Text()
    for viewport in failed:
        text.append(f"   [{viewport.label}]", style=FAILURE_STYLE)
    return text


def _failure_badges(run: TopLevelRunView, *, include_pass: bool = True) -> list[str]:
    badges: list[str] = []
    seen: set[str] = set()
    for session in run.sessions:
        if session.status not in (RunStatus.PASSED, RunStatus.FAILED, RunStatus.BLOCKED):
            continue
        if session.status == RunStatus.PASSED and not include_pass:
            continue
        label = session.viewport_name or "default"
        badge = (
            f"[{label}] FAIL"
            if session.status == RunStatus.FAILED
            else f"[{label}] BLOCKED" if session.status == RunStatus.BLOCKED else "PASS"
        )
        if "FAIL" in badge and "PASS" in seen:
            # replace pass with the specific failure for viewport
            badges = [b for b in badges if b != "PASS"]
            seen.add(badge)
            badges.append(badge)
        # Dont add pass if theres already a viewport with failure
        elif badge not in seen and not (badge == "PASS" and any(b for b in badges if "FAIL" in b)):
            seen.add(badge)
            badges.append(badge)
    return badges


def _append_right_badges(line: Text, badges: list[str], *, width: int | None) -> Text:
    if not badges:
        return line

    badge_text = Text()
    for index, badge in enumerate(badges):
        if index:
            badge_text.append("  ")
        badge_text.append(badge, style="not bold")

    spacing = "   "
    if width:
        pad = max(3, width - len(line.plain) - len(badge_text.plain))
        spacing = " " * pad

    line.append(spacing)
    line.append_text(badge_text)
    return line


def build_step_line(row: StepRowState) -> Text:
    """Render a merged step line."""

    icon, style = status_icon(row.status)
    indent = "  " * row.depth
    line = Text(indent)
    line.append(f"{icon} ", style)
    if row.kind == "atomic":
        line.append_text(format_step_prefix(row.instruction))
        if row.expectation:
            line.append(f" ➜ {row.expectation}", style="white")
    else:
        text_style = (
            "white"
            if row.status == RunStatus.RUNNING
            else (FAILURE_STYLE if row.status == RunStatus.FAILED else "white")
        )
        line.append(row.text, style=text_style)
    line.append_text(build_viewport_markers(row))
    if not any(viewport.active for viewport in row.viewport_status.values()):
        line.append_text(build_failed_viewport_markers(row))
    return line


def _dependency_display_name(session: SessionViewState) -> str:
    return session.file_stem or session.test_name


def _dedup_dependency_sessions(run: TopLevelRunView) -> list[SessionViewState]:
    deduped: dict[str, SessionViewState] = {}
    for session in run.dependencies:
        key = _dependency_display_name(session)
        if key not in deduped:
            deduped[key] = session
            continue
        current = deduped[key]
        if session.status in (RunStatus.FAILED, RunStatus.BLOCKED):
            deduped[key] = session
        elif current.status != RunStatus.PASSED and session.status == RunStatus.PASSED:
            deduped[key] = session
    return list(deduped.values())


# pylint: disable=too-many-locals
def build_dependency_section_lines(run: TopLevelRunView, *, width: int, max_lines: int) -> list[Text]:
    """Render pre-requisite section lines for the focused run."""

    dependency_sessions = _dedup_dependency_sessions(run)
    if not dependency_sessions:
        return []

    total_dependencies = run.dependency_total or len(dependency_sessions)
    header_label = f"Running {total_dependencies} pre-requisite{'s' if total_dependencies != 1 else ''}..."
    completed = sum(
        1
        for session in dependency_sessions
        if session.status in (RunStatus.PASSED, RunStatus.FAILED, RunStatus.BLOCKED)
    )
    header = Text(header_label, style="white")
    progress_width = max(8, width - len(header_label) - 3)
    header.append("   ")
    header.append_text(build_progress_bar(completed, max(1, total_dependencies), width=progress_width))
    lines: list[Text] = [header]

    if max_lines <= 1:
        return lines

    active_dependency = next((session for session in dependency_sessions if session.status == RunStatus.RUNNING), None)
    window_sessions = dependency_sessions[-max_lines:]
    for session in window_sessions:
        if len(lines) >= max_lines:
            break

        style = (
            VIEWPORT_CURSOR_STYLE if active_dependency and session.session_id == active_dependency.session_id else "dim"
        )
        line = Text.assemble(("› ", style), (_dependency_display_name(session), style))
        if session.blocked_reason:
            line.append(f"  {session.blocked_reason}", style=FAILURE_STYLE)
        lines.append(line)

        if active_dependency and session.session_id == active_dependency.session_id:
            action_lines = _build_dependency_actions(session, max_lines=max(1, max_lines - len(lines)))
            for action_line in action_lines:
                if len(lines) >= max_lines:
                    break
                lines.append(action_line)
    return lines


def build_dependency_section(run: TopLevelRunView, *, width: int, max_lines: int) -> Group | None:
    """Wrap dependency section lines in a Rich group when any exist."""
    lines = build_dependency_section_lines(run, width=width, max_lines=max_lines)
    return Group(*lines) if lines else None


def _build_dependency_actions(session: SessionViewState, *, max_lines: int) -> list[Text]:
    """Render only prerequisite parent actions plus substep progress dots."""

    parent_rows = [row for row in session.step_rows if row.depth == 0]
    parent_rows = parent_rows[-max_lines:]
    child_rows = [row for row in session.step_rows if row.depth > 0]
    lines: list[Text] = []
    for row in parent_rows:
        icon, style = status_icon(row.status)
        text_style = (
            PREREQUISITE_CURSOR_STYLE
            if row.status == RunStatus.RUNNING
            else (SUCCESS_STYLE if row.status == RunStatus.PASSED else "white")
        )
        line = Text("  ")
        line.append(f"{icon} ", style)
        line.append(row.text, style=text_style)
        line.append_text(build_viewport_markers(row))
        lines.append(line)

        related_children = [child for child in child_rows if child.parent_step_id == row.step_id]
        if related_children:
            active_markers = build_viewport_markers(row)
            completed_children = sum(
                1 for child in related_children if child.status in (RunStatus.PASSED, RunStatus.FAILED)
            )
            dots = "." * max(1, completed_children)
            dot_line = Text("    ")
            dot_line.append(dots, style=SUCCESS_STYLE)
            dot_line.append_text(active_markers)
            lines.append(dot_line)
    return lines


def build_compact_run_row(
    run: TopLevelRunView,
    *,
    focused: bool = False,
    width: int | None = None,
    include_pass_badges: bool = True,
) -> Text:
    """Render a compact summary line for a top-level run."""

    compact_status = run.summary_status
    status_styles = {
        RunStatus.PASSED: SUCCESS_BOLD_STYLE,
        RunStatus.FAILED: FAILURE_BOLD_STYLE,
        RunStatus.BLOCKED: FAILURE_BOLD_STYLE,
        RunStatus.SKIPPED: "dim",
        RunStatus.RUNNING: "white",
        RunStatus.PENDING: "white",
    }
    style = VIEWPORT_CURSOR_STYLE if focused else status_styles.get(compact_status, "white")
    line = Text(run.display_path, style=style)
    if run.remaining_steps:
        line.append(f"   {run.remaining_steps} steps remaining...", style="dim")
    return _append_right_badges(line, _failure_badges(run, include_pass=include_pass_badges), width=width)


def build_step_list_lines(run: TopLevelRunView, *, max_lines: int) -> list[Text]:
    """Render merged step list lines for the focused run."""

    rows = [row for row in run.merged_step_rows if row.status != RunStatus.SKIPPED]
    active_rows = [row for row in rows if any(viewport.active for viewport in row.viewport_status.values())]
    expanded_parent_id: str | None = None
    active_lane_keys: set[str] = set()
    if active_rows:
        latest_active = max(active_rows, key=lambda item: item.order)
        active_lane_keys = {
            key
            for key, viewport in latest_active.viewport_status.items()
            if viewport.active and viewport.status == RunStatus.RUNNING
        }
        if not active_lane_keys:
            active_lane_keys = {key for key, viewport in latest_active.viewport_status.items() if viewport.active}
        if latest_active.depth > 0:
            expanded_parent_id = latest_active.parent_step_id
        elif latest_active.kind == "parent":
            expanded_parent_id = latest_active.step_id

    if expanded_parent_id is not None:
        rows = [
            row
            for row in rows
            if row.depth == 0
            or (
                row.parent_step_id == expanded_parent_id
                and (not active_lane_keys or any(lane_key in row.viewport_status for lane_key in active_lane_keys))
            )
        ]

    rows = rows[-max_lines:] if max_lines > 0 else rows
    return [build_step_line(row) for row in rows] or [Text("Waiting for steps...", style="dim")]


def build_step_list(run: TopLevelRunView, *, max_lines: int) -> Group:
    """Wrap the rendered step list lines in a Rich group."""
    return Group(*build_step_list_lines(run, max_lines=max_lines))


def build_footer(snapshot: RunSnapshot, run: TopLevelRunView) -> Text:
    """Render the footer for the focused run."""

    if snapshot.run_finished:
        return Text("")
    if run.remaining_steps:
        return Text(f"{run.remaining_steps} steps remaining...", style="dim")
    return Text("")


def build_focused_body(snapshot: RunSnapshot, run: TopLevelRunView, *, height: int, width: int) -> Group:
    """Render the main focused card body without panel chrome."""

    dependency_section_lines: list[Text] = []
    if run.dependencies and height > 2:
        dependency_lines = min(max(2, height // 3), max(2, len(run.dependencies) + 2))
        dependency_section_lines = build_dependency_section_lines(run, width=width, max_lines=dependency_lines)

    title_line = _append_right_badges(
        Text(run.display_path, style="bold white"),
        _failure_badges(run, include_pass=snapshot.run_finished),
        width=width,
    )

    footer = build_footer(snapshot, run)
    detail_lines: list[Text] = []
    if dependency_section_lines:
        detail_lines.extend(dependency_section_lines)
        detail_lines.append(Text(""))

    detail_lines.extend(build_step_list_lines(run, max_lines=max(1, height - 1)))

    if footer.plain:
        detail_lines.append(Text(""))
        detail_lines.append(footer)

    available_detail_lines = max(0, height - 1)
    if len(detail_lines) > available_detail_lines:
        detail_lines = detail_lines[-available_detail_lines:]

    items = [title_line]
    items.extend(detail_lines)
    return Group(*items)
