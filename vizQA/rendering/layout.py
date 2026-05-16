"""Compose the terminal layout from a reporting snapshot."""

from __future__ import annotations

from rich.console import Group
from rich.text import Text

from vizQA.rendering.components import build_compact_run_row, build_focused_body
from vizQA.rendering.models import DisplayMode, RunSnapshot, RunStatus, TopLevelRunView


def _build_silent_detail_rows(run: TopLevelRunView, *, height: int, used_lines: int) -> list[Text]:
    available_lines = min(4, max(1, height - used_lines - 2))
    rows = run.merged_step_rows[-available_lines:]
    if rows:
        rendered_rows: list[Text] = []
        for row in rows:
            rendered_rows.append(Text(f"{'  ' * (row.depth + 1)}{row.text}", style="dim"))
        return rendered_rows

    if any(session.status == RunStatus.RUNNING for session in run.dependencies):
        dependency_total = max(1, run.dependency_total or len(run.dependencies))
        return [
            Text(
                f"  running {dependency_total} pre-requisite{'s' if dependency_total != 1 else ''}...",
                style="dim",
            )
        ]

    return []


def compose_layout(snapshot: RunSnapshot, *, height: int, width: int = 120) -> Group | Text:
    """Build the full terminal layout for a snapshot."""

    if not snapshot.top_level_runs:
        return Text("")

    row_width = max(20, width - 2)
    if snapshot.run_finished:
        return Group(*(build_compact_run_row(run, width=row_width) for run in snapshot.top_level_runs))

    focused_key = snapshot.focused_owner_key or snapshot.top_level_runs[-1].owner_key
    focused_index = next(
        (index for index, run in enumerate(snapshot.top_level_runs) if run.owner_key == focused_key),
        len(snapshot.top_level_runs) - 1,
    )
    focused_run = snapshot.top_level_runs[focused_index]
    previous_runs = snapshot.top_level_runs[:focused_index]

    items = [build_compact_run_row(run, width=row_width) for run in previous_runs]

    if snapshot.display_mode == DisplayMode.SILENT:
        items.append(build_compact_run_row(focused_run, focused=True, width=row_width))
        items.extend(_build_silent_detail_rows(focused_run, height=height, used_lines=len(items)))
        return Group(*items)

    items.append(build_focused_body(snapshot, focused_run, height=height, width=row_width))
    return Group(*items)
