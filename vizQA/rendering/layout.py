"""Compose the terminal layout from a reporting snapshot."""

from __future__ import annotations

from rich.console import Group
from rich.text import Text

from vizQA.rendering.components import build_compact_run_row, build_focused_body
from vizQA.rendering.models import DisplayMode, RunSnapshot


def compose_layout(snapshot: RunSnapshot, *, height: int, width: int = 120) -> Group | Text:
    """Build the full terminal layout for a snapshot."""

    if not snapshot.top_level_runs:
        return Text("")

    focused_key = snapshot.focused_owner_key or snapshot.top_level_runs[-1].owner_key
    focused_index = next(
        (index for index, run in enumerate(snapshot.top_level_runs) if run.owner_key == focused_key),
        len(snapshot.top_level_runs) - 1,
    )
    focused_run = snapshot.top_level_runs[focused_index]
    previous_runs = snapshot.top_level_runs[:focused_index]

    items = [build_compact_run_row(run, width=max(20, width - 2)) for run in previous_runs]

    if snapshot.display_mode == DisplayMode.SILENT:
        items.append(build_compact_run_row(focused_run, focused=True, width=max(20, width - 2)))
        for row in focused_run.merged_step_rows[-min(4, max(1, height - len(items) - 2)) :]:
            items.append(Text(f"  {row.text}", style="dim"))
        return Group(*items)

    items.append(build_focused_body(snapshot, focused_run, height=height, width=max(20, width - 2)))
    return Group(*items)
