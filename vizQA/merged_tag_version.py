"""Resolve a package-style version string for PR CI model weights.

Looks at every tag merged into the given git ref and returns the **maximum** PEP 440
version among tags whose names parse as versions after stripping a leading ``v``.
If none match, returns ``0.1.0``.

Used by CI via ``python -m vizQA.merged_tag_version``; optional second CLI argument is
the git working tree (e.g. a ``base_repo`` checkout) so the command can run from the PR
workspace while inspecting another clone.

This anchors HF weights to the newest release tag still contained in the branch
history, without reading ``pyproject.toml`` (which may be bumped on the PR only).
"""

from __future__ import annotations

import subprocess
import sys
from typing import Optional

from packaging.version import Version
from packaging.version import parse as parse_version


def max_merged_tag_version(ref: str = "HEAD", *, git_cwd: str | None = None) -> Version:
    """Return the highest version among tags merged into *ref*, or 0.1.0 if none.

    *git_cwd* is the repository root for ``git`` (e.g. a second checkout path in CI).
    """
    proc = subprocess.run(
        ["git", "tag", "--merged", ref],
        check=False,
        capture_output=True,
        text=True,
        cwd=git_cwd,
    )
    if proc.returncode != 0:
        return parse_version("0.1.0")

    best: Optional[Version] = None
    for line in proc.stdout.splitlines():
        name = line.strip()
        if not name:
            continue
        raw = name.lstrip("v")
        try:
            v = parse_version(raw)
        except Exception:  # pylint: disable=broad-exception-caught
            continue
        if best is None or v > best:
            best = v

    return best if best is not None else parse_version("0.1.0")


def main() -> None:
    """Print the maximum merged tag version to stdout."""
    ref = sys.argv[1] if len(sys.argv) > 1 else "HEAD"
    git_cwd = sys.argv[2] if len(sys.argv) > 2 else None
    print(max_merged_tag_version(ref, git_cwd=git_cwd))


if __name__ == "__main__":
    main()
