"""
Installation module for vizQA environment setup.
Handles browser binaries and model weights downloading using huggingface_hub.
"""

import asyncio
import sys
import threading
from contextlib import contextmanager
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Any, Optional

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]

from huggingface_hub import HfApi, snapshot_download
from packaging.version import parse as parse_version
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TransferSpeedColumn,
)

console = Console()

REPO_ID = "alieissa/minilm-ui"
WEIGHTS_REL_PATH = Path("weights")


# ---------------------------------------------------------------------------
# Progress helpers
# ---------------------------------------------------------------------------


class SmartProgressColumn(ProgressColumn):
    """Displays download size for byte tasks, or M/N count for everything else."""

    def __init__(self) -> None:
        super().__init__()
        self._download_col = DownloadColumn()
        self._mofn_col = MofNCompleteColumn()

    def render(self, task: Any):
        if task.fields.get("unit") in ("B", "byte", "bytes"):
            return self._download_col.render(task)
        return self._mofn_col.render(task)


class RichHFProgress:
    """A tqdm-compatible wrapper that routes HuggingFace progress into a shared
    Rich Progress instance.

    Only high-level "Fetching …" bars are shown; individual byte-level file
    downloads are suppressed to keep the UI clean.
    """

    _progress: Optional[Progress] = None
    _lock = threading.Lock()

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.iterable = args[0] if args else None
        self._total: Optional[float] = kwargs.get("total")
        self.description: str = kwargs.get("desc", "Downloading")
        self.disable: bool = kwargs.get("disable", False)
        self.unit: str = kwargs.get("unit", "it")
        self.task_id: Optional[int] = None

        if self.disable or self._progress is None:
            return

        # Only show high-level fetching bars, skip per-file byte bars.
        is_byte_task = self.unit in ("B", "byte", "bytes")
        if is_byte_task and "Fetching" not in self.description:
            return

        self.task_id = self._progress.add_task(
            "[yellow]Downloading model weights…",
            total=self._total if self._total and self._total > 0 else None,
            unit=self.unit,
        )

    # -- tqdm API surface used by huggingface_hub --------------------------

    @property
    def total(self) -> Optional[float]:
        """Return the total number of steps/bytes expected."""
        return self._total

    @total.setter
    def total(self, value: Any) -> None:
        """Update the total in the Rich task when huggingface_hub sets it late."""
        self._total = value
        if self._progress and self.task_id is not None:
            self._progress.update(self.task_id, total=value)

    def update(self, n: int = 1) -> None:
        """Advance the progress bar by *n* steps."""
        if self._progress and self.task_id is not None:
            self._progress.update(self.task_id, advance=n)

    def close(self) -> None:
        """Mark the task as complete when the download context exits."""
        if self._progress and self.task_id is not None:
            self._progress.update(self.task_id, completed=self._total or 0)

    def set_description(self, desc: str, refresh: bool = True) -> None:  # pylint: disable=unused-argument
        """Update the task description label."""
        if self._progress and self.task_id is not None:
            self._progress.update(self.task_id, description=desc)

    def refresh(self, *_args: Any, **_kwargs: Any) -> None:
        """No-op; Rich handles refreshes internally."""

    @classmethod
    def get_lock(cls) -> threading.Lock:
        """Return the class-level threading lock (required by tqdm's API)."""
        return cls._lock

    @classmethod
    def set_lock(cls, lock: threading.Lock) -> None:
        """Replace the class-level threading lock (required by tqdm's API)."""
        cls._lock = lock

    def __iter__(self):
        if self.iterable is not None:
            for item in self.iterable:
                yield item
                self.update(1)

    def __enter__(self):
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    @classmethod
    @contextmanager
    def activate(cls, progress: Progress):
        """Context manager that binds/unbinds the shared Progress instance."""
        cls._progress = progress
        try:
            yield
        finally:
            cls._progress = None


# ---------------------------------------------------------------------------
# Version / tag resolution
# ---------------------------------------------------------------------------


def get_package_version() -> str:
    """Return the installed package version, falling back to pyproject.toml."""
    try:
        return _pkg_version("vizQA")
    except Exception:  # pylint: disable=broad-exception-caught
        pass

    try:
        pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
        if pyproject.exists():
            with open(pyproject, "rb") as fh:
                return tomllib.load(fh)["project"]["version"]
    except Exception:  # pylint: disable=broad-exception-caught
        pass

    return "0.1.0"


def get_best_tag(api: HfApi, package_version: str) -> str:
    """Return the latest HF repo tag whose version is ≤ *package_version*.

    Falls back to ``"main"`` when no suitable tag is found or on any error.
    """
    target = parse_version(package_version)

    try:
        tags = api.list_repo_refs(repo_id=REPO_ID).tags
    except Exception:  # pylint: disable=broad-exception-caught
        return "main"

    best_tag = "main"
    best_version = parse_version("0.0.0")

    for ref in tags:
        raw = ref.name.lstrip("v")
        try:
            tag_version = parse_version(raw)
        except Exception:  # pylint: disable=broad-exception-caught
            continue

        if best_version < tag_version <= target:
            best_version = tag_version
            best_tag = ref.name

    return best_tag


# ---------------------------------------------------------------------------
# Installation tasks
# ---------------------------------------------------------------------------


class InstallError(Exception):
    """Raised when an installation step fails."""


async def _install_playwright(progress: Progress, task_id: int) -> None:
    """Install Playwright Chromium browser binary."""
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "playwright",
        "install",
        "chromium",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await process.communicate()

    if process.returncode == 0:
        progress.update(task_id, completed=100, description="[green]✔ Playwright installed")
    else:
        detail = stderr.decode(errors="replace").strip() if stderr else "unknown error"
        progress.update(task_id, description="[red]✘ Playwright failed")
        raise InstallError(f"Playwright install failed (exit {process.returncode}): {detail}")


def _download_weights(tag: str, weights_dir: Path, token: Optional[str]) -> None:
    """Download model weights from HuggingFace Hub (runs in a worker thread)."""
    snapshot_download(
        repo_id=REPO_ID,
        revision=tag,
        local_dir=weights_dir,
        token=token,
        tqdm_class=RichHFProgress,
    )


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------


async def run_install(token: Optional[str] = None) -> None:
    """Set up the vizQA runtime environment.

    Concurrently installs Playwright Chromium and downloads model weights.
    """
    pkg_version = get_package_version()
    console.print(f"\n[bold cyan]Initializing vizQA v{pkg_version} environment…[/]\n")

    weights_dir = Path(__file__).resolve().parent / WEIGHTS_REL_PATH

    api = HfApi(token=token)
    tag = await asyncio.to_thread(get_best_tag, api, pkg_version)

    errors: list[str] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        SmartProgressColumn(),
        TransferSpeedColumn(),
        console=console,
        expand=True,
    ) as progress:

        pw_task = progress.add_task("Playwright Chromium", total=100, unit="step")

        with RichHFProgress.activate(progress):
            results = await asyncio.gather(
                _install_playwright(progress, pw_task),
                asyncio.to_thread(_download_weights, tag, weights_dir, token),
                return_exceptions=True,
            )

    for result in results:
        if isinstance(result, Exception):
            errors.append(str(result))

    if errors:
        console.print()
        for msg in errors:
            console.print(f"[red]  ✘ {msg}[/]")
        console.print("\n[bold red]Installation completed with errors.[/]\n")
    else:
        console.print("\n[bold green]vizQA is ready to use![/]\n")
