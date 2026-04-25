"""Viewport parsing and configuration helpers."""

from __future__ import annotations

import configparser
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from vizQA.app.exceptions import UserFacingException


@dataclass(frozen=True)
class ViewportSpec:
    """Normalized viewport definition."""

    name: str
    width: int
    height: int

    @property
    def slug(self) -> str:
        """Stable filesystem-safe identifier."""
        return self.name.lower().replace(" ", "_")

    @property
    def label(self) -> str:
        """Human-readable reporter label."""
        return f"{self.name} ({self.width}x{self.height})"


@dataclass
class ViewportConfig:
    """Viewport configuration loaded from repo config files."""

    default_viewports: list[str] = field(default_factory=list)
    named_viewports: dict[str, ViewportSpec] = field(default_factory=dict)


BUILTIN_VIEWPORTS: dict[str, ViewportSpec] = {
    "mobile": ViewportSpec(name="mobile", width=390, height=844),
    "tablet": ViewportSpec(name="tablet", width=768, height=1024),
    "desktop": ViewportSpec(name="desktop", width=1440, height=900),
    "widescreen": ViewportSpec(name="widescreen", width=1728, height=1117),
}


_RAW_VIEWPORT_RE = re.compile(r"^(?P<width>\d+)x(?P<height>\d+)$", re.IGNORECASE)


def _parse_viewport_value(name: str, value: str) -> ViewportSpec:
    match = _RAW_VIEWPORT_RE.fullmatch(value.strip())
    if not match:
        raise UserFacingException(f"Invalid viewport '{value}'. Expected a built-in name or WIDTHxHEIGHT.")
    return ViewportSpec(name=name, width=int(match.group("width")), height=int(match.group("height")))


def _load_pyproject_config(pyproject: Path, config: ViewportConfig) -> None:
    if not pyproject.exists():
        return

    try:
        with open(pyproject, "rb") as handle:
            data = tomllib.load(handle)
    except Exception:  # pylint: disable=broad-exception-caught
        return

    vizqa = data.get("tool", {}).get("vizqa", {})
    defaults = vizqa.get("default_viewports", [])
    if isinstance(defaults, list):
        config.default_viewports = [str(item) for item in defaults]

    viewports = vizqa.get("viewports", {})
    for name, spec in viewports.items():
        if not isinstance(spec, dict):
            continue
        width = spec.get("width")
        height = spec.get("height")
        if isinstance(width, int) and isinstance(height, int):
            config.named_viewports[str(name)] = ViewportSpec(name=str(name), width=width, height=height)


def _load_ini_config(ini_path: Path, config: ViewportConfig) -> None:
    if not ini_path.exists():
        return

    try:
        parser = configparser.ConfigParser()
        parser.read(ini_path)
    except Exception:  # pylint: disable=broad-exception-caught
        return

    if parser.has_section("vizqa") and parser.has_option("vizqa", "default_viewports"):
        defaults = parser.get("vizqa", "default_viewports")
        config.default_viewports = [item.strip() for item in defaults.split(",") if item.strip()]

    if parser.has_section("vizqa.viewports"):
        for name, value in parser.items("vizqa.viewports"):
            config.named_viewports[name] = _parse_viewport_value(name, value)


def load_viewport_config() -> ViewportConfig:
    """Load named viewport profiles and defaults from repo config."""
    cwd = Path.cwd()
    config = ViewportConfig()

    _load_pyproject_config(cwd / "pyproject.toml", config)

    for ini_name in ("pytest.ini", "tox.ini", "setup.cfg", "vizqa.ini"):
        _load_ini_config(cwd / ini_name, config)

    return config


def resolve_viewports(tokens: list[str], config: ViewportConfig) -> list[ViewportSpec]:
    """Resolve CLI or config viewport tokens into normalized specs."""
    requested = tokens or config.default_viewports or list(config.named_viewports.keys()) or ["desktop"]
    resolved: list[ViewportSpec] = []

    for token in requested:
        name = token.strip()
        key = name.lower()

        if key in config.named_viewports:
            resolved.append(config.named_viewports[key])
            continue
        if key in BUILTIN_VIEWPORTS:
            resolved.append(BUILTIN_VIEWPORTS[key])
            continue
        if _RAW_VIEWPORT_RE.fullmatch(name):
            resolved.append(_parse_viewport_value(name, name))
            continue
        raise UserFacingException(f"Invalid viewport '{token}'. Expected a built-in name or WIDTHxHEIGHT.")

    return resolved
