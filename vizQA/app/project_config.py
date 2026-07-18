"""Project-level vizQA configuration shared by CLI and library entry points."""

from __future__ import annotations

import configparser
import tomllib
from pathlib import Path


def load_project_language(cwd: Path | None = None) -> str:
    """Return the configured language id, defaulting to the bundled English pack.

    TOML uses ``[tool.vizqa] language = "..."`` and INI files use
    ``[vizqa] language = ...``.  INI values override TOML, matching the
    existing project-header lookup order.
    """
    root = cwd or Path.cwd()
    language = "en"
    pyproject = root / "pyproject.toml"
    if pyproject.exists():
        with pyproject.open("rb") as handle:
            data = tomllib.load(handle)
        value = data.get("tool", {}).get("vizqa", {}).get("language")
        if isinstance(value, str) and value.strip():
            language = value.strip()

    for ini_name in ("pytest.ini", "tox.ini", "setup.cfg", "vizqa.ini"):
        ini_path = root / ini_name
        if not ini_path.exists():
            continue
        parser = configparser.ConfigParser()
        parser.read(ini_path)
        if parser.has_option("vizqa", "language"):
            value = parser.get("vizqa", "language").strip()
            if value:
                language = value
    return language
