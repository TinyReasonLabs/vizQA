"""Tests for process-level project configuration."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from vizQA.app.core import Automator
from vizQA.app.project_config import load_project_language


def test_project_language_defaults_to_english(tmp_path: Path):
    assert load_project_language(tmp_path) == "en"


def test_project_language_loads_toml_then_ini_override(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text('[tool.vizqa]\nlanguage = "toml-pack"\n', encoding="utf-8")
    assert load_project_language(tmp_path) == "toml-pack"

    (tmp_path / "vizqa.ini").write_text("[vizqa]\nlanguage = ini-pack\n", encoding="utf-8")
    assert load_project_language(tmp_path) == "ini-pack"


def test_automator_rejects_unknown_configured_language(tmp_path: Path, monkeypatch):
    (tmp_path / "vizqa.ini").write_text("[vizqa]\nlanguage = missing-pack\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="missing-pack"):
        Automator(perception_client=MagicMock())
