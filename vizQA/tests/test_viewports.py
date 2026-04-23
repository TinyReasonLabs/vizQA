from pathlib import Path
from unittest.mock import patch

import pytest

from vizQA.app.exceptions import UserFacingException
from vizQA.app.viewport import BUILTIN_VIEWPORTS, ViewportConfig, ViewportSpec, load_viewport_config, resolve_viewports


def test_builtin_viewports_include_modern_defaults():
    assert BUILTIN_VIEWPORTS["mobile"] == ViewportSpec(name="mobile", width=390, height=844)
    assert BUILTIN_VIEWPORTS["tablet"] == ViewportSpec(name="tablet", width=768, height=1024)
    assert BUILTIN_VIEWPORTS["desktop"] == ViewportSpec(name="desktop", width=1440, height=900)
    assert BUILTIN_VIEWPORTS["widescreen"] == ViewportSpec(name="widescreen", width=1728, height=1117)


def test_resolve_viewports_accepts_raw_size_tokens():
    resolved = resolve_viewports(["390x844"], ViewportConfig())

    assert resolved == [ViewportSpec(name="390x844", width=390, height=844)]


def test_resolve_viewports_raises_user_facing_error_for_invalid_token():
    with pytest.raises(UserFacingException, match="Invalid viewport"):
        resolve_viewports(["not-a-size"], ViewportConfig())


def test_load_viewport_config_from_pyproject(tmp_path: Path):
    pyproject_content = b"""
[tool.vizqa]
default_viewports = ["desktop", "mobile"]

[tool.vizqa.viewports.app]
width = 1280
height = 720
"""
    with patch("pathlib.Path.cwd", return_value=tmp_path):
        (tmp_path / "pyproject.toml").write_bytes(pyproject_content)

        config = load_viewport_config()

    assert config.default_viewports == ["desktop", "mobile"]
    assert config.named_viewports["app"] == ViewportSpec(name="app", width=1280, height=720)


def test_load_viewport_config_from_ini(tmp_path: Path):
    ini_content = """
[vizqa]
default_viewports = desktop,mobile

[vizqa.viewports]
app = 1280x720
"""
    with patch("pathlib.Path.cwd", return_value=tmp_path):
        (tmp_path / "vizqa.ini").write_text(ini_content, encoding="utf-8")

        config = load_viewport_config()

    assert config.default_viewports == ["desktop", "mobile"]
    assert config.named_viewports["app"] == ViewportSpec(name="app", width=1280, height=720)


def test_load_viewport_config_from_ini_multiple_viewports(tmp_path: Path):
    ini_content = """
[vizqa.viewports]
app = 1280x720
mobile = 390x844
tablet = 768x1024
"""
    with patch("pathlib.Path.cwd", return_value=tmp_path):
        (tmp_path / "vizqa.ini").write_text(ini_content, encoding="utf-8")

        config = load_viewport_config()

    assert config.default_viewports == []
    assert config.named_viewports == {
        "app": ViewportSpec(name="app", width=1280, height=720),
        "mobile": ViewportSpec(name="mobile", width=390, height=844),
        "tablet": ViewportSpec(name="tablet", width=768, height=1024),
    }


def test_resolve_viewports_prefers_cli_tokens_over_config_defaults():
    config = ViewportConfig(default_viewports=["desktop"], named_viewports={"app": ViewportSpec("app", 1280, 720)})

    resolved = resolve_viewports(["app", "390x844"], config)

    assert resolved == [
        ViewportSpec(name="app", width=1280, height=720),
        ViewportSpec(name="390x844", width=390, height=844),
    ]


def test_resolve_viewports_uses_config_defaults_when_cli_omitted():
    config = ViewportConfig(default_viewports=["mobile", "desktop"])

    resolved = resolve_viewports([], config)

    assert resolved == [BUILTIN_VIEWPORTS["mobile"], BUILTIN_VIEWPORTS["desktop"]]


def test_resolve_viewports_uses_named_profiles_when_no_explicit_defaults_exist():
    config = ViewportConfig(
        named_viewports={
            "app": ViewportSpec("app", 1280, 720),
            "mobile": ViewportSpec("mobile", 390, 844),
        }
    )

    resolved = resolve_viewports([], config)

    assert resolved == [
        ViewportSpec(name="app", width=1280, height=720),
        ViewportSpec(name="mobile", width=390, height=844),
    ]


def test_resolve_viewports_falls_back_to_desktop_when_no_defaults_exist():
    resolved = resolve_viewports([], ViewportConfig())

    assert resolved == [BUILTIN_VIEWPORTS["desktop"]]
