import asyncio
import configparser
import tomllib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vizQA.app.cli import _load_config
from vizQA.app.core import Automator
from vizQA.app.memory import TestSession


def test_load_config_pyproject(tmp_path):
    # Create a dummy pyproject.toml
    pyproject_content = b"""
[tool.vizqa.headers]
Authorization = "Bearer global-token"
X-Test = "Global"
"""
    with patch("pathlib.Path.cwd", return_value=tmp_path):
        (tmp_path / "pyproject.toml").write_bytes(pyproject_content)
        headers = _load_config()
        assert headers == {"Authorization": "Bearer global-token", "X-Test": "Global"}


def test_load_config_ini(tmp_path):
    # Create a dummy pytest.ini
    ini_content = """
[vizqa.headers]
Authorization = Bearer ini-token
X-Custom = From-Ini
"""
    with patch("pathlib.Path.cwd", return_value=tmp_path):
        (tmp_path / "pytest.ini").write_text(ini_content)
        headers = _load_config()
        # configparser lowercase keys by default
        assert headers == {"authorization": "Bearer ini-token", "x-custom": "From-Ini"}


def test_load_config_priority(tmp_path):
    # INI takes priority over pyproject in current implementation
    pyproject_content = b"""
[tool.vizqa.headers]
Key = "From-Pyproject"
"""
    ini_content = """
[vizqa.headers]
Key = From-Ini
Other = From-Ini
"""
    with patch("pathlib.Path.cwd", return_value=tmp_path):
        (tmp_path / "pyproject.toml").write_bytes(pyproject_content)
        (tmp_path / "pytest.ini").write_text(ini_content)

        headers = _load_config()
        assert headers["key"] == "From-Ini"
        assert headers["other"] == "From-Ini"


def test_automator_applies_headers():
    # Use asyncio.run for the async part
    async def run_test():
        # Mocking playwright page
        mock_page = MagicMock()
        mock_page.set_extra_http_headers = MagicMock()

        # In async mock, we need to return a future or use AsyncMock
        # But we can also just mock the method to return a dummy awaitable
        async def mock_awaitable(*args, **kwargs):
            return None

        mock_page.set_extra_http_headers.side_effect = mock_awaitable
        mock_page.goto.side_effect = mock_awaitable

        # Automator setup
        automator = Automator(perception_client=MagicMock())
        automator.page = mock_page

        session = TestSession(
            id="test", test_name="Test", url="http://example.com", headers={"Authorization": "Bearer test-token"}
        )

        await automator.run_session(session)

        mock_page.set_extra_http_headers.assert_called_with({"Authorization": "Bearer test-token"})
        mock_page.goto.assert_called_with("http://example.com")

    asyncio.run(run_test())
