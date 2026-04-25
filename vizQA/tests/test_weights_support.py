import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from click.testing import CliRunner

from vizQA.app.cli import cli
from vizQA.app.support.weights import (
    DEFAULT_LEGACY_WEIGHTS_VERSION,
    METADATA_FILE_NAME,
    WeightState,
    inspect_weight_state,
    write_weights_metadata,
)


def test_inspect_weight_state_returns_missing_when_model_dir_absent(tmp_path):
    with patch("vizQA.app.support.weights.resolve_weights_revision", return_value="0.2.0"):
        state = inspect_weight_state(package_version="0.2.0", weights_dir=tmp_path)

    assert state.status == "missing"
    assert state.installed_revision is None
    assert state.expected_revision == "0.2.0"
    assert state.assumed_revision is False


def test_inspect_weight_state_assumes_legacy_version_when_metadata_missing(tmp_path):
    (tmp_path / "minilm").mkdir()

    with patch("vizQA.app.support.weights.resolve_weights_revision", return_value="0.2.0"):
        state = inspect_weight_state(package_version="0.2.0", weights_dir=tmp_path)

    assert state.status == "older than expected"
    assert state.installed_revision == DEFAULT_LEGACY_WEIGHTS_VERSION
    assert state.expected_revision == "0.2.0"
    assert state.assumed_revision is True


def test_inspect_weight_state_reports_aligned_metadata(tmp_path):
    (tmp_path / "minilm").mkdir()
    write_weights_metadata(tmp_path, package_version="0.2.0", revision="0.2.0")

    with patch("vizQA.app.support.weights.resolve_weights_revision", return_value="0.2.0"):
        state = inspect_weight_state(package_version="0.2.0", weights_dir=tmp_path)

    assert state.status == "aligned"
    assert state.installed_revision == "0.2.0"
    assert state.assumed_revision is False


def test_inspect_weight_state_reports_newer_than_expected(tmp_path):
    (tmp_path / "minilm").mkdir()
    write_weights_metadata(tmp_path, package_version="0.3.0", revision="0.3.0")

    with patch("vizQA.app.support.weights.resolve_weights_revision", return_value="0.2.0"):
        state = inspect_weight_state(package_version="0.3.0", weights_dir=tmp_path)

    assert state.status == "newer than expected"
    assert state.installed_revision == "0.3.0"
    assert state.expected_revision == "0.2.0"


def test_write_weights_metadata_persists_expected_fields(tmp_path):
    metadata_path = write_weights_metadata(tmp_path, package_version="0.3.0", revision="0.2.0")

    data = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert metadata_path.name == METADATA_FILE_NAME
    assert data["package_version"] == "0.3.0"
    assert data["weights_revision"] == "0.2.0"
    assert data["metadata_version"] == 1
    assert "installed_at" in data


def test_cli_version_reports_package_and_weights_status():
    runner = CliRunner()
    state = WeightState(
        package_version="0.3.0",
        expected_revision="0.3.0",
        installed_revision="0.2.0",
        status="older than expected",
        assumed_revision=False,
    )

    with (
        patch("vizQA.app.cli.get_package_version", return_value="0.3.0"),
        patch("vizQA.app.cli.inspect_weight_state", return_value=state),
    ):
        result = runner.invoke(cli, ["--version"])

    assert result.exit_code == 0
    assert "vizQA 0.3.0" in result.output
    assert "weights: 0.2.0 (older than expected; expected 0.3.0)" in result.output


def test_run_warns_when_weights_are_older_but_continues(tmp_path):
    test_path = tmp_path / "main.yaml"
    test_path.write_text(
        """
name: "Main"
url: "http://example.com"
steps: []
""".strip(),
        encoding="utf-8",
    )

    runner = CliRunner()
    state = WeightState(
        package_version="0.3.0",
        expected_revision="0.3.0",
        installed_revision="0.2.0",
        status="older than expected",
        assumed_revision=False,
    )

    with (
        patch("vizQA.app.cli.inspect_weight_state", return_value=state),
        patch("vizQA.app.cli.get_logger"),
        patch("vizQA.app.cli.PerceptionClient"),
        patch("vizQA.app.cli.Automator") as mock_automator_cls,
        patch("vizQA.app.cli.run_single_test", new=AsyncMock(return_value=True)),
    ):
        mock_automator_cls.return_value = SimpleNamespace(start=AsyncMock(), stop=AsyncMock())
        result = runner.invoke(cli, ["run", str(test_path)])

    assert result.exit_code == 0
    assert "installed model weights are older than expected" in result.output
    assert "Run `vizqa install` to align them." in result.output


def test_run_warns_when_weights_version_is_assumed(tmp_path):
    test_path = tmp_path / "main.yaml"
    test_path.write_text(
        """
name: "Main"
url: "http://example.com"
steps: []
""".strip(),
        encoding="utf-8",
    )

    runner = CliRunner()
    state = WeightState(
        package_version="0.2.0",
        expected_revision="0.2.0",
        installed_revision=DEFAULT_LEGACY_WEIGHTS_VERSION,
        status="older than expected",
        assumed_revision=True,
    )

    with (
        patch("vizQA.app.cli.inspect_weight_state", return_value=state),
        patch("vizQA.app.cli.get_logger"),
        patch("vizQA.app.cli.PerceptionClient"),
        patch("vizQA.app.cli.Automator") as mock_automator_cls,
        patch("vizQA.app.cli.run_single_test", new=AsyncMock(return_value=True)),
    ):
        mock_automator_cls.return_value = SimpleNamespace(start=AsyncMock(), stop=AsyncMock())
        result = runner.invoke(cli, ["run", str(test_path)])

    assert result.exit_code == 0
    assert "assuming installed weights version 0.1.0 because metadata was not found" in result.output


def test_run_install_writes_metadata_after_success(tmp_path, monkeypatch):
    async def noop_install_playwright(*_args, **_kwargs):
        return None

    def fake_download(_tag, weights_dir, _token):
        (weights_dir / "minilm").mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr("vizQA.app.support.install.get_weights_dir", lambda: tmp_path)

    with (
        patch("vizQA.app.support.install.get_package_version", return_value="0.3.0"),
        patch("vizQA.app.support.install.resolve_weights_revision", return_value="0.2.0"),
        patch("vizQA.app.support.install._install_playwright", new=noop_install_playwright),
        patch("vizQA.app.support.install._download_weights", new=fake_download),
    ):
        from vizQA.app.support.install import run_install

        asyncio.run(run_install())

    metadata = json.loads((tmp_path / METADATA_FILE_NAME).read_text(encoding="utf-8"))
    assert metadata["package_version"] == "0.3.0"
    assert metadata["weights_revision"] == "0.2.0"
