"""Tests for :mod:`vizQA.app.support.merged_tag_version`."""

from unittest.mock import MagicMock, patch

from packaging.version import parse as parse_version

from vizQA.app.support.merged_tag_version import max_merged_tag_version


def test_git_failure_returns_default():
    proc = MagicMock(returncode=1, stdout="")
    with patch("vizQA.app.support.merged_tag_version.subprocess.run", return_value=proc):
        assert max_merged_tag_version() == parse_version("0.1.0")


def test_picks_max_version_strips_v():
    proc = MagicMock(
        returncode=0,
        stdout="v0.1.0\n0.5.0\nnot-a-version\nv2.0.0\n",
    )
    with patch("vizQA.app.support.merged_tag_version.subprocess.run", return_value=proc):
        assert max_merged_tag_version() == parse_version("2.0.0")


def test_empty_tags_returns_default():
    proc = MagicMock(returncode=0, stdout="\n\n")
    with patch("vizQA.app.support.merged_tag_version.subprocess.run", return_value=proc):
        assert max_merged_tag_version() == parse_version("0.1.0")


def test_passes_git_cwd():
    proc = MagicMock(returncode=0, stdout="1.0.0\n")
    with patch("vizQA.app.support.merged_tag_version.subprocess.run", return_value=proc) as run:
        max_merged_tag_version("main", git_cwd="/tmp/repo")
        run.assert_called_once()
        assert run.call_args.kwargs["cwd"] == "/tmp/repo"
        assert run.call_args[0][0] == ["git", "tag", "--merged", "main"]
