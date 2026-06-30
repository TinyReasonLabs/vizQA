"""Utilities for locating, versioning, and reporting installed model weights."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from packaging.version import InvalidVersion, Version

from vizQA.app.support.hf_revision import resolve_weights_revision

MODEL_NAME = "minilm"
DEFAULT_LANGUAGE = "en"
DEFAULT_LANGUAGE_SCHEMA_VERSION = 1
DEFAULT_PROVIDER_ID = "minilm"
DEFAULT_LEGACY_WEIGHTS_VERSION = "0.1.0"
METADATA_FILE_NAME = ".vizqa-weights.json"
METADATA_VERSION = 1


@dataclass(frozen=True)
class WeightState:
    """Resolved view of the installed-vs-expected weights state."""

    package_version: str
    expected_revision: str
    installed_revision: Optional[str]
    status: str
    assumed_revision: bool


def get_weights_dir() -> Path:
    """Return the canonical weights directory."""
    return Path(__file__).resolve().parents[2] / "weights"


def get_model_dir(weights_dir: Optional[Path] = None) -> Path:
    """Return the canonical model directory."""
    base_dir = Path(weights_dir) if weights_dir is not None else get_weights_dir()
    return base_dir / MODEL_NAME


def get_weights_metadata_path(weights_dir: Optional[Path] = None) -> Path:
    """Return the metadata file path stored under the canonical weights dir."""
    base_dir = Path(weights_dir) if weights_dir is not None else get_weights_dir()
    return base_dir / METADATA_FILE_NAME


def read_weights_metadata(weights_dir: Optional[Path] = None) -> Optional[dict[str, Any]]:
    """Read weights metadata if it exists and is valid JSON."""
    metadata_path = get_weights_metadata_path(weights_dir)
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError, ValueError):
        return None


# pylint: disable=too-many-arguments
def write_weights_metadata(
    weights_dir: Path,
    *,
    package_version: str,
    revision: str,
    language: str = DEFAULT_LANGUAGE,
    language_schema_version: int = DEFAULT_LANGUAGE_SCHEMA_VERSION,
    provider_id: str = DEFAULT_PROVIDER_ID,
    provider_revision: Optional[str] = None,
) -> Path:
    """Persist local metadata about the currently installed weights."""
    metadata_path = get_weights_metadata_path(weights_dir)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(
            {
                "metadata_version": METADATA_VERSION,
                "model_name": MODEL_NAME,
                "package_version": package_version,
                "weights_revision": revision,
                "language": language,
                "language_schema_version": language_schema_version,
                "provider_id": provider_id,
                "provider_revision": provider_revision or revision,
                "installed_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return metadata_path


def _parse_revision(raw: Optional[str]) -> Optional[Version]:
    if not raw:
        return None
    try:
        return Version(raw.lstrip("v"))
    except InvalidVersion:
        return None


def _classify_installed_revision(installed_revision: str, expected_revision: str) -> str:
    if installed_revision == expected_revision:
        return "aligned"

    installed_version = _parse_revision(installed_revision)
    expected_version = _parse_revision(expected_revision)

    if installed_version is not None and expected_version is not None:
        if installed_version < expected_version:
            return "older than expected"
        if installed_version > expected_version:
            return "newer than expected"

    return "aligned"


def inspect_weight_state(
    *,
    package_version: str,
    token: Optional[str] = None,
    weights_dir: Optional[Path] = None,
) -> WeightState:
    """Inspect installed weights and compare them with the expected revision."""
    base_dir = Path(weights_dir) if weights_dir is not None else get_weights_dir()
    model_dir = get_model_dir(base_dir)
    expected_revision = resolve_weights_revision(package_version, token=token)

    metadata = read_weights_metadata(base_dir)
    installed_revision = None
    assumed_revision = False

    if metadata and metadata.get("weights_revision"):
        installed_revision = str(metadata["weights_revision"])
    elif model_dir.exists():
        installed_revision = DEFAULT_LEGACY_WEIGHTS_VERSION
        assumed_revision = True

    if installed_revision is None:
        return WeightState(
            package_version=package_version,
            expected_revision=expected_revision,
            installed_revision=None,
            status="missing",
            assumed_revision=False,
        )

    return WeightState(
        package_version=package_version,
        expected_revision=expected_revision,
        installed_revision=installed_revision,
        status=_classify_installed_revision(installed_revision, expected_revision),
        assumed_revision=assumed_revision,
    )
