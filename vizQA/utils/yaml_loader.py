"""YAML loading helpers."""

import os
import re
from pathlib import Path
from typing import Any

import yaml

from vizQA.app.exceptions import TestDefinitionError

_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


# pylint: disable=too-many-ancestors
class LineLoader(yaml.SafeLoader):
    """Custom YAML loader that adds line numbers to mappings."""

    def construct_mapping(self, node, deep=False):
        mapping = super().construct_mapping(node, deep=deep)
        mapping["__line__"] = node.start_mark.line + 1
        return mapping


def load_yaml_with_lines(test_path: Path) -> Any:
    """Load a YAML file with line metadata and env-var interpolation."""
    try:
        data = yaml.load(test_path.read_text(encoding="utf-8"), Loader=LineLoader)
        return expand_env_vars(data, test_path)
    except TestDefinitionError:
        raise
    except Exception as err:
        raise TestDefinitionError(f"Failed to load test file {test_path.name}", internal_detail=str(err)) from err


def expand_env_vars(value: Any, test_path: Path) -> Any:
    """Recursively expand ${VAR} placeholders in YAML string values."""
    if isinstance(value, dict):
        return {key: expand_env_vars(item, test_path) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_env_vars(item, test_path) for item in value]
    if isinstance(value, str):
        return _ENV_VAR_PATTERN.sub(lambda match: _resolve_env_var(match, test_path), value)
    return value


def _resolve_env_var(match: re.Match[str], test_path: Path) -> str:
    """Resolve a single ${VAR} placeholder or raise a test-definition error."""
    var_name = match.group(1)
    var_value = os.environ.get(var_name)
    if var_value is None:
        raise TestDefinitionError(
            f"Failed to load test file {test_path.name}",
            internal_detail=f"Environment variable '{var_name}' referenced in {test_path.name} is not set.",
        )
    return var_value
