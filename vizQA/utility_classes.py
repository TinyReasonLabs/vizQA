"""
Utility classes for the vizQA package.
"""

import yaml

# ---------------------------------------------------------------------------
# YAML Line Tracking
# ---------------------------------------------------------------------------


# pylint: disable=too-many-ancestors
class LineLoader(yaml.SafeLoader):
    """Custom YAML loader that adds line numbers to mappings."""

    def construct_mapping(self, node, deep=False):
        mapping = super().construct_mapping(node, deep=deep)
        mapping["__line__"] = node.start_mark.line + 1
        return mapping
