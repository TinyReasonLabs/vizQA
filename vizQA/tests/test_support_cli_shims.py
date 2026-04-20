"""Regression tests for support CLI modules used by CI workflows."""

import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]


def _load_support_module(name: str, relative_path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestSupportCliModules(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if "huggingface_hub" not in sys.modules:
            huggingface_hub = ModuleType("huggingface_hub")

            class _HfApi:  # pragma: no cover - behavior is irrelevant for this smoke test
                pass

            huggingface_hub.HfApi = _HfApi
            sys.modules["huggingface_hub"] = huggingface_hub

        if "packaging" not in sys.modules:
            packaging = ModuleType("packaging")
            packaging_version = ModuleType("packaging.version")

            class _Version(str):
                pass

            def _parse_version(raw):
                return _Version(raw)

            packaging_version.Version = _Version
            packaging_version.parse = _parse_version
            packaging.version = packaging_version
            sys.modules["packaging"] = packaging
            sys.modules["packaging.version"] = packaging_version

        cls.hf_revision = _load_support_module(
            "test_hf_revision_module",
            "app/support/hf_revision.py",
        )
        cls.merged_tag_version = _load_support_module(
            "test_merged_tag_version_module",
            "app/support/merged_tag_version.py",
        )

    def test_merged_tag_version_module_imports_cleanly(self):
        self.assertTrue(callable(self.merged_tag_version.max_merged_tag_version))
        self.assertTrue(callable(self.merged_tag_version.main))

    def test_hf_revision_module_imports_cleanly(self):
        self.assertEqual(self.hf_revision.REPO_ID, "alieissa/minilm-ui")
        self.assertTrue(callable(self.hf_revision.resolve_weights_revision))
        self.assertTrue(callable(self.hf_revision.main))
