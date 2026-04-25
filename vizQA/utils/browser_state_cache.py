"""Browser state cache helpers."""

import json
from pathlib import Path
from typing import Any, Dict, Optional


class BrowserStateCache:
    """Helpers for persisting browser state snapshots between test runs."""

    CACHE_DIR = Path(".vizQA") / "browser_states"

    @staticmethod
    def build_cache_key(test_stem: str, namespace: str | None = None) -> str:
        """Build a stable cache key for a test, optionally namespaced by lane."""
        if namespace:
            return f"{namespace}__{test_stem}"
        return test_stem

    @staticmethod
    def cache(test_stem: str, state_dict: Dict[str, Any], namespace: str | None = None) -> Path:
        """
        Cache browser state to disk for a test.

        :param test_stem: Stem of the test file (without extension)
        :param state_dict: Browser state dictionary to cache
        :param namespace: Optional execution-lane namespace
        :return: Path to the cached state file
        """
        BrowserStateCache.CACHE_DIR.mkdir(parents=True, exist_ok=True)

        cache_key = BrowserStateCache.build_cache_key(test_stem, namespace=namespace)
        cache_file = BrowserStateCache.CACHE_DIR / f"{cache_key}.json"
        with open(cache_file, "w", encoding="utf-8") as file:
            json.dump(state_dict, file, indent=2)

        return cache_file

    @staticmethod
    def load(test_stem: str, namespace: str | None = None) -> Optional[Dict[str, Any]]:
        """
        Load cached browser state for a test.

        :param test_stem: Stem of the test file (without extension)
        :param namespace: Optional execution-lane namespace
        :return: Browser state dictionary, or None if cache not found
        """
        cache_key = BrowserStateCache.build_cache_key(test_stem, namespace=namespace)
        cache_file = BrowserStateCache.CACHE_DIR / f"{cache_key}.json"

        if not cache_file.exists():
            return None

        try:
            with open(cache_file, "r", encoding="utf-8") as file:
                return json.load(file)
        except Exception:  # pylint: disable=broad-exception-caught
            return None

    @staticmethod
    def clean() -> int:
        """
        Remove all cached browser states.

        :return: Number of cache files deleted
        """
        if not BrowserStateCache.CACHE_DIR.exists():
            return 0

        count = 0
        for cache_file in BrowserStateCache.CACHE_DIR.glob("*.json"):
            cache_file.unlink()
            count += 1

        return count
