"""Browser state cache helpers."""

import json
from pathlib import Path
from typing import Any, Dict, Optional


class BrowserStateCache:
    """Helpers for persisting browser state snapshots between test runs."""

    CACHE_DIR = Path(".vizQA") / "browser_states"

    @staticmethod
    def cache(test_stem: str, state_dict: Dict[str, Any]) -> Path:
        """
        Cache browser state to disk for a test.

        :param test_stem: Stem of the test file (without extension)
        :param state_dict: Browser state dictionary to cache
        :return: Path to the cached state file
        """
        BrowserStateCache.CACHE_DIR.mkdir(parents=True, exist_ok=True)

        cache_file = BrowserStateCache.CACHE_DIR / f"{test_stem}.json"
        with open(cache_file, "w", encoding="utf-8") as file:
            json.dump(state_dict, file, indent=2)

        return cache_file

    @staticmethod
    def load(test_stem: str) -> Optional[Dict[str, Any]]:
        """
        Load cached browser state for a test.

        :param test_stem: Stem of the test file (without extension)
        :return: Browser state dictionary, or None if cache not found
        """
        cache_file = BrowserStateCache.CACHE_DIR / f"{test_stem}.json"

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
