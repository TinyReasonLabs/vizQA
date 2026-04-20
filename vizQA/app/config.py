"""Configuration variables for vizQA."""

import os
from dataclasses import dataclass


def _normalize_base_url(value: str) -> str:
    """Normalize the base URL for the perception backend."""
    value = value.strip()
    if not value:
        return "http://localhost:8228"
    if value.startswith("http://") or value.startswith("https://"):
        return value.rstrip("/")
    return f"http://{value}".rstrip("/")


@dataclass
class ParserConfig:
    """Configuration for SemanticParser thresholds and behavior."""

    use_advanced_ranking: bool = False
    intent_threshold: float = 0.6
    action_threshold: float = 0.52
    semantic_match_threshold: float = 0.70
    verification_timeout: int = 15


PERCEPTION_BACKEND = _normalize_base_url(os.environ.get("PERCEPTION_BACKEND", "localhost:8228"))

use_adv = os.environ.get("VIZQA_ADVANCED_RANKING", "1") == "1"
intent_threq = float(os.environ.get("VIZQA_INTENT_THRESHOLD", "0.6"))
action_threq = float(os.environ.get("VIZQA_ACTION_THRESHOLD", "0.52"))
semantic_threq = float(os.environ.get("VIZQA_SEMANTIC_THRESHOLD", "0.70"))
verif_timeout = int(os.environ.get("VIZQA_VERIFICATION_TIMEOUT", "15"))

CONFIG = ParserConfig(
    use_advanced_ranking=use_adv,
    intent_threshold=intent_threq,
    action_threshold=action_threq,
    semantic_match_threshold=semantic_threq,
    verification_timeout=verif_timeout,
)
