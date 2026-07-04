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


# pylint: disable=too-many-instance-attributes
@dataclass
class ParserConfig:
    """Configuration for SemanticParser thresholds and behavior."""

    use_advanced_ranking: bool
    intent_threshold: float
    action_threshold: float
    semantic_match_threshold: float
    verification_timeout: int
    wait_for_timeout_seconds: int
    wait_for_poll_interval_seconds: float
    scroll_center_band_min: float
    scroll_center_band_max: float
    step_delay_seconds: float


PERCEPTION_BACKEND = _normalize_base_url(os.environ.get("PERCEPTION_BACKEND", "localhost:8228"))

use_adv = os.environ.get("VIZQA_ADVANCED_RANKING", "1") == "1"
intent_threq = float(os.environ.get("VIZQA_INTENT_THRESHOLD", "0.6"))
action_threq = float(os.environ.get("VIZQA_ACTION_THRESHOLD", "0.8"))
semantic_threq = float(os.environ.get("VIZQA_SEMANTIC_THRESHOLD", "0.70"))
verif_timeout = int(os.environ.get("VIZQA_VERIFICATION_TIMEOUT", "5"))
wait_for_timeout = int(os.environ.get("VIZQA_WAIT_FOR_TIMEOUT", "120"))
wait_for_poll_interval = float(os.environ.get("VIZQA_WAIT_FOR_POLL_INTERVAL", "1.0"))
scroll_center_band_min = float(os.environ.get("VIZQA_SCROLL_CENTER_BAND_MIN", "0.35"))
scroll_center_band_max = float(os.environ.get("VIZQA_SCROLL_CENTER_BAND_MAX", "0.65"))
step_delay = float(os.environ.get("VIZQA_STEP_DELAY_SECONDS", "0.5"))

CONFIG = ParserConfig(
    use_advanced_ranking=use_adv,
    intent_threshold=intent_threq,
    action_threshold=action_threq,
    semantic_match_threshold=semantic_threq,
    verification_timeout=verif_timeout,
    wait_for_timeout_seconds=wait_for_timeout,
    wait_for_poll_interval_seconds=wait_for_poll_interval,
    scroll_center_band_min=scroll_center_band_min,
    scroll_center_band_max=scroll_center_band_max,
    step_delay_seconds=step_delay,
)
