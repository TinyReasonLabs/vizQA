"""Configuration variables for vizQA."""

import os
from dataclasses import dataclass


@dataclass
class ParserConfig:
    """Configuration for SemanticParser thresholds and behavior."""

    use_advanced_ranking: bool = False
    intent_threshold: float = 0.6
    action_threshold: float = 0.52
    semantic_match_threshold: float = 0.70


use_adv = os.environ.get("VIZQA_ADVANCED_RANKING", "1") == "1"
intent_threq = float(os.environ.get("VIZQA_INTENT_THRESHOLD", "0.6"))
action_threq = float(os.environ.get("VIZQA_ACTION_THRESHOLD", "0.52"))
semantic_threq = float(os.environ.get("VIZQA_SEMANTIC_THRESHOLD", "0.70"))

CONFIG = ParserConfig(
    use_advanced_ranking=use_adv,
    intent_threshold=intent_threq,
    action_threshold=action_threq,
    semantic_match_threshold=semantic_threq,
)
