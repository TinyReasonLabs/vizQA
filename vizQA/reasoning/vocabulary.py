"""
Shared vocabulary and regex helpers for parser and reasoning modules.
"""

import re


# pylint: disable=too-few-public-methods, duplicate-code
class ParserVocabulary:
    """Central repository for UI grounding vocabulary."""

    ACTION_VERBS = {
        "click": ["click", "tap", "hit", "press"],
        "right-click": ["right-click", "context-click", "right click", "perform right-click", "context click"],
        "type": ["type", "enter", "input", "fill"],
        "hover": ["hover", "move to", "point"],
        "select": ["select", "choose", "pick"],
        "check": ["check", "tick"],
        "drag": ["drag"],
        "drop": ["drop"],
        "scroll": ["scroll"],
        "clear": ["clear", "empty"],
        "wait": ["wait", "pause", "sleep"],
        "find": ["find", "locate"],
    }

    VERIFY_VERBS = ["verify", "ensure", "assert", "check that", "make sure"]
    VERIFY_BOILERPLATE = [
        "should",
        "must",
        "appear",
        "visible",
        "shows",
        "exists",
        "present",
        "displayed",
        "open",
        "pause until",
    ]

    COLORS = ["red", "blue", "green", "yellow", "orange", "purple", "black", "white", "gray", "grey"]
    STATES = ["disabled", "enabled", "checked", "unchecked", "visible", "invisible", "hidden", "displayed", "active"]
    POSITIONS = [
        "top left",
        "top right",
        "bottom left",
        "bottom right",
        "top",
        "bottom",
        "left",
        "right",
        "center",
        "middle",
    ]
    # pylint: disable=line-too-long
    NEGATION_RE = re.compile(
        r"\b(not|no longer|should not|shouldn't|should not be|disappear(?:s|ed)?|gone|invisible|absent|done|finished|closed|close|removed|vanish(?:es|ed)?|gone)\b",
        re.IGNORECASE,
    )
