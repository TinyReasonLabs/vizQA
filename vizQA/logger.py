"""
Logging module for vizQA.

Provides a SessionLogger that writes structured developer-focused log entries
to a timestamped file under .vizQA/.  The CLI never reads this file; it is
solely for debugging.

Usage::

    from vizQA.logger import get_logger
    logger = get_logger()
    logger.log_perception(step_id, query, perception_dict)
    logger.log_step(step_id, "FIND", StepStatus.PASSED)
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

_LOG_DIR = ".vizQA"
_INSTANCE: Optional["SessionLogger"] = None


class SessionLogger:
    """
    Thin wrapper around stdlib ``logging`` that persists developer debug
    information (perception responses, flow events, exceptions) to a file.

    Log levels:
    - DEBUG  — perception API responses, similarity scores, candidate lists
    - INFO   — step / session lifecycle events
    - WARNING — graceful degradation paths (no model, substring fallback, etc.)
    - ERROR  — caught exceptions during step execution
    """

    def __init__(self):
        os.makedirs(_LOG_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(_LOG_DIR, f"run_{timestamp}.log")
        self.log_path = log_path

        self._logger = logging.getLogger(f"vizqa.session.{timestamp}")
        self._logger.setLevel(logging.DEBUG)
        self._logger.propagate = False  # don't bubble up to the root logger

        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setLevel(logging.DEBUG)
        fmt = logging.Formatter(
            fmt="%(asctime)s  %(levelname)-8s  %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        handler.setFormatter(fmt)
        self._logger.addHandler(handler)

    # ------------------------------------------------------------------
    # Structured log helpers
    # ------------------------------------------------------------------

    def log_perception(self, step_id: str, query: str, response: Dict[str, Any]) -> None:
        """Logs a full perception API response at DEBUG level."""
        try:
            payload = json.dumps(response, ensure_ascii=False)
        except (TypeError, ValueError):
            payload = str(response)
        # self._logger.debug("[%s] PERCEPTION query=%r  response=%s", step_id, query, payload)

    def log_step(
        self,
        step_id: str,
        stage: str,
        status: Any,
        reason: Optional[str] = None,
    ) -> None:
        """Logs a step status transition at INFO or ERROR level."""
        msg = "[%s] STEP  stage=%s  status=%s"
        args: tuple = (step_id, stage, str(status))
        if reason:
            msg += "  reason=%r"
            args = args + (reason,)

        if "fail" in str(status).lower() or "error" in str(status).lower():
            self._logger.error(msg, *args)
        else:
            self._logger.info(msg, *args)

    def log_session(self, session_id: str, event: str, detail: str = "") -> None:
        """Logs a session-level lifecycle event (start, end, etc.)."""
        self._logger.info("[%s] SESSION  event=%s  %s", session_id, event, detail)

    def log_warning(self, step_id: str, message: str) -> None:
        """Logs a degradation or fallback warning."""
        self._logger.warning("[%s] WARN  %s", step_id, message)

    def log_exception(self, step_id: str, exc: Exception) -> None:
        """Logs an exception with full traceback."""
        self._logger.error("[%s] EXCEPTION  %s", step_id, str(exc), exc_info=exc)

    def log_debug(self, step_id: str, message: str) -> None:
        """General-purpose DEBUG entry."""
        self._logger.debug("[%s] %s", step_id, message)


def get_logger() -> SessionLogger:
    """
    Returns the process-wide singleton SessionLogger, creating it on first call.

    The log file name is determined at creation time so a single log file
    covers the whole vizQA run.
    """
    global _INSTANCE  # pylint: disable=global-statement
    if _INSTANCE is None:
        _INSTANCE = SessionLogger()
    return _INSTANCE


def reset_logger() -> None:
    """Resets the singleton (mainly useful in tests)."""
    global _INSTANCE  # pylint: disable=global-statement
    _INSTANCE = None
