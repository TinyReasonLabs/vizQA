"""
Logging module for vizQA.

Provides a SessionLogger that writes structured developer-focused log entries
to a timestamped file under .vizQA/.  The CLI never reads this file; it is
solely for debugging.

Usage::

    from vizQA.app.logger import get_logger
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
_INSTANCES: Dict[str, "SessionLogger"] = {}
_RUN_TIMESTAMP: Optional[str] = None


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

    def __init__(self, log_suffix: Optional[str] = None, timestamp: Optional[str] = None):
        os.makedirs(_LOG_DIR, exist_ok=True)
        run_timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = f"_{log_suffix}" if log_suffix else ""
        log_path = os.path.join(_LOG_DIR, f"run_{run_timestamp}{suffix}.log")
        self.log_path = log_path

        logger_name = f"vizqa.session.{run_timestamp}{suffix}"
        self._logger = logging.getLogger(logger_name)
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

    def log_perception(self, _step_id: str, _query: str, response: Dict[str, Any]) -> None:
        """Logs a full perception API response at DEBUG level.

        :param response: The response.
        """
        try:
            _payload = json.dumps(response, ensure_ascii=False)
        except (TypeError, ValueError):
            _payload = str(response)
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


def get_logger(log_suffix: Optional[str] = None) -> SessionLogger:
    """
    Returns a run-scoped SessionLogger, optionally namespaced by lane suffix.

    The timestamp portion is shared across the run so per-lane log files are
    grouped together while remaining isolated.
    """
    global _RUN_TIMESTAMP  # pylint: disable=global-statement
    key = log_suffix or ""
    if key not in _INSTANCES:
        if _RUN_TIMESTAMP is None:
            _RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
        _INSTANCES[key] = SessionLogger(log_suffix=log_suffix, timestamp=_RUN_TIMESTAMP)
    return _INSTANCES[key]


def reset_logger() -> None:
    """Resets the singleton (mainly useful in tests)."""
    global _RUN_TIMESTAMP  # pylint: disable=global-statement
    for logger in _INSTANCES.values():
        for handler in list(logger._logger.handlers):  # pylint: disable=protected-access
            handler.close()
            logger._logger.removeHandler(handler)  # pylint: disable=protected-access
    _INSTANCES.clear()
    _RUN_TIMESTAMP = None
