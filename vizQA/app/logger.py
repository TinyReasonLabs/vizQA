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

import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

_LOG_DIR = ".vizQA"
_INSTANCES: Dict[str, "SessionLogger"] = {}
_RUN_TIMESTAMP: Optional[str] = None
_PERCEPTION_CANDIDATE_LIMIT = 5
_DEBUG_ENABLED = True


class NullLogger:
    """No-op logger used when persistent debug logging is intentionally disabled."""

    log_path: Optional[str] = None

    def log_perception(self, *args, **kwargs) -> None:
        """No-op perception log."""
        del args, kwargs

    def log_step(self, *args, **kwargs) -> None:
        """No-op step log."""
        del args, kwargs

    def log_session(self, *args, **kwargs) -> None:
        """No-op session log."""
        del args, kwargs

    def log_warning(self, *args, **kwargs) -> None:
        """No-op warning log."""
        del args, kwargs

    def log_exception(self, *args, **kwargs) -> None:
        """No-op exception log."""
        del args, kwargs

    def log_debug(self, *args, **kwargs) -> None:
        """No-op debug log."""
        del args, kwargs


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
        run_timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = f"_{log_suffix}" if log_suffix else ""
        log_path = os.path.join(_LOG_DIR, f"run_{run_timestamp}{suffix}.log")
        self.log_path = log_path
        self._handler_attached = False

        logger_name = f"vizqa.session.{run_timestamp}{suffix}"
        self._logger = logging.getLogger(logger_name)
        self._logger.setLevel(logging.DEBUG if _DEBUG_ENABLED else logging.INFO)
        self._logger.propagate = False  # don't bubble up to the root logger

    def _ensure_handler(self) -> None:
        """Attach the file handler only when a log entry is actually emitted."""
        if self._handler_attached:
            return

        os.makedirs(_LOG_DIR, exist_ok=True)
        handler = logging.FileHandler(self.log_path, encoding="utf-8")
        handler.setLevel(logging.DEBUG if _DEBUG_ENABLED else logging.INFO)
        fmt = logging.Formatter(
            fmt="%(asctime)s  %(levelname)-8s  %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        handler.setFormatter(fmt)
        self._logger.addHandler(handler)
        self._handler_attached = True

    # ------------------------------------------------------------------
    # Structured log helpers
    # ------------------------------------------------------------------

    def log_perception(
        self,
        step_id: str,
        query: str,
        response: Dict[str, Any],
        selected: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Logs compact, one-line perception diagnostics at DEBUG level."""
        self._ensure_handler()
        source = "top_matches" if response.get("top_matches") else "elements"
        candidates = response.get(source, [])
        ordered = candidates[:_PERCEPTION_CANDIDATE_LIMIT]

        if ordered:
            for index, element in enumerate(ordered, start=1):
                self._logger.debug(
                    "[%s] PERCEPTION query=%r candidate=%s",
                    step_id,
                    query,
                    self._format_perception_candidate(element, index, source),
                )
        else:
            self._logger.debug("[%s] PERCEPTION query=%r candidate=none", step_id, query)

        if selected is None:
            self._logger.debug("[%s] PERCEPTION selected=none", step_id)
            return

        selected_rank = next((index + 1 for index, element in enumerate(candidates) if element is selected), 1)
        self._logger.debug(
            "[%s] PERCEPTION selected=%s",
            step_id,
            self._format_perception_candidate(selected, selected_rank, source),
        )

    def log_step(
        self,
        step_id: str,
        stage: str,
        status: Any,
        reason: Optional[str] = None,
    ) -> None:
        """Logs a step status transition at INFO or ERROR level."""
        self._ensure_handler()
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
        self._ensure_handler()
        self._logger.info("[%s] SESSION  event=%s  %s", session_id, event, detail)

    def log_warning(self, step_id: str, message: str) -> None:
        """Logs a degradation or fallback warning."""
        self._ensure_handler()
        self._logger.warning("[%s] WARN  %s", step_id, message)

    def log_exception(self, step_id: str, exc: Exception) -> None:
        """Logs an exception with full traceback."""
        self._ensure_handler()
        self._logger.error("[%s] EXCEPTION  %s", step_id, str(exc), exc_info=exc)

    def log_debug(self, step_id: str, message: str) -> None:
        """General-purpose DEBUG entry."""
        self._ensure_handler()
        self._logger.debug("[%s] %s", step_id, message)

    def _format_perception_candidate(self, element: Dict[str, Any], rank: int, source: str) -> str:
        """Formats one candidate into a compact, single-line summary."""
        text = element.get("text") or element.get("placeholder") or element.get("label") or element.get("name") or "-"
        position = element.get("spatial", {}).get("position") or element.get("position") or "-"
        salience = self._format_optional_float(element.get("salience"))
        similarity = self._format_optional_float(element.get("similarity"))
        geometry = self._format_geometry(element)
        return f"#{rank}[src={source} text={text!r} pos={position} " f"sal={salience} sim={similarity} geom={geometry}]"

    def _format_geometry(self, element: Dict[str, Any]) -> str:
        """Returns a compact geometry summary from bounds or normalized location."""
        bounds = element.get("bounds")
        if isinstance(bounds, (list, tuple)) and len(bounds) == 4:
            joined = ",".join(self._format_coord(value) for value in bounds)
            return f"b({joined})"

        location = element.get("location")
        if isinstance(location, (list, tuple)) and len(location) == 4:
            joined = ",".join(f"{float(value):.2f}" for value in location)
            return f"loc({joined})"

        return "-"

    @staticmethod
    def _format_optional_float(value: Any) -> str:
        """Formats optional numeric values with short placeholders when missing."""
        if value is None:
            return "-"
        try:
            return f"{float(value):.2f}"
        except (TypeError, ValueError):
            return "-"

    @staticmethod
    def _format_coord(value: Any) -> str:
        """Formats coordinates without trailing .0 for integer-like values."""
        try:
            number = float(value)
        except (TypeError, ValueError):
            return str(value)
        if number.is_integer():
            return str(int(number))
        return f"{number:.2f}"


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


def configure_logging(*, debug_enabled: bool) -> None:
    """Configure whether new run loggers should emit DEBUG entries."""

    global _DEBUG_ENABLED  # pylint: disable=global-statement
    _DEBUG_ENABLED = debug_enabled


def reset_logger() -> None:
    """Resets the singleton (mainly useful in tests)."""
    global _RUN_TIMESTAMP, _DEBUG_ENABLED  # pylint: disable=global-statement
    for logger in _INSTANCES.values():
        for handler in list(logger._logger.handlers):  # pylint: disable=protected-access
            handler.close()
            logger._logger.removeHandler(handler)  # pylint: disable=protected-access
    _INSTANCES.clear()
    _RUN_TIMESTAMP = None
    _DEBUG_ENABLED = True
