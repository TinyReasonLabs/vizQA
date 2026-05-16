"""Event types consumed by the terminal reporter store."""

from __future__ import annotations

from dataclasses import dataclass

from vizQA.app.memory import TestSession, TestStep
from vizQA.rendering.models import TopLevelRunIdentity


@dataclass(slots=True)
class TopLevelTestStartedEvent(TopLevelRunIdentity):
    """Declare a top-level test owner before dependency execution begins."""


@dataclass(slots=True)
class SessionStartedEvent:
    """A dependency or top-level session has started."""

    owner_key: str
    session: TestSession


@dataclass(slots=True)
class StepStartedEvent:
    """A parent/container step has started running."""

    session_id: str
    step: TestStep


@dataclass(slots=True)
class StepFinishedEvent:
    """A step finished with a final status."""

    session_id: str
    step: TestStep


@dataclass(slots=True)
class SessionBlockedEvent:
    """A session could not run because a prerequisite failed."""

    session_id: str
    reason: str


@dataclass(slots=True)
class SessionFinishedEvent:
    """A session completed running."""

    session: TestSession


@dataclass(slots=True)
class RunFinishedEvent:
    """The entire run has completed."""
