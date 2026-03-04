"""
Memory module for tracking test sessions, steps, and status.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class StepStatus(str, Enum):
    """Enumeration of possible statuses for a test step."""

    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


class FailureType(str, Enum):
    """Enumeration of failure categories for test reporting."""

    TIMEOUT = "timeout"
    PERCEPTION_MISMATCH = "perception_mismatch"
    ACTION_ERROR = "action_error"
    SYSTEM_ERROR = "system_error"
    NONE = "none"


class ActionType(str, Enum):
    """Enumeration of physical interaction types."""

    CLICK = "click"
    HOVER = "hover"
    TYPE = "type"
    NAVIGATE = "navigate"
    ASSERT = "assert"
    CUSTOM = "custom"


# pylint: disable=too-few-public-methods
class TestStep(BaseModel):
    """
    Represents a single step in a test session, including its status and results.
    """

    id: str
    instruction: str
    expectation: Optional[str] = None
    status: StepStatus = StepStatus.PENDING
    action_taken: Optional[str] = None
    perception_result: Optional[Dict[str, Any]] = None
    screenshot_before: Optional[str] = None
    screenshot_after: Optional[str] = None
    action_screenshot: Optional[str] = None
    error: Optional[str] = None
    failure_type: FailureType = FailureType.NONE
    failure_reason: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    sub_steps: List["TestStep"] = []


TestStep.model_rebuild()


class TestSession(BaseModel):
    """
    Represents a full test execution session containing multiple steps.
    """

    id: str
    test_name: str
    url: str
    steps: List[TestStep] = []
    start_time: datetime = Field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    metadata: Dict[str, Any] = {}

    @property
    def duration(self) -> float:
        """Calculates the duration of the session in seconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return (datetime.now() - self.start_time).total_seconds()

    def to_json(self) -> str:
        """Serializes the session to a JSON string."""
        return self.model_dump_json(indent=2)

    def save(self, path: str):
        """Saves the session state to a file."""
        with open(path, "w", encoding="utf-8") as file:
            file.write(self.to_json())


class TestSuite(BaseModel):  # pylint: disable=too-few-public-methods
    """
    Represents a collection of test sessions.
    """

    name: str
    description: Optional[str] = None
    sessions: List[TestSession] = []
