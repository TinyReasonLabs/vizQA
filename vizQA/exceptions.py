"""
Custom exception hierarchy for vizQA.
"""

from typing import Optional


class UserFacingException(Exception):
    """
    Base class for all exceptions that should be reported to the user
    with a clear, non-technical message.
    """

    def __init__(self, user_message: str, internal_detail: Optional[str] = None):
        super().__init__(user_message)
        self.user_message = user_message
        self.internal_detail = internal_detail


class PerceptionServiceError(UserFacingException):
    """Raised when the perception API or network connection fails."""


class ElementNotFoundError(UserFacingException):
    """Raised when a FIND step fails to locate the target."""


class ActionExecutionError(UserFacingException):
    """Raised when a DO interaction fails at runtime."""


class VerificationError(UserFacingException):
    """Raised when a VERIFY step times out or fails."""


class TestDefinitionError(UserFacingException):
    """Raised when a test file (YAML) or its decomposed steps are invalid."""

    __test__ = False


class ArtifactError(UserFacingException):
    """Raised when a test artifact cannot be resolved or used."""


class BrowserError(UserFacingException):
    """Raised when Playwright or the browser encounters a fatal error."""
