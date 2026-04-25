"""Application-layer package exports."""

from vizQA.app.client import PerceptionClient
from vizQA.app.config import CONFIG, PERCEPTION_BACKEND, ParserConfig
from vizQA.app.core import Automator
from vizQA.app.exceptions import (
    ActionExecutionError,
    ArtifactError,
    BrowserError,
    ElementNotFoundError,
    PerceptionServiceError,
    TestDefinitionError,
    UserFacingException,
    VerificationError,
)
from vizQA.app.logger import SessionLogger, get_logger
from vizQA.app.memory import ActionType, FailureType, StepStatus, TestSession, TestStep, TestSuite
from vizQA.app.viewport import BUILTIN_VIEWPORTS, ViewportConfig, ViewportSpec, load_viewport_config, resolve_viewports

__all__ = [
    "PerceptionClient",
    "CONFIG",
    "PERCEPTION_BACKEND",
    "ParserConfig",
    "Automator",
    "ActionExecutionError",
    "ArtifactError",
    "BrowserError",
    "ElementNotFoundError",
    "PerceptionServiceError",
    "TestDefinitionError",
    "UserFacingException",
    "VerificationError",
    "SessionLogger",
    "get_logger",
    "ActionType",
    "FailureType",
    "StepStatus",
    "TestSession",
    "TestStep",
    "TestSuite",
    "BUILTIN_VIEWPORTS",
    "ViewportConfig",
    "ViewportSpec",
    "load_viewport_config",
    "resolve_viewports",
]
