"""Typed parsing for deterministic YAML v2 steps.

V2 deliberately describes browser operations as data.  It does not interpret
author prose: every visible UI reference is passed unchanged to UI Perception.
"""

from enum import Enum
from typing import Any, Dict, Iterable, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from vizQA.app.exceptions import TestDefinitionError
from vizQA.app.memory import StepStatus, TestStep


class OperationName(str, Enum):
    """Operations accepted by deterministic YAML v2."""

    CLICK = "click"
    RIGHT_CLICK = "right_click"
    HOVER = "hover"
    TYPE = "type"
    CLEAR = "clear"
    PRESS_KEY = "press_key"
    CHECK = "check"
    UNCHECK = "uncheck"
    SELECT = "select"
    DRAG = "drag"
    UPLOAD = "upload"
    SCROLL = "scroll"
    WAIT = "wait"
    ASSERT_VISIBLE = "assert_visible"
    ASSERT_NOT_VISIBLE = "assert_not_visible"


_TARGET_OPERATIONS = {
    OperationName.CLICK,
    OperationName.RIGHT_CLICK,
    OperationName.HOVER,
    OperationName.TYPE,
    OperationName.CLEAR,
    OperationName.CHECK,
    OperationName.UNCHECK,
    OperationName.SELECT,
    OperationName.UPLOAD,
    OperationName.ASSERT_VISIBLE,
    OperationName.ASSERT_NOT_VISIBLE,
}


class StructuredOperation(BaseModel):
    """One validated v2 operation and its explicit execution parameters."""

    model_config = ConfigDict(extra="forbid")

    action: OperationName
    target: Optional[str] = Field(default=None, description="Visible UI query sent to UI Perception.")
    text: Optional[str] = None
    key: Optional[str] = None
    option: Optional[str] = None
    source: Optional[str] = None
    file: Optional[str] = None
    position: Optional[str] = None
    seconds: Optional[float] = Field(default=None, ge=0)
    timeout: Optional[float] = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_contract(self) -> "StructuredOperation":
        """Ensure each operation carries only its required deterministic input."""
        if self.action in _TARGET_OPERATIONS and not self.target:
            raise ValueError(f"'{self.action.value}' requires a non-empty target")
        if self.action == OperationName.TYPE and self.text is None:
            raise ValueError("'type' requires text")
        if self.action == OperationName.PRESS_KEY and not self.key:
            raise ValueError("'press_key' requires key")
        if self.action == OperationName.SELECT and not self.option:
            raise ValueError("'select' requires option")
        if self.action == OperationName.DRAG and (not self.source or not self.target):
            raise ValueError("'drag' requires source and target")
        if self.action == OperationName.UPLOAD and not self.file:
            raise ValueError("'upload' requires file")
        if self.action == OperationName.SCROLL and not (self.position or self.target):
            raise ValueError("'scroll' requires position ('top' or 'bottom') or target")
        if self.action == OperationName.SCROLL and self.position not in (None, "top", "bottom"):
            raise ValueError("scroll position must be 'top' or 'bottom'")
        if self.action == OperationName.WAIT and self.seconds is None and not self.target:
            raise ValueError("'wait' requires seconds or target")
        if self.action == OperationName.WAIT and self.seconds is not None and self.target:
            raise ValueError("'wait' accepts either seconds or target, not both")
        return self


def parse_structured_steps(raw_steps: Iterable[Dict[str, Any]]) -> List[TestStep]:
    """Validate v2 YAML steps and adapt them to existing canonical sub-steps."""
    planned: List[TestStep] = []
    for index, raw_step in enumerate(raw_steps):
        line = raw_step.get("__line__", "unknown") if isinstance(raw_step, dict) else "unknown"
        if not isinstance(raw_step, dict):
            raise TestDefinitionError(f"V2 step at YAML line {line} must be a mapping.")
        keys = [key for key in raw_step if key != "__line__"]
        if len(keys) != 1:
            raise TestDefinitionError(
                f"V2 step at YAML line {line} must contain exactly one operation; found {keys!r}."
            )
        action_name = keys[0]
        if action_name in ("action", "expect"):
            raise TestDefinitionError(f"V2 step at YAML line {line} cannot use legacy action/expect fields.")
        payload = raw_step[action_name]
        if payload is None:
            payload = {}
        if not isinstance(payload, dict):
            raise TestDefinitionError(f"V2 operation '{action_name}' at YAML line {line} must be a mapping.")
        try:
            operation = StructuredOperation(action=action_name, **payload)
        except (ValueError, TypeError) as exc:
            detail = str(exc).split("\n")[-2] if "\n" in str(exc) else str(exc)
            raise TestDefinitionError(
                f"Invalid v2 operation '{action_name}' at YAML line {line}: {detail}", internal_detail=str(exc)
            ) from exc
        planned.append(_to_canonical_step(operation, index))
    return planned


# pylint: disable=too-many-branches
def _to_canonical_step(operation: StructuredOperation, index: int) -> TestStep:
    """Compile one typed operation into the runtime's existing internal commands."""
    prefix = f"step_{index:02d}"
    sub_steps: List[TestStep] = []

    # pylint: disable=too-many-arguments
    def add(
        instruction: str,
        *,
        ranked_only: bool = False,
        expect_absent: bool = False,
        timeout: Optional[float] = None,
        wait_seconds: Optional[float] = None,
        wait_for_target: bool = False,
        scroll_position: Optional[str] = None,
    ) -> None:
        sub_steps.append(
            TestStep(
                id=f"{prefix}.{len(sub_steps) + 1:02d}",
                instruction=instruction,
                ranked_only=ranked_only,
                expect_absent=expect_absent,
                timeout_seconds=timeout,
                wait_seconds=wait_seconds,
                wait_for_target=wait_for_target,
                scroll_position=scroll_position,
            )
        )

    action = operation.action
    if action == OperationName.PRESS_KEY:
        add(f"DO: press-key {operation.key}")
    elif action == OperationName.WAIT:
        if operation.seconds is not None:
            add("DO: wait", wait_seconds=operation.seconds)
        else:
            add(f"VERIFY: {operation.target}", ranked_only=True, timeout=operation.timeout, wait_for_target=True)
    elif action == OperationName.SCROLL:
        if operation.position:
            add("DO: scroll", scroll_position=operation.position)
        else:
            add(f"DO: scroll {operation.target}", ranked_only=True)
    elif action == OperationName.ASSERT_VISIBLE:
        add(f"VERIFY: {operation.target}", ranked_only=True, timeout=operation.timeout)
    elif action == OperationName.ASSERT_NOT_VISIBLE:
        add(f"VERIFY: {operation.target}", ranked_only=True, expect_absent=True, timeout=operation.timeout)
    elif action == OperationName.DRAG:
        add(f"FIND: {operation.source}", ranked_only=True)
        add("DO: drag")
        add(f"FIND: {operation.target}", ranked_only=True)
        add("DO: drop")
    elif action == OperationName.UPLOAD:
        add(f"FIND: {operation.file}")
        add("DO: drag")
        add(f"FIND: {operation.target}", ranked_only=True)
        add("DO: drop")
    elif action == OperationName.SELECT:
        add(f"FIND: {operation.target}", ranked_only=True)
        add("DO: click")
        add(f"FIND: {operation.option}", ranked_only=True)
        add("DO: click")
    else:
        interaction = {
            OperationName.RIGHT_CLICK: "right-click",
            OperationName.HOVER: "hover",
            OperationName.CHECK: "click",
            OperationName.UNCHECK: "click",
        }.get(action, action.value)
        add(f"FIND: {operation.target}", ranked_only=True)
        payload = f" {operation.text}" if action == OperationName.TYPE else ""
        add(f"DO: {interaction}{payload}")

    return TestStep(id=prefix, instruction=operation.action.value, status=StepStatus.PENDING, sub_steps=sub_steps)
