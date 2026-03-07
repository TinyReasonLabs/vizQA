"""
Planner module for decomposing high-level instructions into atomic steps.
"""

import re
from typing import Any, Dict, List

from memory import StepStatus, TestStep


class StepPlanner:  # pylint: disable=too-few-public-methods
    """
    Decomposes natural language instructions into FIND, DO, and VERIFY atoms.
    """

    def __init__(self, model_name: str = "minilm"):
        self.model_name = model_name

    def decompose(self, raw_steps: List[Dict[str, Any]]) -> List[TestStep]:
        """
        Decomposes raw YAML steps into TestStep objects with sub-steps.
        Uses heuristics to simulate MiniLM's semantic breakdown.
        """
        refined_steps = []
        for i, step in enumerate(raw_steps):
            instruction = step.get("action", "")
            expectation = step.get("expect")

            # Generate sub-steps for better visibility
            sub_steps = self._decompose_instruction(instruction, i)

            if expectation:
                sub_steps.extend(self._decompose_expectation(expectation, i))

            refined_steps.append(
                TestStep(
                    id=f"step_{i}",
                    instruction=instruction,
                    expectation=expectation,
                    status=StepStatus.PENDING,
                    sub_steps=sub_steps,
                )
            )
        return refined_steps

    def _decompose_instruction(self, instruction: str, parent_idx: int) -> List[TestStep]:
        """Simulates MiniLM breaking down a high-level instruction into FIND and DO atoms."""
        sub_steps = []
        instr = instruction.lower()

        target = ""
        action = ""
        payload = ""

        # Heuristic for finding the "target" and "action"
        if "click" in instr or "tap" in instr:
            action = "click"
            target = re.sub(r"\b(click|tap|the|button)\b", "", instr).strip()
        elif "type" in instr or "enter" in instr:
            action = "type"
            match = re.search(r"['\"](.+?)['\"]", instruction)
            if match:
                payload = match.group(1)
                stripped = re.sub(r"\b(type|enter|into|the|field|input)\b", "", instr)
                target = stripped.replace(payload.lower(), "").replace("''", "").replace('""', "").strip()
            else:
                target = re.sub(r"\b(type|enter|into|the|field|input)\b", "", instr).strip()
        elif "hover" in instr or "move" in instr:
            action = "hover"
            target = re.sub(r"\b(hover|move|to|the)\b", "", instr).strip()

        if target:
            sub_steps.append(TestStep(id=f"step_{parent_idx}_find", instruction=f"FIND: {target}"))

        if action:
            cmd = f"DO: {action}"
            if payload:
                cmd += f" {payload}"
            sub_steps.append(TestStep(id=f"step_{parent_idx}_do", instruction=cmd))

        return sub_steps

    def _decompose_expectation(self, expectation: str, parent_idx: int) -> List[TestStep]:
        """Simulates MiniLM breaking down an expectation into VERIFY atoms."""
        # MiniLM simplifies "Username field should contain 'admin'" to "VERIFY: admin"
        match = re.search(r"['\"](.+?)['\"]", expectation)
        if match:
            query = match.group(1)
        else:
            # Heuristic: exclude boilerplate words
            query = re.sub(r"\b(should|contain|appear|be|shown|in|the)\b", "", expectation.lower()).strip()

        return [TestStep(id=f"step_{parent_idx}_verify", instruction=f"VERIFY: {query}")]
