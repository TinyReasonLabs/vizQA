"""
Planner module for decomposing high-level instructions into atomic steps.
"""

import os
from typing import Any, Dict, List

from vizQA.memory import StepStatus, TestStep
from vizQA.parser import SemanticParser


class StepPlanner:  # pylint: disable=too-few-public-methods
    """
    Decomposes natural language instructions into FIND, DO, and VERIFY atoms.
    """

    def __init__(self, model_name: str = "minilm"):
        # model_name is kept for backwards compatibility but we now use the AST parser
        self.model_name = "ast_parser"
        self.parser = SemanticParser()

    def decompose(self, raw_steps: List[Dict[str, Any]]) -> List[TestStep]:
        """
        Decomposes raw YAML steps into TestStep objects with sub-steps.
        Uses MiniLM ONNX for semantic breakdown.
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
        """Uses SemanticParser to break down a high-level instruction into FIND and DO atoms."""
        try:
            nodes = self.parser.parse(instruction)
            # Filter to only FIND and DO. If a user mixed verify here, we drop or keep it?
            # The parser handles -> so we just take all nodes
            dict_steps = [{"type": n.type, "value": n.value} for n in nodes]
            return self._to_test_steps(dict_steps, parent_idx, "instr")
        except Exception as e:
            raise RuntimeError(f"Semantic decomposition failed for instruction '{instruction}': {e}") from e

    def _decompose_expectation(self, expectation: str, parent_idx: int) -> List[TestStep]:
        """Uses SemanticParser to break down an expectation into VERIFY atoms."""
        try:
            # Force verifications to be treated as such by prefixing slightly or just parsing
            # The parser has a specific _parse_verify we could call, or we just prefix "Verify "
            nodes = self.parser.parse(f"Verify {expectation}")
            dict_steps = [{"type": n.type, "value": n.value} for n in nodes]
            return self._to_test_steps(dict_steps, parent_idx, "expect")
        except Exception as e:
            raise RuntimeError(f"Semantic decomposition failed for expectation '{expectation}': {e}") from e

    def _to_test_steps(self, model_steps: List[Dict[str, str]], parent_idx: int, prefix: str) -> List[TestStep]:
        """Converts model output steps into TestStep objects."""
        test_steps = []
        for j, step in enumerate(model_steps):
            step_type = step["type"].upper()
            value = step["value"]

            # Validation
            if step_type not in ("FIND", "DO", "VERIFY"):
                raise ValueError(f"Invalid step type from model: {step_type}")

            test_steps.append(TestStep(id=f"step_{parent_idx}_{prefix}_{j}", instruction=f"{step_type}: {value}"))
        return test_steps
