"""
Planner module for decomposing high-level instructions into atomic steps.
"""

import os
from typing import Any, Dict, List

from vizQA.memory import StepStatus, TestStep
from vizQA.minilm import MiniLM


class StepPlanner:  # pylint: disable=too-few-public-methods
    """
    Decomposes natural language instructions into FIND, DO, and VERIFY atoms.
    """

    def __init__(self, model_name: str = "minilm"):
        self.model_name = model_name
        self.weights_path = os.path.join(os.path.dirname(__file__), "weights", model_name)

        # Initialize MiniLM ONNX model
        try:
            self.model = MiniLM(self.weights_path)
        except Exception as e:
            # Fallback to regex or raise depending on production requirements.
            # Given the user wants it production ready and solid, we should probably raise
            # if the model fails to load, but for now we'll ensure we have a clear error.
            raise RuntimeError(f"Failed to initialize StepPlanner model: {e}") from e

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
        """Uses MiniLM to break down a high-level instruction into FIND and DO atoms."""
        prompt = f"Decompose instruction into atomic FIND and DO steps: {instruction}"

        try:
            model_steps = self.model.predict(prompt)
            return self._to_test_steps(model_steps, parent_idx, "instr")
        except Exception as e:
            raise RuntimeError(f"Semantic decomposition failed for instruction '{instruction}': {e}") from e

    def _decompose_expectation(self, expectation: str, parent_idx: int) -> List[TestStep]:
        """Uses MiniLM to break down an expectation into VERIFY atoms."""
        prompt = f"Decompose expectation into atomic VERIFY steps: {expectation}"

        try:
            model_steps = self.model.predict(prompt)
            return self._to_test_steps(model_steps, parent_idx, "expect")
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
