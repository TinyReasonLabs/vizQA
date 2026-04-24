"""
Planner module for decomposing high-level instructions into atomic steps.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from vizQA.app.exceptions import TestDefinitionError
from vizQA.app.logger import get_logger
from vizQA.app.memory import StepStatus, TestStep
from vizQA.reasoning import MiniLM, SemanticParser


class StepPlanner:  # pylint: disable=too-few-public-methods
    """
    Decomposes natural language instructions into FIND, DO, and VERIFY atoms.
    """

    def __init__(
        self,
        model_name: str = "minilm",
        parser: Optional[SemanticParser] = None,
        minilm: Optional[Any] = None,
        logger: Optional[Any] = None,
    ):
        self.model_name = model_name
        self._logger = logger or get_logger()

        if parser:
            self.parser = parser
        else:
            self.parser = SemanticParser()

        if minilm:
            self.minilm = minilm
            self.parser.minilm = minilm
        else:
            # Load MiniLM if possible
            self.minilm = None
            if model_name == "minilm":
                model_dir = Path(__file__).resolve().parents[1] / "weights" / "minilm"
                try:
                    self.minilm = MiniLM(str(model_dir), logger=self._logger)
                    # Synergize with parser
                    self.parser.minilm = self.minilm
                except (FileNotFoundError, RuntimeError, ImportError) as e:
                    self._logger.log_debug(
                        0, f"MiniLM model could not be loaded: {e}. Falling back to rule-based parser."
                    )

    def decompose(self, raw_steps: List[Dict[str, Any]]) -> List[TestStep]:
        """
        Decomposes raw YAML steps into TestStep objects with sub-steps.
        Uses MiniLM ONNX for semantic breakdown.
        """
        refined_steps = []
        for i, step in enumerate(raw_steps):
            instruction = step.get("action", "")
            expectation = step.get("expect")
            line_num = step.get("__line__", "unknown")

            try:
                # Generate sub-steps for better visibility
                sub_steps = self._decompose_instruction(instruction, i)

                if expectation:
                    sub_steps.extend(self._decompose_expectation(expectation, i, len(sub_steps)))
            except (ValueError, RuntimeError) as e:
                # Wrap as user-facing definition error
                raise TestDefinitionError(
                    f"Failed to plan steps for instruction '{instruction}' (YAML line {line_num})",
                    internal_detail=str(e),
                ) from e

            refined_steps.append(
                TestStep(
                    id=f"step_{i:02d}",
                    instruction=instruction,
                    expectation=expectation,
                    status=StepStatus.PENDING,
                    sub_steps=sub_steps,
                )
            )
        return refined_steps

    def _decompose_instruction(self, instruction: str, parent_idx: int) -> List[TestStep]:
        """Uses SemanticParser or MiniLM to break down a high-level instruction."""
        try:
            if self.model_name == "minilm" and self.minilm:
                dict_steps = self.minilm.predict(instruction)
            else:
                nodes = self.parser.parse(instruction)
                dict_steps = [{"type": n.type, "value": n.value} for n in nodes]

            return self._to_test_steps(dict_steps, parent_idx, "instr")
        except Exception as e:
            if isinstance(e, RuntimeError):
                raise
            raise RuntimeError(f"Semantic decomposition failed for instruction '{instruction}': {e}") from e

    def _decompose_expectation(self, expectation: str, parent_idx: int, start_idx: int) -> List[TestStep]:
        """Uses SemanticParser to break down an expectation into VERIFY atoms."""
        try:
            # Force verifications to be treated as such by prefixing slightly or just parsing
            # The parser has a specific _parse_verify we could call, or we just prefix "Verify "
            nodes = self.parser.parse(f"Verify {expectation}")
            dict_steps = [{"type": n.type, "value": n.value} for n in nodes]
            return self._to_test_steps(dict_steps, parent_idx, "expect", start_idx)
        except Exception as e:
            raise RuntimeError(f"Semantic decomposition failed for expectation '{expectation}': {e}") from e

    def _to_test_steps(
        self, model_steps: List[Dict[str, str]], parent_idx: int, _prefix: str, start_idx: int = 0
    ) -> List[TestStep]:
        """Converts model output steps into TestStep objects."""
        test_steps = []
        for j, step in enumerate(model_steps):
            step_type = step["type"].upper()
            value = step["value"]

            # Validation
            if step_type not in ("FIND", "DO", "VERIFY"):
                raise ValueError(f"Invalid step type from model: {step_type}")

            p_idx = parent_idx if parent_idx is not None else 0
            test_steps.append(
                TestStep(id=f"step_{p_idx:02d}.{(j+start_idx+1):02d}", instruction=f"{step_type}: {value}")
            )
        return test_steps
