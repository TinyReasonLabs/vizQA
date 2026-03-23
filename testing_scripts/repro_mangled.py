# pylint: disable=all
import os

from vizQA.minilm import MiniLM
from vizQA.parser import SemanticParser
from vizQA.planner import StepPlanner


def test_mangled_output():
    model_dir = os.path.join("vizQA", "weights", "minilm")
    if not os.path.exists(model_dir):
        print("Model not found, skipping...")
        return

    model = MiniLM(model_dir)
    planner = StepPlanner()
    # Support manual injection for testing if needed, though StepPlanner loads its own
    planner.minilm = model
    planner.parser.minilm = model

    parser = SemanticParser(minilm=model)

    instructions = [
        "Type 'wrong_user' into the username field",
        "Click the Submit button",
        "A red error toast 'Invalid credentials' should appear at the bottom right",
        "Type 'admin' into the username field and 'password' into the password field",
        "The 'Sign in' modal should close",
        "The Sign In modal should appear",
        "A popup should NOT be visible",
    ]

    for instr in instructions:
        print(f"\nParsing: {instr}")
        try:
            if "should" in instr or "VERIFY" in instr:
                # Test verification parsing
                intent = parser.parse_verify_intent(instr)
                # The parser prints its own [DEBUG] line now
            else:
                steps = planner._decompose_instruction(instr, parent_idx=None)
                for s in steps:
                    print(f"  {s.instruction}")
        except Exception as e:
            print(f"  ERROR: {e}")


if __name__ == "__main__":
    test_mangled_output()
