# pylint: disable=all
import os

from vizQA.planner import StepPlanner


def repro():
    model_dir = os.path.join("vizQA", "weights", "minilm")
    if not os.path.exists(model_dir):
        print("Model not found")
        return

    planner = StepPlanner()

    instructions = [
        "Click on 'Settings' in the sidebar",
        "Click on 'Overview' in the sidebar",
        "Scroll to the bottom of the page",
    ]

    for instr in instructions:
        print(f"\nInstruction: {instr}")
        steps = planner._decompose_instruction(instr, parent_idx=0)
        for s in steps:
            print(f"  {s.instruction}")


if __name__ == "__main__":
    repro()
