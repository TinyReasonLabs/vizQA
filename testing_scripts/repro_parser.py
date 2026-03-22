# pylint: disable=all
import os

from vizQA.minilm import MiniLM
from vizQA.parser import SemanticParser


def test_parser():
    model_dir = "d:/Manually Transferred/Projects/Coding/UI_testing/vizQA/weights/minilm"
    minilm = None
    if os.path.exists(model_dir):
        minilm = MiniLM(model_dir)

    parser = SemanticParser(minilm=minilm)

    test_cases = [
        "Click the primary Login button header",
        "Right click the 'Customers' row",
        "Click the button on the right",
        "Verify the 'Delete' button is red and aligned to the right",
    ]

    for tc in test_cases:
        print(f"\nInstruction: {tc}")
        nodes = parser.parse(tc)
        for node in nodes:
            print(f"  {node.type}: {node.value}")


if __name__ == "__main__":
    test_parser()
