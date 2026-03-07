import sys

from vizQA.parser import SemanticParser
from vizQA.tests.test_semantic_parser import TEST_CASES


def main():
    parser = SemanticParser()
    failures = 0
    for i, (instr, expected) in enumerate(TEST_CASES):
        actual = parser.parse(instr)
        actual_tups = [(a.type, a.value) for a in actual]
        if actual_tups != expected:
            print(f"[{i+1}/{len(TEST_CASES)}] FAILED: {instr}")
            print(f"  Expected: {expected}")
            print(f"  Actual:   {actual_tups}")
            print("-" * 60)
            failures += 1

    print(f"\nTotal Failures: {failures} / {len(TEST_CASES)}")


if __name__ == "__main__":
    main()
