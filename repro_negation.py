# pylint: disable=all
import os
import sys

# Add the current directory to sys.path to import vizQA
sys.path.append(os.getcwd())

from vizQA.parser import SemanticParser


def test_negation():
    parser = SemanticParser()
    query = "'Sign In' modal should appear in the center of the screen"
    intent = parser.parse_verify_intent(query)
    print(f"Query: {query}")
    print(f"Intent negated: {intent['negated']}")
    print(f"Intent subject: {intent['subject']}")

    query2 = "'Sign In' modal should disappear"
    intent2 = parser.parse_verify_intent(query2)
    print(f"\nQuery: {query2}")
    print(f"Intent negated: {intent2['negated']}")
    print(f"Intent subject: {intent2['subject']}")


if __name__ == "__main__":
    test_negation()
