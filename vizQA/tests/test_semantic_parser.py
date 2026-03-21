import re

import pytest

from vizQA.planner import StepPlanner

# A comprehensive suite of 50 diverse UI testing instructions to test the AST parser's ability
# to extract relations (actions -> subjects, states -> subjects) and clean descriptions.
TEST_CASES = [
    # 1. Simple atomic actions
    ("Click the login button", [("FIND", "login button"), ("DO", "click")]),
    ("Tap 'Submit'", [("FIND", "'Submit'"), ("DO", "tap")]),
    ("Hover over the profile icon", [("FIND", "profile icon"), ("DO", "hover")]),
    # 4. Explicit payloads
    ("Type 'admin' into the username field", [("FIND", "username field"), ("DO", "type 'admin'")]),
    ('Enter "password123" in the password box', [("FIND", "password box"), ("DO", 'type "password123"')]),
    # 6. Implicit payloads (no quotes)
    ("Type admin into the username field", [("FIND", "username field"), ("DO", "type admin")]),
    ("Enter myemail@test.com", [("FIND", "myemail@test.com"), ("DO", "enter")]),
    # 8. Adjectives and location descriptions (Subject + Description)
    ("Click the primary Login button in the header", [("FIND", "primary Login button in the header"), ("DO", "click")]),
    ("Hover the small red warning icon", [("FIND", "small red warning icon"), ("DO", "hover")]),
    ("Tap the first item in the dropdown list", [("FIND", "first item in the dropdown list"), ("DO", "tap")]),
    ("Check the 'I agree' checkbox at the bottom", [("FIND", "'I agree' checkbox at the bottom"), ("DO", "check")]),
    # 12. Multiple targets with one action (Implicit split)
    (
        "Click the submit and cancel buttons",
        [("FIND", "submit buttons"), ("DO", "click"), ("FIND", "cancel buttons"), ("DO", "click")],
    ),
    (
        "Select the 'apple' and 'banana' options",
        [("FIND", "element"), ("DO", "select 'apple'"), ("FIND", "element"), ("DO", "select 'banana' options")],
    ),
    # 14. Complex conjunctions (Multiple actions and targets)
    (
        "Type 'admin' into the username field and 'password' into the password field",
        [("FIND", "username field"), ("DO", "type 'admin'"), ("FIND", "password field"), ("DO", "type 'password'")],
    ),
    (
        "Click 'New Post', then type 'Hello' in the title box",
        [("FIND", "'New Post'"), ("DO", "click"), ("FIND", "title box"), ("DO", "type 'Hello'")],
    ),
    (
        "Hover the menu and click 'Settings'",
        [("FIND", "menu"), ("DO", "hover"), ("FIND", "'Settings'"), ("DO", "click")],
    ),
    # 17. Verifications (States to observe -> Subjects)
    (
        "A 'Sign In' modal should appear in the center of the screen",
        [("VERIFY", "'Sign In' modal should appear in the center of the screen")],
    ),
    (
        "The red error toast 'Invalid credentials' should appear at the bottom right",
        [("VERIFY", "red error toast 'Invalid credentials' should appear at the bottom right")],
    ),
    ("The modal should close", [("VERIFY", "modal should close")]),
    (
        "A 'Login Successful' alert or state change should occur",
        [("VERIFY", "'Login Successful' alert or state change should occur")],
    ),
    ("Verify that the submit button is disabled", [("VERIFY", "submit button is disabled")]),
    ("Ensure the shopping cart counter shows '3'", [("VERIFY", "shopping cart counter shows '3'")]),
    # 23. Verifications with complex contexts
    (
        "The user profile image should load successfully without errors",
        [("VERIFY", "user profile image should load successfully without errors")],
    ),
    (
        "Verify the 'Delete' button is red and aligned to the right",
        [("VERIFY", "'Delete' button is red"), ("VERIFY", "aligned to the right")],
    ),
    # 25. Edge cases: Missing structural words
    ("type john_doe", [("FIND", "john_doe"), ("DO", "type")]),
    (
        "Submit",
        [("FIND", "Submit"), ("DO", "click Submit")],
    ),  # Wait, is submit an action? Generally click is the action on a submit button. Let's parse as click submit.
    ("Click", [("FIND", "element"), ("DO", "click")]),
    # 28. Noise words and complex boilerplate
    (
        "Please navigate ahead and click on the bright blue confirm button located on the top right",
        [("FIND", "bright blue confirm button located on the top right"), ("DO", "click")],
    ),
    (
        "I want you to type 'search query' into the main search bar",
        [("FIND", "main search bar"), ("DO", "type 'search query'")],
    ),
    # 30. Select/Dropdown
    (
        "Select 'United States' from the Country dropdown",
        [("FIND", "Country dropdown"), ("DO", "select 'United States'")],
    ),
    ("Choose the second option in the list", [("FIND", "second option in the list"), ("DO", "choose")]),
    # 32. Drag and drop (Complex relational)
    (
        "Drag the image onto the upload zone",
        [("FIND", "image"), ("DO", "drag"), ("FIND", "upload zone"), ("DO", "drop")],
    ),
    # 33. Scroll actions
    ("Scroll down to the bottom of the page", [("FIND", "down bottom of page"), ("DO", "scroll")]),
    ("Scroll to the 'Reviews' section", [("FIND", "'Reviews' section"), ("DO", "scroll")]),
    # 35. Keyboard actions
    ("Press Enter on the search box", [("FIND", "search box"), ("DO", "press Enter")]),
    ("Hit the Escape key", [("FIND", "key"), ("DO", "press Escape")]),
    # 37. Implicit context from previous clauses
    (
        "Clear the input and type 'new text'",
        [("FIND", "input"), ("DO", "clear"), ("FIND", "'new text'"), ("DO", "type")],
    ),  # Should carry over target
    # 38. Real-world messy inputs
    ("click on settings gear icon", [("FIND", "settings gear icon"), ("DO", "click")]),
    ("input mypassword into pass field", [("FIND", "pass field"), ("DO", "type mypassword")]),
    ("make sure the popup is visible", [("VERIFY", "popup is visible")]),
    ("assert that the header says 'Welcome back'", [("VERIFY", "header says 'Welcome back'")]),
    (
        "Find the 'forgot password' link and click it",
        [("FIND", "'forgot password' link"), ("DO", "find"), ("FIND", "'forgot password' link"), ("DO", "click")],
    ),
    # 43. Extremely long descriptive subjects
    (
        "Click the tiny semi-transparent close icon located in the upper right corner of the overlapping sticky banner",
        [
            (
                "FIND",
                "tiny semi-transparent close icon located in the upper right corner of the overlapping sticky banner",
            ),
            ("DO", "click"),
        ],
    ),
    # 44. Conjunctions with states
    ("Click Submit -> The modal should close", [("FIND", "Submit"), ("DO", "click"), ("VERIFY", "modal should close")]),
    # 45. Multiple implicit payloads
    (
        "Type first name john and last name doe",
        [("FIND", "first name"), ("DO", "type john"), ("FIND", "last name"), ("DO", "type doe")],
    ),
    # 46. Wait or sleep instructions
    ("Wait for 5 seconds", [("FIND", "for 5 seconds"), ("DO", "wait for 5 seconds")]),
    ("Pause until the loader disappears", [("VERIFY", "loader disappears")]),
    # 48. Double quotes and escaped chars
    ('Type "user\'s name" in the field', [("FIND", "field"), ("DO", 'type "user\'s name"')]),
    # 49. Ambiguous verbs used as nouns
    ("Click the 'Click Here' button", [("FIND", "'Click Here' button"), ("DO", "click")]),
    # 50. Negative verifications
    # 50. Negative verifications
    ("Ensure the error message is not displayed", [("VERIFY", "error message is not displayed")]),
    # --- 10 Tricky/Robust Cases ---
    # 51. Multi-word synonym for right-click
    ("Right click the red button", [("FIND", "red button"), ("DO", "right click")]),
    # 52. Common context menu synonym
    ("context-click the icon", [("FIND", "icon"), ("DO", "context-click")]),
    # 53. Word overlap: 'click' + 'right' (positional) -> should NOT be right-click
    ("click on the right side of the banner", [("FIND", "right side of banner"), ("DO", "click")]),
    # 54. Variation of 'tap'
    ("tap the screen twice", [("FIND", "screen twice"), ("DO", "tap")]),
    # 55. 'hit' as action
    ("hit the submit button", [("FIND", "submit button"), ("DO", "hit")]),
    # 56. Longer context phrase for right-click
    ("perform right-click on the link", [("FIND", "link"), ("DO", "perform right-click")]),
    # 57. 'right' as part of an adverbial phrase 'right here'
    ("click right here", [("FIND", "right here"), ("DO", "click")]),
    # 58. Leading noise clause
    ("right then, click the logout", [("FIND", "logout"), ("DO", "click")]),
    # 59. Complex DO with 'and'
    ("press and hold the button", [("FIND", "button"), ("DO", "press-and-hold")]),
    # 60. Chained action and verification
    (
        "right click -> modal should close",
        [("FIND", "element"), ("DO", "right click"), ("VERIFY", "modal should close")],
    ),
]


def normalize_text(text: str) -> str:
    """Collapses whitespace, lowercases, and removes noise words (the, a, an, of, etc)."""
    if not text:
        return ""
    text = text.lower().strip()
    # Normalize punctuation/spaces
    text = re.sub(r"[^a-z0-9'\"]+", " ", text)
    # Remove redundant articles and prepositions that might be stripped inconsistently
    text = re.sub(r"\b(the|a|an|of|to|on|at|in|with|from|by|about|around)\b", " ", text, flags=re.I)
    return " ".join(text.split())


@pytest.fixture(scope="module")
def planner():
    return StepPlanner(model_name="minilm")


@pytest.mark.parametrize("instruction, expected_steps", TEST_CASES)
def test_semantic_parser(planner, instruction, expected_steps):
    raw_steps = [{"action": instruction}]
    test_steps = planner.decompose(raw_steps)

    actual_tuples = []
    if test_steps and test_steps[0].sub_steps:
        for sub in test_steps[0].sub_steps:
            parts = sub.instruction.split(": ", 1)
            if len(parts) == 2:
                actual_tuples.append((parts[0], normalize_text(parts[1])))

    expected_tuples = [(t, normalize_text(v)) for t, v in expected_steps]

    print(f"Instruction: {instruction}")
    print(f"Expected: {expected_tuples}")
    print(f"Actual: {actual_tuples}")

    assert (
        actual_tuples == expected_tuples
    ), f"expected: {expected_tuples}, actual: {actual_tuples}, instruction: {instruction}"
