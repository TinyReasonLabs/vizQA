from vizQA.planner import StepPlanner


def test_planner():
    planner = StepPlanner(model_name="minilm")

    raw_steps = [
        {"action": "Click the primary Login button header"},
        {"action": "Right click the 'Customers' row"},
        {"action": "Click the button on the right"},
        {"action": "Verify the 'Delete' button is red and aligned to the right"},
        {"action": "Hit the Escape key"},
        {"action": "Find the 'forgot password' link and click it"},
    ]

    test_steps = planner.decompose(raw_steps)
    for step in test_steps:
        print(f"\nInstruction: {step.instruction}")
        for sub in step.sub_steps:
            print(f"  {sub.instruction}")


if __name__ == "__main__":
    test_planner()
