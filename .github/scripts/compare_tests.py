import json
import sys
import os

def load_results(filename):
    if not os.path.exists(filename):
        return {}
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
            return {test['nodeid']: test['outcome'] for test in data.get('tests', [])}
    except Exception:
        return {}

def main():
    if len(sys.argv) < 3:
        print("Usage: compare_tests.py <base_results.json> <head_results.json>")
        sys.exit(1)

    base_file = sys.argv[1]
    head_file = sys.argv[2]

    base_results = load_results(base_file)
    head_results = load_results(head_file)

    all_tests = set(base_results.keys()) | set(head_results.keys())
    diffs = []
    
    new_regressions = []
    base_failures = [t for t, o in base_results.items() if o in ["failed", "error"]]
    head_failures = [t for t, o in head_results.items() if o in ["failed", "error"]]

    for test in sorted(all_tests):
        base_outcome = base_results.get(test, "N/A")
        head_outcome = head_results.get(test, "N/A")

        if base_outcome != head_outcome:
            diffs.append((test, base_outcome, head_outcome))
            if base_outcome == "passed" and head_outcome in ["failed", "error"]:
                new_regressions.append(test)

    # Print Markdown Summary
    print("### 🧪 Test Comparison Report")
    
    if new_regressions:
        print(f"#### ❌ ATTENTION: {len(new_regressions)} New Regression(s) Detected!")
        for test in new_regressions:
            print(f"- `{test}`: passed ➡️ **failed**")
        print("\n")

    if not diffs:
        print("✅ No changes in test outcomes.")
    else:
        print("| Test Case | Base Branch | PR Branch |")
        print("| --- | --- | --- |")
        for test, base, head in diffs:
            b_str = f"**{base}**" if base in ["failed", "error"] else base
            h_str = f"**{head}**" if head in ["failed", "error"] else head
            print(f"| `{test}` | {b_str} | {h_str} |")

    print(f"\n**Summary:** Base Failures: {len(base_failures)} | PR Failures: {len(head_failures)}")

    # Failure Logic for CI
    should_fail = False
    if new_regressions:
        print("\n❌ CI Failure: New regressions detected.")
        should_fail = True
    elif len(head_failures) > len(base_failures):
        print(f"\n❌ CI Failure: Total number of failing tests increased ({len(base_failures)} -> {len(head_failures)}).")
        should_fail = True

    if should_fail:
        sys.exit(1)

if __name__ == "__main__":
    main()
