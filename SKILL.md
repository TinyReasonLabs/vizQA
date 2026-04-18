---
name: vizqa
description: Use when helping someone use the vizQA CLI, write vizQA YAML test files, debug failing vizQA test runs, understand artifacts and dependencies, or inspect `.vizQA` screenshots and browser-state output.
---

# vizQA

Use this skill for using the `vizQA` CLI and authoring `vizQA` test cases.

## What vizQA Is

`vizQA` is a vision-driven UI testing tool.

Important user-facing assumptions:
- It is not an LLM-based test runner.
- It is meant for repeatable, idempotent test cases.
- Tests are written in YAML with natural-language `action` and optional `expect`.

## Start Here

Read the smallest relevant doc first:

- For writing test YAMLs: [docs/test_cases.md](docs/test_cases.md)
- For test dependencies: [docs/test_dependencies.md](docs/test_dependencies.md)
- For the perception backend summary: [docs/ui_api_summary.md](docs/ui_api_summary.md)

## Running The CLI

Common commands:

- `vizqa tests/` to run all YAML tests in a directory
- `vizqa path/to/test.yaml` to run one test
- `vizqa --no-headless path/to/test.yaml` to watch the browser
- `vizqa --clean-cache tests/` to clear cached dependency state before a run

Use `--no-headless` when debugging visual issues or incorrect state transitions.

## Writing Good Test YAMLs

Write steps in simple, literal, "caveman English".

Good style:

- `Click the 'Sign In' button`
- `Type {username} into the username field`
- `A 'Request Return' modal should appear`

Prefer:

- one action per step
- short, concrete expectations
- visible UI text and labels
- local `artifacts` for placeholders used in that file

Avoid:

- abstract phrasing
- implementation details like DOM classes
- multi-action steps when one step will do

## Quoted Text

Quoted UI text is treated as the strongest exact-match anchor.

Use quotes when you want exact visible copy matched:

- buttons
- headings
- modals
- badges
- toasts

Examples:

- `Click the 'Continue with SSO' button`
- `A 'Consent required' modal should appear`

If exact matching becomes too strict because the UI wording varies, remove the quotes and use a simpler visible description.

## Artifacts

Artifacts let you reuse values in steps:

```yaml
artifacts:
  username: "analyst.user"
  password: "AnalystPass!23"
```

Then:

```yaml
- action: "Type {username} into the username field"
```

Prefer defining artifacts in the same file that uses them, even if a dependency also defines them.

## Dependencies

Use `requires` to run setup tests first:

```yaml
requires:
  - login
```

Important behavior:

- `requires` uses file stems, not filenames
- dependencies run before the requested test
- if a dependency fails, the requested test is skipped
- dependency browser state can be restored into the dependent test

## Debugging Failures

Check `.vizQA/` first.

This directory contains debugging output such as:

- before screenshots
- action screenshots
- verify or after screenshots
- cached browser state in `.vizQA/browser_states/`

When a test fails:

1. Read the failing YAML step literally.
2. Open the screenshots in `.vizQA/`.
3. Compare the YAML wording against what is actually visible.
4. If dependencies are involved, inspect whether the restored state matches what the child test expects.

For quoted-element issues, compare the exact quoted string in the YAML against the exact visible text in the screenshot.

## Practical Debugging Tips

- If a test is flaky, simplify the language before changing the test flow.
- If a dependent test starts on the wrong screen, try `--clean-cache` and inspect `.vizQA/browser_states/`.
- If an expectation is too broad, rewrite it to mention the visible thing that should appear.
- If a step fails after a click, look at the action screenshot and the following verify screenshot together.
