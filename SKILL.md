---
name: vizqa
description: Use when writing or editing vizQA YAML tests, automating web UI interactions with vizQA, debugging failing runs, interpreting `.vizQA` artifacts, or embedding vizQA in Playwright.
---

# vizQA

## What vizQA Is

`vizQA` is a deterministic, vision-driven web UI automation tool.

It works by:

- taking screenshots
- matching visible text and UI state
- choosing the best visible target
- clicking, typing, scrolling, waiting, or verifying like a user would

It is not an LLM runner. Write natural-language steps, but keep them concrete and visible.

## Core Rule

If a human can see it, say it that way.

Prefer:

- button text
- headings
- labels
- toast messages
- table rows
- modal titles

Avoid:

- CSS classes
- DOM IDs
- implementation details
- hidden state

## Test File Shape

A vizQA test is a YAML file with these common fields:

- `name`: short test title
- `url`: starting page, `https://...` or `file:///...`
- `description`: optional note
- `headers`: optional HTTP headers for the test
- `artifacts`: reusable values referenced as `{name}`
- `requires`: setup tests by file stem
- `steps`: ordered user actions and checks

Strings can use environment interpolation like `${APP_URL}`.

### Minimal Example

```yaml
name: "User Login"
url: "https://example.com/login"

steps:
  - action: "Type 'admin' into the username field"
  - action: "Type 'password123' into the password field"
  - action: "Click the 'Sign in' button"
    expect: "The dashboard should appear"
```

## How To Write Steps

Each step is usually one of:

- an interaction
- a wait
- a scroll
- a visual verification

Use `expect` for the visible result after the action.

Good:

```yaml
- action: "Click the 'Continue with SSO' button"
  expect: "A 'Select identity provider' modal should appear"
```

Less good:

```yaml
- action: "Handle login stuff"
```

### Exact Text

Quoted UI text is the strongest anchor.

Use quotes for exact visible copy:

- `Click the 'Submit' button`
- `A 'Consent required' modal should appear`

If wording varies, remove the quotes and describe the visible thing more loosely.

### Pure Checks

If a step only checks state, write the condition as the action:

- `action: "The success toast should appear"`
- `action: "The submit button should be disabled"`
- `action: "The modal should close"`

## Supported Commands

Use plain language. These forms are the safest authoring choices:

- `Click the ...`
- `Tap the ...`
- `Hover the ...`
- `Type ... into the ...`
- `Enter ... in the ...`
- `Input ... into the ...`
- `Drag the ... onto the ...`
- `Drag and drop ... onto the ...`
- `Scroll to the ...`
- `Scroll to the bottom of the page`
- `Scroll to the top of the page`
- `Scroll down to the bottom of the page`
- `Wait for 5 seconds`
- `Wait for the success toast`

Notes:

- `Type`, `Enter`, and `Input` mean text entry into a field.
- `Tap` behaves like click.
- `Scroll to ...` is target-seeking scroll behavior, not a raw wheel action.
- `Wait for ...` can mean either a time wait or a visible-state wait.

Some parser synonyms exist, but do not rely on them in authoring. Use the safe forms above instead of extra action words.

## Semantic Commands

These two commands are especially important:

- `Scroll to ...` brings a visible target into view. Use it for sections, page-scope phrases like `top` and `bottom`, or stable labels.
- `Wait for ...` waits for a visible state or a time duration. Use it for toasts, tables, modals, loaders, async page changes, or a simple pause.

Examples:

```yaml
- action: "Scroll to the 'Debug controls' section"
  expect: "'Expire current session' button should be visible"

- action: "Scroll to the bottom of the page"

- action: "Wait for the success toast"
  expect: "The success toast should appear"

- action: "Wait for 5 seconds"
```

## Artifacts

Use artifacts for repeated values and files.

```yaml
artifacts:
  username: "analyst.user"
  password: "AnalystPass!23"
  upload_file:
    path: "fixtures/avatar.png"
```

Then reference them in steps:

```yaml
- action: "Type {username} into the username field"
- action: "Drag and drop {upload_file} onto the upload area"
```

Guidelines:

- define artifacts in the same file when possible
- prefer clear names like `admin_password` or `item_name`
- use file-backed artifacts for repeated fixture content

## Dependencies

Use `requires` for setup flows.

```yaml
requires:
  - login
  - checkout_setup
```

Rules:

- use file stems, not filenames
- dependencies run before the main test
- if a dependency fails, the dependent test is skipped
- dependency browser state can be restored into the child test

## CLI Use

Common commands:

- `vizqa tests/` runs all YAML tests in a directory
- `vizqa path/to/test.yaml` runs one test
- `vizqa --no-headless path/to/test.yaml` shows the browser
- `vizqa --clean-cache tests/` clears cached dependency state
- `vizqa run tests/ --viewport mobile --viewport desktop` runs multiple viewport sizes
- `vizqa --silent tests/` uses the compact reporter
- `vizqa --debug-log tests/` keeps the same UI and writes richer logs
- `vizqa -x tests/` stops at the first failure

Built-in viewport names:

- `mobile`
- `tablet`
- `desktop`
- `widescreen`

## Configuration

### Global Headers and Viewports

You can define shared headers and viewport profiles in `pyproject.toml` or an `.ini` file such as `pytest.ini`, `tox.ini`, `setup.cfg`, or `vizqa.ini`.

Example:

```toml
[tool.vizqa.headers]
Authorization = "Bearer global-api-token"

[tool.vizqa.viewports]
app = "1280x720"
mobile = "390x844"
```

Per-test `headers` override global headers.

### Runtime Environment Variables

Useful environment variables:

- `PERCEPTION_BACKEND`: perception service URL, usually `localhost:8228`
- `VIZQA_VERIFICATION_TIMEOUT`: how long visual verification waits
- `VIZQA_WAIT_FOR_TIMEOUT`: how long `wait for ...` waits before timing out
- `VIZQA_WAIT_FOR_POLL_INTERVAL`: polling cadence for `wait for ...`
- `VIZQA_SCROLL_CENTER_BAND_MIN` / `VIZQA_SCROLL_CENTER_BAND_MAX`: center band for `scroll to ...`
- `VIZQA_STEP_DELAY_SECONDS`: pause between interaction steps

Use shorter delays for stable local runs and longer delays when the UI animates or settles slowly.

## Library Use

If you already have a Playwright `page`, you can mix DOM automation with vizQA:

```python
from vizQA import attach

vizqa = attach(page)
await vizqa.click("Sign in button")
await vizqa.type("username field", "analyst.user@example.com")
await vizqa.verify("Dashboard")
```

Useful library functions:

- `attach(page)`
- `click(page, target)`
- `type(page, target, text)`
- `verify(page, target)`
- `run_step(...)`
- `run_steps(...)`

## Debugging

Check `.vizQA/` first.

Important artifacts:

- `*_before.jpg`
- `*_action.jpg`
- `*_verify.jpg`
- `.vizQA/browser_states/`

Debug flow:

1. Read the failing step literally.
2. Compare the YAML wording with what is visible in screenshots.
3. If dependency state looks wrong, inspect the restored browser state.
4. If the page looks right but the step fails, tighten the wording to the exact visible phrase.
5. If a flow is flaky, simplify the step and use a more specific visible anchor.

Use `--clean-cache` when a dependency run starts on the wrong screen.

## Writing Good Tests

Keep these habits:

- one main action per step
- short, visible expectations
- literal on-screen copy when possible
- local artifacts instead of repeated literals
- narrow tests that cover one flow
- stable text anchors over vague descriptions

Good patterns:

- login
- checkout
- returns
- approvals
- modal handling
- table loading
- responsive layout checks

## Common Mistakes

- using hidden implementation details instead of visible text
- making one step do too much
- forgetting to quote exact visible copy
- using `Wait for ...` when you really want a normal action
- using `Scroll to ...` when you only mean a fixed top/bottom scroll
- writing dependency filenames instead of file stems in `requires`
- expecting internal state instead of an observable UI change

## Mental Shortcut

Use this rule of thumb:

- if the user must click or type, write an interaction
- if the user must see it, write an `expect` or verification
- if the user must wait, use `Wait for ...`
- if the user must bring something into view, use `Scroll to ...`
