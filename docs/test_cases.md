# Writing Test YAMLs

This guide is the main reference for authoring `vizQA` test files.

## Deterministic YAML v2

New tests should use `schema: 2`. Each structured step is a single explicit
operation; vizQA does not parse its wording or use MiniLM for that step.
`target`, `source`, and
`option` are visible UI queries sent unchanged to the UI Perception API—never
CSS selectors, DOM IDs, or test IDs.

```yaml
schema: 2
name: "Password Login"
url: "https://example.com/login"

steps:
  - click:
      target: "Sign In button"
  - type:
      target: "username field"
      text: "{username}"
  - press_key:
      key: "Enter"
  - assert_visible:
      target: "Dashboard"
      timeout: 5
```

Supported operations are `click`, `right_click`, `hover`, `type`, `clear`,
`press_key`, `check`, `uncheck`, `select`, `drag`, `upload`, `scroll`, `wait`,
`assert_visible`, and `assert_not_visible`. Actions that need a visible control
use `target`; `drag` uses `source` and `target`; `select` uses `target` and
`option`; `upload` uses an artifact `file`; `wait` uses exactly one of `seconds`
or `target`; and `scroll` uses a `target` or `position: top|bottom`.

The highest-ranked acceptable Perception result is used. If no result is
returned, the step fails with the visible candidates recorded in the run
artifacts. Legacy natural-language `action`/`expect` files remain supported.
They may also be mixed with structured steps in a `schema: 2` test; only those
legacy steps use the semantic parser.
Set `VIZQA_PERCEPTION_MATCH_THRESHOLD` to tune the minimum accepted ranked
similarity (default: `0.5`).

### Editor autocomplete

The versioned schema is at
[`schemas/vizqa-test.schema.json`](../schemas/vizqa-test.schema.json). With the
VS Code YAML extension, map it to your suite in workspace settings:

```json
{
  "yaml.schemas": {
    "./schemas/vizqa-test.schema.json": ["tests/**/*.yaml", "tests/**/*.yml"]
  }
}
```

Alternatively, add this first line to an individual file for yaml-language-server:

```yaml
# yaml-language-server: $schema=https://raw.githubusercontent.com/Spospider/vizQA/main/schemas/vizqa-test.schema.json
```

For pre-requisite-specific behavior, see [test_dependencies.md](test_dependencies.md).

## Legacy YAML v1

The remainder of this guide documents the supported legacy natural-language
format. Use it for existing suites; prefer deterministic YAML v2 for new tests.

## Minimal Example

```yaml
name: "User Login"
url: "https://example.com/login"

steps:
  - action: "Type 'admin' into the username field"
  - action: "Type 'password123' into the password field"
  - action: "Click the 'Sign in' button"
    expect: "The dashboard should load"
```

## File Structure

A test file is a YAML document with these top-level fields:

```yaml
name: "Human-readable test name"
url: "https://example.com/page"
description: "Optional description"

headers:
  Authorization: "Bearer token"

artifacts:
  username: "analyst.user"
  upload_file:
    path: "fixtures/avatar.png"
  payload:
    file: "fixtures/request.json"

requires:
  - login_setup

steps:
  - action: "Click the 'Settings' button"
    expect: "The settings page should load"
```

## Top-Level Fields

### `name`

- Required in practice.
- Used in reporting output.
- Keep it short and task-oriented.

Example:

```yaml
name: "Checkout Flow"
```

### `url`

- The page to open when the test starts.
- Can be `https://...` or `file:///...`.
- For pre-requisite-driven tests, the initial navigation may be skipped if state is restored from a pre-requisite.

Example:

```yaml
url: "https://example.com/dashboard"
```

### `description`

- Optional.
- Useful for open-source examples and longer-lived suites.

### Environment Variables

- Any YAML string value can reference an environment variable with `${VAR}`.
- Interpolation happens when vizQA loads the test file, before pre-requisite resolution and execution.
- If a referenced variable is not set, vizQA raises a `TestDefinitionError`.

Example:

```yaml
url: "${APP_URL}"
headers:
  Authorization: "Bearer ${API_TOKEN}"
steps:
  - action: "Open ${LANDING_PAGE}"
```

### `headers`

- Optional.
- Adds HTTP headers for this test.
- Test-level headers override global headers loaded from config.

Example:

```yaml
headers:
  Authorization: "Bearer test-token"
  X-Environment: "staging"
```

### `artifacts`

- Optional.
- Defines reusable values referenced as `{name}` in steps.
- Good for credentials, file paths, fixture content, and repeated labels.

Supported forms:

```yaml
artifacts:
  username: "analyst.user"         # simple string
  fixture_text:
    file: "fixtures/message.txt"   # loads file contents
  upload_file:
    path: "fixtures/avatar.png"    # resolves to an absolute file path
```

Recommendations:

- Define artifacts in the same file when the test uses them.
- Prefer artifacts over repeating long literals in multiple steps.
- Use descriptive names like `admin_password`, `item_name`, `resume_view`.

### `requires`

- Optional.
- Lists pre-requisite tests by file stem.
- Pre-requisites must live in the same directory as the current test.

Example:

```yaml
requires:
  - login
  - checkout_setup
```

See [test_dependencies.md](test_dependencies.md) for pre-requisite resolution and state handoff details.

### `steps`

- Required.
- Ordered list of user-facing actions and optional expectations.
- Each item should represent one meaningful interaction.

Basic form:

```yaml
steps:
  - action: "Click the 'Profile' button"
  - action: "Type {username} into the email field"
    expect: "The email field should contain 'analyst.user'"
```

## Writing Steps

Each step supports:

- `action`: required natural-language instruction
- `expect`: optional natural-language verification

vizQA parses steps into internal `find`, `do`, and `verify` operations. You do not need to write those internal forms directly in normal test YAMLs.

### Good Step Style

- Use visible UI language: button text, labels, headings, badges, toasts.
- Be specific enough to ground visually.
- Keep one main action per step.
- Put the user-visible outcome in `expect`.

Good:

```yaml
- action: "Click the 'Continue with SSO' button"
  expect: "A 'Select identity provider' modal should appear"
```

Less good:

```yaml
- action: "Handle login stuff"
```

### Common Action Patterns

Typing:

```yaml
- action: "Type {username} into the username field"
- action: "Type 'Nebula' into the search box"
```

Clicking:

```yaml
- action: "Click the 'Submit' button"
- action: "Click on 'Orders' in the navigation menu"
```

Keyboard:

```yaml
- action: "Press key Enter"
- action: "Press keys Ctrl+C"
- action: "Press key '+'"
```

Use `Press key ...` or `Press keys ...` for keyboard input. Bare phrases such as `Press the 'Submit' button` remain semantic UI-control actions and are resolved visually.

Scrolling:

```yaml
- action: "Scroll to the 'Debug controls' section"
  expect: "'Expire current session' button should be visible"
- action: "Scroll to the bottom of the page"
```

Waiting:

```yaml
- action: "Wait for the 'Recent Deals' table to load"
  expect: "Should see Acme Corp and Nebula Inc in the table"
- action: "Wait for the success toast"
  expect: "The success toast should appear"
- action: "Wait for 5 seconds"
```

Drag and drop:

```yaml
- action: "Drag and drop {upload_file} onto the drop zone"
  expect: "The file should be shown as uploaded"
```

State verification:

```yaml
- action: "Click the theme toggle button"
  expect: "The UI should switch to a light background"
```

## Expectations

Use `expect` for visible outcomes:

- page or modal appears
- text becomes visible
- table contents change
- button becomes disabled or enabled
- error toast appears
- route-specific UI is restored

Good expectations are:

- concrete
- observable
- phrased in user terms

Good:

```yaml
expect: "A red error toast should appear saying invalid credentials"
```

Less good:

```yaml
expect: "The internal state should update correctly"
```

## Authoring Tips

### Prefer stable visible anchors

Use text a user can actually see:

- `"Click the 'Sign In' button"`
- `"A 'Request Return' modal should appear"`

Avoid implementation details like CSS classes or DOM IDs in test YAMLs.

### Keep tests narrow

One test should usually cover one flow:

- login
- checkout
- returns
- manager approval

Split broad scenarios into pre-requisites when state handoff matters.

### Keep expectations readable

If an expectation becomes long, break the flow into more steps instead of packing multiple assertions into one sentence.

### Use direct scroll, wait, and keyboard commands

`vizQA` understands common direct commands in test actions:

- `Scroll to ...` for target-seeking scrolling, including section names and page-scope phrases like `top` and `bottom`
- `Wait for ...` for both time-based pauses and semantic UI conditions such as a toast, table, modal, or loader state
- `Press key ...` for targetless keyboard input using Playwright-compatible key names such as `Enter`, `ArrowLeft`, `KeyA`, `Digit1`, or shortcuts like `Ctrl+C`

Prefer these forms when the action is really about reaching a visible target or waiting for a visible state, rather than describing lower-level mechanics.

### Use artifacts for repeated values

Better:

```yaml
artifacts:
  item_name: "Widget Pro Secure Key"
```

Then:

```yaml
- action: "Search for {item_name} in the search field"
```

### Make pre-requisite tests self-describing

Even though pre-requisite artifacts can be inherited, local artifacts are usually better for readability and maintenance.

## Practical Examples

### Login Test

```yaml
name: "Password Login"
url: "https://example.com/login"

artifacts:
  username: "analyst.user"
  password: "AnalystPass!23"

steps:
  - action: "Type {username} into the username field"
    expect: "The username field should contain 'analyst.user'"
  - action: "Type {password} into the password field"
    expect: "The password field should be filled"
  - action: "Click the 'Continue to MFA' button"
    expect: "The 'Complete Multi-factor Authentication' modal should appear"
```

### File Upload Test

```yaml
name: "Avatar Upload"
url: "https://example.com/settings"

artifacts:
  upload_file:
    path: "fixtures/avatar.png"

steps:
  - action: "Drag and drop {upload_file} onto the upload area"
    expect: "The uploaded file should be shown in the file list"
```

### Test With Headers

```yaml
name: "Protected Dashboard"
url: "https://example.com/dashboard"

headers:
  Authorization: "Bearer test-token"

steps:
  - action: "Wait for the dashboard to load"
    expect: "A welcome message should appear"
```

## Common Mistakes

- Using a dependency filename like `login.yaml` instead of the stem `login`
- Referencing `{artifacts}` that are not defined
- Writing steps that rely on hidden implementation details
- Packing multiple unrelated actions into one step
- Using expectations that are not visually verifiable

## See Also

- [test_dependencies.md](test_dependencies.md)
- [README.md](../README.md)
