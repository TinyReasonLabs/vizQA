# Library API

`vizQA` can be used as an embedded Python library inside an existing Playwright test, not only as a CLI runner for YAML files.

This mode is useful when you want to mix:

- normal DOM-based Playwright actions
- visual and semantic `vizQA` actions
- step-scoped assertions that read like user intent

The current library API is:

- async-first
- Playwright-first
- designed to attach to an existing `Page`
- non-owning with respect to the caller's browser lifecycle

`vizQA` uses the current page state you give it. It does not launch a new browser or close your page when attached through the library API.

## Installation and Prerequisites

Install the package and Playwright browsers:

```bash
pip install vizqa
vizqa install
```

You also need a running perception backend. For local development:

```bash
docker run -d -p 8228:8000 --name ui-atlas tinyreasonlabs/ui-atlas:latest
export PERCEPTION_BACKEND=localhost:8228
```

For general runtime settings such as perception backend, verification timeout, and step delay, see [configuration.md](configuration.md).

## Quick Start

Use the top-level helpers when you only need a few visual steps:

```python
from playwright.async_api import async_playwright
from vizQA import click, type, verify


async def smoke_check():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()

        await page.goto("https://example.com/login")

        await type(page, "email field", "analyst.user@example.com")
        await type(page, "password field", "AnalystPass!23")
        await click(page, "Sign in button")
        await verify(page, "Overview dashboard")

        await browser.close()
```

Use `attach(page)` when you want to reuse configuration and keep a stable `vizQA` session object through a test or page object.

```python
from vizQA import attach


async def login(page):
    vizqa = attach(page)

    await vizqa.type("email field", "analyst.user@example.com")
    await vizqa.type("password field", "AnalystPass!23")
    await vizqa.click("Sign in button")
    await vizqa.verify("Overview dashboard")
```

## Public API

### Attach a Session

```python
from vizQA import attach

session = attach(
    page,
    perception_backend="http://localhost:8228",
    verbosity=1,
)
```

Arguments:

- `page`: an existing Playwright `Page`
- `perception_backend`: optional override for the perception service URL
- `verbosity`: optional runtime verbosity for matching and failure details. This is library-only and is independent from the CLI terminal reporter modes such as `--silent` or `--debug-log`.
- `debug_dir`: optional directory for persistent screenshots and debug artifacts

If `debug_dir` is omitted, library calls do not write persistent artifacts under `.vizQA/`. Temporary screenshots may still be created internally to talk to the perception backend, but they are not kept as user-facing artifacts.

### Top-Level Helpers

Available helpers:

```python
from vizQA import click, run_step, run_steps, type, verify
```

Supported calls:

```python
await click(page, "Continue button")
await type(page, "email field", "analyst.user@example.com")
await verify(page, "Success banner")
await run_step(page, "Click the 'Continue' button")
await run_steps(page, [
    "Click the 'Sign In' button",
    "Type 'analyst.user' into the username field",
    "Type 'AnalystPass!23' into the password field",
])
```

These helpers create a short-lived attached session around the provided `page`.

### Session Methods

The attached session mirrors the same operations:

```python
session = attach(page)

await session.click("Continue button")
await session.type("email field", "analyst.user@example.com")
await session.verify("Success banner")
await session.run_step("Click the 'Continue' button")
await session.run_steps([
    "Click the 'Sign In' button",
    "Type 'analyst.user' into the username field",
])
```

## Step Results

Library calls return a `StepResult` object with high-signal execution data.

Fields:

- `success`: whether the step completed successfully
- `instruction`: the instruction that was executed
- `matched_element`: the best matched visual element, when one was found
- `artifacts`: paths to screenshots captured during execution
- `duration`: step duration in seconds
- `raw`: additional status and metadata for debugging

Example:

```python
result = await session.click("Sign in button")

if result.success:
    print(result.matched_element)
    print(result.artifacts)
else:
    print(result.raw["failure_reason"])
```

When `debug_dir` is not set, `result.artifacts` is usually an empty dictionary.

`StepResult` is also truthy on success and falsy on failure, so it works naturally in assertions:

```python
result = await session.verify("Overview dashboard")
assert result, result.raw["failure_reason"]
```

## How It Behaves

When used as a library:

- `vizQA` acts on the current page state
- it does not automatically reset navigation
- it does not own browser startup
- it does not close your page when you call `session.close()`
- it can be mixed freely with normal Playwright selectors and assertions

This makes it a good fit for hybrid test suites where DOM selectors are still valuable, but some interactions are better expressed visually.

## Realistic Use Cases

### 1. Add a Visual Step in the Middle of a Page Object

This is the most common hybrid usage pattern.

```python
from vizQA import attach


class LoginPage:
    def __init__(self, page):
        self.page = page
        self.vizqa = attach(page)

    async def login(self, username: str, password: str) -> None:
        await self.page.get_by_label("Username").fill(username)
        await self.page.get_by_label("Password").fill(password)

        # Use a visual step when the button label or surrounding UI is more stable
        # than the DOM structure behind it.
        await self.vizqa.click("Sign in button")
        await self.vizqa.verify("Overview dashboard")
```

Use this when:

- the control is visually obvious but DOM selectors are brittle
- the page object already exists and you do not want to rewrite it
- you want a user-facing verification after a critical action

### 2. Verify a Modal, Toast, or Banner by Meaning

Some UI feedback is easier to express semantically than structurally.

```python
from vizQA import verify


async def test_invalid_login_shows_error(page):
    await page.goto("https://example.com/login")
    await page.get_by_label("Email").fill("wrong@example.com")
    await page.get_by_label("Password").fill("bad-password")
    await page.get_by_role("button", name="Sign in").click()

    await verify(page, "A red invalid credentials error should appear")
```

Use this when:

- the visual state matters, not just DOM presence
- toast markup is inconsistent across products
- user-facing wording is more stable than implementation details

### 3. Build a Short Embedded Flow Without YAML

If you already have a Playwright suite, the library API can cover a focused visual sub-flow without introducing a YAML test file.

```python
from vizQA import attach


async def test_checkout_entry(page):
    await page.goto("https://example.com/store")

    vizqa = attach(page)

    await vizqa.run_steps([
        "Click the featured product card",
        "Click the Add to Cart button",
        "Click the Checkout button",
    ])

    await vizqa.verify("Payment page")
```

Use this when:

- you want to incrementally adopt `vizQA`
- only part of the test benefits from visual execution
- keeping the flow in one Python test is more practical than extracting YAML

### 4. Capture Step-Scoped Debug Data

`StepResult` can be used for custom logging or richer failure messages in your own framework.

```python
session = attach(page, debug_dir=".vizqa-artifacts")
result = await session.run_step("Click the 'Continue to MFA' button")

assert result, result.raw["failure_reason"]
print(result.artifacts)
```

This is especially useful when building:

- custom pytest helpers
- internal test utilities
- CI diagnostics around flaky UI states

## Hybrid Testing Recommendations

The strongest pattern is usually not "all DOM" or "all visual". It is a deliberate mix.

Prefer Playwright selectors for:

- stable form fields with good labels
- exact URL and network assertions
- low-level state setup
- fast deterministic checks where the DOM contract is intentional

Prefer `vizQA` for:

- visually anchored buttons and menus
- modals, banners, toasts, and callouts
- user-facing assertions
- UI that changes DOM structure more often than appearance

A good hybrid step sequence often looks like this:

```python
await page.get_by_label("Email").fill("analyst.user@example.com")
await page.get_by_label("Password").fill("AnalystPass!23")
await session.click("Sign in button")
await session.verify("Overview dashboard")
```

## Limitations in the Current Version

Current library support is intentionally narrow:

- Playwright `Page` is the only supported attachment target
- the API is async-only
- persistent screenshot artifacts are opt-in through `debug_dir`
- the library API is built on the current runtime rather than a separate backend abstraction

That keeps adoption simple while leaving room for future desktop, mobile, or embedded backends.

## Relationship to YAML Tests

The library API does not replace YAML authoring. It complements it.

Choose YAML when:

- the flow should live as a reusable test artifact
- non-Python contributors need to read or edit flows
- dependency chains and artifact handoff are central to the scenario

Choose the library API when:

- you already have Playwright code
- you want to insert a few visual steps into an existing suite
- page objects or pytest fixtures are already your main abstraction

Both approaches use the same underlying planning and execution model.
