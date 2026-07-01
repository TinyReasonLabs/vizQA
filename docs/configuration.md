# Configuration

This document collects the main global configuration knobs used by `vizQA`.

For YAML authoring, see [test_cases.md](test_cases.md).
For embedded Playwright usage, see [library_api.md](library_api.md).

## Overview

`vizQA` currently uses two broad configuration surfaces:

- file-based configuration for reusable project settings such as headers and viewport profiles
- environment variables for runtime tuning and service connectivity

## File-Based Configuration

You can define global headers and reusable viewport profiles in:

- `pyproject.toml`
- `pytest.ini`
- `tox.ini`
- `setup.cfg`
- `vizqa.ini`

### Headers

Use headers when your app needs shared authentication or environment-specific request metadata.

**`pyproject.toml`**

```toml
[tool.vizqa.headers]
Authorization = "Bearer global-api-token"
X-Custom-Header = "GlobalValue"
```

**`.ini` style**

```ini
[vizqa.headers]
Authorization = Bearer global-api-token
X-Custom-Header = GlobalValue
```

### Viewport Profiles

Viewport profiles define named browser sizes for CLI execution.

**`pyproject.toml`**

```toml
[tool.vizqa.viewports]
app = "1280x720"
mobile = "390x844"
```

**`.ini` style**

```ini
[vizqa.viewports]
app = 1280x720
mobile = 390x844
```

Built-in viewport names:

- `mobile`
- `tablet`
- `desktop`
- `widescreen`

## Environment Variables

### `PERCEPTION_BACKEND`

Controls the URL for the perception service used by `vizQA`.

Examples:

```bash
export PERCEPTION_BACKEND=localhost:8228
export PERCEPTION_BACKEND=http://localhost:8228
export PERCEPTION_BACKEND=https://perception.internal.example
```

If the scheme is omitted, `vizQA` normalizes it to `http://...`.

### `VIZQA_VERIFICATION_TIMEOUT`

Controls how long visual verification waits before timing out.

Example:

```bash
export VIZQA_VERIFICATION_TIMEOUT=8
```

Use this when:

- your UI updates slowly after an action
- remote environments have higher rendering latency
- you want faster local failures during development

### `VIZQA_WAIT_FOR_TIMEOUT`

Controls how long `wait for {elem}` polls before timing out.

Example:

```bash
export VIZQA_WAIT_FOR_TIMEOUT=120
```

Use this when:

- the target element may take time to appear
- your page needs more time after async work or network activity
- you are using semantic actions like `Wait for the success toast` or `Wait for the dashboard to load`

### `VIZQA_WAIT_FOR_POLL_INTERVAL`

Controls the polling cadence for `wait for {elem}`.

Example:

```bash
export VIZQA_WAIT_FOR_POLL_INTERVAL=1.0
```

Use a larger value when:

- you want fewer perception checks
- the UI is expensive to inspect repeatedly

Use a smaller value when:

- you want faster reactions to appearing elements
- the page changes quickly and predictably

This applies to semantic wait steps such as `Wait for the loader to disappear` as well as plain timed pauses.

### `VIZQA_SCROLL_CENTER_BAND_MIN` / `VIZQA_SCROLL_CENTER_BAND_MAX`

Control the vertical band used by `scroll to {elem}` to decide whether the target is close enough to the center of the viewport.

Example:

```bash
export VIZQA_SCROLL_CENTER_BAND_MIN=0.35
export VIZQA_SCROLL_CENTER_BAND_MAX=0.65
```

The defaults treat a target as centered when its vertical midpoint falls roughly in the middle 30% of the screen.

These settings affect target-seeking scroll actions like `Scroll to the 'Debug controls' section` and `Scroll to the bottom of the page`.

### `VIZQA_STEP_DELAY_SECONDS`

Controls the small delay between interaction steps.

Example:

```bash
export VIZQA_STEP_DELAY_SECONDS=0.2
```

Use a lower value when:

- you want faster local execution
- your UI is stable and immediate

Use a slightly higher value when:

- the UI depends on animations or transitions
- focus changes need a moment to settle
- flaky intermediate rendering states appear between actions

### Ranking and Matching Settings

Advanced parser and matching behavior can also be tuned through environment variables:

- `VIZQA_ADVANCED_RANKING`
- `VIZQA_INTENT_THRESHOLD`
- `VIZQA_ACTION_THRESHOLD`
- `VIZQA_SEMANTIC_THRESHOLD`

These are more specialized knobs and are most useful when tuning matching quality during framework development or troubleshooting.

## Recommended Defaults

Most teams should start with:

```bash
export PERCEPTION_BACKEND=localhost:8228
```

and only tune timing values if they observe:

- false verification timeouts
- interactions happening too quickly for the target UI
- unnecessary slowdown in local test execution
