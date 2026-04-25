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
