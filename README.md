# vizQA: Vision-Driven Web UI Testing Framework

[![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()
[![Code Coverage](https://img.shields.io/badge/coverage-55%25-orange.svg)]()

**vizQA** is a lightweight, next-generation UI testing framework that "sees" and interacts with your application like a human does. By combining Playwright's robust automation with advanced visual perception and semantic search, vizQA lets you write tests in natural language without brittle CSS selectors or XPath. It is not an LLM-based test runner: execution is rule-driven, CPU-friendly, and designed for repeatable, idempotent regression.

> [!IMPORTANT]
> **vizQA is currently in its early alpha stage.** We are actively developing and refining the framework. Feedback, bug reports, and contributions are highly encouraged and welcome! Please review our [CONTRIBUTING.md](CONTRIBUTING.md) to get started.


---

## 👁️ Why Vision-Driven?

Traditional UI testing relies on the underlying DOM structure. When a developer changes a class name or wraps an element in a new `div`, tests break—even if the UI looks identical to the user.

**vizQA changes the paradigm by focusing on the visual reality:**
- **Natural Language**: Write tests like "Click the login button" or "Verify the error message appears in red".
- **Visual Intelligence**: Understands visual rules such as **contrast** against background, **salience** (how much an element stands out), and **spatial semantics** ("the button at the top right").
- **Real-World Verification**: Verifies **reachability**, **visibility**, and detects **obstruction** (e.g., an element being covered by a modal or tooltip).
- **Semantic Understanding**: Finds elements based on intent and visual appearance, not just hidden attributes.

---

## Key Features

- **Natural Language Steps**: Define your test flow in simple YAML instructions.
- **Not LLM-Based**: Uses deterministic parsing, semantic matching, and ranking rather than live LLM calls during test execution.
- **Advanced Interactions**: Supports `click`, `hover`, `type`, `scroll`, and even `drag and drop`.
- **Visual Assertions**: Verify UI state, colors, positions, and visibility.
- **Artifact Variables**: Load strings, file contents, or paths as variables (e.g., `{user_name}`) for dynamic test data.
- **Test Dependencies**: Chain setup flows with `requires` and reuse artifacts/browser state across related tests.
- **Repeatable Test Runs**: Optimized for stable, idempotent YAML test cases that can be run consistently in CI and local workflows.
- **Lightweight & Fast**: CPU-only execution with a minimal **~250 MB** memory footprint and sub-second latency.
---

## 🚀 Getting Started

### Prerequisites

- **Python 3.11+**
- **Playwright**
- **UI Perception Backend**: vizQA requires a perception backend. You can run the official lightweight UI-Atlas docker image for fully local operation:
  ```bash
  docker run -d -p 8228:8000 --name ui-atlas tinyreasonlabs/ui-atlas:latest
  ```
  Learn more about [UI-Atlas on Dockerhub](https://hub.docker.com/r/tinyreasonlabs/ui-atlas)

  vizQA connects to the backend using the `PERCEPTION_BACKEND` environment variable (normalized to `http://…`; default `localhost:8228`, i.e. `http://localhost:8228`).
  - Examples:
    ```bash
    export PERCEPTION_BACKEND=localhost:8228   # bash / zsh
    ```

### Installation

```bash
pip install vizqa
vizqa install
```

`vizqa install` installs Playwright browser binaries and model weights under `vizQA/weights`, and refreshes the local weights metadata used by the CLI.

You can inspect both the package version and installed weights revision with:

```bash
vizqa --version
```

---

## 📝 Usage

### Define a Test (`login_test.yaml`)

```yaml
name: "User Login Flow"
url: "https://example.com/login"

steps:
  - action: "Type 'admin' into the username field"
    expect: "Username field contains 'admin'"
  - action: "Type 'password123' into the password field"
  - action: "Click the 'Login' button"
    expect: "dashboard"
```

YAML string values support environment-variable interpolation with `${VAR}`. If a referenced variable is not set, vizQA raises a test-definition error when loading the file.

For the full YAML format and authoring guide, see [docs/test_cases.md](docs/test_cases.md).

### Run the Test

```bash
# Run all YAML tests in a directory (recursively; .yaml / .yml)
vizqa tests/

# Equivalent explicit subcommand
vizqa run tests/
```

### Use As a Library

You can also embed `vizQA` inside an existing Playwright test and mix DOM-based and visual steps:

```python
from vizQA import attach


async def test_login(page):
    vizqa = attach(page)

    await page.get_by_label("Email").fill("analyst.user@example.com")
    await page.get_by_label("Password").fill("AnalystPass!23")
    await vizqa.click("Sign in button")
    await vizqa.verify("Overview dashboard")
```

Library usage is artifact-light by default. If you want persistent screenshots for debugging, pass `debug_dir=...` when attaching.

For the full library guide, see [docs/library_api.md](docs/library_api.md).

### CLI Arguments

| Argument | Description | Default |
| --- | --- | --- |
| `paths` | One or more paths to test files or directories. | Required for a run (omit to print help). |
| `--headless / --no-headless` | Run browser in headless mode. | `True` |
| `-v, --verbose` | Increase output verbosity (-v steps, -vv timing/detail). | `0` |
| `-x, --interactive` | Run in interactive mode; stop at the first failing test. | `False` |
| `--viewport` | Repeatable viewport profile name or raw `WIDTHxHEIGHT` size. | Configured app viewports or `desktop` |

---

## Configuration

vizQA supports global and per-test configuration, including custom HTTP headers for authentication or specialized testing.

For the broader configuration reference, including environment variables see [docs/configuration.md](docs/configuration.md).

### Global Configuration
You can define global headers and reusable viewport profiles in your `pyproject.toml` or a `.ini` file in the current working directory: `pytest.ini`, `tox.ini`, `setup.cfg`, or `vizqa.ini`.

**`pyproject.toml`**
```toml
[tool.vizqa.headers]
Authorization = "Bearer global-api-token"
X-Custom-Header = "GlobalValue"

[tool.vizqa.viewports]
app = "1280x720"
mobile = "390x844"
```

**`pytest.ini` / `tox.ini` / `setup.cfg` / `vizqa.ini`**
```ini
[vizqa.headers]
Authorization = Bearer global-api-token
X-Custom-Header = GlobalValue

[vizqa.viewports]
app = 1280x720
mobile = 390x844
```

Built-in viewport profile names:
- `mobile`
- `tablet`
- `desktop`
- `widescreen`

If you define viewport profiles for your app, `vizqa run ...` uses them by default when `--viewport` is omitted.

### Per-Test Overrides
You can specify or override headers directly in your test YAML file. These take precedence over global settings.

**`my_test.yaml`**
```yaml
name: "Protected API Test"
url: "https://example.com/api"
headers:
  Authorization: "Bearer test-specific-token"
steps:
  - action: "Open the dashboard"
    expect: "A welcome message should appear"
```

---

## Methodology

vizQA follows a three-stage execution cycle for every step:

1.  **Perception**: Takes a screenshot and sends it to the Perception API to identify all visual elements and their properties (bounds, text, color, state).
2.  **Planning**: Uses semantic matching to understand intent and internally breaks down high-level instructions into atomic `find`, `do`, and `verify` commands to handle complex interactions.
3.  **Execution**: Performs the interaction via Playwright using precise pixel coordinates, ensuring we interact exactly with what was "seen."

---

## Contributing

We welcome contributions! Please see our [CONTRIBUTING.md](CONTRIBUTING.md) for details on setting up the environment and submitting PRs.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
