# vizQA: Vision-Driven Web UI Testing Framework

[![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()
[![Code Coverage](https://img.shields.io/badge/coverage-55%25-orange.svg)]()

**vizQA** is a lightweight, next-generation UI testing framework that "sees" and interacts with your application like a human does. By combining Playwright's robust automation with advanced visual perception and semantic search, vizQA allows you to write tests in natural language without brittle CSS selectors or XPath—all running efficiently on **CPU** without the need for LLMs.

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
- **Advanced Interactions**: Supports `click`, `hover`, `type`, `scroll`, and even `drag and drop`.
- **Visual Assertions**: Verify UI state, colors, positions, and visibility.
- **Artifact Variables**: Load strings, file contents, or paths as variables (e.g., `{user_name}`) for dynamic test data.
- **Lightweight & Fast**: CPU-only execution with a minimal **~250 MB** memory footprint and sub-second latency.

## 🚀 Getting Started

### Prerequisites

- **Python 3.11+**
- **Playwright**
- **UI Perception Backend**: vizQA requires a perception backend. You can run the official lightweight UI-Atlas docker image for cully local operation:
  ```bash
  docker run -d -p 8228:8000 --name ui-atlas tinyreasonlabs/ui-atlas:latest
  ```
  Learn more about [UI-Atlas on Dockerhub](https://hub.docker.com/r/tinyreasonlabs/ui-atlas)

  vizQA connects to the backend using the `PERCEPTION_BACKEND` environment variable:
  - Default: `localhost:8228`
  - Example:
    ```bash
    export PERCEPTION_BACKEND=localhost:8228
    ```

### Installation

```bash
pip install vizqa
vizqa install
```

---

## 📝 Usage

### Define a Test (`login_test.yaml`)

```yaml
name: "User Login Flow"
url: "https://example.com/login"
description: "Verify that a user can log in with valid credentials."

steps:
  - action: "Type 'admin' into the username field"
    expect: "The username field should contain 'admin'"
  - action: "Type 'password123' into the password field"
  - action: "Click the 'Login' button"
    expect: "Should redirect to the dashboard and show 'Welcome back!'"
```

### Run the Test

```bash
# Run all tests in a directory
vizqa tests/
```

### CLI Arguments

| Argument | Description | Default |
| --- | --- | --- |
| `paths` | One or more paths to test files or directories. | (Required) |
| `--headless / --no-headless` | Run browser in headless mode. | `True` |
| `-v, --verbose` | Increase output verbosity (-v, -vv). | `0` |
| `-x, --interactive` | Stop and enter interactive mode on first failure. | `False` |
| `--report` | Generate a visual HTML report after execution. | `False` |

---

## Configuration

vizQA supports global and per-test configuration, including custom HTTP headers for authentication or specialized testing.

### Global Configuration
You can define global headers in your `pyproject.toml` or any common `.ini` file (e.g., `pytest.ini`, `tox.ini`).

**`pyproject.toml`**
```toml
[tool.vizqa.headers]
Authorization = "Bearer global-api-token"
X-Custom-Header = "GlobalValue"
```

**`pytest.ini` / `vizqa.ini`**
```ini
[vizqa.headers]
Authorization = Bearer global-api-token
X-Custom-Header = GlobalValue
```

### Per-Test Overrides
You can specify or override headers directly in your test YAML file. These take precedence over global settings.

**`my_test.yaml`**
```yaml
name: "Protected API Test"
url: "https://example.com/api"
headers:
  Authorization: "Bearer test-specific-token"
steps:
  - VERIFY: "Welcome" should appear
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
