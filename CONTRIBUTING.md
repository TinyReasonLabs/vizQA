# Contributing to vizQA

First off, thank you for considering contributing to vizQA! It's people like you that make it a great tool for everyone.

---

## 🏗️ Development Setup

vizQA uses [Poetry](https://python-poetry.org/) for dependency management and packaging.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Spospider/vizQA.git
    cd vizQA
    ```

2.  **Install dependencies:**
    ```bash
    poetry install
    ```

3.  **Install Playwright browsers:**
    ```bash
    poetry run playwright install chromium
    ```

4.  **Set up Pre-commit hooks:**
    ```bash
    poetry run pre-commit install
    ```

---

## 🧪 Running Tests

We use `pytest` for our test suite. **All pull requests must include new tests for added functionality or bug fixes.**

```bash
# Run all tests
poetry run pytest

# Run tests with coverage
poetry run pytest --cov=vizQA
```

---

## ⚖️ Code Style & Architecture

### Architectural Principles
To maintain a high-quality, lightweight codebase, we strictly adhere to a **Separation of Concerns**. When contributing new code, ensure that:
- **Perception** logic (visual analysis) is strictly isolated from **Planning** (semantic understanding).
- **Execution** (Playwright interaction) only depends on the output of the Planner.
- Core utilities remain browser-agnostic where possible.

### Style Guidelines
We follow PEP 8 and use several tools through **pre-commit** hooks:
- **Black**: For consistent code formatting (120 code length).
- **Isort**: For sorting imports.
- **Pylint**: For comprehensive linting and code analysis.

To install pre-commit hooks:
```bash
poetry run pre-commit install
```

## 🤖 Automated Checks

Every Pull Request triggers automated GitHub Actions to ensure stability and quality:
- **Pytest**: Full test suite execution on multiple environments.
- **Linting**: Automated Black, Isort, and Pylint checks.
- **Snyk**: Security vulnerability scans on dependencies and code.
- **Coverage**: Automated reporting on code coverage.

**PRs cannot be merged until all checks are green.**

## 🚀 Submitting a Pull Request

1.  Fork the repo and create your branch from `main`.
2.  **Add tests for your changes.** PRs without tests will be requested to add them before review.
3.  If you've changed APIs, update the documentation.
4.  Ensure the test suite passes locally.
5.  Check your code lints via `pre-commit run --all-files`.
6.  Issue that pull request!

---

## 📝 Pull Request Guidelines

-   **Description**: Provide a clear and concise description of the changes.
-   **Tasks**: Use a checklist for the tasks completed.
-   **Screenshots**: If the change involves UI or reporting, include screenshots.
-   **Linked Issues**: Reference any related issues (e.g., `Closes #123`).

---

## 📄 License

By contributing to vizQA, you agree that your contributions will be licensed under its [MIT License](LICENSE).
