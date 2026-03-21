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

We use `pytest` for our test suite.

```bash
# Run all tests
poetry run pytest

# Run tests with coverage
poetry run pytest --cov=vizQA
```

Before submitting a pull request, please ensure all tests are passing and that you've added tests for any new functionality.

---

## 🎨 Code Style

We follow PEP 8 and use several tools through **pre-commit** hooks to maintain code quality:

- **Black**: For consistent code formatting (120 character line length).
- **Isort**: For sorting imports.
- **Pylint**: For comprehensive linting and code analysis.
- **Pre-commit Hooks**: Includes checks for trailing whitespace, end-of-file, and large files.

### Installing Pre-commit Hooks

To ensure your code matches our style guidelines, please install the pre-commit hooks:

```bash
poetry run pre-commit install
```

Once installed, these tools will run automatically every time you commit. You can also run them manually on all files:

```bash
poetry run pre-commit run --all-files
```

---

## 🚀 Submitting a Pull Request

1.  Fork the repo and create your branch from `main`.
2.  If you've added code that should be tested, add tests.
3.  If you've changed APIs, update the documentation.
4.  Ensure the test suite passes.
5.  Make sure your code lints.
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
