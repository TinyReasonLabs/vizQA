.PHONY: run-demo run-all install

# Default target
all: run-demo

# Run the tool on the demo site using the auth_flow test
run-demo:
	poetry run python -m src.cli run tests/auth_flow.yaml -vv

# Run all test files in the tests directory
run-all:
	poetry run python -m src.cli run tests/

# Install dependencies via poetry
install:
	poetry install
	poetry run playwright install chromium

# Run a specific test file
# Usage: make run test=tests/your_test.yaml
run:
	poetry run python -m src.cli run $(test)
