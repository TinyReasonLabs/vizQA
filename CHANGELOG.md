# Changelog

All notable changes to **vizQA** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **Breaking: CLI reporting system overhauled** - Terminal reporter now uses an event-driven architecture with structured reporting events (`TopLevelTestStartedEvent`, `SessionStartedEvent`, `StepStartedEvent`, `StepFinishedEvent`, `SessionBlockedEvent`, `SessionFinishedEvent`, `RunFinishedEvent`). Custom reporters must implement the new `handle(event)` interface. The reporting store, layout composition, and terminal rendering have been completely redesigned for better multi-viewport support and step-level granularity.
- CLI terminal output now displays shared test flows with moving viewport cursors, improved dependency section rendering with progress bars, and better visual hierarchy during test execution.
- Fixed a bug where Type actions were not cleared before typing on MacOS.
- Adapted interactive cli mode with new reporting flow.

## [0.2.0] - 2026-04-24

### Added

- Community docs: Code of Conduct, security policy, changelog, and pull request template.
- Weight metadata tracking under `vizQA/weights`, version-aware `vizqa --version` output, and non-blocking warnings when installed model weights do not match the package's expected weights revision.
- Test dependencies via `requires`, including dependency graph resolution, circular-dependency and missing-dependency validation, dependency-first execution, dependency skip behavior on failure, artifact inheritance, browser-state reuse, and top-level-only result summaries.
- Dependency-system documentation and a general YAML test authoring guide under `docs/`.
- A local dependency auth lab fixture plus example dependency test flows covering password login, MFA, role elevation, checkout, returns, approvals, session resume, and simulated SSO.
- Custom viewport profiles and viewport-matrix execution for `vizqa run`, including built-in viewport presets, config-defined app viewports, raw `WIDTHxHEIGHT` CLI overrides, and per-viewport browser-state, artifact, and debug-log isolation.
- `${VAR}` environment-variable interpolation for YAML test files, including fail-fast validation for missing variables during test loading and dependency resolution.
- An embedded Python library API for attaching `vizQA` to an existing Playwright `Page`, including `attach`, `click`, `type`, `verify`, `run_step`, and `run_steps`, plus dedicated library documentation.

### Changed

- CLI runs now clear stale screenshots and browser-state caches for the requested tests and their dependency chain before execution, reducing artifact carryover between runs.
- CLI cleanup summaries are now shown only in verbose mode to keep default test output quieter.
- Dependency-related reporting is clearer during execution, including skip messaging and cleaner output for dependency-caused failures.
- Multi-viewport CLI reporting now renders a shared test flow with moving viewport cursors instead of duplicating the full step stream for each viewport lane.
- Library API usage is now artifact-light by default, keeping persistent screenshots only when `debug_dir` is explicitly provided.
- The small delay between interaction steps is now configurable through `VIZQA_STEP_DELAY_SECONDS` instead of being hardcoded.

## [0.1.0] - 2026-04-04

Initial published release line (alpha). See the [README](README.md) for features
and usage.

[Unreleased]: https://github.com/Spospider/vizQA/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/Spospider/vizQA/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Spospider/vizQA/releases/tag/v0.1.0
